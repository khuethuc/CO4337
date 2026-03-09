import copy
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import flatten_tensors, unflatten_tensors


class EDLLoss(nn.Module):
    """
    Evidential Deep Learning loss:
      - MSE term on expected class probabilities
      - KL regularization on non-true classes
    lambda_t should be annealed in trainer over the first half of training,
    which matches the MURMURA paper's recommendation.
    """
    def __init__(self, num_classes: int, lambda_t: float = 0.0, device=None):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_t = float(lambda_t)
        self.device = device if device is not None else torch.device("cpu")

    def forward(self, evidence: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        p_hat = alpha / S

        y_true = F.one_hot(targets, num_classes=self.num_classes).float().to(evidence.device)

        mse = torch.sum((y_true - p_hat) ** 2, dim=1, keepdim=True)

        alpha_tilde = y_true + (1.0 - y_true) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)

        num_classes_t = torch.tensor(
            self.num_classes, dtype=torch.float32, device=evidence.device
        )

        kl_divergence = (
            torch.lgamma(S_tilde)
            - torch.lgamma(num_classes_t)
            - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
            + torch.sum(
                (alpha_tilde - 1.0)
                * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)),
                dim=1,
                keepdim=True,
            )
        )

        return torch.mean(mse + self.lambda_t * kl_divergence)


class EDL_NGC_sender:
    """
    Sender side:
      - loads neighbor model weights
      - computes cross-gradients on local batch
      - computes peer trust statistics on local batch:
          * mean epistemic uncertainty
          * accuracy
    Interface is kept compatible with trainer.py:
      returns (output_grads, output_stats, ref_buf)

    NOTE:
      For full MURMURA faithfulness, trust should ideally be computed on a local
      validation/holdout batch (not the train batch). This implementation keeps
      trainer compatibility and uses the provided batch as a proxy.
    """
    def __init__(self, true_model, device, num_classes: int = 10):
        self.model = copy.deepcopy(true_model).to(device)
        self.model.train()
        self.device = device
        self.num_classes = num_classes
        self.criterion = EDLLoss(num_classes=num_classes, lambda_t=0.0, device=device)
        self.gradient_buffer: Dict[str, torch.Tensor] = {}

    def _update_model(self, state_dict):
        for w, p in zip(state_dict, self.model.parameters()):
            p.data.copy_(w.data)

    def _clear_gradient_buffer(self):
        self.gradient_buffer = {}

    def _flatten_(self, G: Dict[str, torch.Tensor]) -> torch.Tensor:
        grad = [g for g in G.values()]
        return flatten_tensors(grad).to(self.device)

    @torch.no_grad()
    def _evaluate_peer_stats(self, x: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
        """
        Returns:
            mean epistemic uncertainty in [0, 1]
            accuracy in [0, 1]
        """
        self.model.eval()
        output = self.model(x)

        # More stable than ReLU for evidential modeling in practice.
        evidence = F.softplus(output)
        alpha = evidence + 1.0
        S = alpha.sum(dim=1)
        epistemic_uncertainty = (self.num_classes / S).mean().item()

        preds = torch.argmax(output, dim=1)
        acc = (preds == targets).float().mean().item()

        self.model.train()
        return epistemic_uncertainty, acc

    def _accumulate_gradients(self, x: torch.Tensor, targets: torch.Tensor):
        self.model.zero_grad(set_to_none=True)

        output = self.model(x)
        evidence = F.softplus(output)

        loss = self.criterion(evidence, targets)
        loss.backward()

        self._clear_gradient_buffer()
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.gradient_buffer[name] = param.grad.detach().clone()

        return self.gradient_buffer

    def __call__(self, neighbor_weight, batch_x, targets):
        output_grads = {}
        output_stats = {}
        ref_buf = None

        for rank, w in neighbor_weight.items():
            self._update_model(w)

            u_mean, acc = self._evaluate_peer_stats(batch_x, targets)
            gradients = self._accumulate_gradients(batch_x, targets)

            output_grads[rank] = self._flatten_(gradients)

            # Pack [uncertainty, accuracy] into one tensor so trainer.py does not need changes.
            output_stats[rank] = torch.tensor(
                [u_mean, acc], dtype=torch.float32, device=self.device
            )

            if ref_buf is None:
                ref_buf = gradients

        return output_grads, output_stats, ref_buf


class EDL_NGC_receiver:
    """
    Receiver side:
      - decodes received cross-gradients
      - decodes received peer trust stats [uncertainty, accuracy]
      - computes MURMURA-style trust score
      - applies hard filtering using adaptive threshold
      - aggregates trusted peer cross-gradients
      - mixes model-variant and data-variant branches using NGC alpha

    This is a hybrid:
      - trust/filtering logic from MURMURA
      - comp/data gradient fusion from NGC
    """
    def __init__(
        self,
        model,
        device,
        rank,
        lr,
        momentum,
        qgm,
        nesterov=True,
        weight_decay=0.0,
        neighbors=2,
        alpha=0.5,
        # MURMURA-style parameters
        self_weight=0.5,          # omega
        acc_weight=0.5,           # wa
        uncertainty_threshold=0.7,  # tau_u
        init_threshold=0.3,       # tau_min^(0)
        tight_coef=0.5,           # gamma_tau
        tight_speed=1.0,          # kappa
    ):
        self.model = model
        self.rank = rank
        self.device = device

        self.proj_grads: Dict[str, torch.Tensor] = {}

        self.alpha = float(alpha)
        self.self_weight = float(self_weight)
        self.acc_weight = float(acc_weight)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.init_threshold = float(init_threshold)
        self.tight_coef = float(tight_coef)
        self.tight_speed = float(tight_speed)

        self.momentum = momentum
        self.lr = lr
        self.nesterov = nesterov
        self.qgm = qgm
        self.weight_decay = weight_decay

        self.momentum_buff = []
        self.prev_params = []

        for param in self.model.module.parameters():
            self.momentum_buff.append(torch.zeros_like(param.data))
            self.prev_params.append(copy.deepcopy(param.data))

    def _unflatten_(self, flat_tensor: torch.Tensor, ref_buf: Dict[str, torch.Tensor]):
        ref = []
        keys = []
        for key, val in ref_buf.items():
            ref.append(val)
            keys.append(key)

        unflat_tensor = unflatten_tensors(flat_tensor, ref)
        X = {}
        for i, key in enumerate(keys):
            X[key] = unflat_tensor[i]
        return X

    def _adaptive_threshold(self, current_round: int, total_rounds: int) -> float:
        if total_rounds <= 0:
            return self.init_threshold
        t = float(current_round)
        T = float(total_rounds)
        tau_t = self.init_threshold * (
            1.0 - self.tight_coef * math.exp(-self.tight_speed * t / T)
        )
        return max(0.0, min(1.0, tau_t))

    def _compute_trust_score(self, stat_tensor: torch.Tensor) -> Tuple[float, float, float]:
        """
        stat_tensor = [mean_uncertainty, accuracy]
        Returns:
            s_final, u_mean, acc
        """
        u_mean = float(stat_tensor[0].item())
        acc = float(stat_tensor[1].item())

        # MURMURA-style base trust:
        # s_base = (1 - u_bar) * (wa * acc + (1 - wa))
        s_base = max(
            0.0,
            (1.0 - u_mean) * (self.acc_weight * acc + (1.0 - self.acc_weight)),
        )

        if u_mean > self.uncertainty_threshold:
            s_final = s_base * math.exp(-(u_mean - self.uncertainty_threshold))
        else:
            s_final = s_base

        return s_final, u_mean, acc

    def _filter_and_weight_peers(
        self,
        peer_stats: Dict[int, torch.Tensor],
        current_round: int,
        total_rounds: int,
    ):
        tau_t = self._adaptive_threshold(current_round, total_rounds)

        retained_scores = {}
        debug_info = {}

        for rank, stat in peer_stats.items():
            score, u_mean, acc = self._compute_trust_score(stat)
            debug_info[rank] = {
                "score": score,
                "u_mean": u_mean,
                "acc": acc,
                "tau_t": tau_t,
            }
            if score >= tau_t:
                retained_scores[rank] = score

        if len(retained_scores) == 0:
            return {}, [], debug_info

        score_sum = sum(retained_scores.values())
        weights = {r: retained_scores[r] / score_sum for r in retained_scores.keys()}
        retained_ranks = list(retained_scores.keys())
        return weights, retained_ranks, debug_info

    def __call__(
        self,
        neighbor_grads_comm,
        neighbor_grads_comp,
        comm_uncertainty,
        comp_uncertainty,
        ref_buf,
        current_round,
        total_rounds,
    ):
        # Unflatten received gradients
        for rank in list(neighbor_grads_comm.keys()):
            neighbor_grads_comm[rank] = self._unflatten_(neighbor_grads_comm[rank], ref_buf)
        for rank in list(neighbor_grads_comp.keys()):
            neighbor_grads_comp[rank] = self._unflatten_(neighbor_grads_comp[rank], ref_buf)

        # comm_uncertainty / comp_uncertainty actually carry [uncertainty, accuracy]
        weights_comp, ranks_comp, _ = self._filter_and_weight_peers(
            comp_uncertainty, current_round, total_rounds
        )
        weights_comm, ranks_comm, _ = self._filter_and_weight_peers(
            comm_uncertainty, current_round, total_rounds
        )

        for name, self_params in self.model.module.named_parameters():
            if not self_params.requires_grad or self_params.grad is None:
                continue

            local_grad = self_params.grad.data

            # Model-variant branch (comp)
            if len(ranks_comp) == 0:
                p_grads_comp = local_grad
            else:
                g_peers_comp = torch.zeros_like(local_grad)
                for rank in ranks_comp:
                    g_peers_comp.add_(neighbor_grads_comp[rank][name], alpha=weights_comp[rank])
                p_grads_comp = (
                    self.self_weight * local_grad
                    + (1.0 - self.self_weight) * g_peers_comp
                )

            # Data-variant branch (comm)
            if len(ranks_comm) == 0:
                p_grads_comm = local_grad
            else:
                g_peers_comm = torch.zeros_like(local_grad)
                for rank in ranks_comm:
                    g_peers_comm.add_(neighbor_grads_comm[rank][name], alpha=weights_comm[rank])
                p_grads_comm = (
                    self.self_weight * local_grad
                    + (1.0 - self.self_weight) * g_peers_comm
                )

            # NGC fusion
            self.proj_grads[name] = (
                (1.0 - self.alpha) * p_grads_comp + self.alpha * p_grads_comm
            )

    def project_gradients(self, lr):
        for name, p in self.model.module.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.proj_grads:
                continue

            p.grad.data = self.proj_grads[name].data

            if self.weight_decay != 0:
                p.grad.data.add_(p.data, alpha=self.weight_decay)

        if self.momentum != 0:
            if self.qgm:
                for p, p_prev, buf in zip(
                    self.model.module.parameters(),
                    self.prev_params,
                    self.momentum_buff,
                ):
                    buf.mul_(self.momentum).add_(
                        p_prev.data - p.data,
                        alpha=(1.0 - self.momentum) / self.lr,
                    )
                    mom_buff = copy.deepcopy(buf)
                    mom_buff.mul_(self.momentum).add_(p.grad.data)

                    if self.nesterov:
                        p.grad.data.add_(mom_buff, alpha=self.momentum)
                    else:
                        p.grad.data.copy_(mom_buff)

                for p, p_prev in zip(self.model.module.parameters(), self.prev_params):
                    p_prev.data.copy_(p.data)
            else:
                for p, buf in zip(self.model.module.parameters(), self.momentum_buff):
                    buf.mul_(self.momentum).add_(p.grad.data)

                    if self.nesterov:
                        p.grad.data.add_(buf, alpha=self.momentum)
                    else:
                        p.grad.data.copy_(buf)

        self.lr = lr