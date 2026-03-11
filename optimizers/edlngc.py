import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import flatten_tensors, unflatten_tensors


class EDLLoss(nn.Module):
    """
    Class-weighted Evidential Deep Learning loss.

    Expected input:
        evidence: non-negative tensor of shape [B, K]
        target:   integer class labels of shape [B]

    Notes:
    - Works even if class_weights is None.
    - If class_weights is None, batch-wise inverse-frequency weights are used.
    - lambda_t can be updated externally from trainer.
    """

    def __init__(
        self,
        num_classes,
        class_weights=None,
        lambda_t=0.0,
        eps=1e-8,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.lambda_t = float(lambda_t)
        self.eps = float(eps)

        if class_weights is not None:
            cw = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", cw)
        else:
            self.class_weights = None

    def _one_hot(self, target):
        return F.one_hot(target.long(), num_classes=self.num_classes).float()

    def _batch_inverse_frequency_weights(self, target, device):
        counts = torch.bincount(target.long(), minlength=self.num_classes).float().to(device)
        counts = torch.clamp(counts, min=1.0)
        weights = 1.0 / counts
        weights = weights / torch.clamp(weights.mean(), min=self.eps)
        return weights

    def _get_class_weights(self, target, device):
        if self.class_weights is not None:
            return self.class_weights.to(device)
        return self._batch_inverse_frequency_weights(target, device)

    def _kl_dirichlet(self, alpha_tilde):
        """
        KL( Dir(alpha_tilde) || Dir(1) )
        """
        beta = torch.ones((1, self.num_classes), device=alpha_tilde.device, dtype=alpha_tilde.dtype)

        sum_alpha = torch.sum(alpha_tilde, dim=1, keepdim=True)
        sum_beta = torch.sum(beta, dim=1, keepdim=True)

        lnB_alpha = torch.lgamma(sum_alpha) - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
        lnB_beta = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(sum_beta)

        digamma_sum_alpha = torch.digamma(sum_alpha)
        digamma_alpha = torch.digamma(alpha_tilde)

        kl = torch.sum((alpha_tilde - beta) * (digamma_alpha - digamma_sum_alpha), dim=1, keepdim=True)
        kl = kl + lnB_alpha + lnB_beta
        return kl.squeeze(1)

    def forward(self, evidence, target):
        """
        evidence: [B, K], non-negative
        target:   [B]
        """
        evidence = torch.clamp(evidence, min=0.0)
        target = target.long()

        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / torch.clamp(S, min=self.eps)

        y = self._one_hot(target)
        class_weights = self._get_class_weights(target, evidence.device)  # [K]
        class_weights = class_weights.view(1, -1)

        # Class-weighted squared error term
        sq_error = torch.sum(class_weights * (y - probs) ** 2, dim=1)

        # KL annealing term
        alpha_tilde = y + (1.0 - y) * alpha
        kl = self._kl_dirichlet(alpha_tilde)

        loss = sq_error + self.lambda_t * kl
        return loss.mean()


class EDL_NGC_sender:
    """
    Sender side for ENGC / EDL-NGC.

    For each neighbor model weight received from gossip:
      - load that model locally
      - compute gradient on current local batch
      - compute mean epistemic uncertainty on current local batch
      - return flattened gradients and scalar uncertainty per neighbor

    This matches trainer.py, which expects:
        cross_grad, cross_unc, ref_buf = sender(cross_weights, input_var, target_var)
    """

    def __init__(self, true_model, device, num_classes, class_weights=None):
        self.model = copy.deepcopy(true_model)
        self.model.train()
        self.model = self.model.to(device)

        self.gradient_buffer = {}
        self.device = device
        self.num_classes = int(num_classes)
        self.criterion = EDLLoss(
            num_classes=num_classes,
            class_weights=class_weights,
            lambda_t=0.0,
        ).to(device)

    def _update_model(self, state_dict):
        for w, p in zip(state_dict, self.model.parameters()):
            p.data.copy_(w.data)
        return

    def _clear_gradient_buffer(self):
        self.gradient_buffer = {}
        return

    def _flatten_(self, G):
        grad = []
        for g in G.values():
            grad.append(g)
        return flatten_tensors(grad).to(self.device)

    def _mean_epistemic_uncertainty(self, evidence):
        # u = K / S
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1)
        u = float(self.num_classes) / torch.clamp(S, min=1e-8)
        return u.mean()

    def _accumulate_gradients_and_uncertainty(self, x, targets):
        output = self.model(x)
        evidence = F.softplus(output)

        self.model.zero_grad()
        loss = self.criterion(evidence, targets)
        loss.backward()

        self._clear_gradient_buffer()
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.gradient_buffer[name] = param.grad.data.clone()

        mean_unc = self._mean_epistemic_uncertainty(evidence).detach()
        return self.gradient_buffer, mean_unc

    def __call__(self, neighbor_weight, batch_x, targets):
        output_grad = {}
        output_unc = {}
        ref_buf = None

        for rank, w in neighbor_weight.items():
            self._update_model(w)
            g, u = self._accumulate_gradients_and_uncertainty(batch_x, targets)
            output_grad[rank] = self._flatten_(g)
            output_unc[rank] = u.view(1).to(self.device)
            ref_buf = g

        return output_grad, output_unc, ref_buf


class EDL_NGC_receiver:
    """
    Receiver / aggregator for ENGC.

    Core idea:
    - evidential trust is the primary compatibility signal
    - gradient alignment is the auxiliary optimization-aware signal
    - peer weights are computed from a hybrid score
    - final gradient combines:
        self-gradient
        model-variant cross-gradients
        data-variant cross-gradients

    Compatible with trainer.py current call:
        receiver(
            recieved_cross_grad,
            cross_grad_copy,
            recieved_cross_uncertainty,
            cross_uncertainty_copy,
            ref_buf,
            current_round,
            total_rounds
        )
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
        alpha=1.0,
        beta=0.8,
        wa=0.5,
        tau_u=0.6,
        tau_min=0.0,
        gamma_tau=0.5,
        kappa=5.0,
        omega_min=0.4,
        omega_max=0.8,
        softmax_temp=5.0,
        num_classes=None,
        eps=1e-8,
    ):
        self.model = model
        self.rank = rank
        self.device = device

        self.proj_grads = {}
        self.pi = 1.0 / float(neighbors + 1)
        self.alpha = float(alpha)              # NGC mixing weight
        self.beta = float(beta)                # hybrid weight for evidential score
        self.wa = float(wa)                    # weight on local batch accuracy inside trust
        self.tau_u = float(tau_u)
        self.tau_min0 = float(tau_min)
        self.gamma_tau = float(gamma_tau)
        self.kappa = float(kappa)
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.softmax_temp = float(softmax_temp)

        self.momentum = momentum
        self.lr = lr
        self.nesterov = nesterov
        self.qgm = qgm
        self.weight_decay = weight_decay
        self.eps = float(eps)
        self.num_classes = num_classes

        self.momentum_buff = []
        self.prev_params = []
        for param in self.model.module.parameters():
            self.momentum_buff.append(torch.zeros_like(param.data))
            self.prev_params.append(copy.deepcopy(param.data))

        # cache for debugging / analysis
        self.last_peer_scores = {}
        self.last_peer_weights = {}
        self.last_self_weight = None
        self.last_tau_min = None

    def average_gradients(self, grad, weights=None):
        """
        Weighted sum of a list of tensors with same shape.
        If weights is None, uniform average is used.
        """
        if len(grad) == 0:
            raise ValueError("average_gradients received an empty list")

        out = torch.zeros_like(grad[0])

        if weights is None:
            coeff = 1.0 / float(len(grad))
            for g in grad:
                out.add_(g, alpha=coeff)
            return out

        w_sum = sum(float(w) for w in weights)
        if abs(w_sum) < self.eps:
            coeff = 1.0 / float(len(grad))
            for g in grad:
                out.add_(g, alpha=coeff)
            return out

        for g, w in zip(grad, weights):
            out.add_(g, alpha=float(w) / w_sum)
        return out

    def _unflatten_(self, flat_tensor, ref_buf):
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

    def _safe_scalar(self, x, default=0.0):
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return float(default)
            return float(x.detach().view(-1)[0].item())
        try:
            return float(x)
        except Exception:
            return float(default)

    def _adaptive_threshold(self, current_round, total_rounds):
        T = max(float(total_rounds), 1.0)
        return self.tau_min0 * (1.0 - self.gamma_tau * torch.exp(torch.tensor(-self.kappa * current_round / T, device=self.device)).item())

    def _cosine_nonneg(self, a, b):
        denom = torch.norm(a) * torch.norm(b) + self.eps
        if denom <= 0:
            return 0.0
        val = torch.dot(a, b) / denom
        return max(0.0, float(val.item()))

    def _normalize_peer_weights(self, score_dict, tau_min):
        """
        Soft weighting with optional thresholding.
        """
        kept = {}
        for r, s in score_dict.items():
            if s >= tau_min:
                kept[r] = s

        if len(kept) == 0:
            # fallback: if all are filtered out, use the unfiltered scores
            kept = {r: max(0.0, float(s)) for r, s in score_dict.items()}

        if len(kept) == 0:
            return {}

        score_tensor = torch.tensor(
            [self.softmax_temp * kept[r] for r in kept.keys()],
            device=self.device,
            dtype=torch.float32,
        )
        weight_tensor = torch.softmax(score_tensor, dim=0)

        out = {}
        for idx, r in enumerate(kept.keys()):
            out[r] = float(weight_tensor[idx].item())
        return out

    def _compute_local_self_uncertainty(self, local_peer_unc_dict):
        """
        In the current trainer interface, sender returns uncertainty only for neighbor-weight models.
        We do not directly receive local self uncertainty from trainer.

        Practical fallback:
        - use average peer uncertainty as a proxy
        - if no peer uncertainties exist, fall back to 0.5
        """
        if len(local_peer_unc_dict) == 0:
            return 0.5
        vals = [max(0.0, min(1.0, self._safe_scalar(v, 0.5))) for v in local_peer_unc_dict.values()]
        return float(sum(vals) / max(len(vals), 1))

    def __call__(
        self,
        neighbor_grads_comm,
        neighbor_grads_comp,
        neighbor_unc_comm,
        neighbor_unc_comp,
        ref_buf,
        current_round,
        total_rounds,
    ):
        """
        Inputs:
        - neighbor_grads_comm:
            gradients received after communication, interpreted as data-variant cross-gradients
        - neighbor_grads_comp:
            local cross-gradients computed from neighbor models on local batch,
            interpreted as model-variant cross-gradients
        - neighbor_unc_comm:
            uncertainty messages received from neighbors
        - neighbor_unc_comp:
            local uncertainty estimates computed while evaluating neighbor models locally
        """
        # Unflatten gradients first
        for rank, flat_tensor in neighbor_grads_comm.items():
            neighbor_grads_comm[rank] = self._unflatten_(flat_tensor, ref_buf)

        for rank, flat_tensor in neighbor_grads_comp.items():
            neighbor_grads_comp[rank] = self._unflatten_(flat_tensor, ref_buf)

        tau_min_t = self._adaptive_threshold(current_round, total_rounds)
        self.last_tau_min = tau_min_t

        # Self-weight from local uncertainty proxy
        u_local = self._compute_local_self_uncertainty(neighbor_unc_comp)
        omega_i = self.omega_min + (self.omega_max - self.omega_min) * (1.0 - u_local)
        omega_i = float(max(self.omega_min, min(self.omega_max, omega_i)))
        self.last_self_weight = omega_i

        # Compute per-peer scores
        peer_scores = {}
        peer_weights = {}

        # use parameter-wise weighted combination after global peer weighting
        for rank in neighbor_grads_comp.keys():
            # Evidential score
            u_comp = max(0.0, min(1.0, self._safe_scalar(neighbor_unc_comp.get(rank, 1.0), 1.0)))
            # use communicated uncertainty if available as a stabilizer
            u_comm = max(0.0, min(1.0, self._safe_scalar(neighbor_unc_comm.get(rank, u_comp), u_comp)))
            u_bar = 0.5 * (u_comp + u_comm)

            # approximate local-batch compatibility accuracy from model-variant gradient alignment
            # since trainer currently does not pass a_ij explicitly
            if rank in neighbor_grads_comp:
                flat_self = flatten_tensors(
                    [p.grad.data for p in self.model.module.parameters() if p.requires_grad]
                ).to(self.device)
                flat_mv = flatten_tensors(
                    [neighbor_grads_comp[rank][name] for name, p in self.model.module.named_parameters() if p.requires_grad]
                ).to(self.device)
                acc_proxy = 0.5 * (1.0 + self._cosine_nonneg(flat_self, flat_mv))
                acc_proxy = max(0.0, min(1.0, acc_proxy))
            else:
                acc_proxy = 0.5

            s_base = (1.0 - u_bar) * (self.wa * acc_proxy + (1.0 - self.wa))
            if u_bar > self.tau_u:
                s_ev = s_base * float(torch.exp(torch.tensor(-(u_bar - self.tau_u), device=self.device)).item())
            else:
                s_ev = s_base

            # Gradient score
            if rank in neighbor_grads_comp:
                c_mv = self._cosine_nonneg(
                    flat_self,
                    flatten_tensors(
                        [neighbor_grads_comp[rank][name] for name, p in self.model.module.named_parameters() if p.requires_grad]
                    ).to(self.device),
                )
            else:
                c_mv = 0.0

            if rank in neighbor_grads_comm:
                c_dv = self._cosine_nonneg(
                    flat_self,
                    flatten_tensors(
                        [neighbor_grads_comm[rank][name] for name, p in self.model.module.named_parameters() if p.requires_grad]
                    ).to(self.device),
                )
            else:
                c_dv = 0.0

            # time-dependent mixing for mv/dv can reuse alpha as a constant for now
            s_grad = (1.0 - self.alpha) * c_mv + self.alpha * c_dv

            # Hybrid score
            # multiplicative form, but guarded against zeros
            s_ev_safe = max(s_ev, self.eps)
            s_grad_safe = max(s_grad, self.eps)
            s_hybrid = (s_ev_safe ** self.beta) * (s_grad_safe ** (1.0 - self.beta))

            peer_scores[rank] = {
                "u_comp": u_comp,
                "u_comm": u_comm,
                "u_bar": u_bar,
                "acc_proxy": acc_proxy,
                "s_base": s_base,
                "s_ev": s_ev,
                "c_mv": c_mv,
                "c_dv": c_dv,
                "s_grad": s_grad,
                "s_hybrid": s_hybrid,
            }

        # Normalize peer weights from hybrid scores
        hybrid_score_dict = {r: peer_scores[r]["s_hybrid"] for r in peer_scores.keys()}
        peer_weights = self._normalize_peer_weights(hybrid_score_dict, tau_min_t)

        self.last_peer_scores = peer_scores
        self.last_peer_weights = peer_weights

        # Final parameter-wise aggregation
        for name, self_param in self.model.module.named_parameters():
            if not self_param.requires_grad or self_param.grad is None:
                continue

            self_grad = self_param.grad.data

            mv_list = []
            mv_w = []
            for rank, neigh_grad in neighbor_grads_comp.items():
                if rank in peer_weights:
                    mv_list.append(neigh_grad[name])
                    mv_w.append(peer_weights[rank])

            dv_list = []
            dv_w = []
            for rank, neigh_grad in neighbor_grads_comm.items():
                if rank in peer_weights:
                    dv_list.append(neigh_grad[name])
                    dv_w.append(peer_weights[rank])

            if len(mv_list) > 0:
                p_grad_mv = self.average_gradients(mv_list, mv_w)
            else:
                p_grad_mv = torch.zeros_like(self_grad)

            if len(dv_list) > 0:
                p_grad_dv = self.average_gradients(dv_list, dv_w)
            else:
                p_grad_dv = torch.zeros_like(self_grad)

            peer_grad = (1.0 - self.alpha) * p_grad_mv + self.alpha * p_grad_dv
            self.proj_grads[name] = omega_i * self_grad + (1.0 - omega_i) * peer_grad

        return

    def project_gradients(self, lr):
        """
        Applies the projected / aggregated gradients to the wrapped model,
        then applies weight decay and momentum exactly like the NGC baseline.
        """
        for name, p in self.model.module.named_parameters():
            if p.requires_grad:
                if name in self.proj_grads:
                    p.grad.data = self.proj_grads[name].data
                if self.weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=self.weight_decay)

        # momentum
        if self.momentum != 0:
            if self.qgm:
                for p, p_prev, buf in zip(self.model.module.parameters(), self.prev_params, self.momentum_buff):
                    buf.mul_(self.momentum).add_(p_prev.data - p.data, alpha=(1.0 - self.momentum) / self.lr)
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