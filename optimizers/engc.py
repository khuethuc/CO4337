import torch
import copy
import math
import torch.nn.functional as F

from .utils import flatten_tensors, unflatten_tensors

class ENGC_sender():
    def __init__(self, true_model, device, num_classes: int = 10):
        self.model           = copy.deepcopy(true_model)
        self.model.train()
        self.model           = self.model.to(device)
        self.gradient_buffer = {}
        self.device          = device
        self.num_classes     = num_classes
        self.vacuity_scores       = {}
        self.sample_weight_stats  = {}
        self._last_sample_weights = None
        self._last_vacuity        = None
        # Diagnostic: vacuity and weight split by correct vs wrong predictions.
        self.vac_by_pred          = {}
        self._last_vac_by_pred    = {}

    def _update_model(self, state_dict):
        for w, p in zip(state_dict, self.model.parameters()):
            p.data.copy_(w.data)

    def _accumulate_gradients(self, x, targets):
        """
        Cross-gradient with per-sample entropy-based uncertainty weighting.
        """
        self.model.zero_grad()
        output = self.model(x)

        with torch.no_grad():
            K     = output.size(1)
            probs = F.softmax(output.detach(), dim=1)   # [B, K]
            # Shannon entropy-based sample weight
            entropy     = -(probs * (probs + 1e-8).log()).sum(dim=1)    # [B]
            uncertainty = (entropy / math.log(K)).clamp(0.0, 1.0)       # [B]
            w_raw  = (1.0 - uncertainty).clamp(min=0.05)                # [B]
            w_norm = w_raw / (w_raw.mean() + 1e-8)                      # [B]
            self._last_sample_weights = w_raw.cpu()
            self._last_vacuity        = uncertainty.cpu()
            # Diagnostic: split uncertainty and weight by correct vs wrong predictions
            preds        = output.detach().argmax(dim=1)
            correct_mask = (preds == targets)
            n_wrong      = int((~correct_mask).sum().item())
            n_total      = len(targets)
            self._last_vac_by_pred = {
                'vac_correct': uncertainty[correct_mask].mean().item() if correct_mask.any() else float('nan'),
                'vac_wrong': uncertainty[~correct_mask].mean().item() if n_wrong > 0 else float('nan'),
                'w_correct': w_norm[correct_mask].mean().item() if correct_mask.any() else float('nan'),
                'w_wrong': w_norm[~correct_mask].mean().item() if n_wrong > 0 else float('nan'),
                'frac_wrong': n_wrong / max(n_total, 1),
            }

        ce_per_sample = F.cross_entropy(output, targets, reduction='none')  # [B]
        loss = (ce_per_sample * w_norm).mean()
        loss.backward()

        self._clear_gradient_buffer()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.gradient_buffer[name] = param.grad.data.clone()
        return self.gradient_buffer

    def _clear_gradient_buffer(self):
        self.gradient_buffer = {}

    def _flatten_(self, G):
        grad = [g for g in G.values()]
        return flatten_tensors(grad).to(self.device)

    def __call__(self, neighbor_weight, batch_x, targets):
        self.vacuity_scores      = {}
        self.sample_weight_stats = {}
        self.vac_by_pred         = {}
        output = {}
        g      = None
        for rank, w in neighbor_weight.items():
            self._update_model(w)
            with torch.no_grad():
                logits = self.model(batch_x)
                K      = logits.size(1)
                probs  = F.softmax(logits, dim=1)
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
                self.vacuity_scores[rank] = (entropy / math.log(K)).mean().item()
            g = self._accumulate_gradients(batch_x, targets)
            if self._last_vacuity is not None:
                unc = self._last_vacuity            # uncertainty [B]
                sw  = self._last_sample_weights     # w_raw [B]
                w_n = sw / (sw.mean() + 1e-8)       # w_norm [B]
                self.sample_weight_stats[rank] = {
                    'uncertainty_mean': float(unc.mean()),
                    'uncertainty_std': float(unc.std()),
                    'weight_std': float(w_n.std()),
                    'frac_low': float((w_n < 0.5).float().mean()),
                }
            self.vac_by_pred[rank] = self._last_vac_by_pred
            output[rank] = self._flatten_(g)
        return output, g

class ENGC_receiver():
    def __init__(
        self,
        model,
        device,
        rank,
        lr,
        momentum,
        qgm,
        nesterov     = True,
        weight_decay = 0,
        neighbors    = 2,
        alpha        = 1.0,
    ):
        self.model        = model
        self.rank         = rank
        self.device       = device
        self.proj_grads   = {}
        self.pi           = 1.0 / float(neighbors + 1)
        self.alpha        = alpha
        self.momentum     = momentum
        self.lr           = lr
        self.nesterov     = nesterov
        self.qgm          = qgm
        self.weight_decay = weight_decay
        self.momentum_buff = []
        self.prev_params   = []
        for param in self.model.module.parameters():
            self.momentum_buff.append(torch.zeros_like(param.data))
            self.prev_params.append(copy.deepcopy(param.data))
        # Diagnostic: weights assigned to self and each neighbor
        self.last_neighbor_weights = {}

    def _compute_weights(self, neighbor_ranks):
        N                 = len(neighbor_ranks)
        pi_self           = 1.0 / (N + 1)
        w_uniform         = (N / (N + 1)) / N
        weights           = {r: w_uniform for r in neighbor_ranks}
        self.last_neighbor_weights = {'self': pi_self, **weights}
        return pi_self, weights

    def _weighted_average(self, self_grad, neighbor_grads_dict, pi_self, weights):
        result = pi_self * self_grad
        for r, grad in neighbor_grads_dict.items():
            result = result + weights[r] * grad
        return result

    def _unflatten_(self, flat_tensor, ref_buf):
        ref  = list(ref_buf.values())
        keys = list(ref_buf.keys())
        unflat = unflatten_tensors(flat_tensor, ref)
        return {k: v for k, v in zip(keys, unflat)}

    def __call__(self, neighbor_grads_comm, neighbor_grads_comp, ref_buf):
        for rank, ft in neighbor_grads_comm.items():
            neighbor_grads_comm[rank] = self._unflatten_(ft, ref_buf)
        for rank, ft in neighbor_grads_comp.items():
            neighbor_grads_comp[rank] = self._unflatten_(ft, ref_buf)

        pi_self, weights = self._compute_weights(list(neighbor_grads_comm.keys()))

        for name, self_params in self.model.module.named_parameters():
            if not self_params.requires_grad:
                continue
            self_grad     = self_params.grad.data
            comm_neighbor = {r: g[name] for r, g in neighbor_grads_comm.items()}
            comp_neighbor = {r: g[name] for r, g in neighbor_grads_comp.items()}

            p_comm = self._weighted_average(self_grad, comm_neighbor, pi_self, weights)
            p_comp = self._weighted_average(self_grad, comp_neighbor, pi_self, weights)
            self.proj_grads[name] = (1.0 - self.alpha) * p_comp + self.alpha * p_comm

    def project_gradients(self, lr):
        for name, p in self.model.module.named_parameters():
            if not p.requires_grad:
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
                for p, p_prev in zip(
                    self.model.module.parameters(), self.prev_params
                ):
                    p_prev.data.copy_(p.data)
            else:
                for p, buf in zip(
                    self.model.module.parameters(), self.momentum_buff
                ):
                    buf.mul_(self.momentum).add_(p.grad.data)
                    if self.nesterov:
                        p.grad.data.add_(buf, alpha=self.momentum)
                    else:
                        p.grad.data.copy_(buf)

        self.lr = lr


def log_vacuity_stats(rank: int, step: int, vacuity_scores: dict, sample_weight_stats: dict = None):
    N = len(vacuity_scores)
    if N == 0:
        return
    MIN_TRUST = 0.1
    pi_self = 1.0 / (N + 1)
    pi_nbr_tot = N / (N + 1)
    trusts = {r: max(MIN_TRUST, 1.0 - v) for r, v in vacuity_scores.items()}
    total = sum(trusts.values()) or 1.0
    weights = {r: (trusts[r] / total) * pi_nbr_tot for r in trusts}
    parts = [f"pi_self={pi_self:.3f}"]

    for r in sorted(vacuity_scores):
        sw = (sample_weight_stats or {}).get(r, {})
        peer_str = (
            f"peer{r}: vac={vacuity_scores[r]:.3f} "
            f"w={weights.get(r, 0):.3f}"
        )
        if sw:
            peer_str += (
                f" sw_mean={sw.get('mean', 0):.2f}"
                f" sw_fl={sw.get('frac_low', 0):.2f}"
            )
        parts.append(peer_str)

    print(f"[ENGC-Adaptive][Rank {rank}][Step {step}] " + " | ".join(parts))