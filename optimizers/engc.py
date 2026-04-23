import torch
import copy
import math
import torch.nn.functional as F

from .utils import flatten_tensors, unflatten_tensors


# ---------------------------------------------------------------------------
# EDL utility — Sensoy et al. (2018) "Evidential Deep Learning to Quantify
# Classification Uncertainty"
#
# Formulas:
#   evidence = softplus(logits)          # e_k >= 0
#   alpha    = evidence + 1              # Dirichlet params, α_k >= 1
#   S        = Σ alpha_k                 # Dirichlet strength
#   vacuity  = K / S  ∈ (0, 1]          # 0 = confident, 1 = maximally uncertain
# ---------------------------------------------------------------------------

def _edl_params(logits: torch.Tensor):
    evidence = F.softplus(logits)              # [B, K]
    alpha    = evidence + 1.0                  # [B, K]
    S        = alpha.sum(dim=1)                # [B]
    return evidence, alpha, S

def compute_edl_vacuity(logits: torch.Tensor) -> torch.Tensor:
    """
    EDL vacuity = K / S  ∈ (0, 1].
    0 - confident
    1 - uncertain (uniform Dirichlet)
    """
    K = logits.size(1)
    _, _, S = _edl_params(logits)
    return (K / S).clamp(0.0, 1.0)            # [B]


def edl_loss(logits: torch.Tensor, targets: torch.Tensor,
             num_classes: int, lambda_kl: float = 1.0) -> torch.Tensor:
    """
    EDL loss — Sensoy et al. (2018) Eq. 5:

        L = L_NLL + lambda_kl * L_KL

    L_NLL = E_{p~Dir(α)}[-log p(y|p)]
          = Σ_k y_k * (ψ(S) - ψ(α_k))          digamma-based NLL

    L_KL  = KL(Dir(α̃) || Dir(1,...,1))
    α̃_k  = y_k + (1 - y_k) * α_k               zero out true-class evidence
                                                  before penalizing wrong-class confidence

    Properties guaranteed by this loss (unlike weighted-CE):
      - KL term forces evidence → 0 for wrong classes when prediction is wrong
      - Model learns to express high vacuity when uncertain, not just when loss is high
    """
    evidence, alpha, S = _edl_params(logits)                        # [B,K], [B,K], [B]
    y = F.one_hot(targets, num_classes=num_classes).float()         # [B,K]

    # NLL term: Σ_k y_k * (ψ(S) - ψ(α_k))
    l_nll = (y * (torch.digamma(S.unsqueeze(1).expand_as(alpha))
                  - torch.digamma(alpha))).sum(dim=1)               # [B]

    # KL regularizer
    alpha_tilde = y + (1.0 - y) * alpha                            # [B,K]
    S_tilde     = alpha_tilde.sum(dim=1)                           # [B]
    lgamma_K    = torch.lgamma(
        torch.tensor(float(num_classes), device=logits.device)
    )
    l_kl = (
        torch.lgamma(S_tilde)
        - lgamma_K
        - torch.lgamma(alpha_tilde).sum(dim=1)
        + ((alpha_tilde - 1.0) * (
            torch.digamma(alpha_tilde)
            - torch.digamma(S_tilde.unsqueeze(1).expand_as(alpha_tilde))
        )).sum(dim=1)
    )                                                               # [B]

    return (l_nll + lambda_kl * l_kl).mean()


class ENGC_sender():

    def __init__(self, true_model, device, num_classes: int = 10,
                 edl_lambda: float = 1.0):
        self.model           = copy.deepcopy(true_model)
        self.model.train()
        self.model           = self.model.to(device)
        self.gradient_buffer = {}
        self.device          = device
        self.num_classes     = num_classes
        self.edl_lambda      = edl_lambda   # KL weight λ in EDL loss

        # Per-neighbor EDL vacuity on MY local data (no gradient)
        self.vacuity_scores       = {}   # rank -> float in [0, 1]

        # Diagnostic: per-neighbor vacuity split by prediction correctness
        # vac_wrong > vac_correct confirms EDL calibration is working
        self.vacuity_by_pred       = {}
        self._last_vacuity_by_pred = {}

    def _update_model(self, state_dict):
        for w, p in zip(state_dict, self.model.parameters()):
            p.data.copy_(w.data)

    def _accumulate_gradients(self, x, targets):
        """
        Cross-gradient using proper EDL loss (Sensoy et al. 2018):

            L = L_NLL + lambda_kl * L_KL

        Unlike the previous weighted-CE approach, the KL term directly penalizes
        the model for assigning evidence to wrong classes — guaranteeing that
        vac_wrong > vac_correct when the loss is working correctly.
        """
        self.model.zero_grad()
        output = self.model(x)

        with torch.no_grad():
            vacuity = compute_edl_vacuity(output.detach())   # [B]

            # Diagnostic: confirm KL term is calibrating vacuity correctly
            preds        = output.detach().argmax(dim=1)
            correct_mask = (preds == targets)
            n_wrong      = int((~correct_mask).sum().item())
            n_total      = len(targets)
            self._last_vacuity_by_pred = {
                'vac_correct': vacuity[correct_mask].mean().item()  if correct_mask.any() else float('nan'),
                'vac_wrong':   vacuity[~correct_mask].mean().item() if n_wrong > 0        else float('nan'),
                'frac_wrong':  n_wrong / max(n_total, 1),
            }

        loss = edl_loss(output, targets, self.num_classes, lambda_kl=self.edl_lambda)
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
        self.vacuity_scores  = {}
        self.vacuity_by_pred = {}
        output = {}
        g      = None
        for rank, w in neighbor_weight.items():
            self._update_model(w)
            with torch.no_grad():
                logits = self.model(batch_x)
                self.vacuity_scores[rank] = compute_edl_vacuity(logits).mean().item()
            g = self._accumulate_gradients(batch_x, targets)
            self.vacuity_by_pred[rank] = self._last_vacuity_by_pred
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

        # Diagnostic: weights assigned to self and each neighbor in the last step.
        # last_neighbor_weights = {'self': pi_self, rank_r: w_r, ...}
        # Use this to verify: noisy neighbors get lower weights than clean ones.
        self.last_neighbor_weights = {}

    def _compute_weights(self, neighbor_ranks, vacuity_scores=None):
        """
        Compute pi_self and per-neighbor weights in one place.
        Stores result in self.last_neighbor_weights for external logging.
        Returns (pi_self, {rank: weight}).
        """
        N                 = len(neighbor_ranks)
        pi_self           = 1.0 / (N + 1)
        pi_neighbor_total = N   / (N + 1)

        if vacuity_scores:
            MIN_TRUST = 0.1
            trusts    = {r: max(MIN_TRUST, 1.0 - vacuity_scores.get(r, 0.5))
                         for r in neighbor_ranks}
            total     = sum(trusts.values()) or 1.0
            weights   = {r: (trusts[r] / total) * pi_neighbor_total for r in trusts}
        else:
            w_uniform = pi_neighbor_total / N
            weights   = {r: w_uniform for r in neighbor_ranks}

        self.last_neighbor_weights = {'self': pi_self, **weights}
        return pi_self, weights

    def _weighted_average(self, self_grad, neighbor_grads_dict, pi_self, weights):
        """Apply pre-computed weights. Separated from _compute_weights so weights
        are computed once per step, not once per parameter."""
        result = pi_self * self_grad
        for r, grad in neighbor_grads_dict.items():
            result = result + weights[r] * grad
        return result

    def _unflatten_(self, flat_tensor, ref_buf):
        ref  = list(ref_buf.values())
        keys = list(ref_buf.keys())
        unflat = unflatten_tensors(flat_tensor, ref)
        return {k: v for k, v in zip(keys, unflat)}

    def __call__(self, neighbor_grads_comm, neighbor_grads_comp, ref_buf,
                 vacuity_scores=None):
        for rank, ft in neighbor_grads_comm.items():
            neighbor_grads_comm[rank] = self._unflatten_(ft, ref_buf)
        for rank, ft in neighbor_grads_comp.items():
            neighbor_grads_comp[rank] = self._unflatten_(ft, ref_buf)

        # Compute weights ONCE per step (stored in self.last_neighbor_weights)
        pi_self, weights = self._compute_weights(
            list(neighbor_grads_comm.keys()), vacuity_scores
        )

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


# ---------------------------------------------------------------------------
# Vacuity logging helper (used by trainer)
# ---------------------------------------------------------------------------

def log_vacuity_stats(rank: int, step: int, vacuity_scores: dict):
    N = len(vacuity_scores)
    if N == 0:
        return
    MIN_TRUST  = 0.1
    pi_self    = 1.0 / (N + 1)
    pi_nbr_tot = N   / (N + 1)

    trusts  = {r: max(MIN_TRUST, 1.0 - v) for r, v in vacuity_scores.items()}
    total   = sum(trusts.values()) or 1.0
    weights = {r: (trusts[r] / total) * pi_nbr_tot for r in trusts}

    parts = [f"pi_self={pi_self:.3f}"]
    for r in sorted(vacuity_scores):
        parts.append(
            f"peer{r}: vac={vacuity_scores[r]:.3f} w={weights.get(r, 0):.3f}"
        )
    print(f"[ENGC-Adaptive][Rank {rank}][Step {step}] " + " | ".join(parts))
