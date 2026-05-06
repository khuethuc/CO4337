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
             num_classes: int, lambda_kl: float = 1.0,
             sample_weights: torch.Tensor = None,
             return_per_sample: bool = False) -> torch.Tensor:
    """
    EDL loss — Sensoy et al. (2018) Eq. 5:

        L = L_NLL + lambda_kl * L_KL

    L_NLL = E_{p~Dir(α)}[-log p(y|p)]
          = Σ_k y_k * (ψ(S) - ψ(α_k))          digamma-based NLL

    L_KL  = KL(Dir(α̃) || Dir(1,...,1))
    α̃_k  = y_k + (1 - y_k) * α_k               zero out true-class evidence
                                                  before penalizing wrong-class confidence

    Args:
        return_per_sample: if True, return [B] per-sample losses (no reduction).
                           Used for epistemic uncertainty decomposition.
        sample_weights:    [B] tensor; if provided, weighted mean is returned.
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

    per_sample = l_nll + lambda_kl * l_kl                          # [B]

    if return_per_sample:
        return per_sample                                           # [B], no reduction
    if sample_weights is not None:
        return (per_sample * sample_weights).mean()
    return per_sample.mean()


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


class ENGC_sender():

    def __init__(self, true_model, device, num_classes: int = 10,
                 edl_lambda: float = 1.0):
        self.model           = copy.deepcopy(true_model)
        self.model.train()
        self.model           = self.model.to(device)
        self.gradient_buffer = {}
        self.device          = device
        self.num_classes     = num_classes
        self.edl_lambda      = edl_lambda

        self.vacuity_scores          = {}
        self.acc_scores              = {}
        self.loss_scores             = {}

        # Per-neighbor mean epistemic gap U_epi = mean(max(0, loss_A - loss_B))
        # on local data D_B. Used by receiver for per-neighbor trust.
        self.U_epi_scores: dict = {}

        # Per-sample weight stats: did per-sample filtering actually fire?
        # sample_weight_stats[rank] = {'mean', 'min', 'frac_low'}
        # frac_low = fraction of samples with weight < 0.5 (effectively downweighted)
        self.sample_weight_stats: dict = {}
        self._last_sw_stats: dict      = {}

    def _update_model(self, state_dict):
        for w, p in zip(state_dict, self.model.parameters()):
            p.data.copy_(w.data)

    def _accumulate_gradients(self, x: torch.Tensor, targets: torch.Tensor,
                               loss_b_per_sample: torch.Tensor = None):
        """
        Compute cross-gradient ∇L(θ_A, D_B) with per-sample epistemic weighting.

        Per-sample weight:
            loss_a(x_i) = EDL loss of neighbor model A on x_i
            loss_b(x_i) = EDL loss of local model B on x_i  (passed in)

            U_epi(x_i)  = max(0, loss_a - loss_b)
                          high → A worse than B on x_i → OOD for A → downweight
                          low  → both struggle equally (aleatoric) or A knows it

            w_i = 1 / (1 + U_epi(x_i))
            

        Falls back to uniform weights when loss_b_per_sample is None.

        Returns (gradient_buffer, loss_value, mean_U_epi)
        """
        self.model.zero_grad()
        output = self.model(x)

        with torch.no_grad():
            if loss_b_per_sample is not None:
                loss_a_per_sample = edl_loss(
                    output.detach(), targets, self.num_classes,
                    lambda_kl=self.edl_lambda, return_per_sample=True,
                )                                                   # [B]
                U_epi          = (loss_a_per_sample - loss_b_per_sample).clamp(min=0.0)
                sample_weights = (1.0 / (1.0 + U_epi)).clamp(0.0, 1.0)
                mean_U_epi     = U_epi.mean().item()
                self._last_sw_stats = {
                    'mean':     sample_weights.mean().item(),
                    'min':      sample_weights.min().item(),
                    'frac_low': (sample_weights < 0.5).float().mean().item(),
                }
            else:
                sample_weights = None
                mean_U_epi     = float('nan')
                self._last_sw_stats = {}

        loss = edl_loss(output, targets, self.num_classes,
                        lambda_kl=self.edl_lambda, sample_weights=sample_weights)
        loss_val = loss.item()
        loss.backward()

        self._clear_gradient_buffer()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.gradient_buffer[name] = param.grad.data.clone()
        return self.gradient_buffer, loss_val, mean_U_epi

    def _clear_gradient_buffer(self):
        self.gradient_buffer = {}

    def _flatten_(self, G, mean_aleatoric: float = 0.0):
        """Flatten gradient dict and append mean_aleatoric as a trailing scalar.

        The receiver at the neighbor side extracts this scalar to learn about
        our data quality (aleatoric uncertainty) without an extra comm round.
        """
        grad   = [g for g in G.values()]
        flat   = flatten_tensors(grad).to(self.device)
        scalar = torch.tensor([mean_aleatoric], device=self.device, dtype=flat.dtype)
        return torch.cat([flat, scalar])

    def __call__(self, neighbor_weight, batch_x, targets,
                 loss_b_per_sample: torch.Tensor = None):
        """
        Args:
            loss_b_per_sample: [B] per-sample EDL loss of the LOCAL model on batch_x.
                               Passed from trainer so we can compute U_epi without
                               storing a reference to the main model.
        """
        self.vacuity_scores      = {}
        self.loss_scores         = {}
        self.acc_scores          = {}
        self.U_epi_scores        = {}
        self.sample_weight_stats = {}

        # B's aleatoric = mean per-sample loss of B's own model on D_B.
        # Appended to every gradient tensor we send so neighbors can use it
        # for per-neighbor trust computation without an extra round.
        mean_aleatoric_b = (loss_b_per_sample.mean().item()
                            if loss_b_per_sample is not None else 0.0)

        output = {}
        g      = None
        for rank, w in neighbor_weight.items():
            self._update_model(w)
            with torch.no_grad():
                logits = self.model(batch_x)
                self.vacuity_scores[rank] = compute_edl_vacuity(logits).mean().item()
                preds  = logits.argmax(dim=1)
                self.acc_scores[rank] = (preds == targets).float().mean().item()

            g, loss_val, mean_U_epi = self._accumulate_gradients(
                batch_x, targets, loss_b_per_sample
            )
            self.loss_scores[rank]         = loss_val
            self.U_epi_scores[rank]        = mean_U_epi
            self.sample_weight_stats[rank] = self._last_sw_stats
            output[rank]                   = self._flatten_(g, mean_aleatoric_b)
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
        beta         = 0.1,
    ):
        self.model        = model
        self.rank         = rank
        self.device       = device
        self.proj_grads   = {}
        self.pi           = 1.0 / float(neighbors + 1)
        self.alpha        = alpha   # kept for backward compat, unused when beta active
        self.beta         = beta    # blend: beta*p_comp + (1-beta)*p_comm
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

        # Diagnostics
        self.last_weights_comp      = {}   # weights used for model-variant
        self.last_weights_comm      = {}   # weights used for data-variant
        self.last_neighbor_weights  = {}   # legacy: mean of comp+comm weights
        self.last_aleatoric_scores  = {}
        self.last_trust_scores      = {}

    def _normalize(self, neighbor_ranks, trusts):
        """Normalize trust scores to sum to pi_neighbor_total."""
        N                 = len(neighbor_ranks)
        pi_self           = 1.0 / (N + 1)
        pi_neighbor_total = N   / (N + 1)
        total   = sum(trusts.values()) or 1.0
        weights = {r: (trusts[r] / total) * pi_neighbor_total for r in trusts}
        return pi_self, weights

    def _weights_for_comp(self, neighbor_ranks, U_epi_scores):
        """
        Weights for model-variant gradients ∇L(θ_A, D_B).

        Signal: U_epi_A = mean(max(0, loss_A - loss_B)) trên D_B
                HIGH → A kém trên D_B → model-variant gradient kém → weight thấp
        """
        MIN_TRUST = 0.1
        N         = len(neighbor_ranks)
        pi_self   = 1.0 / (N + 1)

        if U_epi_scores:
            trusts = {r: max(MIN_TRUST, 1.0 / (1.0 + U_epi_scores.get(r, 0.0)))
                      for r in neighbor_ranks}
        else:
            pi_nbr  = N / (N + 1)
            weights = {r: pi_nbr / N for r in neighbor_ranks}
            return pi_self, weights

        return self._normalize(neighbor_ranks, trusts)

    def _weights_for_comm(self, neighbor_ranks, aleatoric_scores, vacuity_scores=None,
                           loss_scores=None, acc_scores=None):
        """
        Weights for data-variant gradients ∇L(θ_B, D_A).

        Primary signal: U_ale_A = mean(loss_A trên D_A)
                        HIGH → A's data noisy → data-variant gradient kém → weight thấp
        Fallback: acc_scores, vacuity-based, uniform.
        """
        MIN_TRUST = 0.1
        N         = len(neighbor_ranks)
        pi_self   = 1.0 / (N + 1)

        if aleatoric_scores:
            trusts = {r: max(MIN_TRUST, 1.0 / (1.0 + aleatoric_scores.get(r, 0.0)))
                      for r in neighbor_ranks}
        elif acc_scores:
            trusts = {r: max(MIN_TRUST, acc_scores.get(r, 0.5))
                      for r in neighbor_ranks}
        elif vacuity_scores:
            trusts = {
                r: max(MIN_TRUST,
                       (1.0 - vacuity_scores.get(r, 0.5))
                       / (1.0 + (loss_scores.get(r, 0.0) if loss_scores else 0.0)))
                for r in neighbor_ranks
            }
        else:
            pi_nbr  = N / (N + 1)
            weights = {r: pi_nbr / N for r in neighbor_ranks}
            return pi_self, weights

        return self._normalize(neighbor_ranks, trusts)

    def _weighted_average(self, self_grad, neighbor_grads_dict, pi_self, weights):
        result = pi_self * self_grad
        for r, grad in neighbor_grads_dict.items():
            result = result + weights[r] * grad
        return result

    def _unflatten_(self, flat_tensor, ref_buf):
        ref    = list(ref_buf.values())
        keys   = list(ref_buf.keys())
        unflat = unflatten_tensors(flat_tensor, ref)
        return {k: v for k, v in zip(keys, unflat)}

    def __call__(self, neighbor_grads_comm, neighbor_grads_comp, ref_buf,
                 U_epi_scores=None,
                 vacuity_scores=None, loss_scores=None, acc_scores=None):
        # ------------------------------------------------------------------
        # Step 1: extract trailing aleatoric scalar appended by each sender.
        # ------------------------------------------------------------------
        aleatoric_scores = {}
        for rank in list(neighbor_grads_comm.keys()):
            flat = neighbor_grads_comm[rank]
            aleatoric_scores[rank]    = flat[-1].item()
            neighbor_grads_comm[rank] = flat[:-1]

        for rank in list(neighbor_grads_comp.keys()):
            neighbor_grads_comp[rank] = neighbor_grads_comp[rank][:-1]

        self.last_aleatoric_scores = aleatoric_scores

        # ------------------------------------------------------------------
        # Step 2: unflatten gradient vectors
        # ------------------------------------------------------------------
        for rank, ft in neighbor_grads_comm.items():
            neighbor_grads_comm[rank] = self._unflatten_(ft, ref_buf)
        for rank, ft in neighbor_grads_comp.items():
            neighbor_grads_comp[rank] = self._unflatten_(ft, ref_buf)

        neighbor_ranks = list(neighbor_grads_comm.keys())

        # ------------------------------------------------------------------
        # Step 3: two separate weight sets — each aligned with its gradient type
        #   weights_comp: model-variant ∇L(θ_A, D_B)  ← trust via U_epi_A
        #   weights_comm: data-variant  ∇L(θ_B, D_A)  ← trust via U_ale_A
        # ------------------------------------------------------------------
        pi_self, weights_comp = self._weights_for_comp(neighbor_ranks, U_epi_scores)
        _,       weights_comm = self._weights_for_comm(
            neighbor_ranks, aleatoric_scores,
            vacuity_scores=vacuity_scores, loss_scores=loss_scores, acc_scores=acc_scores,
        )

        self.last_weights_comp     = {'self': pi_self, **weights_comp}
        self.last_weights_comm     = {'self': pi_self, **weights_comm}
        self.last_neighbor_weights = {
            r: self.beta * weights_comp.get(r, 0.0)
               + (1.0 - self.beta) * weights_comm.get(r, 0.0)
            for r in neighbor_ranks
        }
        self.last_neighbor_weights['self'] = pi_self

        # ------------------------------------------------------------------
        # Step 4: weighted gradient blend
        #   proj = beta * p_comp + (1-beta) * p_comm
        # ------------------------------------------------------------------
        for name, self_params in self.model.module.named_parameters():
            if not self_params.requires_grad:
                continue
            self_grad     = self_params.grad.data
            comm_neighbor = {r: g[name] for r, g in neighbor_grads_comm.items()}
            comp_neighbor = {r: g[name] for r, g in neighbor_grads_comp.items()}

            p_comp = self._weighted_average(self_grad, comp_neighbor, pi_self, weights_comp)
            p_comm = self._weighted_average(self_grad, comm_neighbor, pi_self, weights_comm)
            self.proj_grads[name] = self.beta * p_comp + (1.0 - self.beta) * p_comm

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

