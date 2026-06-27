"""
MURMURA — Evidential Trust-Aware Model Aggregation
"""

import copy
import math
import torch
import torch.nn.functional as F

def _edl_params(logits: torch.Tensor):
    evidence = F.softplus(logits)   # [B, K]
    alpha    = evidence + 1.0       # [B, K]
    S        = alpha.sum(dim=1)     # [B]
    return evidence, alpha, S

def compute_edl_vacuity(logits: torch.Tensor) -> torch.Tensor:
    """
    EDL vacuity = K / S [0, 1]
    0: confident
    1: uncertain
    """
    K = logits.size(1)
    _, _, S = _edl_params(logits)
    return (K / S).clamp(0.0, 1.0) 


class MURMURA_sender:
    """
    For each neighbor r, load r's model weights and evaluate on local batch:
        vacuity_r = mean(K / S)           
        accuracy_r = fraction of correct preds
        base_r = (1 - vacuity_r) * (w_a * accuracy_r + (1 - w_a))
        trust_r = base_r * exp(-(vacuity_r - tau_u)) if vacuity_r > tau_u
                = base_r otherwise
    EMA smoothing: trust_r = gamma_ema * trust_r + (1 - gamma_ema) * trust_r_prev
    """

    def __init__(
        self,
        true_model,
        device,
        num_classes: int       = 10,
        vacuity_threshold: float = 0.5,
        accuracy_weight: float   = 0.5,
        use_adaptive_trust: bool = True,
        trust_momentum: float    = 0.7,
        max_eval_samples: int    = 100,
    ):
        self.model            = copy.deepcopy(true_model)
        self.model.eval()
        self.model            = self.model.to(device)
        self.device           = device
        self.num_classes      = num_classes
        self.tau_u            = vacuity_threshold
        self.w_a              = accuracy_weight
        self.use_ema          = use_adaptive_trust
        self.gamma_ema        = trust_momentum
        self.max_eval_samples = max_eval_samples

        self._trust_ema: dict        = {}   # rank -> EMA trust
        self.last_trust_scores: dict = {}
        self.last_vacuities: dict    = {}
        self.last_accuracies: dict   = {}

    def _update_model(self, state_dict):
        for w, p in zip(state_dict, self.model.parameters()):
            p.data.copy_(w.data)

    def _compute_trust(self, batch_x: torch.Tensor, batch_y: torch.Tensor):
        """
        Evaluate model on batch. 
        Returns (trust, vacuity, accuracy).
        """
        self.model.eval()
        with torch.no_grad():
            if len(batch_x) > self.max_eval_samples:
                idx = torch.randperm(len(batch_x), device=self.device)[:self.max_eval_samples]
                x, y = batch_x[idx], batch_y[idx]
            else:
                x, y = batch_x, batch_y

            logits   = self.model(x)
            vacuity  = compute_edl_vacuity(logits).mean().item()
            accuracy = (logits.argmax(dim=1) == y).float().mean().item()

            base_trust = (1.0 - vacuity) * (self.w_a * accuracy + (1.0 - self.w_a))
            if vacuity > self.tau_u:
                trust = base_trust * math.exp(-(vacuity - self.tau_u))
            else:
                trust = base_trust

        return float(trust), float(vacuity), float(accuracy)

    def __call__(
        self,
        neighbor_weights: dict,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
    ) -> dict:
        """
        Returns {rank: trust_score} for all neighbors.
        """
        trust_scores = {}

        for rank, weights in neighbor_weights.items():
            self._update_model(weights)
            trust, vacuity, accuracy = self._compute_trust(batch_x, batch_y)

            if self.use_ema and rank in self._trust_ema:
                trust = (self.gamma_ema * trust
                         + (1.0 - self.gamma_ema) * self._trust_ema[rank])
            self._trust_ema[rank] = trust

            trust_scores[rank]         = trust
            self.last_vacuities[rank]  = vacuity
            self.last_accuracies[rank] = accuracy

        self.last_trust_scores = trust_scores
        return trust_scores

class MURMURA_receiver:
    """
        tau(t) = tau_min * (1 - gamma * exp(-kappa * t / T))  

        accepted   = {r : trust_r >= tau(t)}
        w_r        = trust_r / Σ trust_r   for r in accepted

        param_i = self_weight * param_i + (1 - self_weight) * sum(w_r * neighbor_r_param_i)

    """

    def __init__(
        self,
        model,
        device,
        rank: int,
        lr: float,
        momentum: float,
        qgm: bool,
        nesterov: bool         = True,
        weight_decay: float    = 0.0,
        neighbors: int         = 2,
        self_weight: float     = 0.5,
        trust_threshold: float = 0.1,
        use_tightening: bool   = True,
        tightening_gamma: float = 0.5,
        tightening_kappa: float = 1.0,
        total_rounds: int      = 50,
    ):
        self.model            = model
        self.rank             = rank
        self.device           = device
        self.lr               = lr
        self.self_weight      = self_weight
        self.trust_threshold  = trust_threshold
        self.use_tightening   = use_tightening
        self.t_gamma          = tightening_gamma
        self.t_kappa          = tightening_kappa
        self.total_rounds     = total_rounds

        self._pending_weights: dict = None
        self._pending_trust: dict   = None
        self._pending_step: float   = 0.0

        self.last_accepted: dict    = {}
        self.last_eff_weights: dict = {}
        self.last_threshold: float  = trust_threshold

    def _threshold(self, step: float) -> float:
        if not self.use_tightening:
            return self.trust_threshold
        return float(
            self.trust_threshold * (
                1.0 - self.t_gamma *
                math.exp(-self.t_kappa * step / max(self.total_rounds, 1))
            )
        )

    def prepare(self, neighbor_weights: dict, trust_scores: dict, step: float = 0.0):
        self._pending_weights = neighbor_weights
        self._pending_trust   = trust_scores
        self._pending_step    = step

    def post_step_aggregate(self):
        if self._pending_weights is None:
            return

        tau_t = self._threshold(self._pending_step)
        self.last_threshold = tau_t

        accepted = {r: t for r, t in self._pending_trust.items() if t >= tau_t}
        self.last_accepted = accepted

        if not accepted:
            self.last_eff_weights  = {}
            self._pending_weights  = None
            return

        total  = sum(accepted.values()) or 1.0
        norm_w = {r: t / total for r, t in accepted.items()}
        self.last_eff_weights = norm_w

        nbr_params = {r: list(self._pending_weights[r]) for r in accepted}

        with torch.no_grad():
            for i, p_self in enumerate(self.model.module.parameters()):
                if not p_self.requires_grad:
                    continue
                neighbor_avg = torch.zeros_like(p_self.data)
                for r, w in norm_w.items():
                    neighbor_avg.add_(nbr_params[r][i].data, alpha=w)
                p_self.data.mul_(self.self_weight).add_(
                    neighbor_avg, alpha=(1.0 - self.self_weight)
                )

        self._pending_weights = None
        self._pending_trust   = None

    def project_gradients(self, lr):
        self.lr = lr
