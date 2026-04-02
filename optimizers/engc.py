import torch
import copy
import math
from collections import defaultdict
import torch.nn.functional as F

from .utils import flatten_tensors, unflatten_tensors


class EDLLoss(torch.nn.Module):
    """
    Evidential Deep Learning Loss for medical imaging classification.

    Combines:
    - Likelihood loss (negative log likelihood)
    - KL divergence regularization (for uncertainty)

    Reference: Sensoy et al. "Evidential Deep Learning" (CVPR 2018)
    """
    def __init__(self, num_classes, annealing_step=10, kl_weight=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.kl_weight = kl_weight

    def _kl_divergence(self, alpha, num_classes):
        """
        Compute KL divergence between Dirichlet(alpha) and uniform Dirichlet.
        """
        beta = torch.ones(1, num_classes).to(alpha.device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        sum_beta = torch.sum(beta, dim=1, keepdim=True)

        ln_alpha = torch.lgamma(sum_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        ln_beta = torch.lgamma(sum_beta) - torch.sum(torch.lgamma(beta), dim=1, keepdim=True)

        dg_alpha = torch.digamma(alpha) - torch.digamma(sum_alpha)
        dg_beta = torch.digamma(beta) - torch.digamma(sum_beta)

        kl = ln_alpha - ln_beta + torch.sum((alpha - beta) * dg_alpha, dim=1, keepdim=True)
        return kl

    def forward(self, logits, targets, global_step=0):
        """
        Args:
            logits: raw model outputs [batch_size, num_classes]
            targets: ground truth labels [batch_size]
            global_step: current training step for annealing

        Returns:
            loss: total EDL loss
            alpha: Dirichlet parameters (for uncertainty computation)
            uncertainty: per-sample uncertainty
        """
        # Convert logits to evidence
        evidence = F.softplus(logits)
        alpha = evidence + 1

        # Expected probability
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / torch.clamp(S, min=1e-8)

        # One-hot encoding
        y_one_hot = F.one_hot(targets, self.num_classes).float().to(logits.device)

        # Likelihood loss (negative log likelihood)
        loss_likelihood = torch.sum(
            y_one_hot * torch.log(torch.clamp(probs, min=1e-8)),
            dim=1
        ).mean()

        # KL divergence with annealing
        kl_alpha = (alpha - 1) * (1 - y_one_hot) + 1
        kl_div = self._kl_divergence(kl_alpha, self.num_classes).mean()

        # Annealing coefficient
        annealing_coef = min(1.0, global_step / self.annealing_step)

        # Total loss
        total_loss = -loss_likelihood + annealing_coef * self.kl_weight * kl_div

        # Per-sample uncertainty for logging
        uncertainty = self.num_classes / torch.clamp(S.squeeze(1), min=1e-8)

        return total_loss, alpha, uncertainty


class ENGC_sender():
    """
    ENGC Sender: computes cross-gradients and uncertainty metrics for neighbors.

    Per pseudocode:
    1. Compute neighbor uncertainty on VALIDATION batch (no gradient)
       - alpha_j = Softplus(x_j(x_val)) + 1
       - u_j = K / sum(alpha_j) (vacuity/epistemic uncertainty)
    2. Compute cross-gradients on TRAINING batch with EDL loss
    """

    def __init__(self, true_model, device, num_classes=10, use_edl=False, edl_annealing_step=10):
        """
        Args:
            true_model: local model on current node
            device: cuda device
            num_classes: number of classes for EDL
            use_edl: whether to use EDL loss for uncertainty weighting
            edl_annealing_step: annealing steps for EDL KL divergence
        """
        self.model = copy.deepcopy(true_model)
        self.model.train()
        self.model = self.model.to(device)
        self.gradient_buffer = {}
        self.device = device
        self.num_classes = num_classes
        self.use_edl = use_edl

        # Standard CE loss
        self.criterion = torch.nn.CrossEntropyLoss().to(device)

        # EDL loss (optional)
        if self.use_edl:
            self.edl_criterion = EDLLoss(
                num_classes=num_classes,
                annealing_step=edl_annealing_step
            ).to(device)

        # For storing last uncertainty info
        self.last_uncertainty = None
        self.last_alpha = None

    def _update_model(self, state_dict):
        """
        Args:
            state_dict: parameters of a neighbor model
        """
        for w, p in zip(state_dict, self.model.parameters()):
            p.data.copy_(w.data)
        return

    def _compute_uncertainty_and_accuracy(self, x_val, y_val):
        """
        Compute EDL uncertainty and accuracy on validation batch (no gradient).
        Per pseudocode:
        - alpha_j = Softplus(x_j(x_val)) + 1
        - u_j = K / Σalpha (vacuity = epistemic uncertainty)
        - accuracy = prediction correctness

        Args:
            x_val: validation input batch
            y_val: validation target labels

        Returns:
            uncertainty: mean vacuity (scalar)
            accuracy: classification accuracy (scalar)
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(x_val)
            # alpha = Softplus(logits) + 1
            evidence = F.softplus(output)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            # Vacuity: u = K / Σalpha
            uncertainty = self.num_classes / torch.clamp(S.squeeze(1), min=1e-8)
            # Expected probability for prediction
            probs = alpha / torch.clamp(S, min=1e-8)
            preds = probs.argmax(dim=1)
            accuracy = (preds == y_val).float().mean().item()

        self.model.train()
        return float(uncertainty.mean().item()), float(accuracy)

    def _compute_self_uncertainty_and_accuracy(self, x_val, y_val):
        """
        Compute uncertainty and accuracy for local (self) model on validation batch.

        Args:
            x_val: validation input batch
            y_val: validation target labels

        Returns:
            uncertainty: mean vacuity (scalar)
            accuracy: classification accuracy (scalar)
        """
        return self._compute_uncertainty_and_accuracy(x_val, y_val)

    def _accumulate_gradients(self, x, targets, global_step=0):
        """
        Compute gradient on TRAINING batch (x, targets) for the current sender model.
        With EDL: uses evidential loss (NLL + KL).

        Args:
            x: training input batch
            targets: training target labels
            global_step: current training step for EDL annealing

        Returns:
            gradient_dict: parameter gradients
        """
        output = self.model(x)
        self.model.zero_grad()

        # Compute loss
        if self.use_edl:
            loss, alpha, uncertainty = self.edl_criterion(output, targets, global_step)
            self.last_uncertainty = uncertainty.mean().item()
            self.last_alpha = alpha
        else:
            loss = self.criterion(output, targets)
            uncertainty, alpha = self._compute_uncertainty(output)
            self.last_uncertainty = uncertainty.mean().item()
            self.last_alpha = alpha

        # Backward
        loss.backward()

        # Collect gradients
        self._clear_gradient_buffer()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.gradient_buffer[name] = param.grad.data.clone()

        return self.gradient_buffer

    def _compute_uncertainty(self, logits):
        """
        Compute EDL uncertainty from logits.

        Args:
            logits: model outputs [batch_size, num_classes]

        Returns:
            uncertainty: per-sample uncertainty [batch_size]
            alpha: Dirichlet parameters [batch_size, num_classes]
        """
        evidence = F.softplus(logits)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        uncertainty = self.num_classes / torch.clamp(S.squeeze(1), min=1e-8)
        return uncertainty, alpha

    def _clear_gradient_buffer(self):
        self.gradient_buffer = {}
        return

    def _flatten_(self, G):
        grad = []
        for g in G.values():
            grad.append(g)
        return flatten_tensors(grad).to(self.device)

    def __call__(self, neighbor_weight, batch_x, targets, val_x=None, val_y=None, global_step=0):
        """
        Args:
            neighbor_weight: neighbor model weights
            batch_x, targets: TRAINING mini-batch for gradient computation
            val_x, val_y: VALIDATION mini-batch for uncertainty/accuracy computation
            global_step: current training step for EDL annealing

        Returns:
            output: flattened cross-gradients for each neighbor model
            g: reference unflattened gradient buffer (used for shape only)
            uncertainty_dict: per-neighbor uncertainty and accuracy metrics
        """
        output = {}
        g = None
        uncertainty_dict = {}

        for rank, w in neighbor_weight.items():
            # Load neighbor model weights
            self._update_model(w)

            # 1. Compute uncertainty and accuracy on VALIDATION batch (no gradient)
            # Per pseudocode: alpha_j = Softplus(x_j(x_val)) + 1, u_j = K/Σalpha
            if val_x is not None and val_y is not None:
                vacuity, accuracy = self._compute_uncertainty_and_accuracy(val_x, val_y)
            else:
                vacuity, accuracy = 0.5, 0.5  # fallback defaults

            # 2. Compute cross-gradient on TRAINING batch with EDL loss
            grad_dict = self._accumulate_gradients(batch_x, targets, global_step)
            g = grad_dict
            output[rank] = self._flatten_(g)

            # 3. Store uncertainty metrics for trust score computation
            uncertainty_dict[rank] = {
                "vacuity": vacuity,          # u_j = K/Σalpha (epistemic uncertainty)
                "accuracy": accuracy,        # neighbor accuracy on val batch
                "mean_uncertainty": vacuity, # for backward compatibility
            }

        return output, g, uncertainty_dict


class ENGC_receiver():
    """
    ENGC Receiver: aggregates gradients using trust-weighted averaging.

    Per pseudocode:
    Trust score: s_j = (1 - u_j) * (w_a * acc_j + (1 - w_a))
    Soft penalty: if u_j > τ_u: s_j = s_j * exp(-(u_j - τ_u))

    Aggregation weights:
    - π = 1 / (|N(i)| + 1) (self weight)
    - v_j = (1 - π) * s_j / Σs_j' (neighbor weights)
    - g̃ = π * g^ii + Σ v_j * g^ij
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
        weight_decay=0,
        neighbors=2,
        alpha=1.0,
        # trust score params (per pseudocode)
        tau_u=0.5,           # uncertainty threshold τ_u
        w_a=0.5,             # accuracy weight w_a in trust score
        # NGC mixing params
        self_weight=None,    # deprecated, use uniform π = 1/(|N|+1)
        score_momentum=0.90,
        eps=1e-12,
        # EDL uncertainty params
        use_edl=False,
        edl_weight=0.3,
        num_classes=10,
        edl_uncertainty_threshold=0.5,
    ):
        self.model = model
        self.rank = rank
        self.device = device
        self.proj_grads = {}

        # NGC mixing weight alpha
        self.alpha = alpha

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

        # Trust score parameters (per pseudocode)
        self.tau_u = float(tau_u)           # uncertainty threshold
        self.w_a = float(w_a)               # accuracy weight in trust score
        self.num_neighbors = int(neighbors)
        self.eps = float(eps)

        # EMA for trust scores (optional smoothing)
        self.score_momentum = float(score_momentum)
        self.trust_score_ema = {}  # rank -> EMA trust score

        # For logging/debug
        self.last_trust_scores = {}
        self.last_aggregation_weights = {}

        # EDL settings
        self.use_edl = use_edl
        self.edl_weight = float(edl_weight)
        self.num_classes = num_classes
        self.edl_uncertainty_threshold = float(edl_uncertainty_threshold)

        # For tracking
        self.current_uncertainty = None

    def _unflatten_(self, flat_tensor, ref_buf):
        """
        Args:
            flat_tensor: received flat tensor
            ref_buf: reference gradient dict for shapes
        Returns:
            dict[name] -> tensor
        """
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

    def _compute_trust_score(self, vacuity, accuracy):
        """
        Compute trust score per pseudocode:
        s_j = (1 - u_j) * (w_a * acc_j + (1 - w_a))
        if u_j > τ_u: s_j = s_j * exp(-(u_j - τ_u))

        Args:
            vacuity: u_j (epistemic uncertainty) in [0, K] where K=num_classes
            accuracy: acc_j in [0, 1]

        Returns:
            trust_score: s_j >= 0
        """
        # s_j = (1 - u_j) * (w_a * acc_j + (1 - w_a))
        # Note: vacuity can be > 1 for K classes, so we normalize to [0, 1] range
        # by using (1 - vacuity/K) instead of (1 - vacuity)
        normalized_vacuity = min(vacuity / self.num_classes, 1.0)  # clamp to [0, 1]
        confidence = 1.0 - normalized_vacuity

        # s_j = confidence * (w_a * acc + (1 - w_a))
        trust_score = confidence * (self.w_a * accuracy + (1.0 - self.w_a))

        # Soft penalty for high uncertainty: if u_j > τ_u
        if vacuity > self.tau_u:
            penalty = math.exp(-(vacuity - self.tau_u))
            trust_score = trust_score * penalty

        return max(trust_score, 0.0)  # ensure non-negative

    def _ema_update_trust(self, peer_rank, trust_score):
        """Apply EMA smoothing to trust scores."""
        prev = self.trust_score_ema.get(peer_rank, None)
        if prev is None:
            new_score = trust_score
        else:
            new_score = self.score_momentum * prev + (1.0 - self.score_momentum) * trust_score
        self.trust_score_ema[peer_rank] = new_score
        return new_score

    def _compute_aggregation_weights(self, uncertainty_metrics_dict):
        """
        Compute aggregation weights per pseudocode:
        - π = 1 / (|N(i)| + 1) (self weight)
        - v_j = (1 - π) * s_j / Σs_j' (neighbor weights)

        Args:
            uncertainty_metrics_dict: dict[rank] -> {"vacuity": u_j, "accuracy": acc_j}

        Returns:
            self_weight: π
            peer_weights: dict[rank] -> v_j
        """
        num_neighbors = len(uncertainty_metrics_dict)

        if num_neighbors == 0:
            self.last_aggregation_weights = {"self": 1.0}
            return 1.0, {}

        # π = 1 / (|N| + 1)
        self_weight = 1.0 / (num_neighbors + 1)
        peer_mass = 1.0 - self_weight

        # Compute trust scores for all neighbors
        trust_scores = {}
        for peer_rank, metrics in uncertainty_metrics_dict.items():
            vacuity = metrics.get("vacuity", 0.5)
            accuracy = metrics.get("accuracy", 0.5)
            raw_trust = self._compute_trust_score(vacuity, accuracy)
            # Apply EMA smoothing
            ema_trust = self._ema_update_trust(peer_rank, raw_trust)
            trust_scores[peer_rank] = ema_trust
            self.last_trust_scores[peer_rank] = {
                "raw": raw_trust,
                "ema": ema_trust,
                "vacuity": vacuity,
                "accuracy": accuracy,
            }

        # Normalize trust scores to get neighbor weights
        total_trust = sum(trust_scores.values())
        if total_trust <= self.eps:
            # Fallback to uniform if all trust scores are zero
            peer_weights = {r: peer_mass / num_neighbors for r in trust_scores}
        else:
            peer_weights = {
                r: peer_mass * (s / total_trust)
                for r, s in trust_scores.items()
            }

        self.last_aggregation_weights = {"self": self_weight}
        self.last_aggregation_weights.update(peer_weights)

        return self_weight, peer_weights

    def __call__(
        self,
        neighbor_grads_comm,
        neighbor_grads_comp,
        ref_buf,
        uncertainty_metrics=None
    ):
        """
        Aggregate gradients using trust-weighted averaging.

        Per pseudocode:
        - Compute trust scores: s_j = (1 - u_j) * (w_a * acc_j + 1 - w_a)
        - Apply soft penalty if u_j > τ_u
        - Aggregate: g̃ = π * g^ii + Σ v_j * g^ij

        Args:
            neighbor_grads_comm: dict[rank] -> flattened gradient (received from neighbors)
            neighbor_grads_comp: dict[rank] -> flattened gradient (local recomputed)
            ref_buf: local self gradient dict
            uncertainty_metrics: dict[rank] -> {"vacuity": u_j, "accuracy": acc_j}
        """
        # Unflatten gradients
        for rank, flat_tensor in neighbor_grads_comm.items():
            neighbor_grads_comm[rank] = self._unflatten_(flat_tensor, ref_buf)

        for rank, flat_tensor in neighbor_grads_comp.items():
            neighbor_grads_comp[rank] = self._unflatten_(flat_tensor, ref_buf)

        # Compute aggregation weights using uncertainty metrics from comp branch
        # (these contain the neighbor model uncertainties from validation batch)
        if uncertainty_metrics is not None:
            self_weight, peer_weights = self._compute_aggregation_weights(uncertainty_metrics)
        else:
            # Fallback to uniform weights
            num_neighbors = len(neighbor_grads_comm)
            if num_neighbors == 0:
                self_weight, peer_weights = 1.0, {}
            else:
                self_weight = 1.0 / (num_neighbors + 1)
                peer_weights = {r: (1.0 - self_weight) / num_neighbors for r in neighbor_grads_comm}

        # Track current uncertainty for logging
        if uncertainty_metrics:
            # Average vacuity across neighbors
            total_vacuity = sum(m.get("vacuity", 0.5) for m in uncertainty_metrics.values())
            self.current_uncertainty = total_vacuity / max(len(uncertainty_metrics), 1)

        # Aggregate gradients: g̃ = π * g^ii + Σ v_j * g^ij
        # For NGC: blend comm and comp branches using alpha
        for name, self_params in self.model.module.named_parameters():
            if not self_params.requires_grad:
                continue

            self_grad = self_params.grad.data

            # Communication branch: received gradients from neighbors
            agg_comm = self_weight * self_grad
            for peer_rank, neigh_grad in neighbor_grads_comm.items():
                agg_comm = agg_comm + peer_weights.get(peer_rank, 0.0) * neigh_grad[name]

            # Computation branch: locally recomputed gradients
            agg_comp = self_weight * self_grad
            for peer_rank, neigh_grad in neighbor_grads_comp.items():
                agg_comp = agg_comp + peer_weights.get(peer_rank, 0.0) * neigh_grad[name]

            # NGC branch fusion: g̃ = (1 - alpha) * g_comp + alpha * g_comm
            self.proj_grads[name] = (1.0 - self.alpha) * agg_comp + self.alpha * agg_comm

        return

    def project_gradients(self, lr):
        """
        Apply projected gradients to model.grad and then keep original momentum logic.
        """
        for name, p in self.model.module.named_parameters():
            if p.requires_grad:
                p.grad.data = self.proj_grads[name].data
                if self.weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=self.weight_decay)

        if self.momentum != 0:
            if self.qgm:
                for p, p_prev, buf in zip(self.model.module.parameters(), self.prev_params, self.momentum_buff):
                    buf.mul_(self.momentum).add_(
                        p_prev.data - p.data,
                        alpha=(1.0 - self.momentum) / self.lr
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
