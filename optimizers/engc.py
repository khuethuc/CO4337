import torch
import copy
import math
import torch.nn.functional as F

from .utils import flatten_tensors, unflatten_tensors


# ---------------------------------------------------------------------------
# EDL utility — Sensoy et al. (2018) "Evidential Deep Learning to Quantify
# Classification Uncertainty"
#
# Quy trình:
#   evidence = softplus(logits)          # e_k >= 0
#   alpha    = evidence + 1              # Dirichlet params, α_k >= 1
#   S        = Σ alpha_k                 # Dirichlet strength
#   vacuity  = K / S  ∈ (0, 1]          # 0 = confident, 1 = maximally uncertain
# ---------------------------------------------------------------------------

def _edl_params(logits: torch.Tensor):
    """
    Trả về (evidence, alpha, S) theo EDL.
    evidence = softplus(logits) >= 0
    alpha    = evidence + 1  (Dirichlet params)
    S        = sum(alpha, dim=1)  (Dirichlet strength)
    """
    evidence = F.softplus(logits)              # [B, K]
    alpha    = evidence + 1.0                  # [B, K]
    S        = alpha.sum(dim=1)                # [B]
    return evidence, alpha, S


def compute_edl_vacuity(logits: torch.Tensor) -> torch.Tensor:
    """
    EDL vacuity = K / S  ∈ (0, 1].
    0 → model rất confident, 1 → hoàn toàn uncertain (uniform Dirichlet).
    """
    K = logits.size(1)
    _, _, S = _edl_params(logits)
    return (K / S).clamp(0.0, 1.0)            # [B]


def _kl_dirichlet(alpha: torch.Tensor) -> torch.Tensor:
    """
    KL[Dir(alpha) || Dir(1)]  — uniform Dirichlet prior.

    Công thức (Sensoy et al. 2018, Appendix):
      KL = lgamma(S) - lgamma(K)
           - Σ_k lgamma(α_k)
           + Σ_k (α_k - 1)(digamma(α_k) - digamma(S))

    với S = Σ α_k, K = number of classes.
    lgamma(1) = 0 nên không cần trừ Σ lgamma(1).
    """
    K = alpha.size(1)
    S = alpha.sum(dim=1)                       # [B]

    kl = (torch.lgamma(S)
          - math.lgamma(K)
          - torch.lgamma(alpha).sum(dim=1)
          + ((alpha - 1.0) * (torch.digamma(alpha)
             - torch.digamma(S.unsqueeze(1)))).sum(dim=1))
    return kl                                  # [B]


# ---------------------------------------------------------------------------
# ENGC_sender — giữ nguyên hoàn toàn từ NGC_sender
# Cross-gradient vẫn dùng CE loss, không thay đổi
# Thêm: trả về neighbor model weights để trainer dùng cho KD gate
# ---------------------------------------------------------------------------

class ENGC_sender():
    """
    ENGC Sender (Hướng 3):
    - Giữ nguyên logic cross-gradient từ NGC (CE loss)
    - Thêm: lưu lại neighbor weights để trainer tính KD gate sau

    Không còn trust score, không còn uncertainty per-neighbor.
    EDL chỉ được dùng trong trainer để tạo confident mask per-sample.
    """

    def __init__(self, true_model, device, num_classes: int = 10):
        self.model           = copy.deepcopy(true_model)
        self.model.train()
        self.model           = self.model.to(device)
        self.gradient_buffer = {}
        self.device          = device
        self.num_classes     = num_classes

        # Lưu neighbor weights gần nhất để trainer dùng cho KD
        self.last_neighbor_weights = {}

    def _update_model(self, state_dict):
        for w, p in zip(state_dict, self.model.parameters()):
            p.data.copy_(w.data)

    def _accumulate_gradients(self, x, targets):
        """
        EDL NLL loss — nhất quán với local training loss.
        L_NLL = mean[ ψ(S) - ψ(α_y) ]
        Chỉ dùng NLL (không KL) vì cross-gradient phản ánh fit với data,
        không phải regularization.
        """
        self.model.zero_grad()
        output   = self.model(x)
        evidence = F.softplus(output)
        alpha    = evidence + 1.0
        S        = alpha.sum(dim=1)
        alpha_y  = alpha[torch.arange(len(targets), device=output.device), targets]
        loss     = (torch.digamma(S) - torch.digamma(alpha_y)).mean()
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
        """
        Giống NGC_sender.__call__ hoàn toàn.
        Thêm: lưu neighbor_weight để trainer dùng cho KD gate.

        Returns:
            output : dict[rank] -> flattened gradient
            g      : reference gradient buffer (shapes)
        """
        # Lưu lại để trainer truy cập qua sender.last_neighbor_weights
        self.last_neighbor_weights = neighbor_weight

        output = {}
        g      = None
        for rank, w in neighbor_weight.items():
            self._update_model(w)
            g            = self._accumulate_gradients(batch_x, targets)
            output[rank] = self._flatten_(g)
        return output, g


# ---------------------------------------------------------------------------
# ENGC_receiver — giữ nguyên hoàn toàn từ NGC_receiver
# Gradient aggregation uniform, không thay đổi
# ---------------------------------------------------------------------------

class ENGC_receiver():
    """
    ENGC Receiver (Hướng 3):
    Giữ nguyên hoàn toàn từ NGC_receiver.
    Uniform gradient aggregation — không cần trust score nữa.
    Cải tiến đến từ KD loss trong trainer, không phải ở đây.
    """

    def __init__(
        self,
        model,
        device,
        rank,
        lr,
        momentum,
        qgm,
        nesterov   = True,
        weight_decay = 0,
        neighbors  = 2,
        alpha      = 1.0,
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

    def _average_gradients(self, grad_list):
        """Uniform averaging với weight π = 1/(|N|+1)."""
        new_grad = torch.zeros_like(grad_list[0])
        for g in grad_list:
            new_grad += self.pi * g
        return new_grad

    def _unflatten_(self, flat_tensor, ref_buf):
        ref  = list(ref_buf.values())
        keys = list(ref_buf.keys())
        unflat = unflatten_tensors(flat_tensor, ref)
        return {k: v for k, v in zip(keys, unflat)}

    def __call__(self, neighbor_grads_comm, neighbor_grads_comp, ref_buf):
        """Uniform gradient aggregation — giống NGC_receiver."""
        for rank, ft in neighbor_grads_comm.items():
            neighbor_grads_comm[rank] = self._unflatten_(ft, ref_buf)
        for rank, ft in neighbor_grads_comp.items():
            neighbor_grads_comp[rank] = self._unflatten_(ft, ref_buf)

        for name, self_params in self.model.module.named_parameters():
            if not self_params.requires_grad:
                continue

            # Communication branch
            comm_list = [neigh[name] for neigh in neighbor_grads_comm.values()]
            comm_list.append(self_params.grad.data)
            p_comm = self._average_gradients(comm_list)

            # Computation branch
            comp_list = [neigh[name] for neigh in neighbor_grads_comp.values()]
            comp_list.append(self_params.grad.data)
            p_comp = self._average_gradients(comp_list)

            # NGC branch fusion
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
# EDL-gated Knowledge Distillation loss
# Theo Sensoy et al. (2018) + standard KD (Hinton et al. 2015)
# ---------------------------------------------------------------------------

class EDLGatedKDLoss(torch.nn.Module):
    """
    EDL-gated Knowledge Distillation Loss (Sensoy et al. 2018).

    Local training loss thay CE bằng EDL:
      L_EDL = L_NLL + annealing_coef * lambda_reg * L_KL

      L_NLL(i) = ψ(S_i) - ψ(α_{i,y_i})         ← NLL dưới Dirichlet
      L_KL     = KL[Dir(α̃) || Dir(1)]           ← regularize non-target evidence
                 với α̃_k = y_k + (1-y_k)*α_k

    KD loss chỉ áp dụng khi teacher (neighbor) confident — EDL vacuity < tau_u:
      L_KD(j) = KL(softmax(s/T) || softmax(t/T)) * T²  [chỉ trên conf_mask]

    Tổng:
      L = L_EDL + lambda_kd * mean_j(L_KD_j)

    Args:
        num_classes   : K
        tau_u         : EDL vacuity threshold ∈ (0,1], loại sample nếu u >= tau_u
        temperature   : T cho KD (Hinton)
        lambda_kd     : weight của KD loss
        min_conf_ratio: skip KD nếu confident ratio < giá trị này
        lambda_reg    : weight KL regularization trong EDL loss
    """

    def __init__(
        self,
        num_classes    : int,
        tau_u          : float = 0.4,
        temperature    : float = 2.0,
        lambda_kd      : float = 0.5,
        min_conf_ratio : float = 0.1,
        lambda_reg     : float = 0.1,
    ):
        super().__init__()
        self.K              = num_classes
        self.tau_u          = tau_u
        self.T              = temperature
        self.lambda_kd      = lambda_kd
        self.min_conf_ratio = min_conf_ratio
        self.lambda_reg     = lambda_reg

        # Logging
        self.last_conf_ratios = {}   # rank -> float
        self.last_kd_losses   = {}   # rank -> float
        self.last_n_conf      = {}   # rank -> int
        self.last_edl_loss    = 0.0
        self.last_kl_loss     = 0.0

    # ------------------------------------------------------------------
    # EDL loss components
    # ------------------------------------------------------------------

    def edl_loss(
        self,
        logits         : torch.Tensor,
        targets        : torch.Tensor,
        annealing_coef : float = 1.0,
        class_weights  : torch.Tensor = None,
    ) -> torch.Tensor:
        """
        EDL training loss = L_NLL + annealing_coef * lambda_reg * L_KL

        L_NLL = mean[ ψ(S_i) - ψ(α_{i,y_i}) ]  (class-weighted nếu có)
        L_KL  = mean[ KL[Dir(α̃_i) || Dir(1)] ]
                với α̃_k = y_k + (1 - y_k) * α_k
                (chỉ penalize evidence của các class sai)

        class_weights: [K] inverse-frequency weights để upweight minority classes.
                       Nếu None, dùng plain mean (hành vi cũ).
        """
        _, alpha, S = _edl_params(logits)                # [B,K], [B,K], [B]

        # --- NLL term: ψ(S) - ψ(α_y) ---
        alpha_y = alpha[torch.arange(len(targets), device=logits.device), targets]
        nll_per_sample = torch.digamma(S) - torch.digamma(alpha_y)  # [B]

        if class_weights is not None:
            w   = class_weights[targets]                 # [B] — weight theo class của mỗi sample
            nll = (nll_per_sample * w).sum() / w.sum()  # weighted mean
        else:
            nll = nll_per_sample.mean()

        # --- KL term: only penalize non-target evidence ---
        y_one_hot   = F.one_hot(targets, self.K).float()           # [B, K]
        alpha_tilde = y_one_hot + (1.0 - y_one_hot) * alpha        # zero target evidence
        kl = _kl_dirichlet(alpha_tilde).mean()

        self.last_edl_loss = nll.item()
        self.last_kl_loss  = kl.item()

        return nll + annealing_coef * self.lambda_reg * kl

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _get_neighbor_logits(self, neighbor_model, x):
        """Forward pass neighbor model — no gradient, restore train mode."""
        was_training = neighbor_model.training
        neighbor_model.eval()
        with torch.no_grad():
            logits = neighbor_model(x)
        if was_training:
            neighbor_model.train()
        return logits

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        student_logits   : torch.Tensor,
        targets          : torch.Tensor,
        neighbor_weights : dict,
        input_x          : torch.Tensor,
        neighbor_model   : torch.nn.Module,
        annealing_coef   : float = 1.0,
        class_weights    : torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            student_logits   : [B, K] logits của local model
            targets          : [B]   ground-truth labels
            neighbor_weights : dict  rank -> list[param tensors]
            input_x          : [B, C, H, W] input batch
            neighbor_model   : model dùng để load neighbor weights
            annealing_coef   : λ_anneal ∈ [0,1], tăng dần theo epoch
            class_weights    : [K] inverse-frequency weights (từ local data).
                               Dùng để upweight minority class trong EDL loss
                               và KD loss. None = hành vi cũ (uniform).

        Returns:
            L_EDL + lambda_kd * mean(L_KD_j)
        """
        # --- Local EDL loss (class-weighted nếu có) ---
        total_loss = self.edl_loss(student_logits, targets, annealing_coef,
                                   class_weights=class_weights)

        if not neighbor_weights:
            return total_loss

        kd_losses = []

        for rank, w in neighbor_weights.items():
            # Load teacher weights
            for param_w, param_m in zip(w, neighbor_model.parameters()):
                param_m.data.copy_(param_w.data)

            # Teacher forward — no gradient
            teacher_logits = self._get_neighbor_logits(neighbor_model, input_x)

            # EDL vacuity của teacher: u = K / S ∈ (0, 1]
            vacuity = compute_edl_vacuity(teacher_logits)          # [B]

            # Confident mask: teacher uncertain ít
            conf_mask  = vacuity < self.tau_u
            n_conf     = conf_mask.sum().item()
            conf_ratio = n_conf / max(len(conf_mask), 1)

            self.last_conf_ratios[rank] = conf_ratio
            self.last_n_conf[rank]      = n_conf

            if conf_ratio < self.min_conf_ratio or n_conf == 0:
                self.last_kd_losses[rank] = 0.0
                continue

            # Fix 3: skip nếu tất cả confident samples đều predict cùng 1 class
            # → tránh KD chỉ reinforcing majority class
            teacher_pred = teacher_logits[conf_mask].argmax(dim=1)
            if teacher_pred.unique().numel() < 2:
                self.last_kd_losses[rank] = 0.0
                continue

            # KD loss (Hinton 2015) chỉ trên confident samples
            s_soft = F.log_softmax(student_logits[conf_mask] / self.T, dim=1)
            t_soft = F.softmax(teacher_logits[conf_mask]     / self.T, dim=1)

            if class_weights is not None:
                # Upweight KD signal từ minority class teacher predictions
                kd_w          = class_weights[teacher_pred]        # [N_conf]
                kd_w          = kd_w / kd_w.sum()
                kd_per_sample = F.kl_div(s_soft, t_soft.detach(),
                                         reduction='none').sum(dim=1)  # [N_conf]
                kd_loss       = (kd_per_sample * kd_w).sum() * (self.T ** 2)
            else:
                kd_loss = F.kl_div(s_soft, t_soft.detach(),
                                   reduction='batchmean') * (self.T ** 2)

            kd_losses.append(kd_loss)
            self.last_kd_losses[rank] = kd_loss.item()

        if kd_losses:
            total_loss = total_loss + self.lambda_kd * torch.stack(kd_losses).mean()

        return total_loss

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_stats(self, rank: int, step: int):
        if not self.last_conf_ratios:
            return
        parts = [
            f"edl_nll={self.last_edl_loss:.6f}",
            f"kl_reg={self.last_kl_loss:.6f}",
        ]
        for r in sorted(self.last_conf_ratios):
            parts.append(
                f"peer{r}: conf={self.last_conf_ratios[r]:.2f} "
                f"kd_raw={self.last_kd_losses.get(r, 0.0):.6f} "
                f"kd_weighted={self.last_kd_losses.get(r, 0.0) * self.lambda_kd:.6f}"
            )
        print(f"[ENGC-KD][Rank {rank}][Step {step}] " + " | ".join(parts))