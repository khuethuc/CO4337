import torch
import copy
import math
import torch.nn.functional as F

from .utils import flatten_tensors, unflatten_tensors


# ---------------------------------------------------------------------------
# EDL utility: forward-only uncertainty computation
# Không dùng để train, chỉ dùng để tính confident mask cho KD gate
# ---------------------------------------------------------------------------

# engc.py — compute_edl_vacuity: thay bằng entropy-based uncertainty
def compute_uncertainty_gate(logits: torch.Tensor) -> torch.Tensor:
    """
    Tính uncertainty từ softmax entropy — hoạt động với CE-trained model.
    Normalized về [0, 1]: 0 = confident, 1 = maximum uncertain.
    
    Entropy = -Σ p_k log(p_k), max = log(K) khi uniform
    """
    probs   = F.softmax(logits, dim=1)              # [B, K]
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)  # [B]
    max_entropy = math.log(logits.size(1))           # log(K)
    return (entropy / max_entropy).clamp(0.0, 1.0)  # normalized [0, 1]


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
        self.criterion       = torch.nn.CrossEntropyLoss().to(device)

        # Lưu neighbor weights gần nhất để trainer dùng cho KD
        self.last_neighbor_weights = {}

    def _update_model(self, state_dict):
        for w, p in zip(state_dict, self.model.parameters()):
            p.data.copy_(w.data)

    def _accumulate_gradients(self, x, targets):
        """CE loss — giống NGC_sender hoàn toàn."""
        output = self.model(x)
        self.model.zero_grad()
        loss = self.criterion(output, targets)
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
# Đây là phần mới hoàn toàn — core contribution của Hướng 3
# ---------------------------------------------------------------------------

class EDLGatedKDLoss(torch.nn.Module):
    """
    Knowledge Distillation loss được gate bởi EDL uncertainty.

    Với mỗi neighbor model j và local batch (x, y):
    1. Chạy neighbor model forward (no grad) → teacher_logits, vacuity_j
    2. Tạo confident_mask: vacuity_j < tau_u (normalized [0,1])
    3. KD loss chỉ trên confident samples:
       L_KD = KL( student[mask] / T  ||  teacher[mask] / T )

    Tổng loss:
       L = L_CE + lambda_kd * mean(L_KD_j  for j in neighbors)

    Args:
        num_classes  : K — số class
        tau_u        : ngưỡng uncertainty [0,1], default 0.4
                       sample bị loại nếu vacuity_norm >= tau_u
        temperature  : T cho KD softmax, default 2.0
        lambda_kd    : weight của KD loss, default 0.5
        min_conf_ratio: nếu % confident sample < ratio này thì skip KD
                        tránh KD từ quá ít samples, default 0.1
    """

    def __init__(
        self,
        num_classes    : int,
        tau_u          : float = 0.4,
        temperature    : float = 2.0,
        lambda_kd      : float = 0.5,
        min_conf_ratio : float = 0.1,
    ):
        super().__init__()
        self.num_classes    = num_classes
        self.tau_u          = tau_u
        self.T              = temperature
        self.lambda_kd      = lambda_kd
        self.min_conf_ratio = min_conf_ratio

        # Logging
        self.last_conf_ratios  = {}   # rank -> float
        self.last_kd_losses    = {}   # rank -> float
        self.last_n_conf       = {}   # rank -> int

    def _get_neighbor_logits(self, neighbor_model, x):
        """Forward pass neighbor model, no gradient."""
        was_training = neighbor_model.training
        neighbor_model.eval()
        with torch.no_grad():
            logits = neighbor_model(x)
        if was_training:
            neighbor_model.train()
        return logits

    def forward(
        self,
        student_logits   : torch.Tensor,
        targets          : torch.Tensor,
        neighbor_weights : dict,
        input_x          : torch.Tensor,
        neighbor_model   : torch.nn.Module,
        ce_loss          : torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits   : [B, K] output của local model (đã forward)
            targets          : [B] ground truth labels
            neighbor_weights : dict rank -> state_dict (từ cross_weights)
            input_x          : [B, C, H, W] input batch
            neighbor_model   : copy của model dùng để load neighbor weights
            ce_loss          : CE loss đã tính sẵn (scalar tensor)

        Returns:
            total_loss : ce_loss + lambda_kd * mean_kd_loss
        """
        if not neighbor_weights:
            return ce_loss

        kd_losses = []

        for rank, w in neighbor_weights.items():
            # Load neighbor weights
            for param_w, param_m in zip(w, neighbor_model.parameters()):
                param_m.data.copy_(param_w.data)

            # Forward neighbor — no gradient
            teacher_logits = self._get_neighbor_logits(neighbor_model, input_x)

            # Tính vacuity normalized [0, 1]
            vacuity_norm = compute_uncertainty_gate(teacher_logits)

            # Confident mask: teacher phải biết về sample này
            conf_mask = vacuity_norm < self.tau_u   # [B] bool
            n_conf    = conf_mask.sum().item()
            conf_ratio = n_conf / max(len(conf_mask), 1)

            # Log
            self.last_conf_ratios[rank] = conf_ratio
            self.last_n_conf[rank]      = n_conf

            # Skip nếu quá ít confident samples
            if conf_ratio < self.min_conf_ratio or n_conf == 0:
                self.last_kd_losses[rank] = 0.0
                continue

            # KD loss chỉ trên confident samples
            s_soft = F.log_softmax(student_logits[conf_mask] / self.T, dim=1)
            t_soft = F.softmax(teacher_logits[conf_mask]  / self.T, dim=1)

            kd_loss = F.kl_div(s_soft, t_soft, reduction='batchmean') * (self.T ** 2)
            kd_losses.append(kd_loss)
            self.last_kd_losses[rank] = kd_loss.item()

        if not kd_losses:
            return ce_loss

        mean_kd = torch.stack(kd_losses).mean()
        return ce_loss + self.lambda_kd * mean_kd

    def log_stats(self, rank: int, step: int):
        if not self.last_conf_ratios:
            return
        parts = []
        for r in sorted(self.last_conf_ratios):
            parts.append(
                f"peer{r}: conf={self.last_conf_ratios[r]:.2f} "
                f"kd_raw={self.last_kd_losses.get(r, 0.0):.6f} "   # ← thêm 6 decimal
                f"kd_weighted={self.last_kd_losses.get(r,0.0)*self.lambda_kd:.6f}"
            )
        print(f"[ENGC-KD][Rank {rank}][Step {step}] " + " | ".join(parts))