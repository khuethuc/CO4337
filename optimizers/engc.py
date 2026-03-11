from scipy.linalg import orth
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import copy
import numpy as np
from .utils import flatten_tensors, unflatten_tensors
from collections import defaultdict

# ==========================================
# EVIDENTIAL LOSS FUNCTION
# ==========================================
class EDLLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super(EDLLoss, self).__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights # Tensor trọng số cho bài toán y tế mất cân bằng
        self.lambda_t = 0.0 # Giá trị này sẽ được trainer.py tự động cập nhật liên tục

    def forward(self, evidence, targets):
        """
        evidence: Đầu ra của mô hình sau khi qua hàm kích hoạt (như Softplus hoặc ReLU)
        targets: Nhãn thực tế (chưa one-hot)
        """
        # Tính tham số Dirichlet
        alpha = evidence + 1.0
        
        y = F.one_hot(targets, num_classes=self.num_classes).float()
        S = torch.sum(alpha, dim=1, keepdim=True)
        expected_p = alpha / S
        
        # 1. Thành phần Fit dữ liệu (Error + Variance)
        err_loss = torch.sum((y - expected_p)**2, dim=1, keepdim=True)
        var_loss = torch.sum(expected_p * (1 - expected_p) / (S + 1), dim=1, keepdim=True)
        
        # 2. Thành phần Phạt Epistemic Uncertainty (KL Divergence)
        alpha_tilde = y + (1 - y) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        
        kl_div = torch.lgamma(S_tilde) - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True) \
                 + torch.sum(torch.lgamma(torch.ones_like(alpha_tilde)), dim=1, keepdim=True) \
                 - torch.lgamma(torch.ones_like(S_tilde) * self.num_classes) \
                 + torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1, keepdim=True)
                 
        # Tổng hợp Loss
        loss_batch = err_loss + var_loss + self.lambda_t * kl_div
        
        # 3. Áp dụng trọng số mất cân bằng lớp (Class Weights) cho y tế
        if self.class_weights is not None:
            # Lấy trọng số tương ứng với từng nhãn trong batch hiện tại
            batch_weights = self.class_weights[targets].unsqueeze(1)
            loss_batch = loss_batch * batch_weights
            
        return torch.mean(loss_batch)

# ==========================================
# EVIDENTIAL NGC SENDER
# ==========================================
class ENGC_sender():
    def __init__(self, true_model, device, criterion, num_classes, total_rounds=100, 
                 tau_u=0.7, tau_min_0=0.3, w_a=0.5, gamma_tau=0.5, kappa=1.0):
        self.model           = copy.deepcopy(true_model)
        self.true_model      = true_model 
        self.model.train()
        self.model           = self.model.to(device)
        self.gradient_buffer = {}
        self.device          = device
        self.criterion       = criterion 
        self.num_classes = num_classes
        self.total_rounds = total_rounds
        self.tau_u = tau_u
        self.tau_min_0 = tau_min_0
        self.w_a = w_a
        self.gamma_tau = gamma_tau
        self.kappa = kappa

    def _update_model(self, state_dict):
        for w, p in zip(state_dict, self.model.parameters()):
            p.data.copy_(w.data)
        return

    def _accumulate_gradients(self, x, targets, current_round):
        logits = self.model(x)
        evidence = F.softplus(logits) 
        
        self.model.zero_grad()
        loss = self.criterion(evidence, targets)
        loss.backward()
        
        self._clear_gradient_buffer()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.gradient_buffer[name] = param.grad.data
        return self.gradient_buffer

    def _clear_gradient_buffer(self):
        self.gradient_buffer = {}
        return

    def _flatten_(self, G):
        grad = [g for g in G.values()]
        return flatten_tensors(grad).to(self.device)

    def evaluate_trust(self, val_x, val_targets):
        """Đánh giá mô hình láng giềng để lấy Epistemic Uncertainty & Accuracy."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(val_x)
            evidence = F.softplus(logits)
            alpha = evidence + 1.0
            S = torch.sum(alpha, dim=1, keepdim=True)
            expected_p = alpha / S
            
            preds = torch.argmax(expected_p, dim=1)
            acc = (preds == val_targets).float().mean().item()
            
            u = self.num_classes / S
            mean_u = u.mean().item()
        self.model.train()
        return acc, mean_u

    def __call__(self, neighbor_weight, batch_x, targets, val_x, val_targets, current_round):
        # 1. Tính ngưỡng Hard filter hiện tại
        decay = np.exp(-self.kappa * current_round / self.total_rounds)
        tau_min_t = self.tau_min_0 * (1 - self.gamma_tau * decay)
        
        trust_scores = {}
        total_score = 0.0
        trusted_neighbors = []
        output = {}

        # 2. Đánh giá tất cả láng giềng
        for rank, w in neighbor_weight.items():
            self._update_model(w)
            acc, mean_u = self.evaluate_trust(val_x, val_targets)
            
            # Hybrid Soft Penalty
            s_base = (1 - mean_u) * (self.w_a * acc + (1 - self.w_a))
            s_final = s_base * np.exp(-(mean_u - self.tau_u)) if mean_u > self.tau_u else s_base
            
            # Hard Filtering
            if s_final >= tau_min_t:
                trust_scores[rank] = s_final
                total_score += s_final
                trusted_neighbors.append(rank)

        # 3. Chuẩn hóa trọng số tín nhiệm (v_j)
        if total_score > 0:
            for rank in trusted_neighbors:
                trust_scores[rank] /= total_score
                
        # 4. CHỈ tính toán Cross-gradients cho những node qua bài test
        for rank in trusted_neighbors:
            w = neighbor_weight[rank]
            self._update_model(w)
            g = self._accumulate_gradients(batch_x, targets, current_round)
            output[rank] = self._flatten_(g)
            
        # Tính gradient cho chính mô hình cục bộ (Self-gradient)
        self._update_model([p.data for p in self.true_model.parameters()])
        g_self = self._accumulate_gradients(batch_x, targets, current_round)

        # Trả về output (cross-grads), g_self (local-grad), và trust_scores để Receiver dùng
        return output, g_self, trust_scores

# ==========================================
# EVIDENTIAL NGC RECEIVER
# ==========================================
class ENGC_receiver():
    def __init__(self, model, device, rank, lr, momentum, qgm, nesterov=True, weight_decay=0, neighbors=2, alpha=1.0):
        self.model         = model
        self.rank          = rank
        self.device        = device
        self.proj_grads    = {}
        # Vẫn giữ mức độ tự tin cơ sở cho self-gradient (1 / N+1)
        self.pi            = 1.0/float(neighbors+1)      
        self.alpha         = alpha
        self.momentum      = momentum
        self.lr            = lr
        self.nesterov      = nesterov
        self.qgm           = qgm
        self.weight_decay  = weight_decay
        self.momentum_buff = []
        self.prev_params   = []
        for param in self.model.module.parameters():
            self.momentum_buff.append(torch.zeros_like(param.data))
            self.prev_params.append(copy.deepcopy(param.data))
 
    def _unflatten_(self, flat_tensor, ref_buf):
        ref, keys = [], []
        for key,val in ref_buf.items():
            ref.append(val)
            keys.append(key)
        unflat_tensor =  unflatten_tensors(flat_tensor, ref)
        X = {}
        for i, key in enumerate(keys):
            X[key] = unflat_tensor[i]
        return X

    def __call__(self, neighbor_grads_comm, neighbor_grads_comp, ref_buf, trust_scores):
        ### Unflatten the neighbor grads
        for rank, flat_tenor in neighbor_grads_comm.items():
            neighbor_grads_comm[rank] = self._unflatten_(flat_tenor, ref_buf)
        for rank, flat_tenor in neighbor_grads_comp.items():
            neighbor_grads_comp[rank] = self._unflatten_(flat_tenor, ref_buf)
        
        # Phần dư còn lại phân bổ cho láng giềng
        neighbor_total_weight = 1.0 - self.pi 
        
        for name, self_params in self.model.module.named_parameters():
            if self_params.requires_grad:
                p_grads_comm = torch.zeros_like(self_params.grad.data)
                p_grads_comp = torch.zeros_like(self_params.grad.data)
                
                # Luôn luôn cộng Self-gradient với trọng số self.pi
                p_grads_comm += self.pi * self_params.grad.data
                p_grads_comp += self.pi * self_params.grad.data

                # Cộng Cross-gradients bằng Trust-Aware Weight
                if trust_scores:
                    for rank, normalized_trust in trust_scores.items():
                        v_j = neighbor_total_weight * normalized_trust
                        if rank in neighbor_grads_comm:
                            p_grads_comm += v_j * neighbor_grads_comm[rank][name]
                        if rank in neighbor_grads_comp:
                            p_grads_comp += v_j * neighbor_grads_comp[rank][name]
                
                self.proj_grads[name] = ((1-self.alpha)*p_grads_comp) + (self.alpha*p_grads_comm)
        return 

    def project_gradients(self, lr):
        # [Giữ nguyên logic momentum và update của bạn]
        for name, p in self.model.module.named_parameters():
            if p.requires_grad:
                p.grad.data = self.proj_grads[name].data 
                if self.weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=self.weight_decay)
        
        if self.momentum!=0:
            if self.qgm:
                for p, p_prev, buf in zip(self.model.module.parameters(), self.prev_params, self.momentum_buff):
                    buf.mul_(self.momentum).add_(p_prev.data-p.data, alpha=(1.0-self.momentum)/self.lr)
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