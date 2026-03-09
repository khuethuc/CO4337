from scipy.linalg import orth
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import numpy as np
import math
from .utils import flatten_tensors, unflatten_tensors
from collections import defaultdict

class EDLLoss(nn.Module):
    def __init__(self, num_classes: int, lambda_t: float = 1.0, device=None):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_t = lambda_t
        self.device = device if device else torch.device('cpu')

    def forward(self, evidence: torch.Tensor, targets: torch.Tensor):
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        y_true = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float().to(self.device)
        # MSE
        mse = torch.sum(y_true * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        # KL Divergence Regularization
        alpha_tilde = y_true + (1 - y_true) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        kl_divergence = torch.lgamma(torch.tensor(self.num_classes, dtype=torch.float32).to(self.device)) - \
                 torch.lgamma(S_tilde) + \
                 torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True) + \
                 torch.sum((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1, keepdim=True)

        edl_loss = torch.mean(mse + self.lambda_t * kl_divergence)
        return edl_loss

class EDL_NGC_sender():
    def __init__(self, true_model, device, num_classes: int = 10):
        self.model           = copy.deepcopy(true_model)
        self.model.train()
        self.model           = self.model.to(device)
        self.gradient_buffer = {}
        self.device          = device
        self.num_classes     = num_classes
        self.criterion       = EDLLoss(num_classes, lambda_t=1.0, device=device)
        
    def _update_model(self, state_dict):
        for w, p in zip(state_dict, self.model.parameters()):
            p.data.copy_(w.data)
        return

    def _accumulate_gradients(self, x, targets):
        output = self.model(x)
        self.model.zero_grad()
        
        # Compute evidence by using ReLU and Clamp
        evidence = torch.clamp(torch.relu(output), min=0.0, max=10000.0)
        
        # Compute epistemic uncertainty: u = K / S
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1)
        epistemic_uncertainty = (self.num_classes / S).mean().item()
        
        # Compute loss
        loss = self.criterion(evidence, targets)
        loss.backward()
        
        self._clear_gradient_buffer()
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.gradient_buffer[name] = param.grad.data
        
        return self.gradient_buffer, epistemic_uncertainty

    def _clear_gradient_buffer(self):
        self.gradient_buffer = {}
        return

    def _flatten_(self, G):
        grad = []
        for g in G.values():
            grad.append(g)
        return flatten_tensors(grad).to(self.device)

    def __call__(self, neighbor_weight, batch_x, targets):
        output_grads = {}
        output_unc = {} 
        
        for rank, w in neighbor_weight.items():
            self._update_model(w)
            # Nhận cả gradient và uncertainty từ forward pass
            gradients, epistemic_uncertainty = self._accumulate_gradients(batch_x, targets)
            output_grads[rank] = self._flatten_(gradients)
            output_unc[rank] = torch.tensor([epistemic_uncertainty], dtype=torch.float32).to(self.device)
            
        return output_grads, output_unc, gradients

class EDL_NGC_receiver():
    def __init__(self, model, device, rank, lr, momentum, qgm, 
                 nesterov=True, weight_decay=0, neighbors=2, alpha=0.5,
                 default_threshold=0.7, init_threshold=0.3, tight_coef=0.5, tight_speed=1.0, weight = 0.5):
        self.model         = model
        self.rank          = rank
        self.device        = device
        self.proj_grads    = {}
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
            
        self.default_threshold  = default_threshold
        self.init_threshold     = init_threshold 
        self.tight_coef         = tight_coef
        self.tight_speed        = tight_speed
        self.weight             = weight

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

    def average_gradients_weighted(self, grad_list, weights):
        new_grad = torch.zeros_like(grad_list[-1]) 
        for grad, weight in zip(grad_list, weights):
            new_grad += weight * grad
        return new_grad
    
    def get_peer_aggregation_weights(self, uncertainties, threshold_t):
        scores = {}
        # Computing compatibility from uncertainty
        for rank, uncertainty_value in uncertainties.items():
            uncertainty_value = uncertainty_value.item()
            s_base = 1.0 - uncertainty_value
            if uncertainty_value > self.default_threshold:
                s_final = s_base * math.exp(-(uncertainty_value - self.default_threshold))
            else:
                s_final = s_base
            scores[rank] = s_final
        
        # Keep standard neighbors
        retained_ranks = [r for r, s in scores.items() if s >= threshold_t]
        if not retained_ranks:
            return {}, []
        
        # Normalize scores to weights: w_j = s_j / sum(s_j for j in retained_ranks)
        sum_scores = sum([scores[r] for r in retained_ranks])
        weights = {r: scores[r] / sum_scores for r in retained_ranks}
        
        return weights, retained_ranks


    def __call__(self, neighbor_grads_comm, neighbor_grads_comp, comm_uncertainty, comp_uncertainty, ref_buf, current_round, total_rounds):
        for rank in list(neighbor_grads_comm.keys()):
            neighbor_grads_comm[rank] = self._unflatten_(neighbor_grads_comm[rank], ref_buf)
        for rank in list(neighbor_grads_comp.keys()):
            neighbor_grads_comp[rank] = self._unflatten_(neighbor_grads_comp[rank], ref_buf)
            
        # Adaptive Trust Threshold
        threshold_t = self.init_threshold * (1.0 - self.tight_coef * math.exp(-self.tight_speed * current_round / total_rounds)) # threshold_t gradually increase to nearly init_threshold
        weights_comp, ranks_comp = self.get_peer_aggregation_weights(comp_uncertainty, threshold_t)
        weights_comm, ranks_comm = self.get_peer_aggregation_weights(comm_uncertainty, threshold_t)

        for name, self_params in self.model.module.named_parameters():
            if self_params.requires_grad and self_params.grad is not None:
                local_grad = self_params.grad.data
                
                # Model-variant (comp)
                if not ranks_comp:
                    p_grads_comp = local_grad 
                else:
                    # Aggregate trusted peers
                    g_peers_comp = torch.zeros_like(local_grad)
                    for rank in ranks_comp:
                        g_peers_comp += weights_comp[rank] * neighbor_grads_comp[rank][name]
                    # Personalize
                    p_grads_comp = self.weight * local_grad + (1.0 - self.weight) * g_peers_comp

                # Data-variant (comm)
                if not ranks_comm:
                    # Keep local model
                    p_grads_comm = local_grad 
                else:
                    # Aggregate trusted peers
                    g_peers_comm = torch.zeros_like(local_grad)
                    for rank in ranks_comm:
                        g_peers_comm += weights_comm[rank] * neighbor_grads_comm[rank][name]
                    # Personalize
                    p_grads_comm = self.weight * local_grad + (1.0 - self.weight) * g_peers_comm
                
                self.proj_grads[name] = ((1 - self.alpha) * p_grads_comp) + (self.alpha * p_grads_comm)
        
        return
    
    def project_gradients(self, lr):
        for name, p in self.model.module.named_parameters():
            if p.requires_grad:
                if name not in self.proj_grads:
                    continue
                
                p.grad.data = self.proj_grads[name].data
                
                if self.weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=self.weight_decay)
        
        if self.momentum != 0:
            if self.qgm:
                for p, p_prev, buf in zip(self.model.module.parameters(), 
                                         self.prev_params, self.momentum_buff):
                    buf.mul_(self.momentum).add_(p_prev.data - p.data, 
                                               alpha=(1.0 - self.momentum) / self.lr)
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