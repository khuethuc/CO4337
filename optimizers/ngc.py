
from scipy.linalg import orth
import torch
from torch.autograd import Variable
import copy
import numpy as np
from .utils import flatten_tensors, unflatten_tensors
from collections import defaultdict

class NGC_sender():
    def __init__(self, true_model, device):
        """
            Args
                model: the model on the sender device
                device: device on which the model is 
                include_norm: includes norm weights to gpm computation 
        """
        self.model           = copy.deepcopy(true_model)
        self.model.train()
        self.model           = self.model.to(device)
        self.gradient_buffer = {}
        self.device          = device
        self.criterion       = torch.nn.CrossEntropyLoss().to(device)
        
        

    def _update_model(self, state_dict):
        """
            Args:
                state_dict: list of device parameters 
        """
        for w, p in zip(state_dict, self.model.parameters()):
                p.data.copy_(w.data)
        return

    def _accumulate_gradients(self, x, targets):
        """
            Args:
                x: inputs for which the gradients have to be accumulated
                targets: class labels for x
        """
        output = self.model(x)
        self.model.zero_grad()
        loss   = self.criterion(output, targets)
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
        """
            Args
                G: Input to be flattened
            Returns
                flattened tensor
        """
        grad = []
        for g in G.values():
            grad.append(g)
        return flatten_tensors(grad).to(self.device)

    def __call__(self, neighbor_weight, batch_x, targets):
        """
            Args
                neighbor_weight: weights of the neighbor models
                batches: Input batches to compute variance
            Returns
                flattened gradients for each neighbor
        """

        output = {}
        for rank, w in neighbor_weight.items():
            self._update_model(w)
            g = self._accumulate_gradients(batch_x, targets)
            output[rank] = self._flatten_(g)
        return output, g


class NGC_receiver():
    def __init__(self, model, device, rank, lr, momentum, qgm, nesterov=True, weight_decay=0, neighbors=2, alpha=1.0,
                 confidence_beta=2.0, confidence_floor=1e-4, confidence_eps=1e-12, comm_self_weight_scale=1.0,
                 comp_self_weight_scale=0.5):
        self.model         = model
        self.rank          = rank
        self.device        = device
        self.proj_grads    = {}
        self.pi            = 1.0/float(neighbors+1)      
        self.alpha         = alpha
        self.momentum      = momentum
        self.lr            = lr
        self.nesterov      = nesterov
        self.qgm           = qgm
        self.weight_decay  = weight_decay
        self.momentum_buff = []
        self.prev_params   = []
        self.confidence_beta = confidence_beta
        self.confidence_floor = confidence_floor
        self.confidence_eps = confidence_eps
        self.comm_self_weight_scale = comm_self_weight_scale
        self.comp_self_weight_scale = comp_self_weight_scale
        self.confidence_trace = defaultdict(dict)
        for param in self.model.module.parameters():
            self.momentum_buff.append(torch.zeros_like(param.data))
            self.prev_params.append(copy.deepcopy(param.data))
    
    def _compute_confidence_score(self, neighbor_grad, self_grad):
        """Returns alignment-aware confidence for a neighbor gradient."""
        diff_norm = torch.norm(self_grad - neighbor_grad)
        self_norm = torch.norm(self_grad)
        neigh_norm = torch.norm(neighbor_grad)
        cosine = torch.sum(self_grad * neighbor_grad) / (self_norm * neigh_norm + self.confidence_eps)
        cosine = torch.clamp(cosine, -1.0, 1.0)
        alignment = 0.5 * (cosine + 1.0)
        relative_diff = diff_norm / (self_norm + self.confidence_eps)
        penalty = torch.exp(-self.confidence_beta * relative_diff)
        score = alignment * penalty + self.confidence_floor
        return score


    def _confidence_weighted_average(self, neighbor_grads, self_grad, param_name="", self_weight_scale=1.0):
        if not neighbor_grads:
            return self_grad.clone()

        scores = []
        for neigh_grad in neighbor_grads:
            scores.append(self._compute_confidence_score(neigh_grad, self_grad))

        scores = torch.stack(scores)
        total = scores.sum()
        if total <= self.confidence_eps or torch.isnan(total):
            weights = torch.ones_like(scores) / len(neighbor_grads)
        else:
            weights = scores / total

        neighbor_part = torch.zeros_like(self_grad)
        for w, grad in zip(weights, neighbor_grads):
            neighbor_part.add_(grad, alpha=w.item())

        self_weight = max(0.0, min(1.0, self.pi * self_weight_scale))
        neighbor_budget = max(self.confidence_eps, 1.0 - self_weight)
        neighbor_part.mul_(neighbor_budget)
        neighbor_part.add_(self_grad, alpha=self_weight)

        if param_name:
            self.confidence_trace[param_name] = {
                "weights": [float(w) for w in weights.detach().cpu()],
                "self_weight": self_weight
            }

        return neighbor_part
 

    def _unflatten_(self, flat_tensor, ref_buf):
        """
            Args
                flat_tensor: received flat tensor to be reshaped 
                ref_buf: reference buffer for computing unflattened shape
            Returns
                unflattened tensor based on reference tensor
        """
        ref  = []
        keys = []
        for key,val in ref_buf.items():
            ref.append(val)
            keys.append(key)
        unflat_tensor =  unflatten_tensors(flat_tensor, ref)
        X = {}
        for i, key in enumerate(keys):
            X[key] = unflat_tensor[i]
        return X

    def __call__(self, neighbor_grads_comm, neighbor_grads_comp, ref_buf):
        """
            Args
                flat_tensor: received flat tensor to be reshaped 
                ref_buf: reference buffer for computing unflattened shape
            Returns
                computes orthogonal projection space and stores in self.Z
        """
        ### Unflatten the neighbor grads
        for rank, flat_tenor  in neighbor_grads_comm.items():
            neighbor_grads_comm[rank] = self._unflatten_(flat_tenor, ref_buf)
        for rank, flat_tenor  in neighbor_grads_comp.items():
            neighbor_grads_comp[rank] = self._unflatten_(flat_tenor, ref_buf)
        
        
        #get the projected gradients for each parameter
        for name, self_params in self.model.module.named_parameters():
            if self_params.requires_grad:
                self_grad = self_params.grad.data
                cross_grads_comm = []
                for rank, neigh_grad in neighbor_grads_comm.items():
                    cross_grads_comm.append(neigh_grad[name])
                p_grads_comm  = self._confidence_weighted_average(
                    cross_grads_comm,
                    self_grad,
                    param_name=f"{name}_comm",
                    self_weight_scale=self.comm_self_weight_scale
                )
                
                cross_grads_comp = []
                for rank, neigh_grad in neighbor_grads_comp.items():
                    cross_grads_comp.append(neigh_grad[name])
                p_grads_comp  = self._confidence_weighted_average(
                    cross_grads_comp,
                    self_grad,
                    param_name=f"{name}_comp",
                    self_weight_scale=self.comp_self_weight_scale
                )
                
                self.proj_grads[name] = ((1-self.alpha)*p_grads_comp)+(self.alpha*p_grads_comm)
        return 
                

    def project_gradients(self, lr):
        """
            Returns
                applies the changes to the model
        """
        ### Applies the grad projections
        for name, p in self.model.module.named_parameters():
            if p.requires_grad:
                p.grad.data = self.proj_grads[name].data 
                if self.weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=self.weight_decay)
        
        #apply momentum
        if self.momentum!=0:
            if self.qgm:
                for p, p_prev, buf in zip(self.model.module.parameters(), self.prev_params, self.momentum_buff):
                    buf.mul_(self.momentum).add_(p_prev.data-p.data, alpha=(1.0-self.momentum)/self.lr) #m_hat
                    mom_buff = copy.deepcopy(buf)
                    mom_buff.mul_(self.momentum).add_(p.grad.data) #m
                    if self.nesterov:
                        p.grad.data.add_(mom_buff, alpha=self.momentum) #nestrove momentum
                    else:
                        p.grad.data.copy_(mom_buff) 
                for p, p_prev in zip(self.model.module.parameters(), self.prev_params):
                    p_prev.data.copy_(p.data)
            else:
                for p, buf in zip(self.model.module.parameters(), self.momentum_buff):
                    buf.mul_(self.momentum).add_(p.grad.data)
                    if self.nesterov:
                        p.grad.data.add_(buf, alpha=self.momentum) #nestrove momentum
                    else:
                        p.grad.data.copy_(buf) 

        self.lr = lr
        
        
                
             
