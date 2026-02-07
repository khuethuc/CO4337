
from scipy.linalg import orth
import torch
from torch.autograd import Variable
import copy
import numpy as np
from .utils import flatten_tensors, unflatten_tensors
from collections import defaultdict
import math

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
    def __init__(self, model, device, rank, lr, momentum, qgm, 
                 nesterov=True, weight_decay=0, neighbors=2, alpha=1.0,
                 topk = 1, lambda_1 = 1.0, lambda_2 = 1.0, lambda_3 = 0.5, rho_ema = 0.1,
                 weight_self = 0, weight_model = 0, weight_data = 1.0):
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
        for param in self.model.module.parameters():
            self.momentum_buff.append(torch.zeros_like(param.data))
            self.prev_params.append(copy.deepcopy(param.data))
        self.topk = topk
        self.lambda_1, self.lambda_2, self.lambda_3 = lambda_1, lambda_2, lambda_3
        self.rho_ema = rho_ema
        self.mu = defaultdict(float)
        self.nu = defaultdict(float)
        self.weight_self = weight_self
        self.weight_model = weight_model
        self.weight_data = weight_data
    
    def average_gradients(self, grad):
        new_grad = torch.zeros_like(grad[-1])
        for g in grad:
            new_grad +=self.pi*g
        return new_grad #Ftorch.clamp(new_grad,min=-1.0,max=1.0)
 

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
        
        ### Compute self-gradients
        self_gradients = {}
        for name, self_params in self.model.module.named_parameters():
            if self_params.requires_grad:
                self_gradients[name] = self_params.grad.data

        keys = list(ref_buf.keys())
        self_flatten = flatten_tensors(
            [self_gradients[k] for k in keys]
        ).to(self.device)

        ### Compute utility score
        utilities = {}
        for rank in neighbor_grads_comp.keys():
            g_ji_flatten = flatten_tensors([neighbor_grads_comp[rank][k] for k in keys]).to(self.device)
            g_ij_flatten = flatten_tensors([neighbor_grads_comm[rank][k] for k in keys]).to(self.device)
            utilities[rank] = self.utility_score(rank, self_flatten, g_ij_flatten, g_ji_flatten)

        ### Choose top-k ranks
        ranks_sorted = sorted(utilities.keys(), key = lambda rank: utilities[rank], reverse = True)
        if self.topk is None or self.topk <= 0:
            selected = ranks_sorted
        else:
            selected = ranks_sorted[: min(self.topk, len(ranks_sorted))]

        ### Get the projected gradients for each parameter
        for name, self_params in self.model.module.named_parameters():
            if self_params.requires_grad:
                if len(selected) == 0:
                    self.proj_grads[name] = self_gradients[name]
                    continue

                g_model = torch.zeros_like(self_gradients[name])
                g_data  = torch.zeros_like(self_gradients[name])
                for rank in selected:
                    g_model += neighbor_grads_comp[rank][name]
                    g_data  += neighbor_grads_comm[rank][name]
                g_model /= float(len(selected))
                g_data  /= float(len(selected))

                self.proj_grads[name] = (
                    self.weight_self * self_gradients[name]
                    + self.weight_model * g_model
                    + self.weight_data  * g_data
                )

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
        
    def utility_score(self, rank: int, g_ii: torch.Tensor, g_ij: torch.Tensor, g_ji: torch.Tensor):
        """
        compatibility = lambda_1 * cos(g_ii, g_ji) + lambda_2 * cos(g_ii, g_ij)
        utility_score = compatibility - lambda_3 * uncertainty
        """
        g_ii = g_ii.to(self.device)
        g_ij = g_ij.to(self.device)
        g_ji = g_ji.to(self.device)
        # Compute cosine similarity
        a_model = float(self.cosine_alignment(g_ii, g_ji).item())
        print("a_model:", a_model)
        a_data  = float(self.cosine_alignment(g_ii, g_ij).item())
        print("a_data:", a_data)
        # Compute utility score
        compability = float(self.lambda_1) * a_model + float(self.lambda_2) * a_data
        self.update_running_statics(rank, compability)
        uncertainty = self.uncertainty(rank)
        utility_score = compability - float(self.lambda_3) * uncertainty
        return float(utility_score)       
    
    def cosine_alignment(self, g_ii: torch.Tensor, cross_gradients: torch.Tensor, eps: float = 1e-12):
        """
        returns scalar tensor
        """
        return torch.dot(g_ii, cross_gradients) / (g_ii.norm() * cross_gradients.norm() + eps)

    def uncertainty(self, rank: int):
        """
        EMA stats self.mu[rank], self.nu[rank]
        uncertainty_{ij,k} = sqrt(max(nu - mu^2, 0))
        """
        mu = float(self.mu[rank])
        nu = float(self.nu[rank])
        var = max(nu - mu * mu, 0.0)
        return math.sqrt(var)

    def update_running_statics(self, rank: int, compatibility: float):
        """
        mu = (1 - rho) * mu + rho * compatability
        nu = (1 - rho) * nu + rho * compatability^2
        """
        rho = float(self.rho_ema)
        mu_prev = float(self.mu[rank])
        nu_prev = float(self.nu[rank])
        mu = (1.0 - rho) * mu_prev + rho * compatibility
        nu = (1.0 - rho) * nu_prev + rho * (compatibility**2)
        self.mu[rank] = mu
        self.nu[rank] = nu

     
             
