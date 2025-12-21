
from scipy.linalg import orth
import torch
from torch.autograd import Variable
import copy
import numpy as np
from .utils import flatten_tensors, unflatten_tensors
from collections import defaultdict, OrderedDict

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
             comp_self_weight_scale=0.5,
             prox_mu=0.0, prox_warmup_steps=200,
             blend_lambda_max=0.25, blend_warmup_steps=500, blend_ramp_steps=2000, min_self_weight=0.2
):

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
        self.layer_param_map = OrderedDict()
        # FedProx config
        self.prox_mu = prox_mu
        self.prox_warmup_steps = prox_warmup_steps
        self.step = 0
        self.anchor_params = {}
        # Blend baseline(uniform) + confidence
        self.blend_lambda_max = blend_lambda_max
        self.blend_warmup_steps = blend_warmup_steps
        self.blend_ramp_steps = blend_ramp_steps
        # Optional: tăng self weight để đỡ bị neighbor kéo lệch
        self.min_self_weight = min_self_weight

        for name, param in self.model.module.named_parameters():
            if not param.requires_grad:
                continue
            layer_key = self._layer_key(name)
            self.layer_param_map.setdefault(layer_key, []).append(name)
        for param in self.model.module.parameters():
            self.momentum_buff.append(torch.zeros_like(param.data))
            self.prev_params.append(copy.deepcopy(param.data))

    def _blend_lambda(self):
        # lambda schedule: 0 -> lambda_max
        if self.step < self.blend_warmup_steps:
            return 0.0
        t = self.step - self.blend_warmup_steps
        if self.blend_ramp_steps <= 0:
            return float(self.blend_lambda_max)
        frac = min(1.0, t / float(self.blend_ramp_steps))
        return float(self.blend_lambda_max) * frac


    def _layer_key(self, param_name):
        if '.' in param_name:
            return param_name.rsplit('.', 1)[0]
        return param_name
    
    def _compute_confidence_score(self, neighbor_grad, self_grad):
        device = self_grad.device
        eps = self.confidence_eps

        self_norm = torch.norm(self_grad)
        neigh_norm = torch.norm(neighbor_grad)

        if self_norm < eps or neigh_norm < eps:
            return torch.tensor(1.0, device=device, dtype=self_grad.dtype)

        cosine = torch.sum(self_grad * neighbor_grad) / (self_norm * neigh_norm + eps)
        cosine = torch.clamp(cosine, -1.0, 1.0)

        # Alignment in [0, 1]
        score = 0.5 * (cosine + 1.0)
        return score

    def _confidence_weighted_average_layer(
        self,
        neighbor_vecs,
        self_vec,
        trace_key="",
        self_weight_scale=1.0,
        a_min=0.5,
        temperature=1.0,
    ):
        if not neighbor_vecs:
            return self_vec.clone()

        scores = torch.stack([self._compute_confidence_score(vec, self_vec) for vec in neighbor_vecs])
        scores = torch.clamp(scores, min=a_min)

        weights = scores.pow(temperature)
        weights = weights / (weights.sum() + self.confidence_eps)

        neighbor_part = torch.zeros_like(self_vec)
        for w, vec in zip(weights, neighbor_vecs):
            neighbor_part.add_(vec, alpha=w.item())

        # self mixing (same rule as uniform+self)
        self_weight = max(self.min_self_weight, min(1.0, self.pi * self_weight_scale))
        neighbor_part.mul_(1.0 - self_weight)
        neighbor_part.add_(self_vec, alpha=self_weight)

        if trace_key:
            self.confidence_trace[trace_key] = {
                "weights": weights.detach().cpu().tolist(),
                "self_weight": self_weight,
            }

        return neighbor_part

    def _flatten_layer(self, grads_dict, param_names):
        flats = []
        shapes = []
        sizes = []
        for name in param_names:
            tensor = grads_dict.get(name)
            if tensor is None:
                continue
            flats.append(tensor.view(-1))
            shapes.append(tensor.shape)
            sizes.append(tensor.numel())
        if not flats:
            return None, [], []
        return torch.cat(flats), shapes, sizes

    def _layer_vector_to_params(self, layer_vec, param_names, shapes, sizes):
        result = {}
        offset = 0
        for name, shape, size in zip(param_names, shapes, sizes):
            result[name] = layer_vec[offset:offset+size].view(shape).clone()
            offset += size
        return result

 

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
        # 1) Unflatten neighbor grads
        for r, flat in neighbor_grads_comm.items():
            neighbor_grads_comm[r] = self._unflatten_(flat, ref_buf)
        for r, flat in neighbor_grads_comp.items():
            neighbor_grads_comp[r] = self._unflatten_(flat, ref_buf)

        # 2) step + blend factor (tính 1 lần)
        self.step += 1
        lam = self._blend_lambda()

        # 3) FedProx anchor: chỉ refresh theo chu kỳ (KHÔNG clone mỗi step)
        if (self.prox_mu is not None) and (self.prox_mu > 0):
            if (self.step == 1) or (self.step % 200 == 0) or (len(self.anchor_params) == 0):
                self.anchor_params = {
                    name: p.data.detach().clone()
                    for name, p in self.model.module.named_parameters()
                    if p.requires_grad
                }

        # 4) self grads dict
        self_grad_dict = {
            name: p.grad.data
            for name, p in self.model.module.named_parameters()
            if p.requires_grad
        }

        neighbor_comm_list = list(neighbor_grads_comm.values())
        neighbor_comp_list = list(neighbor_grads_comp.values())

        # 5) layer-wise aggregation
        for layer_name, param_names in self.layer_param_map.items():
            self_layer_vec, shapes, sizes = self._flatten_layer(self_grad_dict, param_names)
            if self_layer_vec is None:
                continue

            # ===== COMM =====
            comm_neighbor_vecs = []
            for neigh in neighbor_comm_list:
                layer_vec, _, _ = self._flatten_layer(neigh, param_names)
                if layer_vec is not None:
                    comm_neighbor_vecs.append(layer_vec)

            # uniform + self
            comm_self_w = max(self.min_self_weight, min(1.0, self.pi * self.comm_self_weight_scale))
            if len(comm_neighbor_vecs) > 0:
                comm_mean = torch.stack(comm_neighbor_vecs, dim=0).mean(dim=0)
            else:
                comm_mean = self_layer_vec
            comm_uniform = (1.0 - comm_self_w) * comm_mean + comm_self_w * self_layer_vec

            # confidence
            comm_conf = self._confidence_weighted_average_layer(
                comm_neighbor_vecs,
                self_layer_vec,
                trace_key=f"{layer_name}_comm",
                self_weight_scale=self.comm_self_weight_scale,
            )

            # blend
            comm_layer = (1.0 - lam) * comm_uniform + lam * comm_conf

            # ===== COMP =====
            comp_neighbor_vecs = []
            for neigh in neighbor_comp_list:
                layer_vec, _, _ = self._flatten_layer(neigh, param_names)
                if layer_vec is not None:
                    comp_neighbor_vecs.append(layer_vec)

            # uniform + self
            comp_self_w = max(self.min_self_weight, min(1.0, self.pi * self.comp_self_weight_scale))
            if len(comp_neighbor_vecs) > 0:
                comp_mean = torch.stack(comp_neighbor_vecs, dim=0).mean(dim=0)
            else:
                comp_mean = self_layer_vec
            comp_uniform = (1.0 - comp_self_w) * comp_mean + comp_self_w * self_layer_vec

            # confidence
            comp_conf = self._confidence_weighted_average_layer(
                comp_neighbor_vecs,
                self_layer_vec,
                trace_key=f"{layer_name}_comp",
                self_weight_scale=self.comp_self_weight_scale,
            )

            # blend
            comp_layer = (1.0 - lam) * comp_uniform + lam * comp_conf

            # final mix (NGC alpha)
            mixed_layer = ((1.0 - self.alpha) * comp_layer) + (self.alpha * comm_layer)

            # write back to per-param grads
            layer_chunks = self._layer_vector_to_params(mixed_layer, param_names, shapes, sizes)
            for n, chunk in layer_chunks.items():
                self.proj_grads[n] = chunk

        return
          

    def project_gradients(self, lr):
        # Applies the grad projections
        for name, p in self.model.module.named_parameters():
            if p.requires_grad:
                # 1) gradient sau aggregation
                p.grad.data = self.proj_grads[name].data

                # 2) weight decay nếu có
                if self.weight_decay != 0:
                    p.grad.data.add_(p.data, alpha=self.weight_decay)

                # 3) FedProx proximal term (KHÔNG nằm trong weight_decay)
                if (self.prox_mu is not None) and (self.prox_mu > 0) and (self.step >= self.prox_warmup_steps):
                    w_anc = self.anchor_params.get(name, None)
                    if w_anc is not None:
                        p.grad.data.add_(p.data - w_anc, alpha=self.prox_mu)

        # apply momentum (giữ nguyên code bạn)
        if self.momentum != 0:
            if self.qgm:
                for p, p_prev, buf in zip(self.model.module.parameters(), self.prev_params, self.momentum_buff):
                    buf.mul_(self.momentum).add_(p_prev.data - p.data, alpha=(1.0-self.momentum)/self.lr)
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
