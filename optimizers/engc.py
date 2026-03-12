import torch
import copy
import math
from collections import defaultdict

from .utils import flatten_tensors, unflatten_tensors


class ENGC_sender():
    def __init__(self, true_model, device):
        """
        Args:
            true_model: local model on current node
            device: cuda device
        """
        self.model = copy.deepcopy(true_model)
        self.model.train()
        self.model = self.model.to(device)
        self.gradient_buffer = {}
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(device)

    def _update_model(self, state_dict):
        """
        Args:
            state_dict: parameters of a neighbor model
        """
        for w, p in zip(state_dict, self.model.parameters()):
            p.data.copy_(w.data)
        return

    def _accumulate_gradients(self, x, targets):
        """
        Compute CE gradient on (x, targets) for the current sender model.
        """
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
        return

    def _flatten_(self, G):
        grad = []
        for g in G.values():
            grad.append(g)
        return flatten_tensors(grad).to(self.device)

    def __call__(self, neighbor_weight, batch_x, targets):
        """
        Args:
            neighbor_weight: neighbor model weights
            batch_x, targets: local mini-batch

        Returns:
            output: flattened cross-gradients for each neighbor model
            g: reference unflattened gradient buffer (used for shape only)
        """
        output = {}
        g = None
        for rank, w in neighbor_weight.items():
            self._update_model(w)
            g = self._accumulate_gradients(batch_x, targets)
            output[rank] = self._flatten_(g)
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
        nesterov=True,
        weight_decay=0,
        neighbors=2,
        alpha=1.0,
        # ---- soft-weight params ----
        self_weight=0.60,
        score_momentum=0.90,
        temperature=0.20,
        align_weight=0.75,
        norm_weight=0.25,
        min_peer_weight=0.00,
        eps=1e-12,
    ):
        self.model = model
        self.rank = rank
        self.device = device
        self.proj_grads = {}

        # Original NGC branch mixer
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

        # ---- new: evidence-inspired soft weighting ----
        self.self_weight = float(self_weight)
        self.score_momentum = float(score_momentum)
        self.temperature = float(temperature)
        self.align_weight = float(align_weight)
        self.norm_weight = float(norm_weight)
        self.min_peer_weight = float(min_peer_weight)
        self.eps = float(eps)

        # EMA branch-wise compatibility scores
        self.branch_score_ema = {
            "comm": {},   # rank -> score
            "comp": {},   # rank -> score
        }

        # for logging/debug
        self.last_weights = {
            "comm": {},
            "comp": {},
        }

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

    def _flatten_grad_dict(self, grad_dict):
        chunks = []
        for _, g in grad_dict.items():
            chunks.append(g.contiguous().view(-1))
        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks, dim=0)

    def _cosine(self, a, b):
        denom = (torch.norm(a) * torch.norm(b)).clamp_min(self.eps)
        return torch.dot(a, b) / denom

    def _norm_agreement(self, a, b):
        """
        Returns a score in (0, 1], high if norms are similar.
        """
        na = torch.norm(a).clamp_min(self.eps)
        nb = torch.norm(b).clamp_min(self.eps)
        ratio = torch.maximum(na / nb, nb / na)
        # ratio >= 1 ; convert to smooth similarity
        return torch.exp(-torch.log(ratio))

    def _raw_branch_score(self, self_flat, neigh_flat):
        """
        Evidence-inspired scalar compatibility score:
        - positive alignment matters most
        - norm mismatch is penalized softly
        """
        cos = self._cosine(self_flat, neigh_flat)
        cos_pos = torch.clamp(cos, min=0.0)  # suppress opposite-direction gradients
        norm_sim = self._norm_agreement(self_flat, neigh_flat)

        score = self.align_weight * cos_pos + self.norm_weight * norm_sim
        return float(score.item())

    def _ema_update(self, branch, peer_rank, score):
        prev = self.branch_score_ema[branch].get(peer_rank, None)
        if prev is None:
            new_score = score
        else:
            new_score = self.score_momentum * prev + (1.0 - self.score_momentum) * score
        self.branch_score_ema[branch][peer_rank] = new_score
        return new_score

    def _normalize_peer_weights(self, peer_scores, branch):
        """
        Convert peer scores -> weights, and reserve self_weight for local gradient.
        peer_scores: dict[rank] -> scalar score
        Returns:
            self_w, peer_w_dict
        """
        if len(peer_scores) == 0:
            self.last_weights[branch] = {"self": 1.0}
            return 1.0, {}

        # stabilize with temperature softmax
        scores = []
        ranks = []
        for r, s in peer_scores.items():
            ranks.append(r)
            scores.append(max(float(s), 0.0))

        score_tensor = torch.tensor(scores, device=self.device, dtype=torch.float32)
        # if all zero, fall back to uniform peers
        if float(score_tensor.sum().item()) <= self.eps:
            peer_prob = torch.ones_like(score_tensor) / float(len(scores))
        else:
            peer_prob = torch.softmax(score_tensor / max(self.temperature, 1e-6), dim=0)

        self_w = min(max(self.self_weight, 0.0), 1.0)
        peer_mass = 1.0 - self_w

        peer_w = {}
        for idx, r in enumerate(ranks):
            peer_w[r] = float((peer_mass * peer_prob[idx]).item())

        # optional floor on peer weights, then renormalize peer mass
        if self.min_peer_weight > 0.0 and len(peer_w) > 0:
            floor = min(self.min_peer_weight, peer_mass / float(len(peer_w)))
            cur = sum(peer_w.values())
            if cur > 0:
                scaled = {}
                for r, w in peer_w.items():
                    scaled[r] = max(w, floor)
                z = sum(scaled.values())
                if z > 0:
                    for r in scaled:
                        scaled[r] = scaled[r] * (peer_mass / z)
                peer_w = scaled

        dbg = {"self": self_w}
        dbg.update(peer_w)
        self.last_weights[branch] = dbg
        return self_w, peer_w

    def _compute_branch_weights(self, neighbor_grad_dicts, ref_buf, branch_name):
        """
        neighbor_grad_dicts: dict[rank] -> dict[param_name] -> tensor
        ref_buf: self/local gradient dict
        """
        self_flat = self._flatten_grad_dict(ref_buf).detach()

        peer_scores = {}
        for peer_rank, grad_dict in neighbor_grad_dicts.items():
            neigh_flat = self._flatten_grad_dict(grad_dict).detach()
            raw_score = self._raw_branch_score(self_flat, neigh_flat)
            ema_score = self._ema_update(branch_name, peer_rank, raw_score)
            peer_scores[peer_rank] = ema_score

        return self._normalize_peer_weights(peer_scores, branch_name)

    def __call__(self, neighbor_grads_comm, neighbor_grads_comp, ref_buf):
        """
        Args:
            neighbor_grads_comm: dict[rank] -> flattened gradient
                gradients received from neighbors through communication
            neighbor_grads_comp: dict[rank] -> flattened gradient
                local recomputed cross-gradients using neighbor weights
            ref_buf: local self gradient dict (used both as shape ref and trust anchor)
        """
        # Unflatten both branches first
        for rank, flat_tensor in neighbor_grads_comm.items():
            neighbor_grads_comm[rank] = self._unflatten_(flat_tensor, ref_buf)

        for rank, flat_tensor in neighbor_grads_comp.items():
            neighbor_grads_comp[rank] = self._unflatten_(flat_tensor, ref_buf)

        # Branch-wise neighbor weights (global scalar per neighbor, not per-parameter)
        self_w_comm, peer_w_comm = self._compute_branch_weights(
            neighbor_grads_comm, ref_buf, branch_name="comm"
        )
        self_w_comp, peer_w_comp = self._compute_branch_weights(
            neighbor_grads_comp, ref_buf, branch_name="comp"
        )

        # Aggregate parameter gradients using learned neighbor weights
        for name, self_params in self.model.module.named_parameters():
            if not self_params.requires_grad:
                continue

            self_grad = self_params.grad.data

            # communication branch
            p_grads_comm = self_w_comm * self_grad
            for peer_rank, neigh_grad in neighbor_grads_comm.items():
                p_grads_comm = p_grads_comm + peer_w_comm.get(peer_rank, 0.0) * neigh_grad[name]

            # computation branch
            p_grads_comp = self_w_comp * self_grad
            for peer_rank, neigh_grad in neighbor_grads_comp.items():
                p_grads_comp = p_grads_comp + peer_w_comp.get(peer_rank, 0.0) * neigh_grad[name]

            # keep original NGC branch fusion
            self.proj_grads[name] = ((1.0 - self.alpha) * p_grads_comp) + (self.alpha * p_grads_comm)

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

        # original momentum logic
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