
from scipy.linalg import orth
import torch
from torch.autograd import Variable
import copy
import numpy as np
from .utils import flatten_tensors, unflatten_tensors
from collections import defaultdict
import math
from .ngc import NGC_sender, NGC_receiver

        

class MedHEAdaptiveTopK:
    """
    MedHE Algorithm 1: Adaptive Top-k sparsification with error feedback + EMA threshold.
    Trạng thái (e, tau) được giữ theo 'key' (ví dụ neighbor rank).
    """
    def __init__(self, sparsity: float = 0.9, tau_alpha: float = 0.9, device=None):
        assert 0.0 <= sparsity < 1.0
        assert 0.0 < tau_alpha < 1.0
        self.sparsity = sparsity
        self.tau_alpha = tau_alpha
        self.device = device

        self.err = {}  # key -> torch.Tensor (same shape as grad)
        self.tau = {}  # key -> float/torch scalar

    @torch.no_grad()
    def compress(self, g: torch.Tensor, key):
        # init state
        if key not in self.err or self.err[key].shape != g.shape:
            self.err[key] = torch.zeros_like(g)
            self.tau[key] = None

        g_comp = g + self.err[key]  # error feedback

        d = g_comp.numel()
        k = int(math.floor((1.0 - self.sparsity) * d))  # number kept
        if k <= 0:
            g_sparse = torch.zeros_like(g_comp)
            self.err[key] = g_comp  # carry all
            return g_sparse

        mags = g_comp.abs()

        # tau_current = k-th largest magnitude
        # dùng topk để lấy ngưỡng; (O(d log k)) nhưng đủ ổn cho demo
        topk_vals = torch.topk(mags, k, largest=True, sorted=False).values
        tau_current = topk_vals.min()

        # EMA threshold
        if self.tau[key] is None:
            tau_t = tau_current
        else:
            tau_t = self.tau_alpha * self.tau[key] + (1.0 - self.tau_alpha) * tau_current
        self.tau[key] = tau_t

        mask = (mags >= tau_t)
        g_sparse = g_comp * mask

        # update error
        self.err[key] = g_comp - g_sparse
        return g_sparse


class MedHE_NGC_sender(NGC_sender):
    """
    NGC_sender nhưng sparsify gradient phẳng trước khi gửi (data-variant cross-gradients).
    """
    def __init__(self, true_model, device, sparsity=0.9, tau_alpha=0.9):
        super().__init__(true_model, device)
        self.compressor = MedHEAdaptiveTopK(sparsity=sparsity, tau_alpha=tau_alpha, device=device)

    def __call__(self, neighbor_weight, batch_x, targets):
        output = {}
        last_ref_buf = None

        for rank, w in neighbor_weight.items():
            self._update_model(w)
            g_dict = self._accumulate_gradients(batch_x, targets)
            flat = self._flatten_(g_dict)

            # MedHE adaptive top-k + error feedback per neighbor rank
            flat_sparse = self.compressor.compress(flat, key=rank)

            output[rank] = flat_sparse
            last_ref_buf = g_dict  # ref for unflatten

        return output, last_ref_buf
                
class MedHE_NGC_receiver(NGC_receiver):
    pass
