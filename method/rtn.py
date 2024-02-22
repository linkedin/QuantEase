# Codes are modified based on the GPTQ algorithm from https://github.com/IST-DASLab/gptq

import time
from logging import getLogger

import torch
import torch.nn as nn
import transformers

from .quant import quantize

logger = getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class RTN:

    def __init__(self, layer, compute_quantization_recon_error=False, use_cpu=False):
        self.use_cpu = use_cpu
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.compute_quantization_recon_error = compute_quantization_recon_error
        if compute_quantization_recon_error:
            self.XtX = torch.zeros((self.columns, self.columns), device=self.dev)

    def add_batch(self, inp, out):
        if self.compute_quantization_recon_error:

            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
                if len(inp.shape) == 3:
                    inp = inp.reshape((-1, inp.shape[-1]))
                inp = inp.t()
            if isinstance(self.layer, nn.Conv2d):
                unfold = nn.Unfold(
                    self.layer.kernel_size,
                    dilation=self.layer.dilation,
                    padding=self.layer.padding,
                    stride=self.layer.stride
                )
                inp = unfold(inp)
                inp = inp.permute([1, 0, 2])
                inp = inp.flatten(1)

            recon_inp = inp.float()
            self.XtX += recon_inp.matmul(recon_inp.t())

    def fasterquant(
            self, group_size=-1, actorder=False, static_groups=False
    ):

        if self.use_cpu:
            self.H = self.H.to('cpu')
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        scale = []
        zero = []
        now_idx = 1

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, group_size):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + group_size)], weight=True)
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)

        if group_size == -1:
            Q = quantize(
                W, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
            )
        else:
            for i in range(self.columns):
                if not static_groups:
                    if i % group_size == 0:
                        self.quantizer.find_params(W[:, i:(i + group_size)], weight=True)

                    if (i // group_size) - now_idx == -1:
                        scale.append(self.quantizer.scale)
                        zero.append(self.quantizer.zero)
                        now_idx += 1
                else:
                    idx = i
                    self.quantizer = groups[idx // group_size]
                Q = quantize(
                    W, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                )
        print('time %.2f' % (time.time() - tick))
        torch.cuda.synchronize()

        if self.compute_quantization_recon_error:
            print(f"Q: {Q.shape}")
            print(f"original_W: {W.shape}")
            diff_QW = Q - W
            Res = torch.diag(torch.matmul(torch.matmul(diff_QW, self.XtX), diff_QW.t()))

            del diff_QW
            Res0 = torch.diag(torch.matmul(torch.matmul(W, self.XtX), W.t()))
            self.error = torch.sum(Res) / torch.sum(Res0)
            logger.info("Error: " + str(self.error.cpu().item() * 100) + " %")

        group_size = group_size if group_size != -1 else self.columns
        g_idx = [i // group_size for i in range(self.columns)]

        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        return scale, zero, g_idx

    def free(self):
        torch.cuda.empty_cache()
