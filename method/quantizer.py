from abc import ABC

import torch
import torch.nn as nn
import transformers


class CD_quantizer(ABC):

    def __init__(self, layer, num_iter, compute_quantization_recon_error=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.num_iter = num_iter
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        self.XtX = torch.zeros((self.columns, self.columns), device=self.dev)
        self.compute_quantization_recon_error = compute_quantization_recon_error

    # The following block, taken from https://github.com/IST-DASLab/gptq (with minor modifications)
    def add_batch(self, inp, out):
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
        inp = inp.float()

        self.XtX += inp.matmul(inp.t())  # Calculating Sigma = X * X^T

    def free(self):
        self.XtX = None
        self.YtX = None
        torch.cuda.empty_cache()
