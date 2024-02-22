# The GPTQ part of codes are taken from https://github.com/IST-DASLab/gptq

import math
import os
import time
from logging import getLogger
import torch
import torch.nn as nn
import transformers

from .quant import quantize

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

logger = getLogger(__name__)


class GPTQ_QuantEase:
    def __init__(self, layer, num_iter=2, compute_quantization_recon_error=False):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.pytorch_utils.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.XtX = torch.zeros((self.columns, self.columns), device=self.dev)
        self.num_iter = num_iter
        self.compute_quantization_recon_error = compute_quantization_recon_error

    def add_batch(self, inp, out):
        if os.environ.get('DEBUG'):
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
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
        # For quant ease case, it needs sum of hessian across data points
        quant_ease_inp = inp.float()
        self.XtX += quant_ease_inp.matmul(quant_ease_inp.t())
        # For gptq case, it needs moving average
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        mat_inc = inp.matmul(inp.t())
        self.H += mat_inc

    def fasterquant(
            self, blocksize=128, percdamp=.01, group_size=-1, actorder=False, static_groups=False
    ):
        starter_all, ender_all = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter_all.record()

        W = self.layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        # Save another copy here for QuantEase quantization range computation and computing the construction error
        original_W = W.clone()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        g_idx = []
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

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if group_size != -1:
                    if not static_groups:
                        if (i1 + i) % group_size == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + group_size)], weight=True)

                        if ((i1 + i) // group_size) - now_idx == -1:
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            now_idx += 1
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // group_size]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if os.environ.get('DEBUG'):
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                logger.debug(torch.sum(Losses))

        torch.cuda.synchronize()
        logger.info(f'GPTQ quantization duration: {(time.time() - tick)}')
        logger.info(f'GPTQ avg loss: {torch.sum(Losses).item() / self.nsamples}')

        group_size = group_size if group_size != -1 else self.columns
        if static_groups and actorder:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)

        GPTQ_W = W.clone()
        if self.compute_quantization_recon_error:
            diff_BW = Q - original_W
            Res = torch.diag(torch.matmul(torch.matmul(diff_BW, self.XtX), diff_BW.t()))

            del diff_BW
            Res0 = torch.diag(torch.matmul(torch.matmul(original_W, self.XtX), original_W.t()))
            self.error = torch.sum(Res) / torch.sum(Res0)
            logger.info("GPTQ Error: " + str(self.error.cpu().item() * 100) + " %")

        # logger.info(f"Output layerwise loss after GPTQ (on last data point in a batch): {torch.sum((self.layer(self.inp1) - self.out1) ** 2)}")
        # We can claim memory back as we no longer need hessian moving average required form GPTQ
        del W, H, Hinv, W1, Q1, Err1, Losses1, Hinv1, Q
        torch.cuda.empty_cache()

        # Initialize quant ease algorithm by using GPTQ quantized weight
        # Weight after gptq initialiation
        W = self.layer.weight.data.clone().float()

        if actorder:
            perm = torch.argsort(torch.diag(self.XtX), descending=True)
            W = W[:, perm]
            original_W = original_W[:, perm]
            self.XtX = self.XtX[perm][:, perm]
            invperm = torch.argsort(perm)

        B = W.clone().t()
        self.XtX = self.XtX.to(self.dev)
        diag_XtX = torch.diagonal(self.XtX, 0)

        norm_XtX = torch.div(self.XtX, diag_XtX)

        self.YtX = torch.matmul(W, norm_XtX)
        norm_XtX = norm_XtX.t()
        norm_XtX.fill_diagonal_(0)  # equivalent to (diagonal - 1) to help absorb -B matrix into XtXB for reducing computation of updating B[i, :]

        for iter_counter in range(self.num_iter):  # Go over columns
            delta_B = B.clone()
            XtXB = torch.matmul(norm_XtX, B)  # Update entire XtXB before each iteration

            if not static_groups:
                scale = []
                zero = []
                now_idx = 1

            for i in range(self.columns):
                if i % 3000 == 0:
                    print(f"Currently processing {str(type(self.layer))} idx {i}.")

                if diag_XtX[i] == 0:  # i.e. the column is equal to zero and is redundant
                    if group_size != -1:
                        if not static_groups:
                            if i % group_size == 0:
                                self.quantizer.find_params(GPTQ_W[:, i:(i + group_size)], weight=True)

                            if (i // group_size) - now_idx == -1:
                                scale.append(self.quantizer.scale)
                                zero.append(self.quantizer.zero)
                                now_idx += 1
                        else:
                            idx = i
                            if actorder:
                                idx = perm[idx]
                            self.quantizer = groups[idx // group_size]
                    tmp_Bi = quantize(
                        B[i, :].unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()
                    B[i, :] = tmp_Bi
                    delta_B[i, :] -= tmp_Bi
                    continue

                tmp_XtXBi = XtXB[i, :]
                if i > 0:
                    tmp_XtXBi -= torch.matmul(norm_XtX[i, :i], delta_B[:i, :])  # Update single row i of XtXB
                tmp_Bi = self.YtX[:, i] - tmp_XtXBi

                # sometimes skip quantization. Last iteration is always quantized.
                if iter_counter % 3 != 1 or iter_counter == self.num_iter - 1 or iter_counter > 25:

                    if group_size != -1:
                        if not static_groups:
                            if i % group_size == 0:
                                self.quantizer.find_params(GPTQ_W[:, i:(i + group_size)], weight=True)

                            if (i // group_size) - now_idx == -1:
                                scale.append(self.quantizer.scale)
                                zero.append(self.quantizer.zero)
                                now_idx += 1
                        else:
                            idx = i
                            if actorder:
                                idx = perm[idx]
                            self.quantizer = groups[idx // group_size]

                    tmp_Bi = quantize(
                        tmp_Bi.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                B[i, :] = tmp_Bi
                delta_B[i, :] -= tmp_Bi

        ender_all.record()
        torch.cuda.synchronize()
        curr_time = starter_all.elapsed_time(ender_all)

        if static_groups and actorder:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=self.dev)

        if self.compute_quantization_recon_error:
            diff_BW = B - original_W.t()
            Res = torch.diag(torch.matmul(torch.matmul(diff_BW.t(), self.XtX), diff_BW))

            del diff_BW
            Res0 = torch.diag(torch.matmul(torch.matmul(original_W, self.XtX), original_W.t()))
            self.error = torch.sum(Res) / torch.sum(Res0)
            logger.info("Error: " + str(self.error.cpu().item() * 100) + " %")

        if actorder:
            B = B[invperm, :]
            g_idx = g_idx[invperm]

        del XtXB

        # Update model weights
        if isinstance(self.layer, transformers.Conv1D) or "Conv1d" in str(type(self.layer)):
            self.layer.weight.data = B.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        else:
            self.layer.weight.data = B.t().reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        return scale, zero, g_idx

    def free(self):
        if os.environ.get('DEBUG'):
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        self.XtX = None
        self.YtX = None
        torch.cuda.empty_cache()
