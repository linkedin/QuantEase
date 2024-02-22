import torch

from .quant import quantize
from .quantizer import CD_quantizer

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class QuantEase(CD_quantizer):
    def __init__(self, layer, num_iter=30, compute_quantization_recon_error=False):
        super().__init__(layer, num_iter, compute_quantization_recon_error)

    def fasterquant(self, group_size=128, static_groups=False, actorder=True):

        starter_all, ender_all = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter_all.record()

        W = self.layer.weight.data.clone()
        print(f"W.shape: {W.shape}")
        W = W.float()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        if static_groups:
            import copy
            groups = []
            scale = []
            zero = []
            for i in range(0, self.columns, group_size):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + group_size)], weight=True)
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(self.XtX), descending=True)
            W = W[:, perm]
            self.XtX = self.XtX[perm][:, perm]
            invperm = torch.argsort(perm)

        W_hat = W.clone()
        self.XtX = self.XtX.to(self.dev)
        diag_Sigma = torch.diagonal(self.XtX, 0)

        norm_Sigma = torch.div(self.XtX, diag_Sigma)

        P = torch.matmul(W, norm_Sigma)
        # equivalent to (diagonal - 1) to help absorb -W matrix into XtXB for reducing computation of updating W[:, i]
        norm_Sigma.fill_diagonal_(0)

        if not self.compute_quantization_recon_error:
            self.XtX = self.XtX.to(
                "cpu")  # if memory is issue we can move it to cpu and do not compute the reconstruction error

        norm_Sigma = norm_Sigma.t()
        for iter_counter in range(self.num_iter):  # Go over columns
            delta_W_hat = W_hat.clone().t()
            P_hat = torch.matmul(norm_Sigma, delta_W_hat).t()  # Update entire XtXB before each iteration

            if not static_groups:
                scale = []
                zero = []
                now_idx = 1

            for j in range(self.columns):
                if j % 3000 == 0:
                    print(f"Currently processing {str(type(self.layer))} idx {j}.")

                if diag_Sigma[j] == 0:  # i.e. the column is equal to zero and is redundant
                    if group_size != -1:
                        if not static_groups:
                            if j % group_size == 0:
                                self.quantizer.find_params(W[:, j:(j + group_size)], weight=True)

                            if (j // group_size) - now_idx == -1:
                                scale.append(self.quantizer.scale)
                                zero.append(self.quantizer.zero)
                                now_idx += 1
                        else:
                            idx = j
                            if actorder:
                                idx = perm[idx]
                            self.quantizer = groups[idx // group_size]
                    u = quantize(
                        W_hat[:, j].unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()
                    W_hat[:, j] = u
                    delta_W_hat[j, :] -= u
                    continue
                u = P[:, j] - P_hat[:, j]
                if j > 0:
                    # Update single row i of XtXB, this is slightly different from the paper algorithm as we transpose
                    # both norm_Sigma and delta_W_hat and do the matmul by switching their order, this is faster and
                    # more memory efficient with the current torch matmul implementation based on CUDA GEMM kernel
                    # though does not have theoretical matmul complexity improvement
                    u += torch.matmul(norm_Sigma[j, :j], delta_W_hat[:j, :])

                # sometimes skip quantization. Last iteration is always quantized.
                if iter_counter % 3 != 1 or iter_counter == self.num_iter - 1 or iter_counter > 25:

                    if group_size != -1:
                        if not static_groups:
                            if j % group_size == 0:
                                self.quantizer.find_params(W[:, j:(j + group_size)], weight=True)

                            if (j // group_size) - now_idx == -1:
                                scale.append(self.quantizer.scale)
                                zero.append(self.quantizer.zero)
                                now_idx += 1
                        else:
                            idx = j
                            if actorder:
                                idx = perm[idx]
                            self.quantizer = groups[idx // group_size]

                    u = quantize(
                        u.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                W_hat[:, j] = u
                delta_W_hat[j, :] -= u

        ender_all.record()
        torch.cuda.synchronize()
        curr_time = starter_all.elapsed_time(ender_all)
        print('time %.2f' % curr_time)

        del P, P_hat

        if self.compute_quantization_recon_error:
            diff_W = W_hat - W
            Res = torch.diag(torch.matmul(torch.matmul(diff_W, self.XtX), diff_W.t()))

            del diff_W
            Res0 = torch.diag(torch.matmul(torch.matmul(W, self.XtX), W.t()))
            self.error = torch.sum(Res) / torch.sum(Res0)
            print("Reconstruction Error: " + str(self.error.cpu().item() * 100) + " %")

        group_size = group_size if group_size != -1 else self.columns
        if static_groups and actorder:
            g_idx = [perm[j] // group_size for j in range(self.columns)]
        else:
            g_idx = [j // group_size for j in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=self.dev)
        if actorder:
            W_hat = W_hat[:, invperm]
            g_idx = g_idx[invperm]

        # Update model weights
        self.layer.weight.data = W_hat.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        return scale, zero, g_idx
