import numpy as np
import torch

from .quant import quantize
from .quantizer import CD_quantizer

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class QuantEaseOutlier(CD_quantizer):

    def __init__(self, layer, num_iter=30, outlier=0, structure_outlier=True, compute_quantization_recon_error=False):
        super().__init__(layer, num_iter, compute_quantization_recon_error)

        self.structure_outlier = structure_outlier
        self.outlier = outlier

    def fasterquant(self, group_size=128, static_groups=False, actorder=True, alpha=1):
        starter_all, ender_all = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter_all.record()

        W = self.layer.weight.data.clone()
        print(f"W.shape: {W.shape}")
        W = W.float()


        if actorder:
            perm = torch.argsort(torch.diag(self.XtX), descending=True)
            W = W[:, perm]
            self.XtX = self.XtX[perm][:, perm]
            invperm = torch.argsort(perm)

        W_hat = W.clone()
        self.XtX = self.XtX.to(self.dev)

        if self.outlier > 0:
            # Calculate the step size and sparsity level
            # Since we are calculating the top k column here, we divide by number of rows
            topk_k = int(np.round(self.columns * self.outlier)) if self.structure_outlier \
                else int(np.round(self.rows * self.columns * self.outlier))
            step = alpha / torch.lobpcg(self.XtX, k=1)[0] / 2
            print(f"Top k is {topk_k}")

            # Initialize B and H (Quantized and Sparse parts, respectivelty)
            # Initilize H with top k largest coordinates of original weights
            # Initialize B such that B + H = original weights
            H = W.clone()
            if self.structure_outlier:
                column_norms = torch.norm(H, dim=0)
                # column_norms = torch.sum(H.abs(), dim=1)
                sorted_norm_indices = torch.argsort(column_norms, descending=True)
                top_indices = sorted_norm_indices[:topk_k]
                reverse_top_indices = sorted_norm_indices[topk_k:]
                temp = H[:, top_indices]
                H = torch.zeros((self.rows, self.columns), device=self.dev)
                H[:, top_indices] = temp
                W_hat[:, top_indices] = torch.zeros_like(W_hat[:, top_indices])
            else:
                H = H.flatten()
                W_hat = W_hat.flatten()
                _, I_mat = H.abs().topk(topk_k)
                temp = H[I_mat]
                H = torch.zeros(self.rows * self.columns, device=self.dev)
                H[I_mat] = temp
                W_hat[I_mat] = torch.zeros_like(H[I_mat])
                H = H.unflatten(0, (self.rows, self.columns))
                W_hat = W_hat.unflatten(0, (self.rows, self.columns))

        else:
            # Initialize normally like QuantEase
            H = torch.zeros_like(W_hat)


        P = torch.matmul(W, self.XtX)
        HSigma = torch.matmul(H, self.XtX)

        P_hat = torch.matmul(W_hat, self.XtX)  # Update entire W_hat * Sigma before each iteration
        for iter_counter in range(self.num_iter):
            pre_W_hat = W_hat.clone()
            delta_W_hat = W_hat.clone().t()

            scale = []
            zero = []
            now_idx = 1

            if group_size == -1:
                if self.structure_outlier:
                    self.quantizer.find_params(W[:, reverse_top_indices], weight=True)
                elif not self.quantizer.ready():
                    self.quantizer.find_params(W, weight=True)
            elif self.structure_outlier:
                boolean_mask = torch.ones(self.columns, dtype=torch.bool)
                boolean_mask[top_indices] = False

            for j in range(self.columns):  # QuantEase updates overall similar to QuantEase from quantease.py

                if j % 3000 == 0:
                    print(f"Currently processing {str(type(self.layer))} idx {j}.")

                Aq = self.XtX[j, j]
                if Aq == 0:  # i.e. the column is equal to zero and is redundant
                    if group_size != -1:
                        if j % group_size == 0:
                            if self.structure_outlier:
                                self.quantizer.find_params(W[:, j:(j + group_size)][:, boolean_mask[j:(j + group_size)]], weight=True)
                            else:
                                self.quantizer.find_params(W[:, j:(j + group_size)], weight=True)

                        if (j // group_size) - now_idx == -1:
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            now_idx += 1
                    u = quantize(
                        W_hat[:, j].unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()
                    W_hat[:, j] = u
                    delta_W_hat[j, :] -= u
                    continue

                # Linear coefficients for calculating the one-dimensional optimal solution (see Lemma 1 in the paper)
                u = (P[:, j] - P_hat[:, j] - HSigma[:, j]) / Aq + pre_W_hat[:, j]
                if j > 0:
                    u += torch.matmul(self.XtX[:j, j].t(), delta_W_hat[:j, :])  # Update single row i of XtXB

                # sometimes skip quantization. Last iteration is always quantized.
                if iter_counter % 3 != 1 or iter_counter == self.num_iter - 1 or iter_counter > 25:
                    if group_size != -1:
                        if j % group_size == 0:
                            if self.structure_outlier:
                                self.quantizer.find_params(W[:, j:(j + group_size)][:, boolean_mask[j:(j + group_size)]], weight=True)
                            else:
                                self.quantizer.find_params(W[:, j:(j + group_size)], weight=True)

                        if (j // group_size) - now_idx == -1:
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            now_idx += 1
                    u = quantize(
                        u.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                P_hat[:, j] = u
                delta_W_hat[j, :] -= u

            P_hat = torch.matmul(W_hat, self.XtX)  # Update entire XtXB before each iteration

            if self.outlier > 0:  # IHT updates for the sparse outliers (Algorithm 2)

                R = HSigma + P_hat - P  # Residuals, used to calculate nabla_H g (see Lemma 2)
                H -= R * step  # Gradient Descent Step (Htilde <- H - step_size * nabla_H g)
                # Projection of top-k (update rule for the sparse part)
                if self.structure_outlier:
                    column_norms = torch.norm(H, dim=0)
                    # column_norms = torch.sum(H.abs(), dim=1)
                    sorted_norm_indices = torch.argsort(column_norms, descending=True)
                    top_indices = sorted_norm_indices[:topk_k]
                    reverse_top_indices = sorted_norm_indices[topk_k:]
                    temp = H[:, top_indices]
                    H = torch.zeros((self.rows, self.columns), device=self.dev)
                    H[:, top_indices] = temp
                else:
                    H = H.flatten()
                    _, I_mat = H.abs().topk(topk_k)
                    temp = H[I_mat]
                    H = torch.zeros(self.rows * self.columns, device=self.dev)
                    H[I_mat] = temp  # Threshold smaller coordinates to zero and keep larger coordinates
                    H = H.unflatten(0, (self.rows, self.columns))
                HSigma = torch.matmul(H, self.XtX)  # Update H * X^T * X

            if self.compute_quantization_recon_error:
                diff_W = W_hat + H - W
                Res = torch.diag(torch.matmul(torch.matmul(diff_W, self.XtX), diff_W.t()))

                del diff_W
                Res0 = torch.sum(torch.diag(torch.matmul(torch.matmul(W, self.XtX), W.t())))
                self.error = torch.sum(Res) / Res0
                print("Error: " + str(self.error.cpu().item() * 100) + " %")

        ender_all.record()
        torch.cuda.synchronize()
        curr_time = starter_all.elapsed_time(ender_all)
        print('time %.2f' % curr_time)


        group_size = group_size if group_size != -1 else self.columns
        g_idx = [j // group_size for j in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=self.dev)
        if actorder:
            W_hat = W_hat[:, invperm]
            H = H[:, invperm]
            g_idx = g_idx[invperm]

        # Save dense (quantized) and sparse matrices
        self.W = W_hat.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype).to('cpu').clone()
        self.S = H.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype).to('cpu').clone()

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        return scale, zero, g_idx
