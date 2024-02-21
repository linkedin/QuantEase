import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from datautils import get_loaders
from method.gptq_quantease import GPTQ_QuantEase
from method.quant import Quantizer
from method.quantease import QuantEase
from method.quantease_outlier import QuantEaseOutlier
from method.rtn import RTN
from modelutils import find_layers

try:
    from modelutils import find_quantlinear_layers
    from packing import make_quant_linear
except ImportError as e:
    print(f"Cannot do model packing due to ImportError: {e}")


@dataclass
class QuantizationConfig:
    model: str = "/shared/public/models/falcon-7b"
    dataset: str = "c4"
    seed: int = 2
    nsamples: int = 128
    wbits: int = 4
    groupsize: int = -1
    sym: bool = False
    num_iter: int = 30
    outlier: float = 0
    alpha: float = 1
    save: str = ''
    true_sequential: bool = False
    quantization_method: str = 'quantease'
    act_order: bool = False
    static_groups: bool = False
    structure_outlier: bool = False
    compute_quantization_recon_error: bool = False
    num_layers_to_quantize: int = 1000


class ModelQuantizerBase:
    inside_layer_modules: List[str]
    quantizer_key_prefix: str = ""
    has_alibi: bool = False

    def __init__(self, quantization_config: Optional[QuantizationConfig] = None):
        if quantization_config is None:
            self.quantization_config = QuantizationConfig()
        else:
            self.quantization_config = quantization_config
        self.model = ModelQuantizerBase.get_model(self.quantization_config.model)
        if hasattr(self.model, 'seqlen'):
            self.model.seqlen = self.model.seqlen if self.model.seqlen <= 2048 else 2048
        else:
            self.model.seqlen = 2048  # replace this with a constant
        self.all_layers = self.get_all_layers()
        self.word_embeddings = self.get_word_embedding_layer()

    @abstractmethod
    def get_word_embedding_layer(self):
        pass

    @abstractmethod
    def get_all_layers(self):
        pass

    @abstractmethod
    def get_transformer_output(self):
        pass

    @abstractmethod
    def get_lm_head(self):
        pass

    def _get_data_loader(self, dataset=""):
        dataloader, test_dataloader = get_loaders(
            dataset if dataset else self.quantization_config.dataset,
            nsamples=self.quantization_config.nsamples,
            seed=self.quantization_config.seed,
            model=self.quantization_config.model,
            seqlen=self.model.seqlen
        )  # Calibration data
        return dataloader, test_dataloader

    @staticmethod
    def get_model(model_name, cached=True):

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip
        from transformers import AutoModelForCausalLM
        print(f"model name: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)
        models_with_seqlen2048 = ["bloom", "falcon"]
        models_with_seqlen2_max_position_embeddings = ["opt", "mistral"]
        if any([m in model_name for m in models_with_seqlen2048]):
            model.seqlen = 2048
        if any([m in model_name for m in models_with_seqlen2_max_position_embeddings]):
            model.seqlen = model.config.max_position_embeddings
        # Set model in evaluation mode so no gradient update will be performed.
        model.eval()
        print(model)
        return model

    @torch.no_grad()
    def sequential(self, dev):
        """
        Method to sequentially quantize the model based on inside_layer_modules.
        """
        # The following block taken from https://github.com/IST-DASLab/gptq with modifications
        print("Prepare dataloader")
        dataloader, _ = self._get_data_loader()
        print(f"Quantization method {self.quantization_config.quantization_method}")

        print('Starting ...')
        has_alibi = self.has_alibi
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        layers = self.all_layers

        self.word_embeddings = self.word_embeddings.to(dev)
        if "bloom" in self.quantization_config.model:
            self.word_embeddings_layer_norm = self.get_word_embedding_layer_norm().to(dev)
        if "opt" in self.quantization_config.model:
            self.embed_positions = self.get_embed_positions().to(dev)
            self.project_in = self.get_project_in().to(dev) if self.get_project_in() else None
            self.project_out = self.get_project_out().to(dev) if self.get_project_out() else None
        layers[0] = layers[0].to(dev)

        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros(
            (self.quantization_config.nsamples, self.model.seqlen, self.model.config.hidden_size), dtype=dtype,
            device="cpu",  # dev
        )
        cache = {'i': 0, 'attention_mask': None, 'alibi': None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp.to(device="cpu")
                # inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                cache['alibi'] = kwargs['alibi'] if has_alibi else None
                raise ValueError

        layers[0] = Catcher(layers[0])
        for batch in dataloader:
            try:
                print(batch)
                self.model(batch[0].to(dev))
            except ValueError:
                pass

        layers[0] = layers[0].module
        layers[0] = layers[0].cpu()
        self.word_embeddings = self.word_embeddings.cpu()
        if "bloom" in self.quantization_config.model:
            self.word_embeddings_layer_norm = self.word_embeddings_layer_norm.cpu()
        if "opt" in self.quantization_config.model:
            self.embed_positions = self.embed_positions.cpu()
            self.project_in = self.project_in.cpu() if self.project_in else None
            self.project_out = self.project_out.cpu() if self.project_out else None
        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        alibi = cache['alibi']

        print('Ready to quantize.')
        quantizers = {}
        error_list = []

        inside_layer_modules = self.inside_layer_modules
        if not self.quantization_config.true_sequential:
            inside_layer_modules = [sum(inside_layer_modules, [])]
        for i in range(len(layers)):
            print(f"Start quantizing layer {i + 1}/{len(layers)}")
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            layer = layers[i].to(dev)
            print("layer_num:", i)
            if i < self.quantization_config.num_layers_to_quantize:
                full = find_layers(layer, enable=True,
                                   num_layers_to_quantize=self.quantization_config.num_layers_to_quantize)
            else:
                full = find_layers(layer, enable=False,
                                   num_layers_to_quantize=self.quantization_config.num_layers_to_quantize)
            quant_algo_dict = {}
            print('----')

            for names in inside_layer_modules:
                subset = {n: full[n] for n in names if n in full}
                if subset == {}:
                    continue
                for name in subset:
                    if self.quantization_config.outlier > 0:  # The quantizer with outliers
                        quant_algo_dict[name] = QuantEaseOutlier(subset[name],
                                                                 num_iter=self.quantization_config.num_iter,
                                                                 outlier=self.quantization_config.outlier,
                                                                 structure_outlier=self.quantization_config.structure_outlier,
                                                                 compute_quantization_recon_error=self.quantization_config.compute_quantization_recon_error)
                        quant_algo_dict[name].quantizer = Quantizer()
                        if self.quantization_config.structure_outlier:
                            quant_algo_dict[name].quantizer.configure(
                                self.quantization_config.wbits, perchannel=True, sym=self.quantization_config.sym,
                                mse=False,
                            )
                        else:
                            quant_algo_dict[name].quantizer.configure(
                                self.quantization_config.wbits, perchannel=True,
                                sym=self.quantization_config.sym, mse=False,
                                outlier=self.quantization_config.outlier
                            )
                    else:
                        if self.quantization_config.quantization_method == "quantease":
                            quant_algo_dict[name] = QuantEase(subset[name], num_iter=self.quantization_config.num_iter,
                                                              compute_quantization_recon_error=self.quantization_config.compute_quantization_recon_error)
                        elif self.quantization_config.quantization_method == "gptq_quantease":
                            quant_algo_dict[name] = GPTQ_QuantEase(subset[name],
                                                                   num_iter=self.quantization_config.num_iter,
                                                                   compute_quantization_recon_error=self.quantization_config.compute_quantization_recon_error)
                        elif self.quantization_config.quantization_method == "rtn":
                            quant_algo_dict[name] = RTN(subset[name],
                                                        compute_quantization_recon_error=self.quantization_config.compute_quantization_recon_error)
                        quant_algo_dict[name].quantizer = Quantizer()
                        quant_algo_dict[name].quantizer.configure(
                            self.quantization_config.wbits, perchannel=True, sym=self.quantization_config.sym, mse=False
                        )

                # The following block taken from https://github.com/IST-DASLab/gptq
                def add_batch(name):
                    def tmp(_, inp, out):
                        quant_algo_dict[name].add_batch(inp[0].data, out.data)  # noqa: F821

                    return tmp

                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                print(f"Preprocessing stage time: {curr_time}, before sample inference")

                starter.record()
                for j in range(self.quantization_config.nsamples):
                    if self.has_alibi:
                        outs[j] = \
                        layer(inps[j].unsqueeze(0).to(device=dev), attention_mask=attention_mask, alibi=alibi)[0].to(
                            device="cpu")
                    else:
                        outs[j] = layer(inps[j].unsqueeze(0).to(device=dev), attention_mask=attention_mask)[0].to(
                            device="cpu")
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                print(inps.device)
                print(f"Preprocessing stage  time: {curr_time}")

                starter.record()
                for h in handles:
                    h.remove()

                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                print(f"handles remove time: {curr_time}")

                for name in subset:
                    print(i, name)
                    print('Quantizing ...')
                    starter.record()
                    if self.quantization_config.outlier > 0:
                        if self.quantization_config.quantization_method == "spqr":
                            scale, zero, g_idx = quant_algo_dict[name].quantize(bits=self.quantization_config.wbits,
                                                                                groupsize=self.quantization_config.groupsize if self.quantization_config.groupsize != -1 else None,
                                                                                # noqa: E501
                                                                                sym=self.quantization_config.sym,
                                                                                outlier_relative_threshold=self.quantization_config.outlier,
                                                                                simplified_outliers=True,
                                                                                save_quantization=False,
                                                                                keep_H=False,
                                                                                )
                        else:
                            scale, zero, g_idx = quant_algo_dict[name].fasterquant(
                                group_size=self.quantization_config.groupsize,
                                static_groups=self.quantization_config.static_groups,
                                actorder=self.quantization_config.act_order,
                                alpha=self.quantization_config.alpha)
                    else:
                        scale, zero, g_idx = quant_algo_dict[name].fasterquant(
                            group_size=self.quantization_config.groupsize,
                            static_groups=self.quantization_config.static_groups,
                            actorder=self.quantization_config.act_order)

                    quant_algo_dict[name].free()

                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    print(f"Quantize layer {name} time: {curr_time}")

                    starter.record()

                    quant_algo_dict[name].quantizer, scale, zero, g_idx = None, scale.to(device="cpu"), zero.to(
                        device="cpu"), g_idx.to(device="cpu")
                    # Save information
                    quantizers[f'{self.quantizer_key_prefix}.{i}.{name}'] = (
                        quant_algo_dict[name].quantizer,
                        scale,
                        zero,
                        g_idx
                    )
                    if self.quantization_config.outlier > 0:
                        quant_algo_dict[name].layer.weight.data = (
                                    quant_algo_dict[name].W + quant_algo_dict[name].S).clone().to(
                            dev)

                    if self.quantization_config.compute_quantization_recon_error:
                        error_list.append(quant_algo_dict[name].error.cpu().item())
                    quant_algo_dict[name].free()

                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    print(f"Save time after quantization for layer {name} time: {curr_time}")

            starter.record()
            for j in range(self.quantization_config.nsamples):
                if self.has_alibi:
                    outs[j] = layer(inps[j].unsqueeze(0).to(device=dev), attention_mask=attention_mask, alibi=alibi)[
                        0].to(device="cpu")
                else:
                    outs[j] = layer(inps[j].unsqueeze(0).to(device=dev), attention_mask=attention_mask)[0].to(
                        device="cpu")

            print(layer)
            print(inps.device)

            layers[i] = layer.cpu()
            del layer
            del quant_algo_dict
            torch.cuda.empty_cache()

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            print(f"Post-quantization inference regeneration time for subset {subset} time: {curr_time}")

            inps, outs = outs, inps

        self.model.config.use_cache = use_cache

        save_dict = {"quantizers": quantizers, "error": error_list}

        return save_dict  # return the network weights

    @torch.no_grad()
    def eval(self, testenc, dev):
        """
        Method to evaluate the model.
        """
        # Taken from https://github.com/IST-DASLab/gptq
        print('Evaluation...')

        testenc = testenc.input_ids
        nsamples = testenc.numel() // self.model.seqlen

        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        layers = self.all_layers

        self.word_embeddings = self.word_embeddings.to(dev)
        if "bloom" in self.quantization_config.model:
            self.word_embeddings_layer_norm = self.get_word_embedding_layer_norm().to(dev)
        if "opt" in self.quantization_config.model:
            self.embed_positions = self.get_embed_positions().to(dev)
            self.project_in = self.get_project_in().to(dev) if self.get_project_in() else None
            self.project_out = self.get_project_out().to(dev) if self.get_project_out() else None
        layers[0] = layers[0].to(dev)

        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros(
            (nsamples, self.model.seqlen, self.model.config.hidden_size), dtype=dtype, device=dev
        )
        cache = {'i': 0, 'attention_mask': None, 'alibi': None}
        has_alibi = self.has_alibi

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                cache['alibi'] = kwargs['alibi'] if has_alibi else None
                raise ValueError

        layers[0] = Catcher(layers[0])
        for i in range(nsamples):
            batch = testenc[:, (i * self.model.seqlen):((i + 1) * self.model.seqlen)].to(dev)
            try:
                self.model(batch)
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        self.word_embeddings = self.word_embeddings.cpu()
        if "bloom" in self.quantization_config.model:
            self.word_embeddings_layer_norm = self.word_embeddings_layer_norm.cpu()
        if "opt" in self.quantization_config.model:
            self.embed_positions = self.embed_positions.cpu()
            self.project_in = self.project_in.cpu() if self.project_in else None
            self.project_out = self.project_out.cpu() if self.project_out else None
        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        alibi = cache['alibi']

        for i in range(len(layers)):
            print(i)
            layer = layers[i].to(dev)

            for j in range(nsamples):
                if self.has_alibi:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
            inps, outs = outs, inps

        transformer_output = self.get_transformer_output()
        if transformer_output is not None:
            transformer_output = transformer_output.to(dev)
        # For falcon
        if callable(getattr(self, 'get_project_out', None)):
            if self.get_project_out() is None:
                project_out = None
            else:
                project_out = self.get_project_out().to(dev)
        else:
            project_out = None
        lm_head = self.get_lm_head().to(dev)
        testenc = testenc.to(dev)
        nlls = []
        for i in range(nsamples):
            hidden_states = inps[i].unsqueeze(0)
            if transformer_output is not None:
                hidden_states = transformer_output(hidden_states)
            # For falcon
            if project_out is not None:
                hidden_states = project_out(hidden_states)
            lm_logits = lm_head(hidden_states)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = testenc[:, (i * self.model.seqlen):((i + 1) * self.model.seqlen)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * self.model.seqlen
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * self.model.seqlen))
        print(ppl.item())

        self.model.config.use_cache = use_cache
        return ppl.item()

    def pack(self, quantizers):
        try:
            layers = find_layers(self.model, enable=False,
                                 num_layers_to_quantize=self.quantization_config.num_layers_to_quantize)
            layers = {n: layers[n] for n in quantizers}
            make_quant_linear(self.model, quantizers, self.quantization_config.wbits,
                              self.quantization_config.groupsize, outlier=self.quantization_config.outlier)
            from auto_gptq.nn_modules.qlinear.qlinear_cuda import QuantLinear
            qlayers = find_quantlinear_layers(self.model, [QuantLinear], enable=False,
                                              num_layers_to_quantize=self.quantization_config.num_layers_to_quantize)
            print('Packing ...')
            for name in qlayers:
                if name == "lm_head":
                    continue
                print(name)

                scale = quantizers[name][1].clone().to("cpu")
                zero = quantizers[name][2].clone().to("cpu")
                g_idx = quantizers[name][3].clone().to("cpu")

                print(f"scale max {torch.max(scale)}, min {torch.min(scale)}, shape {scale.shape}")
                print(f"zero max {torch.max(zero)}, min {torch.min(zero)}, shape {zero.shape}")
                print(f"g_idx max {torch.max(g_idx)}, min {torch.min(g_idx)}, shape {g_idx.shape}")

                qlayers[name].pack(layers[name], scale, zero, g_idx)
            print('Done.')
        except Exception as e:
            print(f"Cannot do model packing due to Exception: {e}. Return original quantized model without packing.")
