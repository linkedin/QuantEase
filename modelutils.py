# Codes are token from https://github.com/IST-DASLab/gptq with modifications

import torch
import torch.nn as nn

DEV = torch.device('cuda:0')
print(DEV)


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name='', enable=True, num_layers_to_quantize=1000):
    layer_ids = [str(i) for i in list(range(num_layers_to_quantize))]
    if (type(module) in layers or "Linear" in str(type(module)) or "Conv2D" in str(
            type(
                module))) and (enable or any([s in name.split(".") for s in layer_ids])):
        print(name)
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1, enable=enable,
            num_layers_to_quantize=num_layers_to_quantize
        ))
    return res


def find_quantlinear_layers(module, layers=[nn.Conv2d, nn.Linear], name='', enable=True, num_layers_to_quantize=1000):
    layer_ids = [str(i) for i in list(range(num_layers_to_quantize))]
    try:
        from auto_gptq.nn_modules.qlinear.qlinear_cuda import QuantLinear
        if layers == [QuantLinear]:
            if type(module) in layers and (enable or any([s in name.split(".") for s in layer_ids])):
                print(name)
                return {name: module}
            res = {}
            for name1, child in module.named_children():
                res.update(find_layers(
                    child, layers=layers, name=name + '.' + name1 if name != '' else name1, enable=enable,
                    num_layers_to_quantize=num_layers_to_quantize
                ))
        else:
            if (type(module) in layers or "Linear" in str(type(module)) or "Conv2D" in str(
                    type(
                        module))) and (enable or any([s in name.split(".") for s in layer_ids])):
                print(name)
                return {name: module}
            res = {}
            for name1, child in module.named_children():
                res.update(find_layers(
                    child, layers=layers, name=name + '.' + name1 if name != '' else name1, enable=enable,
                    num_layers_to_quantize=num_layers_to_quantize
                ))
    except Exception as e:
        print(f"Cannot find QuantLinear layers due to Exception: {e}. Return empty dict")
        res = {}
    return res


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
    models_with_seqlen2_max_position_embeddings = ["opt"]
    if any([m in model_name for m in models_with_seqlen2048]):
        model.seqlen = 2048
    if any([m in model_name for m in models_with_seqlen2_max_position_embeddings]):
        model.seqlen = model.config.max_position_embeddings
    print(model)
    return model
