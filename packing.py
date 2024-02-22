# Codes are taken from https://github.com/qwopqwop200/GPTQ-for-LLaMa with modifications

def make_quant_linear(module, names, bits, groupsize, outlier, name=''):
    try:
        from auto_gptq.nn_modules.qlinear.qlinear_cuda import QuantLinear
        if isinstance(module, QuantLinear):
            return
        for attr in dir(module):
            tmp = getattr(module, attr)
            name1 = name + '.' + attr if name != '' else attr
            if name1 in names:
                delattr(module, attr)
                setattr(module, attr, QuantLinear(bits, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None))
        for name1, child in module.named_children():
            make_quant_linear(child, names, bits, groupsize, outlier, name + '.' + name1 if name != '' else name1)
    except Exception as e:
        print(f"Cannot replace Linear layer with QuantLinear layer due to Exception: {e}")
