import json
import time

import torch
from transformers import AutoConfig

from modeling import CAUSAL_LM_QUANTIZER_MAP, QuantizationConfig
from modelutils import DEV, find_layers

try:
    from modelutils import find_quantlinear_layers
    from packing import make_quant_linear
except ImportError as e:
    print(f"Cannot do model packing due to ImportError: {e}")


def evaluate_quantized_model(quantizer, datasets):
    print("Start eval the quantized model")
    ppl = {}
    for dataset in datasets:
        _, test_dataloader = quantizer._get_data_loader(dataset)
        ppl[dataset] = quantizer.eval(test_dataloader, DEV)  # Evaluate on test data
    return ppl


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        help='LLM model to load; pass `tiiuae/falcon-X`.'
    )

    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )

    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )

    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )

    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 10, 12, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )

    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')

    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )

    parser.add_argument(
        '--num-iter', type=int, default=30,
        help='Number of iterations for QuantEase.'
    )

    parser.add_argument(
        '--outlier', type=float, default=0,
        help='fraction of outlier. eg: 0.01 corresponds to 1%, roughly 0.16 bits.'
    )

    parser.add_argument(
        '--alpha', type=float, default=1,
        help='step size multiplier for outlier IHT step'
    )

    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )

    parser.add_argument(
        '--true-sequential', action='store_true', default='',
        help='If using true sequential order to quantize layer by layer.'
    )

    parser.add_argument(
        '--quantization-method', type=str, default='quantease',
        help='The quantization method to use "gptq_quantease" or "quantease".'
    )

    parser.add_argument(
        '--num-layers-to-quantize', type=int, default=1000,
        help='How many blocks to quantize from beginning (mainly used for debugging purpose).'
    )

    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic, recommend to use'
    )

    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--act_order` for more efficient inference.'
    )

    parser.add_argument(
        '--structure-outlier', action='store_true', default='',
        help='If using structure outlier detection'
    )

    parser.add_argument(
        '--compute-quantization-recon-error', action='store_true', default='',
        help='If computing the quantization reconstruction error'
    )

    args = parser.parse_args()

    assert args.outlier >= 0, "outlier should be non-negative"
    # assert args.outlier < 1, "outlier should be less than one"
    assert args.alpha > 0, "alpha should be positive"

    quantization_config = QuantizationConfig(
        model=args.model,
        dataset=args.dataset,
        seed=args.seed,
        nsamples=args.nsamples,
        wbits=args.wbits,
        groupsize=args.groupsize,
        sym=args.sym,
        num_iter=args.num_iter,
        outlier=args.outlier,
        alpha=args.alpha,
        save=args.save,
        true_sequential=args.true_sequential,
        quantization_method=args.quantization_method,
        act_order=args.act_order,
        static_groups=args.static_groups,
        structure_outlier=args.structure_outlier,
        compute_quantization_recon_error=args.compute_quantization_recon_error,
        num_layers_to_quantize=args.num_layers_to_quantize
    )

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    from modeling import SUPPORTED_MODELS
    if config.model_type not in SUPPORTED_MODELS:
        raise TypeError(f"{config.model_type} isn't supported yet.")
    model_type = config.model_type
    print(f"model type of model: {args.model} is {model_type}")
    quantizer = CAUSAL_LM_QUANTIZER_MAP[model_type](quantization_config)

    runtime = 0
    save_dict = {}
    if args.wbits < 16:
        tick = time.time()
        save_dict = quantizer.sequential(DEV)  # run quantization
        runtime = time.time() - tick

    print(f"runtime: {runtime}")
    print("Start eval the quantized model")
    datasets = ['wikitext2', 'ptb']  # Test datasets
    ppl = evaluate_quantized_model(quantizer, datasets)
    print(f"ppl quant before packing: {ppl}")

    data_to_save = {'args': vars(args), 'ppl_before_packing': ppl, "error": save_dict.get("error", []),
                    "runtime": runtime}

    if args.save:
        if args.outlier == 0:
            save_model_path = f"{args.save}_seed_{args.seed}"
            quantizer.pack(save_dict["quantizers"])
            torch.save(quantizer.model.state_dict(), save_model_path)
            print("Start eval the packed model")
            datasets = ['wikitext2', 'ptb']
            ppl_quant_before_saving = evaluate_quantized_model(quantizer, datasets)
            print(f"ppl quant before packing: {ppl}")
            print(f"ppl quant before saving: {ppl_quant_before_saving}")

            del quantizer
            torch.cuda.empty_cache()
            quantizer = CAUSAL_LM_QUANTIZER_MAP[model_type](quantization_config)
            new_model = quantizer.model

            layers = find_layers(new_model, enable=True, num_layers_to_quantize=args.num_layers_to_quantize)
            for name in ['lm_head']:
                if name in layers:
                    del layers[name]
            make_quant_linear(new_model, layers, args.wbits, args.groupsize, outlier=args.outlier)
            del layers

            new_model.load_state_dict(torch.load(save_model_path), strict=False)
            print(new_model)

            datasets = ['wikitext2', 'ptb']
            ppl_quant_after_loading = evaluate_quantized_model(quantizer, datasets)
            print(f"ppl quant before packing: {ppl}")
            print(f"ppl quant before saving: {ppl_quant_before_saving}")
            print(f"ppl quant after model loading: {ppl_quant_after_loading}")

            data_to_save.update({'ppl_quant_packed_before_saving': ppl_quant_before_saving,
                                 'ppl_quant_after_loading': ppl_quant_after_loading})
    else:
        # Save results and required logs to a JSON file
        with open(f"{args.model.split('/')[-1]}_seed_{args.seed}_data.json", 'w') as json_file:
            json.dump(data_to_save, json_file)
