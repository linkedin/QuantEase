# Codes are taken from https://github.com/IST-DASLab/gptq with modifications

import json
import os

import numpy as np
import torch

ROOT_DIR = "."

os.environ["HF_DATASETS_OFFLINE"] = "true"


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext(nsamples, seed, seqlen, model, cached=True, read_json=True):
    if not cached:
        from datasets import load_dataset, load_from_disk
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train',
                                 cache_dir=os.path.join(ROOT_DIR, 'data'))
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir=os.path.join(ROOT_DIR, 'data'))

        traindata.save_to_disk(os.path.join(ROOT_DIR, 'data', 'wikitext', 'wiki-train'))
        testdata.save_to_disk(os.path.join(ROOT_DIR, 'data', 'wikitext', 'wiki-test'))
    elif read_json is False:
        from datasets import load_dataset, load_from_disk
        traindata = load_from_disk(os.path.join(ROOT_DIR, 'data', 'wikitext', 'wiki-train'))
        testdata = load_from_disk(os.path.join(ROOT_DIR, 'data', 'wikitext', 'wiki-test'))
        traindata = traindata['text']
        testdata = testdata['text']
    else:
        f = open(os.path.join(ROOT_DIR, 'data', 'wikitrain.json'))
        traindata = json.load(f)
        f = open(os.path.join(ROOT_DIR, 'data', 'wikitest.json'))
        testdata = json.load(f)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, local_files_only=cached,
                                              cache_dir=os.path.join(ROOT_DIR, 'data'), trust_remote_code=True,
                                              use_auth_token=False)
    trainenc = tokenizer("\n\n".join(traindata), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model, cached=True, read_json=True):
    if not cached:
        from datasets import load_dataset, load_from_disk
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train',
                                 cache_dir=os.path.join(ROOT_DIR, 'data'))
        valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation',
                               cache_dir=os.path.join(ROOT_DIR, 'data'))
        traindata.save_to_disk(os.path.join(ROOT_DIR, 'data', 'ptb', 'ptb-train'))
        valdata.save_to_disk(os.path.join(ROOT_DIR, 'data', 'ptb', 'ptb-val'))
    elif not read_json:
        from datasets import load_dataset, load_from_disk
        traindata = load_from_disk(os.path.join(ROOT_DIR, 'data', 'ptb', 'ptb-train'))
        valdata = load_from_disk(os.path.join(ROOT_DIR, 'data', 'ptb', 'ptb-val'))
        traindata = traindata['sentence']
        valdata = valdata['sentence']
    else:
        f = open(os.path.join(ROOT_DIR, 'data', 'ptbtrain.json'))
        traindata = json.load(f)
        f = open(os.path.join(ROOT_DIR, 'data', 'ptbval.json'))
        valdata = json.load(f)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, local_files_only=cached,
                                              cache_dir=os.path.join(ROOT_DIR, 'data'), trust_remote_code=True)
    trainenc = tokenizer("\n\n".join(traindata), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_val(nsamples, seed, seqlen, model, cached=True):
    if not cached:
        from datasets import load_dataset
        valdata = load_dataset(
            'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
            split='validation',
            cache_dir=os.path.join(ROOT_DIR, 'data')
        )
        valdata.save_to_disk(os.path.join(ROOT_DIR, 'data', 'c4', 'c4-val'))
    else:
        from datasets import load_from_disk
        valdata = load_from_disk(os.path.join(ROOT_DIR, 'data', 'c4', 'c4-val'))

    from transformers import AutoTokenizer
    print(model)
    print(os.path.join(ROOT_DIR, 'data'))
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    import random
    random.seed(seed)
    trainloader = []
    for jjj in range(nsamples):
        while True:
            i = random.randint(0, len(valdata) - 1)
            trainenc = tokenizer(valdata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random

    # valenc = tokenizer("\n\n".join(valdata['text']), return_tensors='pt')
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    if 'wikitext2' in name:
        return get_wikitext(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        return get_c4_val(nsamples, seed, seqlen, model)
