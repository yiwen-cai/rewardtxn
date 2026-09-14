#!/usr/bin/env python3
"""Full-text deterministic diagnosis; never writes historical eval results."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from e7_eval_checkpoint_incontainer import load_eval_split, format_prompt
from day2_custom_rm import _v1_reward


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--compare', help='Second HF model for weight/logprob roundtrip audit')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(29)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).cuda().eval()
    samples = load_eval_split(args.split)
    ids = json.loads(args.split.read_text())['eval_indices']
    if args.limit:
        samples, ids = samples[:args.limit], ids[:args.limit]
    result = {'checkpoint': args.checkpoint, 'split': str(args.split),
              'split_sha256': hashlib.sha256(args.split.read_bytes()).hexdigest(),
              'max_new_tokens': 2048, 'batch_size': 8, 'seed': 29, 'results': []}
    if args.compare:
        other = AutoModelForCausalLM.from_pretrained(args.compare, torch_dtype=torch.bfloat16).cuda().eval()
        a, b = model.state_dict(), other.state_dict()
        assert a.keys() == b.keys()
        weight_errors = {k: float((a[k].float()-b[k].float()).abs().max()) for k in a}
        text = tok.apply_chat_template(format_prompt(samples[0]), tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors='pt').to('cuda')
        with torch.no_grad():
            la = model(**inputs).logits.float().log_softmax(-1)
            lb = other(**inputs).logits.float().log_softmax(-1)
            diff = (la-lb).abs()
            ga = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            gb = other.generate(**inputs, max_new_tokens=64, do_sample=False)
        result['roundtrip'] = {'compare': args.compare, 'max_weight_abs_diff': max(weight_errors.values()),
                               'nonidentical_tensors': {k:v for k,v in weight_errors.items() if v},
                               'mean_logprob_abs_diff': float(diff.mean()),
                               'max_logprob_abs_diff': float(diff.max()),
                               'greedy64_equal': torch.equal(ga,gb),
                               'pass': max(weight_errors.values()) == 0 and float(diff.mean()) <= .02
                               and float(diff.max()) <= .2 and torch.equal(ga,gb)}
        del a, b, other, la, lb, diff, ga, gb
        torch.cuda.empty_cache()
    for start in range(0, len(samples), 8):
        batch = samples[start:start+8]
        texts = [tok.apply_chat_template(format_prompt(s), tokenize=False, add_generation_prompt=True) for s in batch]
        inp = tok(texts, padding=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = model.generate(**inp, max_new_tokens=2048, do_sample=False,
                                     pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos,list) else [eos])
        for j, output in enumerate(outputs):
            tokens = output[inp.input_ids.shape[1]:].tolist()
            end = next((i for i,t in enumerate(tokens) if t in eos),len(tokens))
            valid = tokens[:end]
            response = tok.decode(valid, skip_special_tokens=True)
            grams = Counter(tuple(valid[i:i+4]) for i in range(max(0,len(valid)-3)))
            repeated = sum(v-1 for v in grams.values()) / max(1,len(valid)-3)
            label = batch[j].get('label', batch[j].get('answer',''))
            result['results'].append({'source_index': ids[start+j], 'prompt': texts[j],
                                      'response': response, 'label': label,
                                      'correct': _v1_reward(response,label) > .5,
                                      'generated_tokens': len(valid), 'truncated': end == len(tokens) and len(tokens) == 2048,
                                      'repeated_4gram_fraction': repeated})
        print(f"{args.checkpoint}: {len(result['results'])}/{len(samples)}", flush=True)
    result['n_total'] = len(samples)
    result['n_correct'] = sum(r['correct'] for r in result['results'])
    result['accuracy'] = result['n_correct'] / len(samples)
    result['truncated_fraction'] = sum(r['truncated'] for r in result['results']) / len(samples)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'results'}),flush=True)


if __name__ == '__main__':
    main()
