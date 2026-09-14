#!/usr/bin/env python3
"""E7 checkpoint evaluation (runs inside Docker container).

This script must run inside the Slime container where torch/vllm/transformers are available.
It loads a checkpoint, generates responses for the eval split, and grades them with the verifier.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import verifier functions
sys.path.insert(0, str(Path(__file__).parent))
from day2_custom_rm import _v1_reward, _v2_reward, extract_answer


def load_eval_split(split_path: Path) -> list[dict]:
    """Load eval samples from frozen split."""
    split_spec = json.loads(split_path.read_text())
    dataset_path = Path(split_spec["source"])
    
    # Resolve relative path from split file location
    if not dataset_path.is_absolute():
        # split_path is /workspace/prereg/eval_splits/...
        # source is models/datasets/gsm8k/...
        # Resolve from /workspace
        dataset_path = Path("/workspace") / split_spec["source"]
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with dataset_path.open() as f:
        all_samples = [json.loads(line) for line in f if line.strip()]
    
    eval_samples = [all_samples[idx] for idx in split_spec["eval_indices"]]
    assert len(eval_samples) == split_spec["eval_count"]
    return eval_samples


def format_prompt(sample: dict):
    """Return the prompt in the training rollout format (chat message list).

    The DAPO GSM8K split stores prompts as ``[{role, content}, ...]`` message
    lists under the ``prompt`` key; the training rollout applies the chat
    template with ``add_generation_prompt=True``.  Fall back to raw text for
    plain-text datasets (instruction/problem keys).
    """
    prompt = sample.get("prompt")
    if isinstance(prompt, list):
        return prompt
    instruction = sample.get("instruction", sample.get("problem", ""))
    return [{"role": "user", "content": instruction}]


def generate_responses_hf(checkpoint_path: Path, prompts: list, batch_size: int = 8) -> list[str]:
    """Generate responses using HuggingFace transformers (greedy decoding)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading checkpoint from {checkpoint_path}...", file=sys.stderr)
    
    # Attempt to load as HF checkpoint
    # For Megatron checkpoints, this will fail and we need a conversion step
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        # Fix: decoder-only models need left padding for batched generation
        tokenizer.padding_side = "left"
        
        # Fix: use single GPU to avoid device_map="auto" CPU offload issues
        # The docker wrapper maps 4 GPUs, but we only need 1 for 1.5B inference
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",  # explicit single GPU instead of "auto"
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading checkpoint: {e}", file=sys.stderr)
        print("This checkpoint may be in Megatron format; HF conversion required.", file=sys.stderr)
        raise
    
    model.eval()
    responses = []
    
    print(f"Generating {len(prompts)} responses...", file=sys.stderr)
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        # Match the training rollout formatting (--apply-chat-template with
        # add_generation_prompt=True in slime/utils/data.py).
        texts = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in batch
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        for output_ids, input_ids in zip(outputs, inputs.input_ids):
            response = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True)
            responses.append(response)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Generated {i + len(batch)}/{len(prompts)}", file=sys.stderr)
    
    return responses


def grade_responses(responses: list[str], labels: list[str], verifier_version: str) -> tuple[list[float], list[bool]]:
    """Grade responses with specified verifier."""
    verifier_fn = _v1_reward if verifier_version == "v1" else _v2_reward
    rewards = []
    corrects = []
    
    print(f"Grading {len(responses)} responses with verifier={verifier_version}...", file=sys.stderr)
    for response, label in zip(responses, labels):
        reward = verifier_fn(response, label)
        correct = reward > 0.5
        rewards.append(reward)
        corrects.append(correct)
    
    return rewards, corrects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eval-split", required=True, type=Path)
    parser.add_argument("--verifier-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    
    # Load eval split
    eval_samples = load_eval_split(args.eval_split)
    print(f"Loaded {len(eval_samples)} eval samples", file=sys.stderr)
    
    # Format prompts
    prompts = [format_prompt(s) for s in eval_samples]
    labels = [s.get("label", s.get("answer", "")) for s in eval_samples]
    
    # Generate responses
    try:
        responses = generate_responses_hf(args.checkpoint, prompts, args.batch_size)
    except Exception as e:
        print(f"Generation failed: {e}", file=sys.stderr)
        print("Note: Megatron checkpoints require HF conversion first", file=sys.stderr)
        sys.exit(1)
    
    # Grade responses
    rewards, corrects = grade_responses(responses, labels, args.verifier_version)
    
    # Compute accuracy
    n_correct = sum(corrects)
    accuracy = n_correct / len(eval_samples)
    
    # Prepare output
    results = []
    for idx, (sample, prompt, response, label, reward, correct) in enumerate(
        zip(eval_samples, prompts, responses, labels, rewards, corrects)
    ):
        # prompt 可能是消息 list（chat 格式）或 str；统一转紧凑字符串再截断
        if isinstance(prompt, str):
            prompt_text = prompt
        else:
            prompt_text = " ".join(
                m.get("content", "") for m in prompt if isinstance(m, dict)
            ) or json.dumps(prompt, ensure_ascii=False)
        results.append({
            "index": idx,
            "prompt": (prompt_text[:100] + "...") if len(prompt_text) > 100 else prompt_text,
            "response": (response[:200] + "...") if len(response) > 200 else response,
            "label": label,
            "reward": reward,
            "correct": correct,
        })
    
    output = {
        "checkpoint": str(args.checkpoint),
        "eval_split": str(args.eval_split),
        "verifier_version": args.verifier_version,
        "n_total": len(eval_samples),
        "n_correct": n_correct,
        "final_eval_accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "results": results,
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    
    print(f"\nEval complete: {n_correct}/{len(eval_samples)} correct ({accuracy*100:.2f}%)", file=sys.stderr)
    print(f"Results written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
