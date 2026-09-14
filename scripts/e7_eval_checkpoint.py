#!/usr/bin/env python3
"""E7 evaluation entry point: checkpoint → final_eval_accuracy.

This script loads a trained checkpoint, generates responses for the evaluation
split, grades them with the same verifier used during training, and computes
pass@1 accuracy. The evaluation split is fixed before training (frozen in
prereg/eval_splits/), and the verifier is the same rule-based grader used
during RM calls (v1 or v2 from day2_custom_rm.py).

Usage:
  python3 scripts/e7_eval_checkpoint.py \\
    --checkpoint runs/pilot-E7-clean-oracle-1.5B-4gpu-s17-170145/checkpoints/iter_0000029 \\
    --eval-split prereg/eval_splits/gsm8k_eval500_seed42.json \\
    --model-name Qwen2.5-1.5B-Instruct \\
    --verifier-version v1 \\
    --output runs/pilot-E7-clean-oracle-1.5B-4gpu-s17-170145/eval_iter_0000029.json \\
    --gpus 0,1,2,3

The script will:
1. Load the checkpoint into vLLM or a compatible inference backend
2. Read the eval split indices
3. Generate completions for each eval prompt
4. Grade each completion with the specified verifier
5. Write {accuracy, n_correct, n_total, per_sample_results} to --output

For formal E7, run this on the final checkpoint (e.g., iter_0000500) of every
training run, then aggregate accuracy values across seeds using
scripts/paper_statistics.py paired_tost.

Dependencies:
  - vLLM or sglang for inference
  - The same verifier logic from day2_custom_rm.py
  - Access to the eval split file and the original dataset

This is a placeholder scaffold. The actual implementation requires:
  - Checkpoint loading from Megatron/DeepSpeed format
  - vLLM or sglang engine initialization
  - Prompt formatting matching the training format
  - Verifier invocation (v1 or v2)
  - JSON output serialization
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# TODO: Import actual checkpoint loading, inference engine, and verifier
# from slime.backends.megatron_utils import load_checkpoint
# from vllm import LLM, SamplingParams
# from scripts.day2_custom_rm import _v1_reward, _v2_reward


def load_eval_split(split_path: Path) -> List[Dict[str, Any]]:
    """Load the frozen eval split indices and return eval samples."""
    split_spec = json.loads(split_path.read_text(encoding="utf-8"))
    dataset_path = Path(split_spec["source"])
    if not dataset_path.is_file():
        # Try relative to repo root
        dataset_path = split_path.parent.parent / split_spec["source"]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {split_spec['source']}")
    
    # Load full dataset
    with dataset_path.open(encoding="utf-8") as handle:
        all_samples = [json.loads(line) for line in handle if line.strip()]
    
    # Extract eval subset
    eval_indices = split_spec["eval_indices"]
    eval_samples = [all_samples[idx] for idx in eval_indices]
    
    if len(eval_samples) != split_spec["eval_count"]:
        raise RuntimeError(
            f"Expected {split_spec['eval_count']} eval samples, found {len(eval_samples)}"
        )
    
    return eval_samples


def format_prompt(sample: Dict[str, Any]) -> str:
    """Format a sample into the prompt used during training."""
    # This must match the prompt format used during training rollout.
    # For DAPO-style datasets, the typical format is:
    # <question>\n\n{instruction/problem text}
    instruction = sample.get("instruction", sample.get("problem", ""))
    return f"<question>\n\n{instruction}"


def grade_response(response: str, label: str, verifier_version: str) -> float:
    """Grade a response using the specified verifier version."""
    # TODO: Import and call the actual verifier from day2_custom_rm
    # if verifier_version == "v1":
    #     return _v1_reward(response, label)
    # elif verifier_version == "v2":
    #     return _v2_reward(response, label)
    # else:
    #     raise ValueError(f"Unknown verifier version: {verifier_version}")
    
    # Placeholder: always return 0.0
    print(f"[PLACEHOLDER] Grading response (verifier={verifier_version}): {response[:80]}...", file=sys.stderr)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to checkpoint directory")
    parser.add_argument("--eval-split", required=True, type=Path, help="Path to eval split JSON")
    parser.add_argument("--model-name", required=True, help="Model name for inference engine")
    parser.add_argument("--verifier-version", choices=["v1", "v2"], default="v1", help="Verifier version")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path for eval results")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU indices")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0=greedy)")
    args = parser.parse_args()

    if not args.checkpoint.is_dir():
        print(f"Error: Checkpoint directory not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    
    if not args.eval_split.is_file():
        print(f"Error: Eval split file not found: {args.eval_split}", file=sys.stderr)
        sys.exit(1)
    
    # Load eval split
    print(f"Loading eval split from {args.eval_split}...", file=sys.stderr)
    eval_samples = load_eval_split(args.eval_split)
    print(f"Loaded {len(eval_samples)} eval samples.", file=sys.stderr)
    
    # TODO: Load checkpoint into inference engine
    print(f"[PLACEHOLDER] Loading checkpoint from {args.checkpoint}...", file=sys.stderr)
    print(f"[PLACEHOLDER] Model: {args.model_name}, GPUs: {args.gpus}", file=sys.stderr)
    # llm = LLM(model=str(args.checkpoint), tensor_parallel_size=len(args.gpus.split(",")))
    # sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    
    # Generate responses
    print(f"[PLACEHOLDER] Generating {len(eval_samples)} responses...", file=sys.stderr)
    prompts = [format_prompt(sample) for sample in eval_samples]
    # outputs = llm.generate(prompts, sampling_params)
    # responses = [output.outputs[0].text for output in outputs]
    
    # Placeholder: empty responses
    responses = [""] * len(eval_samples)
    
    # Grade responses
    print(f"Grading responses with verifier={args.verifier_version}...", file=sys.stderr)
    results = []
    correct_count = 0
    for idx, (sample, response) in enumerate(zip(eval_samples, responses)):
        label = sample.get("label", sample.get("answer", ""))
        reward = grade_response(response, label, args.verifier_version)
        correct = reward > 0.5
        if correct:
            correct_count += 1
        results.append({
            "index": idx,
            "prompt": prompts[idx][:100] + "...",  # truncated
            "response": response[:200] + "...",  # truncated
            "label": label,
            "reward": reward,
            "correct": correct,
        })
    
    accuracy = correct_count / len(eval_samples) if eval_samples else 0.0
    
    # Write output
    output_payload = {
        "checkpoint": str(args.checkpoint),
        "eval_split": str(args.eval_split),
        "model_name": args.model_name,
        "verifier_version": args.verifier_version,
        "n_total": len(eval_samples),
        "n_correct": correct_count,
        "final_eval_accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "results": results,
        "note": "This is a PLACEHOLDER implementation. Actual checkpoint loading, inference, and verifier invocation are not yet wired.",
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    
    print(f"Eval complete: {correct_count}/{len(eval_samples)} correct ({accuracy*100:.2f}%)", file=sys.stderr)
    print(f"Results written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
