#!/usr/bin/env python3
"""Smoke test: verify HF checkpoint loads and generates 2 samples correctly."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/scripts")

from e7_eval_checkpoint_incontainer import (
    format_prompt,
    generate_responses_hf,
    grade_responses,
)

ckpt_path = Path("/workspace/runs/pilot-E7-clean-oracle-1.5B-4gpu-s17-170145/checkpoints/iter_0000029_hf")
split_path = Path("/workspace/prereg/eval_splits/gsm8k_eval500_seed42.json")

# Load only first 2 samples
split = json.load(open(split_path))
all_samples = [json.loads(line) for line in open(Path("/workspace") / split["source"])]
eval_indices = split["eval_indices"][:2]
eval_samples = [all_samples[idx] for idx in eval_indices]

print(f"Smoke test: {len(eval_samples)} samples", file=sys.stderr)

# Format prompts
prompts = [format_prompt(s) for s in eval_samples]
labels = [s.get("label", s.get("answer", "")) for s in eval_samples]

# Generate
start = time.time()
responses = generate_responses_hf(ckpt_path, prompts, batch_size=2)
elapsed = time.time() - start

# Grade
rewards, corrects = grade_responses(responses, labels, "v1")
n_correct = sum(corrects)

# Output
print(f"\n=== Smoke test result ===", file=sys.stderr)
print(f"Samples: {len(eval_samples)}", file=sys.stderr)
print(f"Correct: {n_correct}/{len(eval_samples)} ({n_correct/len(eval_samples)*100:.1f}%)", file=sys.stderr)
print(f"Time: {elapsed:.1f}s", file=sys.stderr)
for i, (r, l, c) in enumerate(zip(responses, labels, corrects)):
    status = "CORRECT" if c else "WRONG"
    print(f"\nSample {i}: {status}", file=sys.stderr)
    print(f"  Label: {l[:80]}", file=sys.stderr)
    print(f"  Response: {r[:150]}...", file=sys.stderr)
