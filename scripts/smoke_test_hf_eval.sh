#!/usr/bin/env bash
# Smoke test: verify HF checkpoint loads and generates 2 samples correctly.
# Usage: bash scripts/smoke_test_hf_eval.sh <hf_checkpoint_dir>
set -euo pipefail

HF_CKPT=${1:?Usage: $0 <hf_checkpoint_dir>}
BASE=$(cd "$(dirname "$0")/.." && pwd)
EVAL_SPLIT="$BASE/prereg/eval_splits/gsm8k_eval500_seed42.json"
OUTPUT="/tmp/smoke_test_eval_$(date +%s).json"

if [ ! -d "$HF_CKPT" ]; then
  echo "Error: HF checkpoint not found: $HF_CKPT" >&2
  exit 1
fi

echo "=== Smoke test: 2-sample eval ==="
echo "Checkpoint: $HF_CKPT"
echo "Output: $OUTPUT"
echo ""

docker run --rm \
  --gpus '"device=4"' \
  --shm-size 8g \
  -v "$BASE":/workspace \
  -v "$BASE/third_party/slime":/root/slime \
  -v "$BASE/models":/root/models \
  -e PYTHONPATH=/root/Megatron-LM/:/root/slime:/workspace/scripts \
  slimerl/slime:v0.3.1 \
  python3 - <<PYEOF
import json, sys
from pathlib import Path
sys.path.insert(0, "/workspace/scripts")

from e7_eval_checkpoint_incontainer import load_eval_split, format_prompt, generate_responses_hf, grade_responses

# Load only first 2 samples
split = json.load(open("$EVAL_SPLIT"))
all_samples = [json.loads(line) for line in open(Path("/workspace") / split["source"])]
eval_indices = split["eval_indices"][:2]  # only first 2
eval_samples = [all_samples[idx] for idx in eval_indices]

print(f"Loaded {len(eval_samples)} samples for smoke test", file=sys.stderr)

# Format prompts
prompts = [format_prompt(s) for s in eval_samples]
labels = [s.get("label", s.get("answer", "")) for s in eval_samples]

# Generate (should take ~10-20 seconds for 2 samples)
try:
    responses = generate_responses_hf(Path("$HF_CKPT"), prompts, batch_size=2)
except Exception as e:
    print(f"Generation failed: {e}", file=sys.stderr)
    sys.exit(1)

# Grade
rewards, corrects = grade_responses(responses, labels, "v1")
n_correct = sum(corrects)

# Output
result = {
    "smoke_test": True,
    "n_samples": len(eval_samples),
    "n_correct": n_correct,
    "accuracy": n_correct / len(eval_samples),
    "samples": [
        {
            "prompt": str(p)[:100] + "...",
            "response": r[:200] + "...",
            "label": l,
            "reward": rw,
            "correct": c,
        }
        for p, r, l, rw, c in zip(prompts, responses, labels, rewards, corrects)
    ]
}

with open("$OUTPUT", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\nSmoke test: {n_correct}/{len(eval_samples)} correct", file=sys.stderr)
print(f"Results: $OUTPUT", file=sys.stderr)
PYEOF

echo ""
echo "=== Smoke test complete ==="
cat "$OUTPUT" | python3 -m json.tool | head -30
