#!/usr/bin/env bash
# 500-step confirmation, gated by the prior fixed-endpoint validation result.
set -euo pipefail
BASE=$(cd "$(dirname "$0")/.." && pwd)
GROUP=${1:?usage: e7_diagnosis_stability.sh oracle|rewardtxn}
case "$GROUP" in
  oracle) PARENT=diagnosis-E7-E-s29-20260910; STEPS=50; MODE=group_rm ;;
  rewardtxn) PARENT=diagnosis-E7-D3-oracle-s29-20260910; STEPS=500; MODE=b6 ;;
  *) exit 2 ;;
esac
python3 - "$BASE" "$PARENT" "$STEPS" <<'PY'
import json, sys
from pathlib import Path
base, parent, steps = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
ref = json.loads((base/'runs/diagnosis-20260910/base_validation.json').read_text())
run = base/'runs'/parent
result = json.loads((run/f'diagnostic_eval/step{steps}_n100.json').read_text())
audit = json.loads((run/'diagnosis_summary.json').read_text())
assert result['n_total'] == ref['n_total'] == 100
assert result['split_sha256'] == ref['split_sha256']
assert result['checkpoint'] == f'/workspace/runs/{parent}/checkpoints/iter_{steps-1:07d}_hf'
assert [r['source_index'] for r in result['results']] == [r['source_index'] for r in ref['results']]
assert result['n_correct'] == sum(r['correct'] for r in result['results'])
assert result['accuracy'] >= ref['accuracy'] - .10 - 1e-12, 'accuracy gate failed'
assert result['truncated_fraction'] <= ref['truncated_fraction'] + .10 + 1e-12, 'truncation gate failed'
assert audit['train_steps'] == list(range(steps)) and audit['all_logged_metrics_finite']
assert audit['consumed_split_overlap'] == 0 and len(audit['consumed_rollouts']) == steps
print('Fixed-endpoint quality and consumption gates passed:', parent)
PY
export RTX_BASE="$BASE" RTX_GPUS=${RTX_GPUS:-device=1,2,3,4}
export RTX_SEED=29 RTX_BASELINE_MODE="$MODE" RTX_PAPER_MODE=0
export RTX_MODEL_DIR=/root/models/Qwen2.5-0.5B-Instruct RTX_MODEL_CONFIG=qwen2.5-0.5B.sh
export RTX_DATA_PATH=/workspace/runs/diagnosis-20260910/train.jsonl
export RTX_NO_SAVE_OPTIM=1 RTX_CKPT_KEEP=0 RTX_SAVE_HF=1 RTX_SAVE_INTERVAL=50
export RTX_MAX_TOKENS_PER_GPU=3072 RTX_SGLANG_MEM_FRACTION_STATIC=0.45
export RTX_SGLANG_CONCURRENCY=24 RTX_NUM_ROLLOUT=500 RTX_FULLY_ASYNC=1
export RTX_TRAIN_ENTRY=train_async.py RTX_LR=1e-6
export RTX_EXP_ID="diagnosis-E7-D3-${GROUP}-s29-20260910"
export RTX_EXTRA_MODEL_ARGS="--use-rollout-logprobs --save-debug-rollout-data /workspace/runs/${RTX_EXP_ID}/rollout_debug/{rollout_id}.pt --check-weight-update-equal"
bash "$BASE/scripts/phase2_run.sh" none 0 0 500 "$RTX_EXP_ID"
