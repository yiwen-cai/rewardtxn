#!/usr/bin/env bash
# Independent diagnostic repeats; never replace the pilot manifest or results.
set -euo pipefail
BASE=/public/home/caiyiwen/rewardtxn
export RTX_FORMAL_GPUS=${RTX_FORMAL_GPUS:-device=0,1,2,3}
STAMP=$(date -u +%Y%m%d-%H%M%S)
python3 "$BASE/scripts/resource_gate.py" check --gpus "$RTX_FORMAL_GPUS" --scratch-path /tmp/rewardtxn
# Preselected seed 11, first planned pilot seed; reverse the original group order.
for GROUP in b6 group_rm; do
  bash "$BASE/scripts/e7_restart_run.sh" pilot 11 "$GROUP" "e7restart-diagnostic-repeat-${GROUP}-s11-${STAMP}"
done
