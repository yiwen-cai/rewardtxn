#!/usr/bin/env bash
# Versioned 0.5B clean matrix. Default formal mode refuses missing pilot/freeze.
# --stage pilot runs the separately planned pilot; --check never launches training.
set -euo pipefail
BASE=${RTX_BASE:-/public/home/caiyiwen/rewardtxn}
exec python3 "$BASE/scripts/e7_restart_matrix.py" "$@"
