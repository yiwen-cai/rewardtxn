#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the RewardTxn CPU verification harness.
#
# Scope note: GPU training, proprietary AReaL/slime Docker images, large model
# weights, and the ~33 GB local-only evidence directories are intentionally out
# of scope here (see HANDOFF.md). This prepares the CPU-only fault-tolerance and
# paper-statistics test suites, which are the portable verification harness.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# System package: python venv support (stable, idempotent).
if ! python3.12 -m venv --help >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3.12-venv
fi

# Project venv with the documented CPU-only dependencies.
if [ ! -x ".venv/bin/python" ]; then
  python3.12 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-phase3-archive.txt

echo "RewardTxn CPU harness ready. Activate with: source .venv/bin/activate"
