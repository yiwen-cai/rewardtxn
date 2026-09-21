#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the RewardTxn CPU verification harness.
#
# Source-of-truth note: the Cloud Agent environment for this repo is
# dashboard-managed (DB-backed). The authoritative install lives in the
# environment settings; this file mirrors it under version control so the
# harness is reviewable and runnable locally. It is intentionally self-contained
# (locates the repo via BASH_SOURCE) so `bash .cursor/install.sh` works from any
# checkout without a committed .cursor/environment.json (whose presence would
# flip the environment to repo-file-managed and override the dashboard).
#
# Scope note: GPU training, proprietary AReaL/slime Docker images, large model
# weights, and the ~33 GB local-only evidence directories are intentionally out
# of scope here (see HANDOFF.md). This prepares the CPU-only fault-tolerance and
# paper-statistics test suites, which are the portable verification harness.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Always use the *system* interpreter, never a project .venv copy that may be on
# PATH (an activated venv would otherwise be used to rebuild itself).
PYTHON_BIN=/usr/bin/python3.12
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3.12 || true)"
fi
[ -n "$PYTHON_BIN" ] || { echo "python3.12 not found on this host" >&2; exit 1; }

# System package: python venv support. Gate on ensurepip (not `venv --help`,
# which succeeds even when python3.12-venv/ensurepip is missing and would leave
# a broken venv with no pip). apt-get install is idempotent.
if ! "$PYTHON_BIN" -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3.12-venv
fi

# (Re)create the project venv. A valid venv always has bin/pip; if that is
# missing the venv is absent or partial, so remove it and rebuild cleanly.
if [ ! -x ".venv/bin/pip" ]; then
  rm -rf .venv
  "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-phase3-archive.txt
# Fail fast on an inconsistent dependency set (pinned sympy/pylatexenc + mpmath).
pip check
python -c "import sympy, pylatexenc; print('deps ok:', 'sympy', sympy.__version__, '| pylatexenc', pylatexenc.__version__)"

# CPU verification smoke: prove the portable harness actually runs, not just
# that dependencies import. These modules exercise the paper-statistics and
# fault-tolerance state/oracle/fault logic and depend only on the pinned deps
# plus in-repo source, so they never touch out-of-scope GPU code, Docker images,
# model weights, or the local-only evidence directories. With environment
# builds this runs once at build time and gates the snapshot; keeping it small
# and deterministic ensures the harness ships in a known-good state.
CPU_SMOKE_MODULES=(
  tests.paper.test_statistics_core
  tests.paper.test_statistics_bootstrap
  tests.paper.test_schedule_and_split
  tests.ft.test_state
  tests.ft.test_oracle
  tests.ft.test_faults
)
echo ">>> CPU verification smoke: ${#CPU_SMOKE_MODULES[@]} modules"
python -m unittest "${CPU_SMOKE_MODULES[@]}"

echo "RewardTxn CPU harness ready. Activate with: source .venv/bin/activate"
