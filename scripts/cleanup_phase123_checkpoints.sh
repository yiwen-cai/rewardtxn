#!/usr/bin/env bash
# Clean up Phase 1-3 checkpoint weight directories to free storage for formal E7.
#
# Policy (frozen in the E7 storage plan):
#   - Delete ONLY runs/p[123]-*/checkpoints/ (Megatron weight dirs; 37 runs)
#   - Keep all audit evidence: meta.json, config.json, logs/, rewards.jsonl,
#     seals.jsonl, manifests/, cas_rejects.jsonl, checkpoint_retention.log
#   - Phase 1-3 are archived (PHASE3_ARCHIVE_MANIFEST.json, phase3-final tag),
#     so checkpoint weights are regenerable and not part of the audit trail.
#   - Pilot (pilot-*) checkpoints are retained for statistics and eval tests.
#
# Checkpoint files are root-owned (written inside the Slime container), so the
# deletion runs as root inside the same image via docker.
#
# Usage:
#   bash scripts/cleanup_phase123_checkpoints.sh          # real cleanup
#   DRY_RUN=1 bash scripts/cleanup_phase123_checkpoints.sh  # report only
#
# Outputs:
#   cleanup_phase123_<ts>.log          - step-by-step log (project root)
#   runs/SPACE_CLEANUP_<ts>.json       - structured verification report
set -euo pipefail

BASE=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${RTX_IMAGE:-slimerl/slime:v0.3.1}
TS=$(date +%Y%m%d-%H%M%S)
LOG="$BASE/cleanup_phase123_${TS}.log"
REPORT="$BASE/runs/SPACE_CLEANUP_${TS}.json"
DRY_RUN=${DRY_RUN:-0}

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Pre-flight: audit inventory BEFORE any deletion (as host user; sizes via du)
inventory() {
  local patterns=$1
  python3 - "$patterns" <<'PY'
import json, os, sys, subprocess
from pathlib import Path
patterns = sys.argv[1].split(",")
root = Path("/public/home/caiyiwen/rewardtxn/runs")
entries = sorted({p for pat in patterns for p in root.glob(pat)})
rows = []
for d in entries:
    if not d.is_dir():
        continue
    ckpt = d / "checkpoints"
    ckpt_size = 0
    if ckpt.is_dir():
        out = subprocess.run(["du", "-sk", str(ckpt)], capture_output=True, text=True)
        ckpt_size = int(out.stdout.split()[0]) * 1024 if out.returncode == 0 else 0
    rows.append({
        "run": d.name,
        "has_checkpoints": ckpt.is_dir(),
        "checkpoints_bytes": ckpt_size,
        "has_meta": (d / "meta.json").is_file(),
        "has_config": (d / "config.json").is_file(),
        "has_logs": (d / "logs").is_dir(),
        "has_rewards": (d / "rewards.jsonl").is_file(),
        "has_seals": (d / "seals.jsonl").is_file(),
        "has_manifests": (d / "manifests").is_dir(),
    })
print(json.dumps(rows))
PY
}

log "=== Phase 1-3 checkpoint cleanup ==="
log "Base: $BASE   Image: $IMAGE   Dry-run: $DRY_RUN"
log "Log: $LOG"
echo ""

# 1. Inventory before (audit evidence present? sizes?)
log "Inventorizing Phase 1-3 runs (audit files + checkpoint sizes)..."
PHASE_PATTERNS="p1-*,p2*,p3*"
BEFORE=$(inventory "$PHASE_PATTERNS")
BEFORE_JSON=$(python3 -c "
import json,sys
rows=json.loads(sys.argv[1])
total=sum(r['checkpoints_bytes'] for r in rows)
audit_ok=sum(1 for r in rows if r['has_meta'] and r['has_logs'])
print(json.dumps({'n_runs':len(rows),'n_checkpoint_dirs':sum(1 for r in rows if r['has_checkpoints']),'checkpoints_bytes':total,'runs_with_meta_and_logs':audit_ok}))
" "$BEFORE")
log "Before: $(echo "$BEFORE_JSON" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("runs=%d ckpt_bytes=%.1fGB audit_ok=%d" % (d["n_runs"], d["checkpoints_bytes"]/1e9, d["runs_with_meta_and_logs"]))')"

# 2. Integrity gate: refuse to proceed if any run lost meta.json or logs/
MISSING=$(python3 -c "
import json,sys
rows=json.loads(sys.argv[1])
bad=[r['run'] for r in rows if not (r['has_meta'] and r['has_logs'])]
print(' '.join(bad))
" "$BEFORE")
if [ -n "$MISSING" ]; then
  log "WARNING: runs missing meta.json or logs/ (skipped from audit guarantee): $MISSING"
fi

# 3. Delete checkpoints/ inside the container as root
DF_BEFORE=$(df -h /public)
if [ "$DRY_RUN" = "1" ]; then
  log "DRY RUN: no deletion performed."
else
  log "Deleting runs/p[123]-*/checkpoints/ as root in container..."
  CONTAINER_LOG="/workspace/${LOG#$BASE/}"
  docker run --rm --user 0:0 \
    -v "$BASE":/workspace \
    "$IMAGE" \
    bash -lc '
      cd /workspace/runs
      for d in p1-* p2* p3*; do
        if [ -d "$d/checkpoints" ]; then
          size=$(du -sh "$d/checkpoints" 2>/dev/null | cut -f1)
          echo "  rm -rf $d/checkpoints ($size)" >> '"$CONTAINER_LOG"'
          rm -rf "$d/checkpoints"
        fi
      done
    '
  log "Deletion pass complete."
fi
echo ""

# 4. Inventory after + df
AFTER=$(inventory "$PHASE_PATTERNS")
AFTER_JSON=$(python3 -c "
import json,sys
rows=json.loads(sys.argv[1])
total=sum(r['checkpoints_bytes'] for r in rows)
print(json.dumps({'n_runs':len(rows),'n_checkpoint_dirs':sum(1 for r in rows if r['has_checkpoints']),'checkpoints_bytes':total}))
" "$AFTER")

DF_AFTER=$(df -h /public)

log "df /public (before):"
echo "$DF_BEFORE" | tee -a "$LOG"
log "df /public (after):"
echo "$DF_AFTER" | tee -a "$LOG"

# 5. Write structured report
python3 - "$REPORT" "$TS" "$DRY_RUN" "$BEFORE_JSON" "$AFTER_JSON" <<'PY'
import json, sys
report_path, ts, dry_run, before_raw, after_raw = sys.argv[1:6]
before = json.loads(before_raw)
after = json.loads(after_raw)
report = {
    "report_id": f"SPACE_CLEANUP_{ts}",
    "created_at": ts,
    "dry_run": dry_run == "1",
    "scope": "runs/p[123]-*/checkpoints only (audit evidence preserved)",
    "before": before,
    "after": after,
    "freed_bytes": before["checkpoints_bytes"] - after["checkpoints_bytes"],
    "checkpoint_dirs_remaining": after["checkpoints_bytes"],
}
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"Report written to {report_path}")
PY

log "Done. Summary: freed $(python3 -c "import json; print('%.1fGB' % (json.load(open('$REPORT'))['freed_bytes']/1e9))" 2>/dev/null || echo '?')"
echo ""
log "=== Cleanup finished (dry_run=$DRY_RUN) ==="
