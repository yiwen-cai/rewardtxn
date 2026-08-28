#!/usr/bin/env python3
"""Safely retain only the newest completed Megatron checkpoints.

The checkpoint writer updates ``latest_checkpointed_iteration.txt`` only after a
checkpoint is complete.  We use that marker as the commit point and never remove
an iteration newer than it (it may still be being written).  The newest ``keep``
completed iterations are retained so Phase 3B can resume from the latest one or
fall back to the preceding one if the latest directory is damaged.

This is deliberately an external housekeeping process: it does not change
Megatron checkpoint contents or the resume protocol.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


_ITER_PREFIX = "iter_"


def _iteration(path: Path) -> int | None:
    name = path.name
    if name.isdigit():
        return int(name)
    if name.startswith(_ITER_PREFIX) and name[len(_ITER_PREFIX) :].isdigit():
        return int(name[len(_ITER_PREFIX) :])
    return None


def _latest_iteration(save_dir: Path) -> int | None:
    marker = save_dir / "latest_checkpointed_iteration.txt"
    try:
        return int(marker.read_text().strip())
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return None


def _checkpoint_dirs(save_dir: Path) -> list[tuple[int, Path]]:
    try:
        entries = save_dir.iterdir()
    except (FileNotFoundError, OSError):
        return []
    found = []
    for path in entries:
        # Do not follow/remove symlinks or temporary writer directories.
        if path.is_symlink() or not path.is_dir():
            continue
        iteration = _iteration(path)
        if iteration is not None:
            found.append((iteration, path))
    return sorted(found)


def prune(save_dir: str | Path, keep: int = 2, min_age_seconds: float = 30.0) -> dict:
    """Remove old completed checkpoint directories and return an audit record.

    No deletion occurs until the completion marker is readable.  ``min_age``
    protects against deleting a directory while an asynchronous filesystem
    operation is still settling; normal training uses synchronous checkpoint
    saves, but the guard is cheap.
    """
    if keep < 1:
        raise ValueError("keep must be >= 1")
    root = Path(save_dir)
    latest = _latest_iteration(root)
    result = {"save_dir": str(root), "latest": latest, "keep": keep, "removed": [], "errors": []}
    if latest is None:
        result["status"] = "marker-unavailable"
        return result

    completed = [(iteration, path) for iteration, path in _checkpoint_dirs(root) if iteration <= latest]
    protected = {iteration for iteration, _path in completed[-keep:]}
    cutoff = time.time() - max(0.0, min_age_seconds)
    for iteration, path in completed:
        if iteration in protected:
            continue
        try:
            if path.stat().st_mtime > cutoff:
                continue
            shutil.rmtree(path)
            result["removed"].append(iteration)
        except (FileNotFoundError, OSError) as exc:
            result["errors"].append({"iter": iteration, "error": str(exc)})
    result["kept"] = sorted(protected)
    result["status"] = "ok" if not result["errors"] else "partial"
    return result


def watch(save_dir: str | Path, keep: int, interval_seconds: float, min_age_seconds: float) -> None:
    root = Path(save_dir)
    print(
        f"[ckpt-retention] watching {root} keep={keep} interval={interval_seconds}s "
        f"min_age={min_age_seconds}s",
        flush=True,
    )
    while True:
        result = prune(root, keep=keep, min_age_seconds=min_age_seconds)
        if result["removed"] or result["errors"]:
            print(f"[ckpt-retention] {json.dumps(result, ensure_ascii=False)}", flush=True)
        time.sleep(interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prune_parser = sub.add_parser("prune", help="prune once")
    prune_parser.add_argument("save_dir")
    prune_parser.add_argument("--keep", type=int, default=2)
    prune_parser.add_argument("--min-age-seconds", type=float, default=30.0)

    watch_parser = sub.add_parser("watch", help="prune periodically")
    watch_parser.add_argument("save_dir")
    watch_parser.add_argument("--keep", type=int, default=2)
    watch_parser.add_argument("--interval-seconds", type=float, default=15.0)
    watch_parser.add_argument("--min-age-seconds", type=float, default=30.0)

    args = parser.parse_args()
    if args.command == "prune":
        print(json.dumps(prune(args.save_dir, args.keep, args.min_age_seconds), ensure_ascii=False))
        return 0
    watch(args.save_dir, args.keep, args.interval_seconds, args.min_age_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
