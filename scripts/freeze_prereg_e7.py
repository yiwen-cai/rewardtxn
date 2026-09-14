#!/usr/bin/env python3
"""Freeze E7 prereg parameters before formal training.

This script writes frozen_margin_pp, frozen_commit, and frozen_at into
prereg/tost_margin.json. It can only be run once; subsequent calls will fail
if the margin is already frozen.

Usage:
  python3 scripts/freeze_prereg_e7.py --margin-pp 1.0 --commit $(git rev-parse HEAD)

The frozen commit should be the current HEAD after pilot completion and before
formal training launch. The frozen margin must be a positive number in absolute
percentage points (e.g., 1.0 means ±1.0 pp equivalence bounds).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--margin-pp", type=float, required=True, help="Frozen TOST margin in absolute percentage points")
    parser.add_argument("--commit", required=True, help="Git commit SHA at freeze time")
    parser.add_argument("--prereg-path", type=Path, default=Path("prereg/tost_margin.json"), help="Path to prereg file")
    args = parser.parse_args()
    
    if args.margin_pp <= 0:
        raise ValueError(f"margin-pp must be positive, got {args.margin_pp}")
    
    if not args.prereg_path.is_file():
        raise FileNotFoundError(f"Prereg file not found: {args.prereg_path}")
    
    spec = json.loads(args.prereg_path.read_text(encoding="utf-8"))
    
    if spec.get("frozen_margin_pp") is not None:
        raise RuntimeError(
            f"frozen_margin_pp is already set to {spec['frozen_margin_pp']}; "
            "cannot re-freeze. If this is intentional, manually edit the JSON."
        )
    
    frozen_at = dt.datetime.now(dt.timezone.utc).isoformat()
    
    spec["frozen_margin_pp"] = args.margin_pp
    spec["frozen_commit"] = args.commit
    spec["frozen_at"] = frozen_at
    spec["changelog"].append({
        "action": "freeze_margin_and_commit",
        "margin_pp": args.margin_pp,
        "commit": args.commit,
        "timestamp": frozen_at,
    })
    
    args.prereg_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    
    print(f"Frozen: margin={args.margin_pp} pp, commit={args.commit}, at={frozen_at}")
    print(f"Updated: {args.prereg_path}")
    print("")
    print("Next steps:")
    print("  1. Commit the frozen prereg:")
    print(f"     git add {args.prereg_path}")
    print(f'     git commit -m "freeze: E7 margin={args.margin_pp}pp before formal training"')
    print("  2. Launch formal training:")
    print("     bash scripts/e7_formal_launch.sh")


if __name__ == "__main__":
    main()
