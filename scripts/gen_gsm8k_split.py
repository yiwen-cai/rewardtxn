#!/usr/bin/env python3
"""
P0: GSM8K 固定评测集划分生成器 — 附录 C.3

从转换后的 DAPO 格式 GSM8K (models/datasets/gsm8k/dapo-gsm8k-train.jsonl, 7,473 条)
生成固定 random split：train 6,973 + eval 500。划分只记录行索引（不复制数据），
写入 prereg/eval_splits/gsm8k_eval500_seed42.json 并带内容哈希，一经生成不可变。

用法:
  python3 scripts/gen_gsm8k_split.py [--seed 42] [--eval 500] [--dry-run]
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "models" / "datasets" / "gsm8k" / "dapo-gsm8k-train.jsonl"
OUT = BASE / "prereg" / "eval_splits"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"ERROR: 未找到转换后的 GSM8K 数据集: {SRC}", file=sys.stderr)
        sys.exit(2)

    total = sum(1 for _ in open(SRC, encoding="utf-8"))
    if args.eval >= total:
        print(f"ERROR: eval 数 {args.eval} >= 总数 {total}", file=sys.stderr)
        sys.exit(2)

    rng = random.Random(args.seed)
    idx = list(range(total))
    rng.shuffle(idx)
    eval_idx = sorted(idx[: args.eval])
    train_count = total - args.eval
    content = json.dumps({"eval": eval_idx, "train_count": train_count}, separators=(",", ":"))
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    rec = {
        "dataset": "gsm8k-dapo-format",
        "source": str(SRC.relative_to(BASE)),
        "total": total,
        "train_count": train_count,
        "eval_count": args.eval,
        "seed": args.seed,
        "split_hash": content_hash,
        "eval_indices": eval_idx,
    }
    if args.dry_run:
        print(json.dumps({k: v for k, v in rec.items() if k != "eval_indices"}, indent=2, ensure_ascii=False))
        print(f"eval 示例索引: {eval_idx[:10]} ...")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"gsm8k_eval{args.eval}_seed{args.seed}.json"
    p.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {p.relative_to(BASE)} (total={total}, train={train_count}, eval={args.eval})")
    print(f"split_hash={content_hash[:16]}...")


if __name__ == "__main__":
    main()
