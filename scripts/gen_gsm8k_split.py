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


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_record(source, seed, eval_count):
    with source.open(encoding="utf-8") as handle:
        total = sum(1 for _ in handle)
    if eval_count >= total:
        raise ValueError(f"eval 数 {eval_count} >= 总数 {total}")

    source_hash = sha256_file(source)
    rng = random.Random(seed)
    idx = list(range(total))
    rng.shuffle(idx)
    eval_idx = sorted(idx[:eval_count])
    train_count = total - eval_count
    identity = {
        "source_sha256": source_hash,
        "total": total,
        "train_count": train_count,
        "eval_count": eval_count,
        "seed": seed,
        "eval_indices": eval_idx,
    }
    split_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    try:
        source_ref = str(source.relative_to(BASE))
    except ValueError:
        source_ref = str(source)
    return {
        "dataset": "gsm8k-dapo-format",
        "source": source_ref,
        "source_sha256": source_hash,
        "split_algorithm": "python-random-shuffle-v1",
        **identity,
        "split_hash": split_hash,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"ERROR: 未找到转换后的 GSM8K 数据集: {SRC}", file=sys.stderr)
        sys.exit(2)

    try:
        rec = build_record(SRC, args.seed, args.eval)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    p = OUT / f"gsm8k_eval{args.eval}_seed{args.seed}.json"
    if args.verify:
        if not p.exists():
            print(f"ERROR: split file missing: {p}", file=sys.stderr)
            sys.exit(2)
        stored = json.loads(p.read_text())
        if stored != rec:
            print("ERROR: split/source hash mismatch", file=sys.stderr)
            sys.exit(1)
        print(f"PASS {p.relative_to(BASE)} source_sha256={rec['source_sha256'][:16]}...")
        return
    if args.dry_run:
        print(json.dumps({k: v for k, v in rec.items() if k != "eval_indices"}, indent=2, ensure_ascii=False))
        print(f"eval 示例索引: {rec['eval_indices'][:10]} ...")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {p.relative_to(BASE)} (total={rec['total']}, train={rec['train_count']}, eval={args.eval})")
    print(f"source_sha256={rec['source_sha256'][:16]}... split_hash={rec['split_hash'][:16]}...")


if __name__ == "__main__":
    main()
