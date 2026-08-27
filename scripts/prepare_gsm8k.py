#!/usr/bin/env python3
"""GSM8K (parquet) -> DAPO 格式 jsonl 转换器
输出: {"prompt": [{"content": "<DAPO 模板+题目>"}], "label": "<数字答案>"}
"""
import json
import re
import sys

import pyarrow.parquet as pq

SRC = "/public/home/caiyiwen/rewardtxn/models/datasets/gsm8k-train.parquet"
OUT = "/public/home/caiyiwen/rewardtxn/models/datasets/gsm8k/dapo-gsm8k-train.jsonl"

TEMPLATE = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: \\boxed{{$Answer}} where $Answer "
    "is the answer to the problem.\n\n{question}"
)


def extract_label(answer: str) -> str | None:
    m = re.search(r"####\s*(-?[\d.]+)", answer)
    return m.group(1) if m else None


def main():
    t = pq.read_table(SRC)
    rows = t.to_pylist()
    n, skipped = 0, 0
    with open(OUT, "w") as f:
        for r in rows:
            label = extract_label(r["answer"])
            if label is None:
                skipped += 1
                continue
            rec = {
                "prompt": [{"role": "user", "content": TEMPLATE.format(question=r["question"])}],
                "label": label,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"written {n} rows -> {OUT} (skipped {skipped})")


if __name__ == "__main__":
    sys.exit(main())
