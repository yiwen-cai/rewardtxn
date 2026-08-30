#!/usr/bin/env python3
"""
P0: run 产物数据契约校验器 — §10 实验产物与数据契约

paper run 必须包含:
  meta.json / config.json / events.jsonl / manifests/ / metrics.json /
  resource.jsonl / verdict.json / artifact_manifest.json / stdout+stderr 与退出状态

校验内容:
  1. 必需文件存在性（--paper 严格模式 / 默认 legacy 报告模式）;
  2. meta.json 必填字段（exp_id、commit_sha、seed、K/U、fault schedule 引用等）;
  3. events.jsonl 每行满足 configs/paper_event.schema.json（无第三方依赖的手写校验）;
  4. manifests/ 非空且包含 GroupManifest/StepManifest/StepToken 类文件。

用法:
  python3 scripts/validate_run_artifacts.py <run_dir> [--paper] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((BASE / "configs" / "paper_event.schema.json").read_text())

REQUIRED_PAPER_FILES = [
    "meta.json", "config.json", "events.jsonl", "manifests/",
    "metrics.json", "resource.jsonl", "verdict.json", "artifact_manifest.json",
]
REQUIRED_META_FIELDS = [
    "exp_id", "phase", "stack", "baseline", "commit_sha", "seed",
    "group_size_K", "batch_groups_U", "fault_injection", "created_at",
]
EVENT_TYPES = {"group", "reward", "queue", "learner", "checkpoint", "ack", "recovery", "fault"}


def validate_event_line(line: str, lineno: int) -> list[str]:
    errs = []
    try:
        ev = json.loads(line)
    except json.JSONDecodeError as e:
        return [f"events.jsonl:{lineno} JSON 解析失败: {e}"]
    if not isinstance(ev, dict):
        return [f"events.jsonl:{lineno} 行不是对象"]
    for f in ("type", "ts", "exp_id"):
        if f not in ev:
            errs.append(f"events.jsonl:{lineno} 缺少必填字段 {f}")
    if "type" in ev and ev["type"] not in EVENT_TYPES:
        errs.append(f"events.jsonl:{lineno} 未知事件类型 {ev['type']!r}")
    if "ts" in ev and not isinstance(ev["ts"], (int, float)):
        errs.append(f"events.jsonl:{lineno} ts 非数值")
    if ev.get("type") == "group" and "group_ids" not in ev:
        errs.append(f"events.jsonl:{lineno} group 事件缺少 group_ids")
    if ev.get("type") == "checkpoint" and "checkpoint_hash" not in ev:
        errs.append(f"events.jsonl:{lineno} checkpoint 事件缺少 checkpoint_hash")
    if ev.get("type") == "recovery" and "recovery_decision" not in ev:
        errs.append(f"events.jsonl:{lineno} recovery 事件缺少 recovery_decision")
    return errs


def validate_run(run_dir: Path, paper: bool) -> dict:
    report = {"run_dir": str(run_dir), "paper": paper, "checks": {}}
    missing = []
    for name in REQUIRED_PAPER_FILES:
        target = run_dir / name if not name.endswith("/") else run_dir / name
        if not target.exists():
            missing.append(name)
    report["checks"]["required_files"] = {"pass": not missing, "missing": missing}

    meta_errs = []
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            for f in REQUIRED_META_FIELDS:
                if f not in meta:
                    meta_errs.append(f"meta.json 缺少 {f}")
        except json.JSONDecodeError as e:
            meta_errs.append(f"meta.json 解析失败: {e}")
    else:
        meta_errs.append("meta.json 不存在")
    report["checks"]["meta"] = {"pass": not meta_errs, "errors": meta_errs}

    ev_errs = []
    ev_path = run_dir / "events.jsonl"
    n_events = 0
    if ev_path.exists():
        for i, line in enumerate(ev_path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            n_events += 1
            ev_errs.extend(validate_event_line(line, i))
            if len(ev_errs) > 20:
                ev_errs.append("... 错误过多，截断")
                break
    report["checks"]["events"] = {"pass": not ev_errs, "errors": ev_errs, "lines": n_events}

    manif = run_dir / "manifests"
    manif_errs = []
    if manif.exists():
        names = sorted(p.name for p in manif.iterdir())
        if not names:
            manif_errs.append("manifests/ 为空")
        for kw in ("GroupManifest", "StepManifest", "StepToken", "step_token", "manifest"):
            if any(kw in n for n in names):
                break
        else:
            manif_errs.append("manifests/ 缺少 GroupManifest/StepManifest/StepToken 类文件")
    else:
        manif_errs.append("manifests/ 不存在")
    report["checks"]["manifests"] = {"pass": not manif_errs, "errors": manif_errs}

    ok = all(c["pass"] for c in report["checks"].values())
    if paper and missing:
        ok = False
    report["status"] = "PASS" if ok else "FAIL"
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--paper", action="store_true", help="严格模式：缺失任一必需文件即 FAIL")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rep = validate_run(args.run_dir, args.paper)
    if args.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    else:
        print(f"status: {rep['status']}")
        for name, c in rep["checks"].items():
            detail = c.get("missing") or c.get("errors") or []
            print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {name}: {detail[:3]}")
    sys.exit(0 if rep["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
