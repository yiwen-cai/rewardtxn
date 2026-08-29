#!/usr/bin/env python3
"""生成并验证 Phase 3 结项归档。

归档分两层：
1. portable evidence：报告、门禁、脚本、JSONL 和训练日志，必须进入 Git；
2. local large artifacts：checkpoint 大权重，仅记录文件清单摘要和本地保留状态。

用法：
  python3 scripts/phase3_archive.py write
  python3 scripts/phase3_archive.py verify --require-local-sources
  python3 scripts/phase3_archive.py verify --require-final-tag --require-clean
"""

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
MANIFEST_PATH = BASE / "runs/PHASE3_ARCHIVE_MANIFEST.json"
CONVERGENCE_PATH = BASE / "runs/PHASE3_CONVERGENCE_EVIDENCE.json"

GATE_FILES = [
    "runs/PHASE3_GATE3A.json",
    "runs/PHASE3_GATE3B.json",
    "runs/PHASE3_GATE3C.json",
]

CORE_FILES = [
    ".gitattributes",
    ".gitignore",
    "ENVIRONMENT_SETUP.md",
    "MEMORY_OPTIMIZATION.md",
    "MODEL_INVENTORY.md",
    "PHASE3_PLAN.md",
    "RewardTxn H100 自包含实验指导.md",
    "requirements-phase3-archive.txt",
    "patches/slime-qwen2.5-1.5b-rotary-base.patch",
    "runs/PHASE3_3A.md",
    "runs/PHASE3_3B.md",
    "runs/PHASE3_3C.md",
    "runs/PHASE3_FINAL.md",
    "runs/PHASE3_GATE3A.json",
    "runs/PHASE3_GATE3B.json",
    "runs/PHASE3_GATE3C.json",
    "runs/PHASE3_GATE3B_MEMIO.json",
    "runs/PHASE3_EXP_MEMIO.json",
    "runs/PHASE3_REGRESSION.json",
    "runs/PHASE3_CONVERGENCE_EVIDENCE.json",
    "runs/TRACE_REPORT.json",
    "runs/rtx_recovery_history.jsonl",
    "runs/p1-slime-L0-K8-s42-20260825/recovery_plan.json",
    "runs/p1-slime-L2-K8-s42-20260825/recovery_plan.json",
    "runs/p1-tq-Q0Q1-s42-20260825/tq_crash_probe.json",
    "runs/p2c-slime-crash-sealresp-K8-s42-20260827/recovery_plan.json",
    "runs/p2c-slime-crash-sealresp-K8-s42-20260827/replay_result.json",
    "runs/p2c-slime-crash-sealresp-K8-s42-20260827/rewards.jsonl",
    "runs/p2c-slime-skew-sealresp-K8-s42-20260827/recovery_audit_report.md",
    "runs/p2c-slime-skew-sealresp-K8-s42-20260827/replay_result.json",
    "runs/p2c-slime-skew-sealresp-K8-s42-20260827/rewards.jsonl",
    "runs/p2c-slime-skew-sealresp-K8-s42-20260827/seals.jsonl",
    "scripts/checkpoint_retention.py",
    "scripts/experiment_profiles.sh",
    "scripts/phase2_manifest.py",
    "scripts/phase2_reconciler.py",
    "scripts/phase2_seal_rm.py",
    "scripts/phase2_trace_runner.py",
    "scripts/phase2_verifiers.py",
    "scripts/phase3_archive.py",
    "scripts/phase3_auto_recover.sh",
    "scripts/phase3_gate3a.py",
    "scripts/phase3_gate3b.py",
    "scripts/phase3_regress.sh",
    "scripts/resource_gate.py",
]

CANONICAL_RUNS = [
    {
        "id": "p3a-slime-skew-autofix-K8-s42-20260828",
        "role": "3A 注入组消费侧 0 混算",
    },
    {
        "id": "p3a-slime-none-autofix-K8-s42-20260828",
        "role": "3A 干净组零误改",
    },
    {
        "id": "p3b-probe-stage1-K8-s42-20260828",
        "role": "3B resume 探针 checkpoint 生成",
    },
    {
        "id": "p3b-probe-stage2-K8-s42-20260828",
        "role": "3B resume 探针续跑",
    },
    {
        "id": "p3b-slime-crash-autorecover-K8-s42-20260828",
        "role": "3B RM 异常边界探针",
    },
    {
        "id": "p3b-slime-kill-autorecover-K8-s42-20260828",
        "role": "3B 自动恢复初次实弹",
    },
    {
        "id": "p3b-slime-kill-autorecover2-K8-s42-20260828",
        "role": "3B 自动恢复门禁实弹",
    },
    {
        "id": "smoke-K8-s42-20260828-232940",
        "role": "内存/IO 新配置冒烟",
    },
    {
        "id": "p1-slime-skew-K8-s42-20260828-233354",
        "role": "新配置 Seal/CAS/Manifest 复验",
    },
    {
        "id": "p1-slime-none-K8-s42-20260828-234039",
        "role": "新配置自动恢复复验",
    },
    {
        "id": "p3c-slime-long-seal-K8-s42-20260828",
        "role": "3C 历史 0.5B Seal 100 步对照",
    },
    {
        "id": "p1-slime-none-K8-s42-20260829-001020",
        "role": "3C 新配置 0.5B 100 步长程",
    },
    {
        "id": "p1-slime-skew-K8-s42-20260829-100342",
        "role": "3C Qwen2.5-1.5B 协议升级复验",
    },
]

SUPERSEDED_RUNS = [
    {
        "id": "smoke-K8-s42-20260828-232347",
        "steps": None,
        "reason": "冒烟配置修复前的失败尝试，仅保留本地诊断日志/checkpoint",
    },
    {
        "id": "p3c-slime-long-clean-K8-s42-20260828",
        "steps": 7,
        "reason": "训练失败，未形成 100 步对照",
    },
    {
        "id": "p3c-slime-long-clean2-K8-s42-20260828",
        "steps": 9,
        "reason": "训练中断，未形成 100 步对照",
    },
    {
        "id": "p3c-slime-long-clean-dm-K8-s42-20260828",
        "steps": 70,
        "reason": "训练中断，未形成 100 步对照",
    },
    {
        "id": "p3c-slime-long-seal-dm-K8-s42-20260828",
        "steps": 10,
        "reason": "CUDA invalid argument，已由完整 100 步 Seal 实验替代",
    },
]

LOSS_RE = re.compile(r"step (\d+): \{'train/loss': ([^,}]+)")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args, check=True):
    proc = subprocess.run(
        ["git", *args], cwd=str(BASE), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "git command failed")
    return proc.stdout.strip()


def read_json(relative):
    return json.loads((BASE / relative).read_text())


def jsonl(relative):
    with (BASE / relative).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_loss_log(relative, expected_steps):
    path = BASE / relative
    matches = LOSS_RE.findall(path.read_text(errors="replace"))
    seen = {}
    duplicates = []
    for step_text, loss_text in matches:
        step = int(step_text)
        if step in seen:
            duplicates.append(step)
        seen[step] = float(loss_text)
    expected = list(range(expected_steps))
    missing = [step for step in expected if step not in seen]
    extras = sorted(step for step in seen if step not in expected)
    if duplicates or missing or extras:
        raise RuntimeError(
            "loss 轨迹不完整 {}: duplicates={} missing={} extras={}".format(
                relative, duplicates, missing, extras
            )
        )
    values = [seen[step] for step in expected]
    tail = values[expected_steps // 2:]
    return {
        "source": relative,
        "source_sha256": sha256_file(path),
        "steps": expected_steps,
        "loss": values,
        "mean": statistics.mean(values),
        "population_std": statistics.pstdev(values),
        "tail_half_mean": statistics.mean(tail),
        "tail_half_population_std": statistics.pstdev(tail),
    }


def protocol_counts(run_id):
    seals = list(jsonl("runs/{}/seals.jsonl".format(run_id)))
    rewards = list(jsonl("runs/{}/rewards.jsonl".format(run_id)))
    return {
        "seals_total": len(seals),
        "sealed": sum(item.get("status") == "SEALED" for item in seals),
        "aborted": sum(item.get("status") == "ABORTED" for item in seals),
        "autofix_groups": sum(item.get("autofix") is True for item in seals),
        "rewards_total": len(rewards),
        "non_v1_records": sum(item.get("verifier") != "v1" for item in rewards),
        "autofix_samples": sum(item.get("autofix") is True for item in rewards),
    }


def write_convergence_evidence():
    new = parse_loss_log(
        "runs/p1-slime-none-K8-s42-20260829-001020/logs/train.log", 100
    )
    historical = parse_loss_log(
        "runs/p3c-slime-long-seal-K8-s42-20260828/logs/train.log", 100
    )
    upgrade = parse_loss_log(
        "runs/p1-slime-skew-K8-s42-20260829-100342/logs/train.log", 20
    )
    mean_abs_diff = statistics.mean(
        abs(left - right) for left, right in zip(new["loss"], historical["loss"])
    )
    evidence = {
        "schema_version": 1,
        "stage": "3C",
        "comparison_scope": (
            "0.5B 新配置 100 步与历史同 Seal 配置 100 步对齐；"
            "1.5B 为 20 步协议升级复验，不冒充 100 步长程对照"
        ),
        "long_run_new": new,
        "long_run_historical": historical,
        "paired_mean_absolute_loss_diff": mean_abs_diff,
        "threshold": 0.05,
        "pass": mean_abs_diff < 0.05,
        "model_upgrade_1_5b": {
            "loss_trace": upgrade,
            "protocol": protocol_counts("p1-slime-skew-K8-s42-20260829-100342"),
        },
        "long_run_protocol": protocol_counts(
            "p1-slime-none-K8-s42-20260829-001020"
        ),
    }
    if not evidence["pass"]:
        raise RuntimeError("G3C2 loss 对齐门禁失败")
    CONVERGENCE_PATH.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n"
    )
    return evidence


def portable_files_for_run(run_id):
    root = BASE / "runs" / run_id
    if not root.is_dir():
        raise RuntimeError("缺少归档实验目录: {}".format(root.relative_to(BASE)))
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if "checkpoints" in rel_parts:
            continue
        if path.name.endswith((".sqlite", ".sqlite-shm", ".sqlite-wal")):
            continue
        files.append(str(path.relative_to(BASE)))
    return sorted(files)


def checkpoint_summary(run_id):
    root = BASE / "runs" / run_id / "checkpoints"
    if not root.is_dir():
        return {"present": False, "retention": "not present / already cleaned"}
    files = sorted(path for path in root.rglob("*") if path.is_file())
    iterations = sorted(
        path.name for path in root.iterdir()
        if path.is_dir() and (path.name.startswith("iter_") or path.name.isdigit())
    )
    if not files and not iterations:
        return {"present": False, "retention": "empty directory / checkpoint already cleaned"}
    inventory = hashlib.sha256()
    total = 0
    for path in files:
        size = path.stat().st_size
        total += size
        rel = str(path.relative_to(root))
        inventory.update("{}:{}\n".format(rel, size).encode())
    latest = root / "latest_checkpointed_iteration.txt"
    return {
        "present": True,
        "retention": "local large artifact; Git excludes checkpoint weights",
        "file_count": len(files),
        "total_bytes": total,
        "iterations": iterations,
        "latest_checkpointed_iteration": (
            latest.read_text().strip() if latest.exists() else None
        ),
        "path_size_inventory_sha256": inventory.hexdigest(),
        "note": "摘要绑定路径与大小，不等同于大权重内容哈希",
    }


def evidence_entry(relative):
    path = BASE / relative
    if not path.is_file():
        raise RuntimeError("缺少 portable evidence: {}".format(relative))
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def expected_portable_files():
    portable = set(CORE_FILES)
    for run in CANONICAL_RUNS:
        portable.update(portable_files_for_run(run["id"]))
    return portable


def build_manifest():
    gates = {}
    for relative in GATE_FILES:
        data = read_json(relative)
        gates[data["stage"]] = {
            "path": relative,
            "status": data.get("status"),
            "passed": sum(item.get("pass") is True for item in data["gates"].values()),
            "total": len(data["gates"]),
        }
    regression = read_json("runs/PHASE3_REGRESSION.json")
    run_records = []
    portable = expected_portable_files()
    for run in CANONICAL_RUNS:
        files = portable_files_for_run(run["id"])
        run_records.append({
            "id": run["id"],
            "role": run["role"],
            "portable_file_count": len(files),
            "checkpoint": checkpoint_summary(run["id"]),
        })
    entries = [evidence_entry(relative) for relative in sorted(portable)]
    manifest = {
        "schema_version": 1,
        "archive": "RewardTxn Phase 3 final",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_base_commit": git("rev-parse", "HEAD"),
        "expected_final_tag": "phase3-final",
        "verdict": "PASS",
        "gate_summary": gates,
        "regression": {
            "path": "runs/PHASE3_REGRESSION.json",
            "status": regression.get("status"),
            "passed": regression.get("passed"),
            "total": regression.get("total"),
        },
        "portable_evidence": entries,
        "canonical_runs": run_records,
        "superseded_local_runs": SUPERSEDED_RUNS,
        "policy": {
            "portable": "进入 Git，可在 checkout 后校验 SHA-256",
            "checkpoints": (
                "大权重不进入 Git；保留本地路径/大小清单摘要。"
                "Phase 3 回归不依赖大 checkpoint。"
            ),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def ensure_tracked(relative):
    proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=str(BASE), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def verify(args):
    errors = []
    manifest = None
    if not MANIFEST_PATH.is_file():
        errors.append("缺少 {}".format(MANIFEST_PATH.relative_to(BASE)))
    else:
        try:
            loaded = json.loads(MANIFEST_PATH.read_text())
            if isinstance(loaded, dict):
                manifest = loaded
            else:
                errors.append("archive manifest 根节点必须是 object")
        except (OSError, ValueError) as exc:
            errors.append("archive manifest 无法解析: {}".format(exc))

    expected_paths = expected_portable_files()
    expected_run_ids = [run["id"] for run in CANONICAL_RUNS]
    if manifest is not None:
        identity = {
            "schema_version": 1,
            "archive": "RewardTxn Phase 3 final",
            "expected_final_tag": "phase3-final",
            "verdict": "PASS",
        }
        for key, expected in identity.items():
            if manifest.get(key) != expected:
                errors.append("manifest identity 不匹配: {}={!r}".format(key, manifest.get(key)))

        entries = manifest.get("portable_evidence")
        if not isinstance(entries, list) or not entries:
            errors.append("manifest portable_evidence 必须是非空数组")
            entries = []
        entry_paths = [entry.get("path") for entry in entries if isinstance(entry, dict)]
        valid_entry_paths = [path for path in entry_paths if isinstance(path, str)]
        if (
            len(entry_paths) != len(entries)
            or len(valid_entry_paths) != len(entry_paths)
            or len(valid_entry_paths) != len(set(valid_entry_paths))
        ):
            errors.append("manifest portable_evidence 含非法项或重复路径")
        actual_paths = set(valid_entry_paths)
        if actual_paths != expected_paths:
            errors.append(
                "manifest portable 集合不完整: missing={} extra={}".format(
                    sorted(expected_paths - actual_paths), sorted(actual_paths - expected_paths)
                )
            )

        archived_runs = manifest.get("canonical_runs")
        if not isinstance(archived_runs, list) or not archived_runs:
            errors.append("manifest canonical_runs 必须是非空数组")
            archived_runs = []
        archived_ids = [run.get("id") for run in archived_runs if isinstance(run, dict)]
        if (
            len(archived_ids) != len(archived_runs)
            or archived_ids != expected_run_ids
            or len(archived_ids) != len(set(archived_ids))
        ):
            errors.append("manifest canonical run 集合/顺序不匹配")

        expected_superseded_ids = [run["id"] for run in SUPERSEDED_RUNS]
        superseded = manifest.get("superseded_local_runs")
        superseded_ids = [
            run.get("id") for run in superseded if isinstance(run, dict)
        ] if isinstance(superseded, list) else []
        if superseded_ids != expected_superseded_ids:
            errors.append("manifest superseded run 集合/顺序不匹配")

        gate_summary = manifest.get("gate_summary")
        if not isinstance(gate_summary, dict) or set(gate_summary) != {"3A", "3B", "3C"}:
            errors.append("manifest gate_summary 必须精确包含 3A/3B/3C")
        archived_regression = manifest.get("regression")
        if not isinstance(archived_regression, dict) or not (
            archived_regression.get("status") == "PASS"
            and archived_regression.get("passed") == archived_regression.get("total") == 9
        ):
            errors.append("manifest regression 摘要不是 9/9 PASS")

        base_commit = manifest.get("archive_base_commit")
        if not isinstance(base_commit, str) or subprocess.run(
            ["git", "merge-base", "--is-ancestor", base_commit or "invalid", "HEAD"],
            cwd=str(BASE), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode:
            errors.append("archive_base_commit 不是当前 HEAD 的有效祖先")

        for entry in entries:
            if not isinstance(entry, dict) or not {"path", "bytes", "sha256"} <= set(entry):
                continue
            relative = entry["path"]
            path = BASE / relative
            if not path.is_file():
                errors.append("portable evidence 缺失: {}".format(relative))
                continue
            if path.stat().st_size != entry["bytes"]:
                errors.append("文件大小变化: {}".format(relative))
            elif sha256_file(path) != entry["sha256"]:
                errors.append("SHA-256 不匹配: {}".format(relative))
            if not ensure_tracked(relative):
                errors.append("portable evidence 未进入 Git: {}".format(relative))

    for relative in GATE_FILES:
        data = read_json(relative)
        passed = sum(item.get("pass") is True for item in data.get("gates", {}).values())
        if data.get("status") != "PASS" or passed != 3:
            errors.append("门禁未通过: {} ({}/3)".format(relative, passed))
    regression = read_json("runs/PHASE3_REGRESSION.json")
    if not (
        regression.get("status") == "PASS"
        and regression.get("passed") == regression.get("total") == 9
    ):
        errors.append("Phase 3 回归不是 9/9 PASS")
    convergence = read_json("runs/PHASE3_CONVERGENCE_EVIDENCE.json")
    if not (
        convergence.get("pass") is True
        and convergence["long_run_new"].get("steps") == 100
        and convergence["long_run_historical"].get("steps") == 100
        and convergence.get("paired_mean_absolute_loss_diff", 1) < 0.05
    ):
        errors.append("长程收敛归档证据不满足 G3C2")
    protocol = convergence["model_upgrade_1_5b"]["protocol"]
    if not (
        protocol.get("aborted") == 20
        and protocol.get("autofix_groups") == 20
        and protocol.get("non_v1_records") == 0
    ):
        errors.append("1.5B 协议升级证据不满足 20/20 autofix + 0 非 v1")

    for tag in ("phase3a", "phase3b", "phase3c"):
        if subprocess.run(
            ["git", "rev-parse", "--verify", "refs/tags/{}^{{}}".format(tag)],
            cwd=str(BASE), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode:
            errors.append("缺少阶段 tag: {}".format(tag))
    if args.require_final_tag:
        proc = subprocess.run(
            ["git", "rev-parse", "--verify", "refs/tags/phase3-final^{}"],
            cwd=str(BASE), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True,
        )
        if proc.returncode:
            errors.append("缺少 phase3-final tag")
        else:
            tag_commit = proc.stdout.strip()
            head = git("rev-parse", "HEAD")
            if tag_commit != head:
                errors.append("phase3-final 未指向当前 HEAD")

    if args.require_local_sources and manifest:
        for run in manifest.get("canonical_runs", []):
            if not isinstance(run, dict) or run.get("id") not in expected_run_ids:
                continue
            if "checkpoint" not in run:
                errors.append("manifest 缺少 checkpoint 摘要: {}".format(run["id"]))
                continue
            current = checkpoint_summary(run["id"])
            archived = run["checkpoint"]
            if current != archived:
                errors.append("本地 checkpoint 清单变化: {}".format(run["id"]))
        for key in ("long_run_new", "long_run_historical"):
            source = BASE / convergence[key]["source"]
            if not source.is_file():
                errors.append("本地源日志缺失: {}".format(source.relative_to(BASE)))
            elif sha256_file(source) != convergence[key]["source_sha256"]:
                errors.append("本地源日志哈希变化: {}".format(source.relative_to(BASE)))

    if args.require_clean:
        status = git("status", "--porcelain")
        if status:
            errors.append("工作区不干净:\n{}".format(status))

    if errors:
        print("PHASE3 ARCHIVE VERIFY: FAIL", file=sys.stderr)
        for error in errors:
            print("- {}".format(error), file=sys.stderr)
        return 1
    count = len(manifest.get("portable_evidence", [])) if manifest else 0
    print("PHASE3 ARCHIVE VERIFY: PASS ({} portable files)".format(count))
    return 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("write")
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--require-local-sources", action="store_true")
    verify_parser.add_argument("--require-final-tag", action="store_true")
    verify_parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()

    if args.command == "write":
        convergence = write_convergence_evidence()
        manifest = build_manifest()
        print(
            "wrote {} ({} portable files, loss MAE {:.6f})".format(
                MANIFEST_PATH.relative_to(BASE),
                len(manifest["portable_evidence"]),
                convergence["paired_mean_absolute_loss_diff"],
            )
        )
        return 0
    return verify(args)


if __name__ == "__main__":
    sys.exit(main())
