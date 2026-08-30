#!/usr/bin/env python3
"""Validate RewardTxn paper-run artifacts against the frozen data contract.

Paper mode validates structure and provenance: JSON/JSONL payloads, event
schema, config/schedule hashes, exit metadata, and SHA-256 manifest entries.
It does not turn a scientifically failing run into an invalid artifact;
``verdict.json`` is validated but may legitimately say FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((BASE / "configs" / "paper_event.schema.json").read_text())
REQUIRED_PAPER_FILES = (
    "meta.json", "config.json", "schedule.json", "events.jsonl", "manifests/",
    "metrics.json", "resource.jsonl", "verdict.json", "artifact_manifest.json",
    "stdout.log", "stderr.log", "exit_status.json",
)
REQUIRED_META_FIELDS = (
    "exp_id", "phase", "stack", "baseline", "commit_sha", "seed",
    "group_size_K", "batch_groups_U", "fault_injection", "created_at",
    "config_sha256", "schedule_sha256",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, errors: list[str], label: str):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        errors.append(f"{label} 无法解析: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label} 根节点必须是 object")
        return None
    return value


def _is_integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _type_matches(value, schema_type) -> bool:
    types = schema_type if isinstance(schema_type, list) else [schema_type]
    for kind in types:
        if kind == "null" and value is None:
            return True
        if kind == "object" and isinstance(value, dict):
            return True
        if kind == "array" and isinstance(value, list):
            return True
        if kind == "string" and isinstance(value, str):
            return True
        if kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if kind == "integer" and _is_integer(value):
            return True
        if kind == "boolean" and isinstance(value, bool):
            return True
    return False


def _validate_property(name: str, value, rule: dict, prefix: str, errors: list[str]) -> None:
    schema_type = rule.get("type")
    if schema_type is not None and not _type_matches(value, schema_type):
        errors.append(f"{prefix}.{name} 类型不符合 schema: {schema_type!r}")
        return
    if "enum" in rule and value not in rule["enum"]:
        errors.append(f"{prefix}.{name} 不在 enum 中: {value!r}")
    if isinstance(value, str) and len(value) < rule.get("minLength", 0):
        errors.append(f"{prefix}.{name} 长度小于 {rule['minLength']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in rule and value < rule["minimum"]:
            errors.append(f"{prefix}.{name} 小于 minimum={rule['minimum']}")
    if isinstance(value, list) and isinstance(rule.get("items"), dict):
        for index, item in enumerate(value):
            _validate_property(f"{name}[{index}]", item, rule["items"], prefix, errors)


def validate_event(event, lineno: int) -> list[str]:
    errors: list[str] = []
    prefix = f"events.jsonl:{lineno}"
    if not isinstance(event, dict):
        return [f"{prefix} 行不是 object"]
    for field in SCHEMA.get("required", []):
        if field not in event:
            errors.append(f"{prefix} 缺少必填字段 {field}")
    for name, value in event.items():
        rule = SCHEMA.get("properties", {}).get(name)
        if isinstance(rule, dict):
            _validate_property(name, value, rule, prefix, errors)
    for conditional in SCHEMA.get("allOf", []):
        expected = conditional.get("if", {}).get("properties", {})
        applies = all(event.get(name) == rule.get("const") for name, rule in expected.items() if "const" in rule)
        if applies:
            for field in conditional.get("then", {}).get("required", []):
                if field not in event:
                    errors.append(f"{prefix} 条件必填字段缺失 {field}")
    return errors


def _validate_jsonl(path: Path, kind: str) -> tuple[list[str], int]:
    errors: list[str] = []
    count = 0
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        return [f"{kind} 无法读取: {exc}"], 0
    with handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            count += 1
            try:
                value = json.loads(line)
            except ValueError as exc:
                errors.append(f"{kind}:{lineno} JSON 解析失败: {exc}")
                continue
            if kind == "events.jsonl":
                errors.extend(validate_event(value, lineno))
            elif not isinstance(value, dict):
                errors.append(f"{kind}:{lineno} 必须是 object")
            elif "ts" not in value or not isinstance(value["ts"], (int, float)):
                errors.append(f"{kind}:{lineno} 缺少数值 ts")
            if len(errors) >= 50:
                errors.append("错误过多，截断")
                break
    if count == 0:
        errors.append(f"{kind} 不得为空")
    return errors, count


def _validate_manifest(run_dir: Path, required_paths: set[str]) -> list[str]:
    errors: list[str] = []
    manifest = _load_json(run_dir / "artifact_manifest.json", errors, "artifact_manifest.json")
    if manifest is None:
        return errors
    if manifest.get("schema_version") != "1.0":
        errors.append("artifact_manifest.json schema_version 必须为 '1.0'")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        return errors + ["artifact_manifest.json files 必须是非空数组"]
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"artifact_manifest files[{index}] 不是 object")
            continue
        path_text = entry.get("path")
        if not isinstance(path_text, str) or not path_text or Path(path_text).is_absolute() or ".." in Path(path_text).parts:
            errors.append(f"artifact_manifest files[{index}] path 非法")
            continue
        if path_text == "artifact_manifest.json":
            errors.append("artifact_manifest.json 不得自哈希")
            continue
        if path_text in seen:
            errors.append(f"artifact_manifest 重复路径: {path_text}")
            continue
        seen.add(path_text)
        path = run_dir / path_text
        if not path.is_file():
            errors.append(f"artifact manifest 文件缺失: {path_text}")
            continue
        if entry.get("bytes") != path.stat().st_size:
            errors.append(f"artifact bytes 不匹配: {path_text}")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            errors.append(f"artifact sha256 非法: {path_text}")
        elif sha256_file(path) != digest:
            errors.append(f"artifact sha256 不匹配: {path_text}")
    missing = sorted(required_paths - seen)
    if missing:
        errors.append(f"artifact manifest 未覆盖必需文件: {missing}")
    manifest_files = {str(path.relative_to(run_dir)) for path in (run_dir / "manifests").rglob("*") if path.is_file()}
    missing_manifests = sorted(manifest_files - seen)
    if missing_manifests:
        errors.append(f"artifact manifest 未覆盖 manifests: {missing_manifests}")
    return errors


def validate_run(run_dir: Path, paper: bool) -> dict:
    report = {"run_dir": str(run_dir), "paper": paper, "checks": {}}
    required = REQUIRED_PAPER_FILES if paper else REQUIRED_PAPER_FILES[:8]
    missing = [name for name in required if not (run_dir / name).exists()]
    report["checks"]["required_files"] = {"pass": not missing, "missing": missing}

    meta_errors: list[str] = []
    meta = _load_json(run_dir / "meta.json", meta_errors, "meta.json") if (run_dir / "meta.json").exists() else None
    if meta is None and not meta_errors:
        meta_errors.append("meta.json 不存在")
    if meta is not None:
        for field in REQUIRED_META_FIELDS if paper else REQUIRED_META_FIELDS[:10]:
            if field not in meta:
                meta_errors.append(f"meta.json 缺少 {field}")
        for field in ("exp_id", "phase", "stack", "baseline", "created_at"):
            if field in meta and (not isinstance(meta[field], str) or not meta[field]):
                meta_errors.append(f"meta.json {field} 必须是非空字符串")
        if "commit_sha" in meta and not re.fullmatch(r"[0-9a-f]{40}", str(meta["commit_sha"])):
            meta_errors.append("meta.json commit_sha 必须是完整 40 位 Git SHA")
        for field in ("seed", "group_size_K", "batch_groups_U"):
            if field in meta and (not _is_integer(meta[field]) or (field != "seed" and meta[field] <= 0)):
                meta_errors.append(f"meta.json {field} 必须是有效整数")
        if "fault_injection" in meta and not isinstance(meta["fault_injection"], dict):
            meta_errors.append("meta.json fault_injection 必须是 object")
        for field, filename in (("config_sha256", "config.json"), ("schedule_sha256", "schedule.json")):
            if field in meta and (run_dir / filename).is_file():
                if not isinstance(meta[field], str) or sha256_file(run_dir / filename) != meta[field]:
                    meta_errors.append(f"meta.json {field} 与 {filename} 不匹配")
    report["checks"]["meta"] = {"pass": not meta_errors, "errors": meta_errors}

    json_errors: list[str] = []
    for filename in ("config.json", "schedule.json", "metrics.json", "verdict.json"):
        path = run_dir / filename
        if path.exists():
            _load_json(path, json_errors, filename)
    exit_path = run_dir / "exit_status.json"
    if paper and exit_path.exists():
        exit_status = _load_json(exit_path, json_errors, "exit_status.json")
        if exit_status is not None:
            if exit_status.get("completed") is not True:
                json_errors.append("exit_status.json completed 必须为 true")
            if not _is_integer(exit_status.get("return_code")):
                json_errors.append("exit_status.json return_code 必须为整数")
            if not isinstance(exit_status.get("finished_at"), str) or not exit_status["finished_at"]:
                json_errors.append("exit_status.json finished_at 必须是非空字符串")
    report["checks"]["json_documents"] = {"pass": not json_errors, "errors": json_errors}

    event_errors, event_count = _validate_jsonl(run_dir / "events.jsonl", "events.jsonl") if (run_dir / "events.jsonl").exists() else (["events.jsonl 不存在"], 0)
    report["checks"]["events"] = {"pass": not event_errors, "errors": event_errors, "lines": event_count}
    resource_errors, resource_count = _validate_jsonl(run_dir / "resource.jsonl", "resource.jsonl") if (run_dir / "resource.jsonl").exists() else (["resource.jsonl 不存在"], 0)
    report["checks"]["resources"] = {"pass": not resource_errors, "errors": resource_errors, "lines": resource_count}

    manifest_errors: list[str] = []
    manifest_dir = run_dir / "manifests"
    manifest_files = []
    if not manifest_dir.is_dir():
        manifest_errors.append("manifests/ 不存在")
    else:
        manifest_files = [path for path in manifest_dir.rglob("*") if path.is_file()]
        if not manifest_files:
            manifest_errors.append("manifests/ 为空")
        for path in manifest_files:
            _load_json(path, manifest_errors, str(path.relative_to(run_dir)))
    report["checks"]["manifests"] = {"pass": not manifest_errors, "errors": manifest_errors, "files": len(manifest_files)}

    artifact_errors: list[str] = []
    if paper and (run_dir / "artifact_manifest.json").exists() and not missing:
        required_manifest_paths = {name for name in REQUIRED_PAPER_FILES if not name.endswith("/") and name != "artifact_manifest.json"}
        artifact_errors = _validate_manifest(run_dir, required_manifest_paths)
    elif paper:
        artifact_errors.append("无法验证 artifact manifest：必需文件缺失")
    report["checks"]["artifact_manifest"] = {"pass": not artifact_errors, "errors": artifact_errors}

    report["status"] = "PASS" if all(check["pass"] for check in report["checks"].values()) else "FAIL"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--paper", action="store_true", help="严格论文产物模式")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = validate_run(args.run_dir, args.paper)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"status: {report['status']}")
        for name, check in report["checks"].items():
            detail = check.get("missing") or check.get("errors") or []
            print(f"  [{'PASS' if check['pass'] else 'FAIL'}] {name}: {detail[:3]}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
