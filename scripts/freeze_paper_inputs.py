#!/usr/bin/env python3
"""Generate and verify a content-addressed paper experiment input manifest.

The manifest records source repositories, the locally runnable container image,
and the exact model/data/schedule paths used by an experiment.  Registry
digests are copied only from ``docker image inspect`` RepoDigests; when Docker
is unavailable they remain null rather than being guessed from a tag or a
short local image ID.

Examples:
  python3 scripts/freeze_paper_inputs.py generate --out runs/PAPER_INPUTS.json \
    --image slimerl/slime:v0.3.1 \
    --asset model=models/Qwen2.5-1.5B-Instruct \
    --asset torch_dist=models/Qwen2.5-1.5B-Instruct_torch_dist \
    --asset data=models/datasets/dapo-math-17k/dapo-math-17k.jsonl \
    --asset schedule=prereg/fault_schedules/e7-s42-faulted.json
  python3 scripts/freeze_paper_inputs.py verify \
    --manifest runs/PAPER_INPUTS.json --require-image slimerl/slime:v0.3.1
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


BASE = Path(__file__).resolve().parent.parent
CHUNK_SIZE = 8 * 1024 * 1024

DEFAULT_REPOS = (
    ("rewardtxn", BASE),
    ("slime", BASE / "third_party" / "slime"),
    ("areal", BASE / "third_party" / "areal"),
    ("transferqueue", BASE / "third_party" / "transferqueue"),
)

# This is deliberately an audit superset.  Missing optional paper inputs make
# the default manifest a draft; callers freezing one concrete run should pass
# explicit --asset entries for exactly the inputs that run uses.
DEFAULT_ASSETS = (
    ("model_qwen2_5_0_5b", BASE / "models" / "Qwen2.5-0.5B-Instruct"),
    ("torch_dist_qwen2_5_0_5b", BASE / "models" / "Qwen2.5-0.5B-Instruct_torch_dist"),
    ("model_qwen2_5_1_5b", BASE / "models" / "Qwen2.5-1.5B-Instruct"),
    ("torch_dist_qwen2_5_1_5b", BASE / "models" / "Qwen2.5-1.5B-Instruct_torch_dist"),
    ("model_qwen2_5_3b", BASE / "models" / "Qwen2.5-3B-Instruct"),
    ("torch_dist_qwen2_5_3b", BASE / "models" / "Qwen2.5-3B-Instruct_torch_dist"),
    ("data_dapo_math_17k", BASE / "models" / "datasets" / "dapo-math-17k" / "dapo-math-17k.jsonl"),
    ("data_gsm8k", BASE / "models" / "datasets" / "gsm8k"),
    ("data_humaneval", BASE / "models" / "datasets" / "humaneval"),
    ("data_mbpp", BASE / "models" / "datasets" / "mbpp"),
)


def _run(argv: Sequence[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(argv), cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, universal_newlines=False, timeout=120,
    )


def _sha256_file(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _tree_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(str(root), followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d != ".git")
        for filename in sorted(filenames):
            yield Path(dirpath) / filename


def hash_path(path: Path) -> Dict[str, object]:
    """Hash a file or directory without loading large files into memory."""
    path = path.resolve()
    if not path.exists():
        return {"status": "missing", "content_sha256": None, "files": 0, "bytes": 0}
    if path.is_file():
        digest, size = _sha256_file(path)
        return {
            "status": "ok", "kind": "file", "content_sha256": digest,
            "files": 1, "bytes": size, "hash_method": "sha256(file-bytes)",
        }
    if not path.is_dir():
        return {"status": "unsupported", "content_sha256": None, "files": 0, "bytes": 0}

    aggregate = hashlib.sha256()
    files = 0
    total = 0
    for child in _tree_files(path):
        rel = child.relative_to(path).as_posix()
        if child.is_symlink():
            payload = os.readlink(str(child)).encode("utf-8", "surrogateescape")
            child_digest = hashlib.sha256(payload).hexdigest()
            size = len(payload)
            kind = "symlink"
        elif child.is_file():
            child_digest, size = _sha256_file(child)
            kind = "file"
        else:
            continue
        aggregate.update(kind.encode("ascii") + b"\0")
        aggregate.update(rel.encode("utf-8", "surrogateescape") + b"\0")
        aggregate.update(str(size).encode("ascii") + b"\0")
        aggregate.update(child_digest.encode("ascii") + b"\n")
        files += 1
        total += size
    return {
        "status": "ok", "kind": "directory", "content_sha256": aggregate.hexdigest(),
        "files": files, "bytes": total,
        "hash_method": "sha256(sorted(type,NUL,relative-path,NUL,size,NUL,file-sha256))",
        "excluded": [".git/"],
    }


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(BASE).as_posix()
    except ValueError:
        return str(resolved)


def _untracked_paths(repo: Path, status: bytes) -> List[Path]:
    paths = []
    for raw in status.decode("utf-8", "surrogateescape").splitlines():
        if not raw.startswith("?? "):
            continue
        value = raw[3:]
        # porcelain v1 quotes unusual paths.  Such paths remain represented in
        # the status bytes, but are not opened here; the status still changes
        # the aggregate dirty hash.
        if value.startswith('"'):
            continue
        paths.append((repo / value).resolve())
    return paths


def git_state(repo: Path, ignored_paths: Sequence[Path] = ()) -> Dict[str, object]:
    repo = repo.resolve()
    if not (repo / ".git").exists():
        return {"path": _display_path(repo), "status": "missing_or_not_git", "clean": False}
    head = _run(["git", "rev-parse", "HEAD"], repo)
    status_proc = _run(["git", "status", "--porcelain=v1", "--untracked-files=all"], repo)
    diff = _run(["git", "diff", "--binary", "HEAD", "--"], repo)
    if head.returncode or status_proc.returncode or diff.returncode:
        return {"path": _display_path(repo), "status": "git_error", "clean": False}

    ignored = {p.resolve() for p in ignored_paths}
    status_lines = []
    for line in status_proc.stdout.decode("utf-8", "surrogateescape").splitlines():
        value = line[3:] if len(line) >= 4 else ""
        candidate = (repo / value).resolve() if value and not value.startswith('"') else None
        if candidate in ignored:
            continue
        status_lines.append(line)
    canonical_status = ("\n".join(status_lines) + ("\n" if status_lines else "")).encode(
        "utf-8", "surrogateescape"
    )

    dirty_digest = hashlib.sha256()
    dirty_digest.update(b"status\0" + canonical_status)
    dirty_digest.update(b"diff\0" + diff.stdout)
    for untracked in sorted(_untracked_paths(repo, canonical_status), key=lambda p: str(p)):
        if untracked in ignored or not untracked.is_file():
            continue
        rel = untracked.relative_to(repo).as_posix()
        file_digest, size = _sha256_file(untracked)
        dirty_digest.update(rel.encode("utf-8", "surrogateescape") + b"\0")
        dirty_digest.update(str(size).encode("ascii") + b"\0" + file_digest.encode("ascii"))
    return {
        "path": _display_path(repo),
        "status": "ok",
        "commit_sha": head.stdout.decode().strip(),
        "clean": not status_lines,
        "dirty_diff_sha256": dirty_digest.hexdigest(),
        "dirty_entries": status_lines,
    }


def inspect_image(reference: str) -> Dict[str, object]:
    try:
        proc = _run(["docker", "image", "inspect", reference])
    except OSError as exc:
        return {
            "reference": reference, "inspect_status": "unavailable",
            "local_image_id": None, "registry_digest": None, "repo_digests": [],
            "error": str(exc),
        }
    if proc.returncode:
        return {
            "reference": reference, "inspect_status": "unavailable",
            "local_image_id": None, "registry_digest": None, "repo_digests": [],
            "error": proc.stderr.decode("utf-8", "replace").strip()[:500],
        }
    try:
        record = json.loads(proc.stdout.decode("utf-8"))[0]
    except (ValueError, IndexError, TypeError) as exc:
        return {
            "reference": reference, "inspect_status": "invalid_output",
            "local_image_id": None, "registry_digest": None, "repo_digests": [],
            "error": str(exc),
        }
    repo_digests = sorted(record.get("RepoDigests") or [])
    registry_digest = None
    if repo_digests:
        # This is registry-provided metadata stored in the local image object,
        # not an inferred digest from the tag or local config ID.
        registry_digest = repo_digests[0].rsplit("@", 1)[-1]
    return {
        "reference": reference,
        "inspect_status": "ok",
        "local_image_id": record.get("Id"),
        "registry_digest": registry_digest,
        "repo_digests": repo_digests,
    }


def _parse_named_paths(values: Sequence[str], defaults: Sequence[Tuple[str, Path]]) -> List[Tuple[str, Path]]:
    if not values:
        return list(defaults)
    parsed = []
    names = set()
    for value in values:
        if "=" not in value:
            raise ValueError("expected NAME=PATH, got {!r}".format(value))
        name, raw_path = value.split("=", 1)
        if not name or name in names:
            raise ValueError("asset/repo names must be non-empty and unique: {!r}".format(name))
        path = Path(raw_path)
        if not path.is_absolute():
            path = BASE / path
        parsed.append((name, path.resolve()))
        names.add(name)
    return parsed


def _parse_repositories(values: Sequence[str]) -> List[Tuple[str, Path]]:
    repositories = {name: path.resolve() for name, path in DEFAULT_REPOS}
    for name, path in _parse_named_paths(values, ()):
        if name not in repositories:
            raise ValueError("unknown repository name {!r}".format(name))
        repositories[name] = path.resolve()
    return [(name, repositories[name]) for name, _ in DEFAULT_REPOS]


def build_provenance(repo_values: Sequence[str], ignored_paths: Sequence[Path] = ()) -> Dict[str, object]:
    repos = _parse_repositories(repo_values)
    return {
        name: git_state(path, ignored_paths if name == "rewardtxn" else ())
        for name, path in repos
    }


def generate(args: argparse.Namespace) -> int:
    output = args.out.resolve()
    repos = build_provenance(args.repo, [output])
    assets = []
    for name, path in _parse_named_paths(args.asset, DEFAULT_ASSETS):
        item = {"name": name, "path": _display_path(path)}
        item.update(hash_path(path))
        assets.append(item)
    image = inspect_image(args.image)
    frozen = (
        all(item.get("clean") is True for item in repos.values())
        and all(item.get("status") == "ok" for item in assets)
        and image.get("inspect_status") == "ok"
        and bool(image.get("local_image_id"))
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "frozen": frozen,
        "repositories": repos,
        "image": image,
        "assets": assets,
        "notes": {
            "registry_digest": "null unless present in local Docker RepoDigests; never inferred from a tag or short image ID",
            "verification": "formal runs must verify this manifest against current repositories, image ID, and required assets",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS" if frozen else "DRAFT", "manifest": str(output)}, ensure_ascii=False))
    return 0 if frozen else 2


def _asset_absolute(item: Dict[str, object]) -> Path:
    path = Path(str(item["path"]))
    return path if path.is_absolute() else (BASE / path).resolve()


def verify(args: argparse.Namespace) -> int:
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        report = {"status": "FAIL", "errors": ["cannot read manifest: {}".format(exc)]}
        print(json.dumps(report, ensure_ascii=False))
        return 2
    errors = []
    if manifest.get("schema_version") != 1:
        errors.append("unsupported manifest schema_version")
    if not manifest.get("frozen"):
        errors.append("manifest is a draft (frozen is not true)")

    expected_repos = manifest.get("repositories", {})
    try:
        required_repos = dict(_parse_repositories(args.require_repo_path))
    except ValueError as exc:
        errors.append(str(exc))
        required_repos = {name: path.resolve() for name, path in DEFAULT_REPOS}
    for name, required_path in required_repos.items():
        expected = expected_repos.get(name)
        if not expected:
            errors.append("required repository is not frozen: {}".format(name))
            continue
        raw_path = Path(str(expected.get("path", "")))
        recorded_path = raw_path if raw_path.is_absolute() else BASE / raw_path
        if recorded_path.resolve() != required_path:
            errors.append("repository {} points to unexpected path {}".format(name, recorded_path))
    for name, expected in expected_repos.items():
        raw_path = Path(str(expected.get("path", "")))
        repo = raw_path if raw_path.is_absolute() else BASE / raw_path
        current = git_state(repo, [args.manifest.resolve()] if name == "rewardtxn" else ())
        if current.get("clean") is not True:
            errors.append("repository {} is dirty or unavailable (dirty_diff_sha256={})".format(
                name, current.get("dirty_diff_sha256")
            ))
        if current.get("commit_sha") != expected.get("commit_sha"):
            errors.append("repository {} commit changed: {} != {}".format(
                name, current.get("commit_sha"), expected.get("commit_sha")
            ))
        if current.get("dirty_diff_sha256") != expected.get("dirty_diff_sha256"):
            errors.append("repository {} dirty diff hash changed".format(name))

    manifest_assets = manifest.get("assets", [])
    by_path = {str(_asset_absolute(item).resolve()): item for item in manifest_assets}
    for item in manifest_assets:
        current = hash_path(_asset_absolute(item))
        if current.get("status") != "ok":
            errors.append("asset {} is missing/unreadable".format(item.get("name")))
        elif current.get("content_sha256") != item.get("content_sha256"):
            errors.append("asset {} content hash changed".format(item.get("name")))
    for raw in args.require_asset_path:
        required = Path(raw)
        if not required.is_absolute():
            required = BASE / required
        if str(required.resolve()) not in by_path:
            errors.append("required asset path is not frozen: {}".format(required.resolve()))

    expected_image = manifest.get("image", {})
    requested_image = args.require_image or expected_image.get("reference")
    if args.require_image and args.require_image != expected_image.get("reference"):
        errors.append("requested image reference is not the frozen reference")
    if not requested_image:
        errors.append("manifest has no image reference")
    else:
        current_image = inspect_image(str(requested_image))
        if current_image.get("inspect_status") != "ok":
            errors.append("cannot inspect required local image")
        elif current_image.get("local_image_id") != expected_image.get("local_image_id"):
            errors.append("local image ID changed for {}".format(requested_image))
        expected_registry = expected_image.get("registry_digest")
        if expected_registry and expected_registry not in {
            str(value).rsplit("@", 1)[-1] for value in current_image.get("repo_digests", [])
        }:
            errors.append("frozen registry digest is absent from current RepoDigests")

    report = {"status": "PASS" if not errors else "FAIL", "errors": errors}
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if not errors else 2


def provenance(args: argparse.Namespace) -> int:
    ignored = []
    for raw in args.ignore_path:
        path = Path(raw)
        ignored.append(path if path.is_absolute() else BASE / path)
    report = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repositories": build_provenance(args.repo, ignored),
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="hash inputs and write a freeze manifest")
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--image", required=True)
    gen.add_argument("--repo", action="append", default=[], metavar="NAME=PATH")
    gen.add_argument("--asset", action="append", default=[], metavar="NAME=PATH")

    check = sub.add_parser("verify", help="verify repositories, image, and content hashes")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--require-image")
    check.add_argument("--require-asset-path", action="append", default=[])
    check.add_argument("--require-repo-path", action="append", default=[], metavar="NAME=PATH")
    check.add_argument("--report", type=Path)

    prov = sub.add_parser("provenance", help="record repository commit and dirty diff hashes")
    prov.add_argument("--repo", action="append", default=[], metavar="NAME=PATH")
    prov.add_argument("--out", type=Path)
    prov.add_argument("--ignore-path", action="append", default=[])

    args = parser.parse_args()
    try:
        if args.command == "generate":
            return generate(args)
        if args.command == "verify":
            return verify(args)
        return provenance(args)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print("freeze_paper_inputs: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
