#!/usr/bin/env python3
"""Prepare and verify an immutable source tree from a pinned Git commit.

The active third-party checkout may contain user changes.  Paper runs mount a
separate clean clone at the frozen commit instead, so those changes are neither
overwritten nor accidentally included in an experiment.  The clone retains
Git metadata because formal launch validation checks its commit and cleanliness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


MANIFEST_RELATIVE = Path(".git") / "paper_source_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_identity(root: Path) -> dict:
    entries = []
    total_bytes = 0
    links = 0
    for path in sorted(root.rglob("*")):
        relative_path = path.relative_to(root)
        if ".git" in relative_path.parts:
            continue
        relative = str(relative_path)
        if path.is_symlink():
            entries.append((relative, "link", os.readlink(str(path))))
            links += 1
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        entries.append((relative, size, sha256_file(path)))
        total_bytes += size
    digest = hashlib.sha256()
    for relative, size, file_hash in entries:
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode())
    return {"entries": len(entries), "symlinks": links, "bytes": total_bytes, "content_sha256": digest.hexdigest()}


def git_output(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "git command failed")
    return proc.stdout.strip()


def prepare(repo: Path, commit: str, output: Path) -> dict:
    full_commit = git_output(repo, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(repo), str(output)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode(errors="replace").strip() or "git clone failed")
    git_output(output, "checkout", "--quiet", "--detach", full_commit)
    identity = tree_identity(output)
    manifest = {
        "schema_version": "1.0",
        "source_repository": str(repo.resolve()),
        "commit_sha": full_commit,
        **identity,
    }
    (output / MANIFEST_RELATIVE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify(output: Path) -> list[str]:
    path = output / MANIFEST_RELATIVE
    if not path.is_file():
        return [f"missing {MANIFEST_RELATIVE}"]
    try:
        manifest = json.loads(path.read_text())
    except ValueError as exc:
        return [f"invalid manifest: {exc}"]
    if manifest.get("schema_version") != "1.0":
        return ["unsupported schema_version"]
    identity = tree_identity(output)
    return [f"{key} mismatch" for key, value in identity.items() if manifest.get(key) != value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--repo", type=Path, required=True)
    create.add_argument("--commit", required=True)
    create.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        try:
            manifest = prepare(args.repo, args.commit, args.out)
        except (OSError, RuntimeError) as exc:
            print(f"PREPARE PAPER SOURCE: FAIL: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(manifest, indent=2))
        return 0
    errors = verify(args.out)
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
        return 1
    print(json.dumps({"status": "PASS", "path": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
