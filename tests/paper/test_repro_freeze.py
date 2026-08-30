import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "freeze_paper_inputs", ROOT / "scripts" / "freeze_paper_inputs.py"
)
freeze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze)


class FreezeInputTests(unittest.TestCase):
    def test_hash_path_is_content_sensitive_and_streamed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "a.bin").write_bytes(b"abcdefgh")
            (root / "nested").mkdir()
            (root / "nested" / "b.bin").write_bytes(b"ijklmnop")
            with mock.patch.object(freeze, "CHUNK_SIZE", 3):
                first = freeze.hash_path(root)
                (root / "nested" / "b.bin").write_bytes(b"ijklmnox")
                second = freeze.hash_path(root)
            self.assertEqual(first["files"], 2)
            self.assertEqual(first["bytes"], 16)
            self.assertNotEqual(first["content_sha256"], second["content_sha256"])

    def test_image_id_is_not_reported_as_registry_digest(self):
        payload = json.dumps([{
            "Id": "sha256:" + "a" * 64,
            "RepoDigests": [],
        }]).encode()
        completed = subprocess.CompletedProcess([], 0, payload, b"")
        with mock.patch.object(freeze, "_run", return_value=completed):
            image = freeze.inspect_image("example:test")
        self.assertEqual(image["local_image_id"], "sha256:" + "a" * 64)
        self.assertIsNone(image["registry_digest"])

    def test_verify_rejects_mutated_required_asset(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repos = []
            repo_records = {}
            for name in ("rewardtxn", "slime", "areal", "transferqueue"):
                path = root / name
                path.mkdir()
                repos.append((name, path))
                repo_records[name] = {
                    "path": str(path), "commit_sha": "abc", "clean": True,
                    "dirty_diff_sha256": hashlib.sha256(b"").hexdigest(), "status": "ok",
                }
            asset = root / "model.bin"
            asset.write_bytes(b"frozen")
            asset_record = {"name": "model", "path": str(asset)}
            asset_record.update(freeze.hash_path(asset))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": 1, "frozen": True, "repositories": repo_records,
                "image": {"reference": "image:test", "inspect_status": "ok",
                          "local_image_id": "sha256:" + "b" * 64,
                          "registry_digest": None, "repo_digests": []},
                "assets": [asset_record],
            }))
            asset.write_bytes(b"mutated")
            args = argparse.Namespace(
                manifest=manifest, require_image="image:test",
                require_asset_path=[str(asset)], require_repo_path=[], report=None,
            )
            current_repo = {
                "path": "unused", "commit_sha": "abc", "clean": True,
                "dirty_diff_sha256": hashlib.sha256(b"").hexdigest(), "status": "ok",
            }
            current_image = {
                "reference": "image:test", "inspect_status": "ok",
                "local_image_id": "sha256:" + "b" * 64,
                "registry_digest": None, "repo_digests": [],
            }
            with mock.patch.object(freeze, "DEFAULT_REPOS", tuple(repos)), \
                    mock.patch.object(freeze, "git_state", return_value=current_repo), \
                    mock.patch.object(freeze, "inspect_image", return_value=current_image):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = freeze.verify(args)
            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
