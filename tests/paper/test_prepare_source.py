import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


BASE = Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_source", BASE / "scripts/prepare_paper_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrepareSourceTests(unittest.TestCase):
    def test_archive_ignores_dirty_checkout_and_verifies(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            out = root / "out"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "paper@test"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Paper Test"], check=True)
            (repo / "source.txt").write_text("frozen\n")
            subprocess.run(["git", "-C", str(repo), "add", "source.txt"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
            commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            (repo / "source.txt").write_text("dirty\n")
            manifest = module.prepare(repo, commit, out)
            self.assertEqual((out / "source.txt").read_text(), "frozen\n")
            self.assertEqual(manifest["commit_sha"], commit)
            self.assertEqual(manifest["entries"], 1)
            self.assertEqual(subprocess.check_output(["git", "-C", str(out), "status", "--porcelain"], text=True), "")
            self.assertEqual(module.verify(out), [])
            (out / "source.txt").write_text("tampered\n")
            self.assertTrue(module.verify(out))


if __name__ == "__main__":
    unittest.main()
