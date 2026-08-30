import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


BASE = Path(__file__).resolve().parents[2]


def load_validator():
    spec = importlib.util.spec_from_file_location("artifact_validator", BASE / "scripts/validate_run_artifacts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


class ArtifactContractTests(unittest.TestCase):
    def make_run(self, root):
        run = Path(root)
        (run / "manifests").mkdir()
        write_json(run / "config.json", {"mode": "formal"})
        write_json(run / "schedule.json", {"seed": 7, "events": []})
        write_json(run / "meta.json", {
            "exp_id": "paper-test", "phase": "P1", "stack": "model", "baseline": "B6",
            "commit_sha": "a" * 40, "seed": 7, "group_size_K": 8, "batch_groups_U": 4,
            "fault_injection": {}, "created_at": "2026-08-30T00:00:00Z",
            "config_sha256": hashlib.sha256((run / "config.json").read_bytes()).hexdigest(),
            "schedule_sha256": hashlib.sha256((run / "schedule.json").read_bytes()).hexdigest(),
        })
        write_json(run / "metrics.json", {"n": 1})
        write_json(run / "verdict.json", {"status": "PASS"})
        write_json(run / "exit_status.json", {"completed": True, "return_code": 0, "finished_at": "now"})
        write_json(run / "manifests/fixture_manifest.json", {"kind": "protocol_model"})
        (run / "events.jsonl").write_text(json.dumps({"type": "fault", "ts": 0.0, "exp_id": "paper-test"}) + "\n")
        (run / "resource.jsonl").write_text(json.dumps({"ts": 0.0, "cpu": 0}) + "\n")
        (run / "stdout.log").write_text("")
        (run / "stderr.log").write_text("")
        files = []
        for path in sorted(p for p in run.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"):
            files.append({"path": str(path.relative_to(run)), "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        write_json(run / "artifact_manifest.json", {"schema_version": "1.0", "files": files})
        return run

    def test_valid_formal_run_passes(self):
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            report = validator.validate_run(self.make_run(tmp), True)
        self.assertEqual(report["status"], "PASS", report)

    def test_tampered_artifact_fails(self):
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            run = self.make_run(tmp)
            (run / "metrics.json").write_text("{}\n")
            report = validator.validate_run(run, True)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("sha256 不匹配" in error for error in report["checks"]["artifact_manifest"]["errors"]))

    def test_schema_rejects_invalid_fields(self):
        validator = load_validator()
        event = {"type": "recovery", "ts": 0.0, "exp_id": "", "step": -1, "recovery_decision": "invented"}
        self.assertTrue(validator.validate_event(event, 1))


if __name__ == "__main__":
    unittest.main()
