import json
import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "scripts" / "phase2_run.sh"
TRAIN_ENTRY = ROOT / "scripts" / "day2_slime_train.sh"


class Phase2EntryTests(unittest.TestCase):
    def _environment(self, base):
        (base / "third_party" / "slime").mkdir(parents=True, exist_ok=True)
        env = {key: value for key, value in os.environ.items() if not key.startswith("RTX_")}
        env.update({
            "RTX_BASE": str(base),
            "RTX_META_ONLY": "1",
            "RTX_SKIP_GATE": "1",
            "RTX_GPUS": "device=3,4,6,7",
            "RTX_SEED": "73",
            "RTX_BASELINE_MODE": "b6",
            "RTX_SCHEDULE": "prereg/schedule.json",
            "RTX_EXP_ID": "repro-entry-test",
            "RTX_LOCAL_SCRATCH": str(base / "scratch"),
        })
        return env

    def _init_repo(self, path):
        path.mkdir(parents=True, exist_ok=True)
        quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        subprocess.run(["git", "init", "-q", str(path)], check=True, **quiet)
        subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.invalid"], check=True, **quiet)
        subprocess.run(["git", "-C", str(path), "config", "user.name", "test"], check=True, **quiet)
        (path / "tracked.txt").write_text("tracked\n")
        subprocess.run(["git", "-C", str(path), "add", "."], check=True, **quiet)
        subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True, **quiet)

    def test_meta_records_effective_gpu_seed_baseline_and_schedule(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            (base / "scripts").mkdir()
            shutil.copy2(ROOT / "scripts" / "freeze_paper_inputs.py", base / "scripts")
            (base / "prereg").mkdir()
            (base / "prereg" / "schedule.json").write_text(
                json.dumps({"schedule_id": "test-s73", "events": []}) + "\n"
            )
            result = subprocess.run(
                ["bash", str(ENTRY), "none", "0", "3"],
                env=self._environment(base), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            run_dir = base / "runs" / "repro-entry-test"
            meta = json.loads((run_dir / "meta.json").read_text())
            config = json.loads((run_dir / "config.json").read_text())
            self.assertEqual(meta["params"]["gpus"], [3, 4, 6, 7])
            self.assertEqual(meta["seed"], 73)
            self.assertEqual(meta["baseline_mode"], "b6")
            self.assertEqual(
                meta["baseline_effective"],
                {"seal": True, "group_rm": True, "seal_auto_fix": True,
                 "custom_rm": "phase2_seal_rm.rm_function"},
            )
            self.assertEqual(meta["fault_schedule"]["schedule_id"], "test-s73")
            self.assertEqual(
                meta["fault_schedule"]["container_path"], "/workspace/prereg/schedule.json"
            )
            self.assertEqual(config["gpus"], [3, 4, 6, 7])
            self.assertEqual(config["seed"], 73)

    def test_formal_unimplemented_baseline_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            (base / "prereg").mkdir()
            (base / "prereg" / "schedule.json").write_text("{}\n")
            env = self._environment(base)
            env.update({"RTX_PAPER_MODE": "1", "RTX_BASELINE_MODE": "b3"})
            result = subprocess.run(
                ["bash", str(ENTRY), "none", "0", "3"], env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("mechanism implementation missing", result.stderr)
            self.assertFalse((base / "runs").exists())

    def test_formal_mode_requires_explicit_seed(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            (base / "prereg").mkdir()
            (base / "prereg" / "schedule.json").write_text("{}\n")
            env = self._environment(base)
            env.update({"RTX_PAPER_MODE": "1", "RTX_BASELINE_MODE": "b0"})
            env.pop("RTX_SEED")
            result = subprocess.run(
                ["bash", str(ENTRY), "none", "0", "3"], env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("formal run requires explicit RTX_SEED", result.stderr)

    def test_train_entry_forwards_seed_baseline_and_schedule_to_ray(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            model_scripts = base / "slime" / "scripts" / "models"
            model_scripts.mkdir(parents=True)
            (model_scripts / "test.sh").write_text("MODEL_ARGS=()\n")
            schedule = base / "schedule.json"
            schedule.write_text("{}\n")
            fake_bin = base / "bin"
            fake_bin.mkdir()
            ray_capture = base / "ray-args.txt"
            ray = fake_bin / "ray"
            ray.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$RAY_CAPTURE"\n')
            ray.chmod(0o755)
            env = {key: value for key, value in os.environ.items() if not key.startswith("RTX_")}
            env.update({
                "PATH": str(fake_bin) + ":" + env.get("PATH", ""),
                "RAY_CAPTURE": str(ray_capture),
                "SLIME_ROOT": str(base / "slime"),
                "RTX_MODEL_CONFIG": "test.sh",
                "MODEL_DIR": str(base / "model"),
                "DATA_PATH": str(base / "data.jsonl"),
                "SAVE_DIR": str(base / "checkpoints"),
                "RTX_SEED": "101",
                "RTX_BASELINE_MODE": "b6",
                "RTX_SEAL": "1",
                "RTX_GROUP_RM": "1",
                "RTX_SEAL_AUTO_FIX": "1",
                "RTX_SCHEDULE": str(schedule),
                "RTX_SKIP_GATE": "1",
                "RTX_CKPT_KEEP": "0",
                "RTX_RAY_TMP_DIR": str(base / "ray-tmp"),
                "RTX_RAY_SPILL_DIR": str(base / "ray-spill"),
                "NUM_GPUS": "4",
            })
            result = subprocess.run(
                ["bash", str(TRAIN_ENTRY)], env=env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            captured = ray_capture.read_text()
            self.assertIn("--seed 101", captured)
            self.assertIn("--group-rm", captured)
            self.assertIn('"RTX_BASELINE_MODE": "b6"', captured)
            self.assertIn(str(schedule), captured)

    def test_formal_meta_only_run_verifies_clean_custom_source_and_inputs(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            (base / "scripts").mkdir()
            for name in ("freeze_paper_inputs.py", "resource_gate.py"):
                shutil.copy2(ROOT / "scripts" / name, base / "scripts" / name)
            for name in ("slime", "areal", "transferqueue"):
                self._init_repo(base / "third_party" / name)
            (base / "models" / "model").mkdir(parents=True)
            (base / "models" / "model" / "weights.bin").write_bytes(b"model")
            (base / "models" / "model_torch_dist").mkdir()
            (base / "models" / "model_torch_dist" / "weights.bin").write_bytes(b"dist")
            (base / "models" / "datasets").mkdir()
            (base / "models" / "datasets" / "data.jsonl").write_text("{}\n")
            (base / "prereg").mkdir()
            (base / "prereg" / "schedule.json").write_text(
                json.dumps({"schedule_id": "formal-test", "events": []}) + "\n"
            )
            fake_bin = base / "bin"
            fake_bin.mkdir()
            image_id = "sha256:" + "a" * 64
            registry_digest = "sha256:" + "b" * 64
            docker = fake_bin / "docker"
            docker.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' '[{\"Id\":\"%s\",\"RepoDigests\":[\"slimerl/slime@%s\"]}]'\n"
                % ("%s", image_id, registry_digest)
            )
            docker.chmod(0o755)
            nvidia = fake_bin / "nvidia-smi"
            nvidia.write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  *--query-gpu=*) printf '%s\\n' '3, GPU-3, 81920, 0, 81920, 0' '4, GPU-4, 81920, 0, 81920, 0' ;;\n"
                "  *--query-compute-apps=*) : ;;\n"
                "esac\n"
            )
            nvidia.chmod(0o755)
            self._init_repo(base)
            env = {key: value for key, value in os.environ.items() if not key.startswith("RTX_")}
            env["PATH"] = str(fake_bin) + ":" + env.get("PATH", "")
            manifest = base / "paper-inputs.json"
            generated = subprocess.run([
                "python3", str(base / "scripts" / "freeze_paper_inputs.py"), "generate",
                "--out", str(manifest), "--image", "slimerl/slime:v0.3.1",
                "--asset", "model=models/model",
                "--asset", "torch_dist=models/model_torch_dist",
                "--asset", "data=models/datasets/data.jsonl",
                "--asset", "schedule=prereg/schedule.json",
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)

            env.update({
                "RTX_BASE": str(base), "RTX_PAPER_MODE": "1", "RTX_META_ONLY": "1",
                "RTX_GPUS": "device=3,4", "RTX_SEED": "29", "RTX_BASELINE_MODE": "b0",
                "RTX_SCHEDULE": "prereg/schedule.json", "RTX_FREEZE_MANIFEST": str(manifest),
                "RTX_MODEL_DIR": "/root/models/model", "RTX_DATA_PATH": "/root/datasets/data.jsonl",
                "RTX_EXP_ID": "formal-repro-test", "RTX_LOCAL_SCRATCH": str(base / "scratch"),
                "RTX_GATE_MIN_PUB_GB": "0", "RTX_GATE_MIN_ROOT_GB": "0",
                "RTX_GATE_MIN_GPU_FREE_GB": "1",
            })
            result = subprocess.run(
                ["bash", str(ENTRY), "none", "0", "3"], env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            run_dir = base / "runs" / "formal-repro-test"
            meta = json.loads((run_dir / "meta.json").read_text())
            verification = json.loads((run_dir / "logs" / "freeze_verify.json").read_text())
            self.assertEqual(verification["status"], "PASS")
            self.assertEqual(meta["source_repositories"]["slime"]["path"], "third_party/slime")
            self.assertTrue(meta["source_repositories"]["slime"]["clean"])
            self.assertEqual(meta["freeze_manifest"]["sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
