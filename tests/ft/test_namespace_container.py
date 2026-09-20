"""Opt-in, CPU-only real Docker tests. No Docker is launched without image opt-in."""
import json
import contextlib
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
import uuid
from unittest.mock import patch

from scripts.ft import container_run

IMAGE = os.environ.get("FT_NAMESPACE_DOCKER_IMAGE")
ROOT = Path(__file__).resolve().parents[2]
FIXTURE = "/workspace/tests/ft/namespace_fixture.py"


@unittest.skipUnless(IMAGE, "set FT_NAMESPACE_DOCKER_IMAGE to explicitly opt into CPU Docker tests")
class NamespaceContainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.python = os.environ.get("FT_NAMESPACE_DOCKER_PYTHON", "python")
        evidence = os.environ.get("FT_NAMESPACE_EVIDENCE_DIR")
        cls.sentinel_temporary = None
        if evidence:
            root = Path(evidence); root.mkdir(parents=True, exist_ok=True)
            cls.sentinel_output = Path(tempfile.mkdtemp(prefix="sentinel-", dir=root))
        else:
            cls.sentinel_temporary = tempfile.TemporaryDirectory()
            cls.sentinel_output = Path(cls.sentinel_temporary.name)
        cls.sentinel_nonce = uuid.uuid4().hex
        create = ["create", "--label", f"rewardtxn.ft.nonce={cls.sentinel_nonce}", "--network=none", "--user", f"{os.getuid()}:{os.getgid()}",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--entrypoint", cls.python,
            IMAGE, "-c", "import time; time.sleep(180)"]
        container_run.atomic_json(cls.sentinel_output / "launch.json", {"argv": ["docker", *create]})
        cls.sentinel = container_run.docker(create)
        (cls.sentinel_output / "container.id").write_text(cls.sentinel + "\n")
        container_run.docker(["start", cls.sentinel])
        container_run.atomic_json(cls.sentinel_output / "inspect_started.json", container_run.inspect_owned(cls.sentinel, cls.sentinel_nonce))

    @classmethod
    def tearDownClass(cls):
        error = None
        try:
            info = container_run.inspect_owned(cls.sentinel, cls.sentinel_nonce)
            container_run.atomic_json(cls.sentinel_output / "inspect_before_cleanup.json", info)
            if info["State"]["Running"]:
                container_run.docker(["kill", cls.sentinel])
            info = container_run.inspect_owned(cls.sentinel, cls.sentinel_nonce)
            container_run.atomic_json(cls.sentinel_output / "inspect_final.json", info)
            if info["State"]["Running"]:
                raise RuntimeError("sentinel cleanup not confirmed")
            container_run.docker(["rm", cls.sentinel])
        except BaseException as exc:
            error = str(exc)
            raise
        finally:
            container_run.atomic_json(cls.sentinel_output / "cleanup.json", {"container_id": cls.sentinel, "error": error})
            if cls.sentinel_temporary:
                cls.sentinel_temporary.cleanup()

    def case_directory(self):
        artifacts = os.environ.get("FT_NAMESPACE_EVIDENCE_DIR")
        if artifacts:
            root = Path(artifacts); root.mkdir(parents=True, exist_ok=True)
            return contextlib.nullcontext(tempfile.mkdtemp(prefix="namespace-", dir=root))
        return tempfile.TemporaryDirectory()

    def config(self, directory, events=2, dormant=False):
        # sys.executable inside the image may differ from the host interpreter.
        # /usr/bin/env resolves the image Python from an explicitly frozen PATH.
        config = {"argv": ["/usr/bin/env", self.python, FIXTURE, "dormant" if dormant else "launcher", "--count", str(events)],
                  "env": {"PYTHONPATH": "/workspace", "PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"},
                  "timeouts": {"run": 20, "handshake": 2, "lease": 4},
                  "schedule": [] if dormant else [{"event_id": f"event-{i}", "target": "target", "waiters": ["target", "waiter"], "evidence": {"boundary": "cpu_fixture"}} for i in range(events)]}
        path = directory / "input.json"; path.write_text(json.dumps(config))
        return path

    def assert_sentinel(self):
        info = json.loads(container_run.docker(["inspect", self.sentinel]))[0]
        self.assertTrue(info["State"]["Running"])

    def run_case(self, directory, fault=None, intervention=None):
        path = self.config(directory, dormant=fault == "pidfd_failure" or intervention is not None)
        output = directory / "run"
        real_docker, real_json = container_run.docker, container_run.atomic_json
        def altered_docker(argv, timeout=10):
            argv = list(argv)
            if fault and argv[0] == "create":
                index = argv.index("scripts.ft.namespace_run")
                argv[index-1:index+1] = [FIXTURE, f"controller_{fault}"]
            return real_docker(argv, timeout)
        def altered_json(path, value):
            if intervention == "lost_host" and Path(path).name == "host_lease.json" and value["counter"] > 0:
                return
            real_json(path, value)
        errors = []
        def interrupt():
            try:
                deadline = time.monotonic() + 20
                while not (output / "controller_heartbeat.json").exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("fixture controller never started")
                    time.sleep(0.02)
                cid = (output / "container.id").read_text().strip()
                nonce = json.loads((output / "launch.json").read_text())["nonce"]
                container_run.inspect_owned(cid, nonce)
                real_docker(["kill", "--signal", "STOP" if intervention == "hang" else "KILL", cid])
            except BaseException as exc:
                errors.append(exc)
        thread = None
        if intervention in ("hang", "crash"):
            thread = threading.Thread(target=interrupt)
            thread.start()
        try:
            with patch.object(container_run, "docker", side_effect=altered_docker), patch.object(container_run, "atomic_json", side_effect=altered_json):
                result = container_run.supervise(path, ROOT, output, IMAGE, self.python, cleanup_timeout=1)
        finally:
            if thread:
                thread.join(timeout=25)
                self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        self.assertTrue(result["cleanup_confirmed"], result)
        self.assert_sentinel()
        final = json.loads((output / "inspect_final.json").read_text())
        self.assertFalse(final["State"]["Running"])
        return output, result

    def test_two_events_one_launcher_nonchild_exit_and_parent_receipts(self):
        with self.case_directory() as tmp:
            output, result = self.run_case(Path(tmp))
            self.assertIsNone(result["failure"], result)
            records = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual(sum(r["kind"] == "launcher_created" for r in records), 1)
            self.assertEqual([r["event_id"] for r in records if r["kind"] == "signal_sent"], ["event-0", "event-1"])
            self.assertEqual(sum(r["kind"] == "process_exit_observed" for r in records), 2)
            self.assertEqual(sum(r["kind"] == "release_receipt" for r in records), 2)
            for index in range(2):
                self.assertEqual(json.loads((output / f"parent_wait-{index}.json").read_text()), [-9, 0])
            self.assertEqual(records[-1]["classification"], "execution_complete")

    def test_first_launcher_pidfd_failure_kills_unregistered_subtree(self):
        with self.case_directory() as tmp:
            output, result = self.run_case(Path(tmp), fault="pidfd_failure")
            self.assertIsNotNone(result["failure"])
            self.assertTrue((output / "dormant.json").exists())
            records = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual(sum(r["kind"] == "launcher_created" for r in records), 1)
            self.assertFalse(any(r["kind"] == "launcher_registered" for r in records))
            self.assertEqual(records[-1]["classification"], "technical_invalid")

    def test_registration_failure_closes_namespace(self):
        with self.case_directory() as tmp:
            _, result = self.run_case(Path(tmp), fault="registration_failure")
            self.assertIsNotNone(result["failure"])

    def test_controller_crash_hang_and_lost_host_cleanup(self):
        for intervention in ("crash", "hang", "lost_host"):
            with self.subTest(intervention=intervention), self.case_directory() as tmp:
                _, result = self.run_case(Path(tmp), intervention=intervention)
                self.assertIsNotNone(result["failure"], result)
                self.assertLess(result["duration_seconds"], 35)
