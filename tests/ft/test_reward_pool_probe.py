"""Protocol tests plus explicitly opted-in, unmodified official CPU library runs."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest

from scripts.ft.container_run import supervise
from scripts.ft.descendants import Client
from scripts.ft.faults import ControlError
from scripts.ft.namespace_run import injection_assignment, read_config
from scripts.ft.reward_pool_probe import EVENT_ID, EVIDENCE

ROOT = Path(__file__).resolve().parents[2]
IMAGE = os.environ.get("FT_REWARD_POOL_DOCKER_IMAGE")


class InjectionAssignmentTests(unittest.TestCase):
    def test_replacement_receives_consumed_status_without_recovery_data(self):
        events = {"one": {"waiters": ["reward"]}}
        states = {"one": {"phase": "waiting", "allocated": {}}}
        first = SimpleNamespace(requested_event="one", role="reward", incarnation="first")
        replacement = SimpleNamespace(requested_event="one", role="reward", incarnation="replacement")
        assignment = injection_assignment(first, events, states, "nonce")
        self.assertEqual(assignment["status"], "pending")
        self.assertEqual(set(assignment), {"event_id", "status", "event_nonce"})
        with self.assertRaises(ControlError):
            injection_assignment(replacement, events, states, "nonce")
        states["one"]["phase"] = "released"
        second = injection_assignment(replacement, events, states, "nonce")
        self.assertEqual(second["status"], "already_fired")
        self.assertEqual(second["event_nonce"], assignment["event_nonce"])
        states["one"]["phase"] = "uncertain"
        with self.assertRaises(ControlError):
            injection_assignment(replacement, events, states, "nonce")

    def test_observer_role_has_no_injection_authority(self):
        observer = SimpleNamespace(requested_event=None, role="observer", incarnation="one")
        self.assertIsNone(injection_assignment(observer, {}, {}, "nonce"))
        observer.requested_event = "not-scheduled"
        with self.assertRaises(ControlError):
            injection_assignment(observer, {}, {}, "nonce")
        observer.requested_event = "one"
        with self.assertRaises(ControlError):
            injection_assignment(observer, {"one": {"waiters": ["reward"]}}, {}, "nonce")

    def test_already_fired_client_never_sends_another_ready(self):
        client = Client.__new__(Client)
        client.pid = os.getpid()
        client.injection = {"event_id": "one", "status": "already_fired"}
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client.channel = child
        parent.settimeout(0.02)
        try:
            with self.assertRaises(ControlError):
                client.ready("one", {})
            with self.assertRaises(TimeoutError):
                parent.recv(1024)
        finally:
            parent.close(); child.close()

    def test_namespace_only_identity_env_and_observer_role_schema(self):
        config = {"argv": ["/usr/bin/python3"], "env": {"USER": "test", "LOGNAME": "test"},
                  "observer_roles": ["reward"], "schedule": [],
                  "timeouts": {"run": 10, "handshake": 2, "lease": 4}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"; path.write_text(json.dumps(config))
            self.assertEqual(read_config(path)[0], config)
            config["observer_roles"] = ["reward", "reward"]
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                read_config(path)


@unittest.skipUnless(IMAGE, "set FT_REWARD_POOL_DOCKER_IMAGE to opt into real official CPU pool probes")
class RealRewardPoolTests(unittest.TestCase):
    def case_directory(self):
        evidence = os.environ.get("FT_REWARD_POOL_EVIDENCE_DIR")
        if evidence:
            root = Path(evidence); root.mkdir(parents=True, exist_ok=True)
            return contextlib.nullcontext(tempfile.mkdtemp(prefix="reward-pool-", dir=root))
        return tempfile.TemporaryDirectory()

    def run_probe(self, directory, fault):
        python = os.environ.get("FT_NAMESPACE_DOCKER_PYTHON", "/opt/.venv/bin/python")
        argv = [python, "-m", "scripts.ft.reward_pool_probe", "--max-retries", "1"]
        if fault:
            argv.append("--fault")
        config = {"argv": argv, "observer_roles": ["reward"],
                  "env": {"PYTHONPATH": "/workspace:/workspace/third_party/areal", "PATH": "/opt/.venv/bin:/usr/local/bin:/usr/bin:/bin",
                          "USER": "ft_cpu_probe", "LOGNAME": "ft_cpu_probe", "OMP_NUM_THREADS": "1"},
                  "timeouts": {"run": 120, "handshake": 5, "lease": 10},
                  "schedule": [{"event_id": EVENT_ID, "target": "reward", "waiters": ["reward"], "evidence": EVIDENCE}] if fault else []}
        path = directory / "input.json"; path.write_text(json.dumps(config))
        output = directory / "run"
        supervisor = supervise(path, ROOT, output, IMAGE, python)
        self.assertTrue(supervisor["cleanup_confirmed"], supervisor)
        self.assertFalse(supervisor["cleanup_errors"], supervisor)
        log = (output / "launcher.log").read_text()
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        diagnosis = {"supervisor": supervisor, "launcher_log_tail": log[-4000:],
                     "method_observations": [e for e in events if e["kind"] == "method_observation"]}
        self.assertIsNone(supervisor["failure"], diagnosis)
        self.assertTrue(any(e["kind"] == "method_observation" and e.get("launcher_exit_code") == 0 for e in events), diagnosis)
        result = json.loads((output / "probe_result.json").read_text())
        self.assertEqual(result["status"], "returned")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["max_workers"], 1)
        self.assertEqual(result["max_retries"], 1)
        for path, digest in result["source_sha256"].items():
            relative = Path(path).relative_to("/workspace")
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest)
        self.assertEqual(sum(e["kind"] == "launcher_created" for e in events), 1)
        workers = [e for e in events if e["kind"] == "descendant_registered"]
        self.assertEqual(len(workers), 2 if fault else 1)
        self.assertEqual(len({w["identity"]["pid"] for w in workers}), len(workers))
        for worker in workers:
            self.assertEqual(worker["identity"]["ppid"], result["trainer_identity"]["pid"])
            self.assertEqual(worker["ancestry"][-1], result["trainer_identity"])
        calls = [[json.loads(line) for line in file.read_text().splitlines()] for file in sorted(output.glob("reward-*.jsonl"))]
        self.assertEqual(len(calls), 2 if fault else 1)
        self.assertEqual(sum(any(e["kind"] == "official_score_returned" and e["score"] == 1.0 for e in call) for call in calls), 1)
        sent = [e for e in events if e["kind"] == "signal_sent"]
        observed = [e for e in events if e["kind"] == "process_exit_observed"]
        if fault:
            self.assertEqual(len(sent), 1)
            self.assertEqual(len(observed), 1)
            self.assertEqual(sent[0]["identity"], observed[0]["identity"])
            assignments = [e["injection"]["status"] for e in events if e["kind"] == "injection_assignment"]
            self.assertEqual(assignments, ["pending", "already_fired"])
            self.assertIn("ProcessPoolExecutor broken (attempt 1/2)", log)
            self.assertIn("Recreated ProcessPoolExecutor with 1 workers", log)
            killed_call = next(call for call in calls if call[0]["identity"]["pid"] == sent[0]["identity"]["pid"])
            self.assertFalse(any(e["kind"] == "official_score_start" for e in killed_call))
        else:
            self.assertFalse(sent)
            self.assertFalse(observed)
            self.assertNotIn("ProcessPoolExecutor broken", log)
        return result

    def test_nofault_and_native_pool_replacement_use_identical_scoring(self):
        results = []
        for fault in (False, True):
            with self.subTest(fault=fault), self.case_directory() as tmp:
                results.append(self.run_probe(Path(tmp), fault))
        if len(results) == 2:
            self.assertEqual(results[0]["input"], results[1]["input"])
            self.assertEqual(results[0]["score"], results[1]["score"])
            self.assertEqual(results[0]["source_sha256"], results[1]["source_sha256"])
