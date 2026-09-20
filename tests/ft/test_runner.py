import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest

from scripts.ft.faults import ControlError
from scripts.ft.run import Journal, execute, preflight

PIDFD = hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")
ROOT = str(Path(__file__).resolve().parents[2])
WORKER = """
import os, socket
from scripts.ft.faults import barrier, event_nonce, send, receive
key = 'event-' + os.environ['FT_ATTEMPT']
barrier(key, {'boundary': 'cpu_fixture'})
"""
DUPLICATE = """
import os, socket
from scripts.ft.faults import event_nonce, send, receive
key = 'event-' + os.environ['FT_ATTEMPT']; attempt = int(os.environ['FT_ATTEMPT'])
nonce = os.environ['FT_RUN_NONCE']
s = socket.socket(fileno=int(os.environ['FT_CONTROL_FD'])); s.settimeout(2)
m = dict(event_id=key, attempt=attempt, run_nonce=nonce, event_nonce=event_nonce(nonce,key,attempt),role=os.environ['FT_ROLE'],pid=os.getpid())
send(s,dict(m,kind='ready',evidence={'boundary':'cpu_fixture'}))
send(s,dict(m,kind='ready',evidence={'boundary':'cpu_fixture'}))
r = receive(s)
assert r['kind'] == 'release', r
send(s,dict(m,kind='released'))
send(s,dict(m,kind='released'))
"""


class RunnerTests(unittest.TestCase):
    def fixture(self, path, attempts=2, worker=WORKER):
        config = {"attempts": [{"target": [sys.executable, "-c", WORKER],
                                  "waiter": [sys.executable, "-c", worker]} for _ in range(attempts)],
                  "env": {"PYTHONPATH": ROOT}, "handshake_timeout": 2,
                  "run_timeout": 8, "cleanup_timeout": 2}
        schedule = [{"event_id": f"event-{i}", "attempt": i, "target": "target",
                     "waiters": ["target", "waiter"], "evidence": {"boundary": "cpu_fixture"}}
                    for i in range(attempts)]
        cp, sp = path / "config.json", path / "schedule.json"
        cp.write_text(json.dumps(config)); sp.write_text(json.dumps(schedule))
        return preflight(cp, sp)

    def test_readonly_freeze_and_no_secret_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = self.fixture(root)
            self.assertEqual(set(p.name for p in root.iterdir()), {"config.json", "schedule.json"})
            self.assertEqual(len(frozen["config_sha256"]), 64)
            cp = root / "config.json"
            config = json.loads(cp.read_text()); config["env"]["API_TOKEN"] = "secret"
            cp.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                preflight(cp, root / "schedule.json")

    @unittest.skipUnless(PIDFD, "requires Linux pidfd Python")
    def test_two_scheduled_attempts_release_and_duplicate_notifications(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); frozen = self.fixture(root, worker=DUPLICATE)
            result = execute(frozen, root / "run")
            self.assertEqual(result["classification"], "execution_complete", result)
            records = Journal(root / "run").records
            self.assertEqual([r["attempt"] for r in records if r["kind"] == "observed"], [0, 1])
            self.assertEqual(len([r for r in records if r["kind"] == "fired"]), 2)
            self.assertEqual(len([r for r in records if r["kind"] == "release_receipt"]), 2)
            self.assertEqual(result["oracle_status"], "not_evaluated")
            with self.assertRaises(ControlError):
                execute(frozen, root / "run", resume=True)

    @unittest.skipUnless(PIDFD, "requires Linux pidfd Python")
    def test_bad_pid_and_failed_run_preserved_not_retried(self):
        bad = DUPLICATE.replace('pid=os.getpid()', 'pid=1')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); frozen = self.fixture(root, attempts=1, worker=bad)
            result = execute(frozen, root / "run")
            self.assertEqual(result["classification"], "technical_invalid")
            before = (root / "run" / "events.jsonl").read_bytes()
            with self.assertRaises(ControlError):
                execute(frozen, root / "run", resume=True)
            self.assertEqual(before, (root / "run" / "events.jsonl").read_bytes())
            self.assertFalse(any(r["kind"] == "fired" for r in Journal(root / "run").records))

    @unittest.skipUnless(PIDFD, "requires Linux pidfd Python")
    def test_uncertain_fired_never_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); frozen = self.fixture(root)
            directory = root / "run"; directory.mkdir()
            (directory / "frozen.json").write_text(json.dumps(dict(frozen, run_nonce="retained")))
            journal = Journal(directory)
            journal.add("attempt_start", attempt=0)
            journal.add("fired", attempt=0, event_id="event-0")
            result = execute(frozen, directory, resume=True)
            self.assertEqual(result["classification"], "technical_invalid")
            self.assertEqual(len([r for r in Journal(directory).records if r["kind"] == "fired"]), 1)
            self.assertEqual(json.loads((directory / "frozen.json").read_text())["schedule"], frozen["schedule"])

    @unittest.skipUnless(PIDFD, "requires Linux pidfd Python")
    def test_resume_closed_attempt_keeps_later_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); frozen = self.fixture(root)
            directory = root / "run"; directory.mkdir()
            (directory / "frozen.json").write_text(json.dumps(dict(frozen, run_nonce="retained")))
            journal = Journal(directory)
            journal.add("attempt_start", attempt=0)
            journal.add("attempt_end", attempt=0)
            result = execute(frozen, directory, resume=True)
            self.assertEqual(result["classification"], "execution_complete", result)
            records = Journal(directory).records
            self.assertEqual([r["attempt"] for r in records if r["kind"] == "process_registered"], [1, 1])
            self.assertEqual([r["attempt"] for r in records if r["kind"] == "observed"], [1])

    @unittest.skipUnless(PIDFD, "requires Linux pidfd Python")
    def test_missing_release_receipt_is_technical_not_method_timeout(self):
        no_receipt = DUPLICATE.replace("send(s,dict(m,kind='released'))", "import time; time.sleep(3)")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); frozen = self.fixture(root, attempts=1, worker=no_receipt)
            frozen["config"]["handshake_timeout"] = 0.2
            frozen["config"]["cleanup_timeout"] = 0.2
            result = execute(frozen, root / "run")
            self.assertEqual(result["classification"], "technical_invalid", result)
            records = Journal(root / "run").records
            self.assertEqual(len([r for r in records if r["kind"] == "observed"]), 1)
            self.assertFalse(any(r.get("outcome") == "timeout" for r in records))

    @unittest.skipUnless(PIDFD, "requires Linux pidfd Python")
    def test_registration_failure_reaps_launched_child(self):
        from unittest.mock import patch
        import subprocess
        launched = []
        popen = subprocess.Popen
        def capture(*args, **kwargs):
            child = popen(*args, **kwargs)
            launched.append(child)
            return child
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); frozen = self.fixture(root, attempts=1)
            with patch("scripts.ft.run.subprocess.Popen", side_effect=capture), patch(
                    "scripts.ft.run.OwnedProcess.register", side_effect=ControlError("registration rejected")):
                result = execute(frozen, root / "run")
            self.assertEqual(result["classification"], "technical_invalid", result)
            self.assertEqual(len(launched), 1)
            self.assertIsNotNone(launched[0].poll())

    @unittest.skipUnless(PIDFD, "requires Linux pidfd Python")
    def test_cleanup_error_does_not_skip_other_children(self):
        from unittest.mock import patch
        from scripts.ft.faults import OwnedProcess
        import subprocess
        registered = []
        register = OwnedProcess.register
        def capture(process, pidfd=None):
            entry = register(process, pidfd=pidfd)
            registered.append(entry)
            if len(registered) == 1:
                real_wait = process.wait
                calls = [0]
                def failing_wait(*args, **kwargs):
                    # Real child is killed/reaped; emulate a wait bookkeeping
                    # timeout to check classification and remaining cleanup.
                    value = real_wait(*args, **kwargs)
                    calls[0] += 1
                    if calls[0] == 1:
                        raise subprocess.TimeoutExpired(process.args, kwargs.get("timeout"))
                    return value
                process.wait = failing_wait
            return entry
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); frozen = self.fixture(root, attempts=1)
            frozen["config"]["attempts"][0] = {role: [sys.executable, "-c", "import time; time.sleep(5)"] for role in ("target", "waiter")}
            frozen["config"]["handshake_timeout"] = 0.1
            with patch("scripts.ft.run.OwnedProcess.register", side_effect=capture):
                result = execute(frozen, root / "run")
            self.assertEqual(result["classification"], "technical_invalid", result)
            self.assertEqual(len(registered), 2)
            self.assertTrue(all(entry.process.poll() is not None for entry in registered))
            self.assertTrue(any(r["kind"] == "cleanup_failure" for r in Journal(root / "run").records))
