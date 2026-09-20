import array
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.ft.descendants import (Descendant, exited, recv_packet, send_packet,
                                    snapshot)
from scripts.ft.faults import ControlError
from scripts.ft.namespace_run import check_namespace, control, read_config

PIDFD = hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")


@unittest.skipUnless(PIDFD, "requires Linux Python with pidfd")
class DescendantTests(unittest.TestCase):
    def setUp(self):
        self.server, self.client = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        self.server.settimeout(3)
        self.message = dict(kind="register", pid=os.getpid(), nonce="test", role="worker")

    def tearDown(self):
        self.server.close(); self.client.close()

    def registration(self, fd, root=None, message=None):
        send_packet(self.client, message or self.message, fd)
        body, cred, rights = recv_packet(self.server)
        return Descendant(self.server, body, cred, rights, root or snapshot(os.getpid()), "test", {"worker"})

    def test_valid_self_pidfd_transport_and_per_message_identity(self):
        fd = os.pidfd_open(os.getpid())
        try:
            entry = self.registration(fd)
            try:
                self.assertEqual(entry.original["pid"], os.getpid())
                self.assertFalse(exited(entry.pidfd))
                send_packet(self.client, dict(pid=os.getpid(), nonce="test", incarnation="stale"))
                with self.assertRaises(ControlError):
                    entry.receive()
            finally:
                entry.close()
        finally:
            os.close(fd)

    def test_invalid_descriptor_missing_and_extra_rights_are_closed(self):
        for count in (0, 1, 2, 80):
            with self.subTest(count=count):
                fd = os.open("/dev/null", os.O_RDONLY)
                before = len(list(Path("/proc/self/fd").iterdir()))
                try:
                    rights = [] if not count else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd] * count))]
                    self.client.sendmsg([json.dumps(self.message).encode()], rights)
                    with self.assertRaises(ControlError):
                        body, cred, received = recv_packet(self.server)
                        Descendant(self.server, body, cred, received, snapshot(os.getpid()), "test", {"worker"})
                    self.assertEqual(len(list(Path("/proc/self/fd").iterdir())), before)
                finally:
                    os.close(fd)

    def test_external_launcher_root_and_wrong_pidfd_are_rejected(self):
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        ownfd, otherfd = os.pidfd_open(os.getpid()), os.pidfd_open(other.pid)
        try:
            with self.assertRaises(ControlError):
                self.registration(ownfd, root=snapshot(other.pid))
            with self.assertRaises(ControlError):
                self.registration(otherfd)
            self.assertIsNone(other.poll())
        finally:
            os.close(ownfd); os.close(otherfd)
            other.kill(); other.wait(timeout=3)

    def test_forked_sender_cannot_borrow_registered_connection(self):
        child = os.fork()
        if child == 0:
            try:
                send_packet(self.client, self.message)  # Claimed parent PID, actual child credentials.
            finally:
                os._exit(0)
        try:
            with self.assertRaises(ControlError):
                recv_packet(self.server, expected_pid=os.getpid())
        finally:
            os.waitpid(child, 0)

    def test_nonchild_kill_readiness_does_not_reap_real_parents_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            listener.bind(str(Path(tmp) / "control.sock")); listener.listen(1); listener.settimeout(5)
            worker = "from scripts.ft.descendants import Client; import time; c=Client('worker'); time.sleep(30)"
            launcher_code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',sys.argv[1]],start_new_session=True); print(p.wait(),flush=True)"
            env = dict(os.environ, FT_RUN_NONCE="test", FT_CONTROL_SOCKET=str(Path(tmp) / "control.sock"), FT_HANDSHAKE_TIMEOUT="5")
            launcher = subprocess.Popen([sys.executable, "-c", launcher_code, worker], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            entry = None
            try:
                channel, _ = listener.accept()
                body, cred, rights = recv_packet(channel)
                entry = Descendant(channel, body, cred, rights, snapshot(launcher.pid), "test", {"worker"})
                self.assertEqual(entry.original["ppid"], launcher.pid)
                send_packet(channel, dict(kind="registered", nonce="test", incarnation=entry.incarnation))
                original = entry.original
                entry.original = dict(original, start_time="wrong")
                with self.assertRaises(ControlError):
                    entry.kill()
                self.assertFalse(exited(entry.pidfd))
                entry.original = original
                with patch("os.waitpid", side_effect=AssertionError("must not reap descendant")):
                    entry.kill()
                    self.assertTrue(exited(entry.pidfd, 3))
                stdout, stderr = launcher.communicate(timeout=5)
                self.assertEqual(launcher.returncode, 0, stderr)
                self.assertEqual(stdout.strip(), b"-9")
                with self.assertRaises(ControlError):
                    entry.kill()  # Stale pidfd must not signal any subsequent process.
            finally:
                if entry:
                    entry.close()
                listener.close()
                if launcher.poll() is None:
                    launcher.kill(); launcher.wait(timeout=3)

    def test_namespace_requires_real_pid_one(self):
        with self.assertRaises(ControlError):
            check_namespace(os.stat("/proc/self/ns/pid").st_ino)
        if os.getpid() != 1:
            with self.assertRaises(ControlError):
                check_namespace(os.stat("/proc/self/ns/pid").st_ino + 1)


class NamespaceConfigTests(unittest.TestCase):
    def test_lease_covers_normal_handshake_and_secret_env_rejected(self):
        config = {"argv": ["/usr/bin/python3", "-c", "pass"], "env": {}, "schedule": [],
                  "timeouts": {"run": 5, "handshake": 2, "lease": 2}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                read_config(path)
            config["timeouts"]["lease"] = 3
            path.write_text(json.dumps(config))
            self.assertEqual(read_config(path)[0], config)
            config["env"]["API_TOKEN"] = "not-allowed"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                read_config(path)

    def test_existing_uncertain_signal_evidence_cannot_be_replayed(self):
        for last_kind in ("signal_intent", "signal_sent"):
            with self.subTest(last_kind=last_kind), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                evidence = json.dumps({"kind": last_kind, "event_id": "one"}) + "\n"
                (output / "events.jsonl").write_text(evidence)
                with patch("scripts.ft.namespace_run.check_namespace", return_value={}), patch(
                        "scripts.ft.namespace_run.subprocess.Popen", side_effect=AssertionError("must not relaunch")):
                    with self.assertRaises(ControlError):
                        control({}, "hash", output, "nonce", 1)
                self.assertEqual((output / "events.jsonl").read_text(), evidence)
