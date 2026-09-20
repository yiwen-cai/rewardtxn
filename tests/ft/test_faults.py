import os
import signal
import socket
import subprocess
import sys
import unittest

from scripts.ft.faults import ControlError, OwnedProcess, identity

PIDFD = hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")


class FaultTests(unittest.TestCase):
    @unittest.skipUnless(PIDFD, "requires Linux pidfd Python")
    def test_precise_identity_and_external_process_rejection(self):
        external = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
        owned = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"], start_new_session=True)
        registration = None
        try:
            with self.assertRaises(ControlError):
                OwnedProcess.register(external)  # Not a controller-created session leader.
            registration = OwnedProcess.register(owned)
            registration.original["start_time"] = "wrong"
            with self.assertRaises(ControlError):
                registration.send(signal.SIGKILL)
            self.assertIsNone(external.poll())
            self.assertIsNone(owned.poll())
            registration.original = identity(owned.pid)
            registration.send(signal.SIGKILL)
            self.assertEqual(owned.wait(timeout=2), -signal.SIGKILL)
            self.assertIsNone(external.poll())
        finally:
            if registration:
                registration.close()
            # Test owns both Popen handles; this is fixture cleanup, not runner fallback.
            for p in (external, owned):
                if p.poll() is None:
                    p.kill()
                p.wait(timeout=2)

    def test_disconnected_controller_and_timeout_abort_worker(self):
        for disconnect in (False, True):
            with self.subTest(disconnect=disconnect):
                parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                env = dict(os.environ, FT_CONTROL_FD=str(child.fileno()), FT_ATTEMPT="0",
                           FT_RUN_NONCE="test", FT_ROLE="waiter", FT_HANDSHAKE_TIMEOUT="0.1")
                p = subprocess.Popen([sys.executable, "-c", "from scripts.ft.faults import barrier; barrier('one', {'boundary':'fixture'})"],
                                     pass_fds=(child.fileno(),), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                child.close()
                if disconnect:
                    parent.close()
                _, error = p.communicate(timeout=3)
                parent.close()
                self.assertNotEqual(p.returncode, 0)
                self.assertIn(b"ControlError", error)

    def test_malformed_message_is_not_normal_disconnect(self):
        from scripts.ft.faults import ChannelClosed, receive
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            child.send(b"[]")
            with self.assertRaises(ControlError) as raised:
                receive(parent)
            self.assertNotIsInstance(raised.exception, ChannelClosed)
            child.close()
            with self.assertRaises(ChannelClosed):
                receive(parent)
        finally:
            parent.close(); child.close()
