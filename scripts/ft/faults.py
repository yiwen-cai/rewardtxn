"""Linux direct-child identity and bounded semantic-barrier transport.

The adapter supplies real semantic evidence; this module verifies its frozen
contract, not the experiment's oracle or the truth of application evidence.
"""
import hashlib
import json
import os
import signal
import socket
from dataclasses import dataclass
from pathlib import Path


class ControlError(RuntimeError):
    pass


class ChannelClosed(ControlError):
    pass


def event_nonce(run_nonce, event_id, attempt):
    return hashlib.sha256(json.dumps([run_nonce, event_id, attempt]).encode()).hexdigest()


def identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "ppid": int(fields[1]), "pgid": int(fields[2]),
            "start_time": fields[19], "cgroup": Path(f"/proc/{pid}/cgroup").read_text().strip(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


@dataclass
class OwnedProcess:
    process: object
    original: dict
    pidfd: int

    @classmethod
    def register(cls, process, pidfd=None):
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise ControlError("pidfd unavailable: injection fails closed")
        original = identity(process.pid)
        if original["ppid"] != os.getpid() or original["pgid"] != process.pid:
            raise ControlError("only directly launched session leaders are supported")
        fd = os.pidfd_open(process.pid) if pidfd is None else pidfd
        if identity(process.pid) != original:
            if pidfd is None:
                os.close(fd)
            raise ControlError("identity changed during registration")
        return cls(process, original, fd)

    def send(self, sig):
        if self.process.poll() is not None:
            return False
        if identity(self.process.pid) != self.original:
            raise ControlError("process identity mismatch")
        signal.pidfd_send_signal(self.pidfd, sig)
        return True

    def close(self):
        os.close(self.pidfd)


def send(channel, message):
    channel.send(json.dumps(message, sort_keys=True).encode())


def receive(channel):
    data = channel.recv(65536)
    if not data:
        raise ChannelClosed("controller/worker disconnected")
    try:
        message = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise ControlError("malformed control message") from exc
    if not isinstance(message, dict):
        raise ControlError("control message must be an object")
    return message


def barrier(event_id, evidence):
    """Called synchronously at an adapter's real boundary; disconnect aborts.

    Only the direct launcher process can use this endpoint. Each notification
    binds event ID, attempt, run nonce, role, PID, and deterministic event nonce.
    """
    attempt = int(os.environ["FT_ATTEMPT"])
    nonce = os.environ["FT_RUN_NONCE"]
    message = {"event_id": event_id, "attempt": attempt, "run_nonce": nonce,
               "event_nonce": event_nonce(nonce, event_id, attempt),
               "role": os.environ["FT_ROLE"], "pid": os.getpid()}
    channel = socket.socket(fileno=os.dup(int(os.environ["FT_CONTROL_FD"])))
    channel.settimeout(float(os.environ["FT_HANDSHAKE_TIMEOUT"]))
    try:
        send(channel, dict(message, kind="ready", evidence=evidence))
        while True:
            response = receive(channel)
            if any(response.get(key) != value for key, value in message.items()):
                raise ControlError("barrier identity mismatch")
            if response.get("kind") == "release":
                send(channel, dict(message, kind="released"))
                return
            if response.get("kind") == "abort":
                send(channel, dict(message, kind="aborted"))
                raise ControlError(response.get("reason", "controller aborted"))
            raise ControlError("unexpected barrier response")
    except (OSError, ValueError) as exc:
        raise ControlError(f"barrier watchdog: {exc}") from exc
    finally:
        channel.close()
