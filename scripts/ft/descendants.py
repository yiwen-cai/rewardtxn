"""Linux per-message sender authentication and launcher-owned descendant pidfds.

Only ordinary, trusted fixture processes are supported here. A role name is a
frozen protocol slot, not proof of an AReaL role or native recovery epoch.
"""
import array
import json
import os
from pathlib import Path
import select
import signal
import socket
import struct
import uuid

from .faults import ControlError, ChannelClosed, identity


MAX_MESSAGE = 65536
UCRED = struct.Struct("3i")


def snapshot(pid):
    value = identity(pid)
    value["pid_namespace"] = os.stat(f"/proc/{pid}/ns/pid").st_ino
    status = Path(f"/proc/{pid}/status").read_text().splitlines()
    value["nspid"] = next(line.split()[1:] for line in status if line.startswith("NSpid:"))
    return value


def exited(pidfd, timeout=0):
    poller = select.poll()
    poller.register(pidfd, select.POLLIN)
    events = poller.poll(max(0, int(timeout * 1000)))
    if any(flags & (select.POLLNVAL | select.POLLERR) for _, flags in events):
        raise ControlError("invalid/error pidfd readiness")
    return any(flags & (select.POLLIN | select.POLLHUP) for _, flags in events)


def validate_pidfd(fd, pid):
    if os.readlink(f"/proc/self/fd/{fd}") != "anon_inode:[pidfd]":
        raise ControlError("SCM_RIGHTS descriptor is not a pidfd")
    info = dict(line.split(":", 1) for line in Path(f"/proc/self/fdinfo/{fd}").read_text().splitlines() if ":" in line)
    if int(info.get("Pid", "-1").strip()) != pid or exited(fd):
        raise ControlError("pidfd does not identify the live kernel sender")
    signal.pidfd_send_signal(fd, 0)


def recv_packet(channel, expected_pid=None):
    """Return (object, kernel credential, received FDs); close all FDs on error.

    SO_PASSCRED must be enabled before the peer sends. MSG_CMSG_CLOEXEC prevents
    accidental leakage into another launcher exec. Truncation is never accepted.
    """
    fds = []
    try:
        data, ancillary, flags, _ = channel.recvmsg(
            MAX_MESSAGE, socket.CMSG_SPACE(UCRED.size) + socket.CMSG_SPACE(64 * array.array("i").itemsize),
            socket.MSG_CMSG_CLOEXEC)
        credentials = []
        unexpected = False
        for level, kind, payload in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                values = array.array("i")
                if len(payload) % values.itemsize:
                    unexpected = True
                values.frombytes(payload[:len(payload) - len(payload) % values.itemsize])
                fds.extend(values)
            elif level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS and len(payload) == UCRED.size:
                credentials.append(UCRED.unpack(payload))
            else:
                unexpected = True
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or unexpected:
            raise ControlError("truncated/unexpected control data")
        if not data:
            raise ChannelClosed("peer disconnected")
        if len(credentials) != 1:
            raise ControlError("one kernel credential required per message")
        pid, uid, gid = credentials[0]
        if pid <= 0 or uid != os.getuid() or (expected_pid is not None and pid != expected_pid):
            raise ControlError("kernel sender differs from registered process")
        message = json.loads(data)
        if not isinstance(message, dict) or message.get("pid") != pid:
            raise ControlError("claimed PID differs from kernel sender")
        return message, (pid, uid, gid), fds
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise


def send_packet(channel, message, fd=None):
    ancillary = [] if fd is None else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
    data = json.dumps(message, sort_keys=True).encode()
    if len(data) > MAX_MESSAGE:
        raise ControlError("control message too large")
    if channel.sendmsg([data], ancillary) != len(data):
        raise ControlError("short control send")


def lineage(pid, root):
    chain = []
    while len(chain) < 64:
        current = snapshot(pid)
        if current["pid_namespace"] != root["pid_namespace"] or current["cgroup"] != root["cgroup"]:
            raise ControlError("process outside launcher namespace/cgroup")
        chain.append(current)
        if pid == root["pid"]:
            if current != root:
                raise ControlError("launcher identity changed")
            for expected in reversed(chain):
                if snapshot(expected["pid"]) != expected:
                    raise ControlError("ancestry changed during validation")
            return chain
        pid = current["ppid"]
        if pid <= 1 or any(item["pid"] == pid for item in chain):
            break
    raise ControlError("sender is not a current launcher descendant")


class Descendant:
    def __init__(self, channel, message, credentials, fds, root, nonce, roles):
        self.pidfd = None
        try:
            pid, _, _ = credentials
            if (not {"kind", "pid", "nonce", "role"} <= set(message)
                    or not set(message) <= {"kind", "pid", "nonce", "role", "event_id"}
                    or ("event_id" in message and not isinstance(message["event_id"], str))
                    or message.get("kind") != "register" or message.get("nonce") != nonce
                    or message.get("role") not in roles or len(fds) != 1):
                raise ControlError("invalid descendant registration")
            validate_pidfd(fds[0], pid)
            chain = lineage(pid, root)
            validate_pidfd(fds[0], pid)
            if snapshot(pid) != chain[0]:
                raise ControlError("sender identity changed")
            self.channel, self.root, self.original = channel, root, chain[0]
            self.chain = chain
            self.role, self.nonce = message["role"], nonce
            self.requested_event = message.get("event_id")
            self.incarnation = uuid.uuid4().hex
            self.pidfd = fds.pop()
        finally:
            for fd in fds:
                os.close(fd)

    def receive(self):
        message, _, fds = recv_packet(self.channel, self.original["pid"])
        try:
            if fds or message.get("nonce") != self.nonce or message.get("incarnation") != self.incarnation:
                raise ControlError("invalid registered message or stale incarnation")
            return message
        finally:
            for fd in fds:
                os.close(fd)

    def verify(self):
        validate_pidfd(self.pidfd, self.original["pid"])
        if snapshot(self.original["pid"]) != self.original:
            raise ControlError("descendant identity changed")
        if lineage(self.original["pid"], self.root) != self.chain:
            raise ControlError("descendant ancestry changed")

    def kill(self):
        self.verify()
        signal.pidfd_send_signal(self.pidfd, signal.SIGKILL)

    def close(self):
        if self.pidfd is not None:
            os.close(self.pidfd)
            self.pidfd = None
        self.channel.close()


class Client:
    """CPU hook transport; each process must connect independently after fork."""
    def __init__(self, role, event_id=None):
        self.pid, self.nonce = os.getpid(), os.environ["FT_RUN_NONCE"]
        self.channel = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.channel.settimeout(float(os.environ["FT_HANDSHAKE_TIMEOUT"]))
        try:
            self.channel.connect(os.environ["FT_CONTROL_SOCKET"])
            fd = os.pidfd_open(self.pid)
            try:
                registration = dict(kind="register", pid=self.pid, nonce=self.nonce, role=role)
                if event_id is not None:
                    registration["event_id"] = event_id
                send_packet(self.channel, registration, fd)
            finally:
                os.close(fd)
            reply = self._reply()
            if reply.get("kind") != "registered":
                raise ControlError("registration rejected")
            self.incarnation = reply["incarnation"]
            self.injection = reply.get("injection")
            if event_id is not None and (not isinstance(self.injection, dict)
                    or self.injection.get("event_id") != event_id
                    or self.injection.get("status") not in ("pending", "already_fired")):
                raise ControlError("invalid injection assignment")
        except BaseException:
            self.channel.close()
            raise

    def _reply(self):
        data = self.channel.recv(MAX_MESSAGE)
        if not data:
            raise ChannelClosed("controller disconnected")
        value = json.loads(data)
        if not isinstance(value, dict) or value.get("nonce") != self.nonce:
            raise ControlError("invalid controller response")
        return value

    def ready(self, event_id, evidence):
        if os.getpid() != self.pid:
            raise ControlError("inherited client cannot send for its parent")
        if self.injection is not None and (self.injection["event_id"] != event_id or self.injection["status"] != "pending"):
            raise ControlError("this incarnation has no pending injection assignment")
        send_packet(self.channel, dict(kind="ready", pid=self.pid, nonce=self.nonce,
                    incarnation=self.incarnation, event_id=event_id, evidence=evidence))

    def wait_release(self, event_id):
        reply = self._reply()
        if reply.get("kind") != "release" or reply.get("event_id") != event_id or reply.get("incarnation") != self.incarnation:
            raise ControlError("barrier aborted or invalid release")
        send_packet(self.channel, dict(kind="released", pid=self.pid, nonce=self.nonce,
                    incarnation=self.incarnation, event_id=event_id))

    def close(self):
        self.channel.close()
