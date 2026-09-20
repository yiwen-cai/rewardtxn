"""Single-launcher CPU controller, required to run as private PID-namespace PID 1."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import time

from .descendants import Descendant, exited, recv_packet, send_packet, snapshot
from .faults import ChannelClosed, ControlError, event_nonce
from .run import ENV_ALLOWLIST, Journal, encoded


def read_config(path, profile="cpu"):
    from .native_gpu import PROFILE, GPU_ENV
    if profile not in ("cpu", PROFILE):
        raise ValueError("unknown execution profile")
    extra_env = GPU_ENV if profile == PROFILE else set()
    raw = Path(path).read_bytes()
    config = json.loads(raw)
    required = {"argv", "env", "schedule", "timeouts"}
    if not required <= set(config) or not set(config) <= required | {"observer_roles"}:
        raise ValueError("requires argv, env, schedule, timeouts; optional observer_roles")
    observer_roles = config.get("observer_roles", [])
    if not isinstance(observer_roles, list) or not all(isinstance(role, str) and role.isidentifier() for role in observer_roles) or len(set(observer_roles)) != len(observer_roles):
        raise ValueError("invalid observer_roles")
    argv = config["argv"]
    if (not isinstance(argv, list) or not argv or not all(isinstance(s, str) and "\0" not in s for s in argv)
            or not Path(argv[0]).is_absolute()):
        raise ValueError("argv must be a full array with an absolute executable")
    if not isinstance(config["env"], dict) or not set(config["env"]) <= ENV_ALLOWLIST | {"USER", "LOGNAME"} | extra_env or not all(isinstance(v, str) for v in config["env"].values()):
        raise ValueError("invalid effective environment allowlist")
    limits = config["timeouts"]
    if set(limits) != {"run", "handshake", "lease"} or any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0.1 <= v <= 7200 for v in limits.values()):
        raise ValueError("timeouts require finite run/handshake/lease seconds in [0.1,7200]")
    if limits["lease"] < limits["handshake"] + 1:
        raise ValueError("lease must exceed handshake by at least one second")
    seen = set()
    if not isinstance(config["schedule"], list):
        raise ValueError("schedule must be a list")
    for event in config["schedule"]:
        if set(event) != {"event_id", "target", "waiters", "evidence"}:
            raise ValueError("invalid scheduled event")
        key, waiters = event["event_id"], event["waiters"]
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("duplicate/invalid event id")
        seen.add(key)
        if (not isinstance(waiters, list) or not waiters or not all(isinstance(r, str) and r.isidentifier() for r in waiters)
                or len(set(waiters)) != len(waiters) or event["target"] not in waiters
                or not isinstance(event["evidence"], dict) or not event["evidence"]):
            raise ValueError("invalid roles or semantic evidence contract")
    return config, hashlib.sha256(raw).hexdigest()


def injection_assignment(entry, events, states, nonce):
    """Injection bookkeeping only: never return score, samples, or recovery data."""
    key = entry.requested_event
    if key is None:
        return None
    if key not in events or entry.role not in events[key]["waiters"]:
        raise ControlError("unknown requested injection event")
    state = states[key]
    if state["phase"] in ("releasing", "released"):
        status = "already_fired"
    elif state["phase"] == "waiting":
        previous = state["allocated"].get(entry.role)
        if previous is not None and previous != entry.incarnation:
            raise ControlError("pending assignment lost its process; no reassignment")
        state["allocated"][entry.role] = entry.incarnation
        status = "pending"
    else:
        raise ControlError("uncertain injection state; no reassignment")
    return {"event_id": key, "status": status,
            "event_nonce": event_nonce(nonce, key, list(events).index(key))}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded(value))
    os.replace(temporary, path)


def check_namespace(host_pidns):
    if os.getpid() != 1 or os.stat("/proc/self/ns/pid").st_ino == host_pidns:
        raise ControlError("controller must be PID 1 in a private namespace")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise ControlError("pidfd capability missing")
    fd = os.pidfd_open(1)
    try:
        signal.pidfd_send_signal(fd, 0)
    finally:
        os.close(fd)
    return snapshot(1)


def control(config, digest, output, nonce, host_pidns):
    controller_identity = check_namespace(host_pidns)
    if (output / "events.jsonl").exists() or (output / "frozen.json").exists():
        raise ControlError("namespace runs never resume or overwrite existing evidence")
    journal = Journal(output)
    atomic_json(output / "frozen.json", dict(config=config, config_sha256=digest, run_nonce=nonce))
    journal.add("controller_started", identity=controller_identity, run_nonce=nonce)
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    def interrupted(signum, frame):
        raise ControlError(f"controller interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
    listener.bind(str(output / "control.sock"))
    os.chmod(output / "control.sock", 0o600)
    listener.listen(32)
    listener.setblocking(False)
    selector = selectors.DefaultSelector()
    selector.register(listener, selectors.EVENT_READ, "listener")
    root_fd = None
    connections, registrations = {}, []
    active_roles = {}
    events = {e["event_id"]: e for e in config["schedule"]}
    states = {key: {"ready": {}, "receipt": set(), "phase": "waiting", "since": None, "allocated": {}} for key in events}
    roles = set().union(*(set(e["waiters"]) for e in events.values())) if events else set()
    roles.update(config.get("observer_roles", []))
    started = time.monotonic()
    lease_version, lease_at, heartbeat = None, started, 0
    failure = None
    try:
        # Only one Popen. No automatic launcher retry, including acquisition failure.
        env = dict(config["env"], FT_RUN_NONCE=nonce, FT_CONTROL_SOCKET=str(output / "control.sock"),
                   FT_HANDSHAKE_TIMEOUT=str(config["timeouts"]["handshake"]))
        with (output / "launcher.log").open("xb") as stream:
            launcher = subprocess.Popen(config["argv"], env=env, stdin=subprocess.DEVNULL,
                                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        journal.add("launcher_created", pid=launcher.pid, argv=config["argv"], env=config["env"])
        root_fd = os.pidfd_open(launcher.pid)
        root = snapshot(launcher.pid)
        if root["ppid"] != 1 or root["pid_namespace"] != controller_identity["pid_namespace"] or root["cgroup"] != controller_identity["cgroup"]:
            raise ControlError("launcher outside controller containment")
        journal.add("launcher_registered", identity=root)
        while True:
            now = time.monotonic()
            if now - started >= config["timeouts"]["run"]:
                if all(s["phase"] == "released" for s in states.values()):
                    journal.add("method_observation", outcome="timeout")
                    break
                raise ControlError("run deadline before injection/release completion")
            try:
                lease = json.loads((output / "host_lease.json").read_text())
            except (OSError, ValueError) as exc:
                raise ControlError("missing/corrupt host lease") from exc
            if lease.get("nonce") != nonce or type(lease.get("counter")) is not int:
                raise ControlError("invalid host lease")
            if lease["counter"] != lease_version:
                if lease_version is not None and lease["counter"] < lease_version:
                    raise ControlError("host lease counter regressed")
                lease_version, lease_at = lease["counter"], now
            if now - lease_at >= config["timeouts"]["lease"]:
                raise ControlError("host lease expired")
            atomic_json(output / "controller_heartbeat.json", {"nonce": nonce, "counter": heartbeat})
            heartbeat += 1
            for channel, pending in connections.items():
                if pending["registered"] is None and now - pending["since"] >= config["timeouts"]["handshake"]:
                    raise ControlError("registration deadline")
            for key, state in states.items():
                if state["since"] is not None and state["phase"] != "released" and now-state["since"] >= config["timeouts"]["handshake"]:
                    raise ControlError(f"barrier watchdog: {key}")
            for selected, _ in selector.select(0.02):
                if selected.data == "listener":
                    channel, _ = listener.accept()
                    channel.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
                    channel.settimeout(min(0.5, config["timeouts"]["handshake"]))
                    connections[channel] = {"registered": None, "since": now}
                    selector.register(channel, selectors.EVENT_READ, "worker")
                    continue
                channel = selected.fileobj
                entry = connections[channel]["registered"]
                if entry is None:
                    message, credentials, fds = recv_packet(channel)
                    entry = Descendant(channel, message, credentials, fds, root, nonce, roles)
                    registrations.append(entry)
                    previous = active_roles.get(entry.role)
                    if previous is not None and not exited(previous.pidfd):
                        raise ControlError("role already has a live registered incarnation")
                    active_roles[entry.role] = entry
                    connections[channel]["registered"] = entry
                    journal.add("descendant_registered", role=entry.role, incarnation=entry.incarnation,
                                identity=entry.original, ancestry=entry.chain)
                    assignment = injection_assignment(entry, events, states, nonce)
                    journal.add("injection_assignment", role=entry.role, incarnation=entry.incarnation, injection=assignment)
                    send_packet(channel, dict(kind="registered", nonce=nonce, incarnation=entry.incarnation, injection=assignment))
                    continue
                try:
                    message = entry.receive()
                except ChannelClosed:
                    selector.unregister(channel)
                    del connections[channel]
                    for key, state in states.items():
                        if state["ready"].get(entry.role) is entry:
                            target_done = entry.role == events[key]["target"] and state["phase"] in ("releasing", "released")
                            if not target_done and entry.role not in state["receipt"]:
                                raise ControlError("waiter disconnected before release receipt")
                    continue
                key = message.get("event_id")
                if key not in events or entry.role not in events[key]["waiters"]:
                    raise ControlError("unknown event/role")
                state, event = states[key], events[key]
                if entry.requested_event is not None and entry.requested_event != key:
                    raise ControlError("message outside assigned event")
                kind = message.get("kind")
                if kind == "ready":
                    if message.get("evidence") != event["evidence"]:
                        raise ControlError("semantic contract mismatch")
                    prior = state["ready"].get(entry.role)
                    if prior is not None:
                        if prior is not entry:
                            raise ControlError("old event reused by a new incarnation")
                        continue
                    entry.verify()
                    state["ready"][entry.role] = entry
                    state["since"] = state["since"] if state["since"] is not None else now
                    journal.add("ready", event_id=key, incarnation=entry.incarnation, identity=entry.original,
                                evidence=message["evidence"])
                elif kind == "released":
                    if state["phase"] not in ("releasing", "released") or state["ready"].get(entry.role) is not entry or entry.role == event["target"]:
                        raise ControlError("unexpected release receipt")
                    if entry.role not in state["receipt"]:
                        state["receipt"].add(entry.role)
                        journal.add("release_receipt", event_id=key, role=entry.role, incarnation=entry.incarnation)
                    if state["receipt"] == set(event["waiters"]) - {event["target"]}:
                        state["phase"] = "released"
                else:
                    raise ControlError("unknown registered message")
            # Ordered schedule: later ready messages may arrive, but never fire early.
            for index, (key, event) in enumerate(events.items()):
                state = states[key]
                if state["phase"] == "released":
                    continue
                if state["phase"] == "waiting" and set(state["ready"]) == set(event["waiters"]):
                    target = state["ready"][event["target"]]
                    fields = dict(event_id=key, event_nonce=event_nonce(nonce, key, index),
                                  incarnation=target.incarnation, identity=target.original)
                    for waiter in state["ready"].values():
                        waiter.verify()
                    journal.add("armed", **fields)
                    journal.add("signal_intent", **fields)
                    target.kill()
                    journal.add("signal_sent", **fields, signal=int(signal.SIGKILL))
                    if not exited(target.pidfd, min(config["timeouts"]["handshake"], max(0, config["timeouts"]["run"]-(time.monotonic()-started)))):
                        raise ControlError("pidfd exit observation deadline")
                    journal.add("process_exit_observed", **fields, evidence="same pidfd became readable; no wait status claimed")
                    state["phase"] = "releasing"
                    for role, waiter in state["ready"].items():
                        if role == event["target"]:
                            continue
                        if exited(waiter.pidfd):
                            raise ControlError("surviving waiter exited before release")
                        send_packet(waiter.channel, dict(kind="release", nonce=nonce, event_id=key, incarnation=waiter.incarnation))
                        journal.add("release", event_id=key, role=role, incarnation=waiter.incarnation)
                    if len(event["waiters"]) == 1:
                        state["phase"] = "released"
                break
            if launcher.poll() is not None:
                if all(s["phase"] == "released" for s in states.values()):
                    journal.add("method_observation", launcher_exit_code=launcher.returncode)
                    break
                # Drain pending receipts before concluding failure; handshake bounds it.
                if not connections:
                    raise ControlError("launcher exited before schedule completion")
        journal.add("result", classification="execution_complete", oracle_status="not_evaluated")
        return 0
    except BaseException as exc:
        failure = str(exc)
        journal.add("result", classification="technical_invalid", reason=failure, oracle_status="not_evaluated")
        return 2
    finally:
        if failure:
            for entry in registrations:
                try:
                    send_packet(entry.channel, dict(kind="abort", nonce=nonce, incarnation=entry.incarnation, reason=failure))
                except OSError:
                    pass
        for channel in connections:
            channel.close()
        for entry in registrations:
            entry.close()
        selector.close()
        listener.close()
        if root_fd is not None:
            os.close(root_fd)
        # No descendant wait/kill fallback here. main MUST exit PID 1, which is
        # the containment cleanup even when initial launcher pidfd acquisition failed.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default="cpu", choices=["cpu", "native-trainer-4gpu"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--host-pidns", required=True, type=int)
    args = parser.parse_args()
    code = 2
    try:
        config, digest = read_config(args.config, args.profile)
        code = control(config, digest, Path(args.output), args.nonce, args.host_pidns)
    except BaseException as exc:
        print(f"namespace controller failed: {exc}", flush=True)
    finally:
        # No Python finalizer/thread can keep the PID namespace alive after failure.
        os._exit(code)


if __name__ == "__main__":
    main()
