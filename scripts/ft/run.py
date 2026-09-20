"""Bounded CPU-capable FT controller; no AReaL recovery adapter is implied."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import time
import uuid

from .faults import ChannelClosed, ControlError, OwnedProcess, event_nonce, receive, send

# Child environment is constructed, never copied wholesale or logged wholesale.
ENV_ALLOWLIST = {"PATH", "LANG", "LC_ALL", "PYTHONPATH", "CUDA_VISIBLE_DEVICES",
                 "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM"}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def preflight(config_path, schedule_path):
    config_bytes = Path(config_path).read_bytes()
    schedule_bytes = Path(schedule_path).read_bytes()
    config, schedule = json.loads(config_bytes), json.loads(schedule_bytes)
    if set(config) != {"attempts", "env", "handshake_timeout", "run_timeout", "cleanup_timeout"}:
        raise ValueError("config requires attempts/env/handshake_timeout/run_timeout/cleanup_timeout")
    if not set(config["env"]) <= ENV_ALLOWLIST or not all(isinstance(v, str) for v in config["env"].values()):
        raise ValueError("environment is not in the effective allowlist")
    if not 1 <= len(config["attempts"]) <= 4:
        raise ValueError("one initial attempt and at most three restart attempts")
    for key in ("handshake_timeout", "run_timeout", "cleanup_timeout"):
        if not isinstance(config[key], (int, float)) or not 0 < config[key] <= 7200:
            raise ValueError(f"invalid {key}")
    for workers in config["attempts"]:
        if not workers or not all(isinstance(role, str) and role.replace('_', '').isalnum()
                                  for role in workers):
            raise ValueError("invalid role map")
        for argv in workers.values():
            if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
                raise ValueError("argv must be a nonempty string array")
            if not Path(argv[0]).is_absolute():
                raise ValueError("executable must be absolute")
    seen = set()
    for event in schedule:
        if set(event) != {"event_id", "attempt", "target", "waiters", "evidence"}:
            raise ValueError("invalid event fields")
        key = (event["event_id"], event["attempt"])
        if not isinstance(key[0], str) or key in seen:
            raise ValueError("duplicate/invalid event identity")
        seen.add(key)
        if type(key[1]) is not int or not 0 <= key[1] < len(config["attempts"]):
            raise ValueError("invalid event attempt")
        roles = config["attempts"][key[1]]
        if (event["target"] not in event["waiters"] or not event["waiters"]
                or len(set(event["waiters"])) != len(event["waiters"])
                or not set(event["waiters"]) <= set(roles)
                or not isinstance(event["evidence"], dict) or not event["evidence"]):
            raise ValueError("invalid target/waiters/semantic evidence contract")
    return {"config": config, "schedule": schedule,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "schedule_sha256": hashlib.sha256(schedule_bytes).hexdigest()}


class Journal:
    def __init__(self, directory):
        self.path = directory / "events.jsonl"
        self.records = [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []

    def add(self, kind, **fields):
        record = dict(kind=kind, controller_monotonic_ns=time.monotonic_ns(), **fields)
        with self.path.open("ab") as stream:
            stream.write(encoded(record) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records.append(record)


def execute(frozen, directory, resume=False):
    # Capability is checked before launching any child.
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise ControlError("pidfd unavailable: use a supported Linux Python; no kill fallback")
    probe_fd = os.pidfd_open(os.getpid())
    os.close(probe_fd)
    directory = Path(directory)
    if resume:
        if not directory.is_dir():
            raise ControlError("resume directory missing")
    else:
        directory.mkdir(parents=True, exist_ok=False)
    with (directory / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_path = directory / "frozen.json"
        if resume:
            saved = json.loads(freeze_path.read_text())
            if {key: saved[key] for key in frozen} != frozen:
                raise ControlError("frozen inputs changed")
            frozen = saved
        else:
            frozen = dict(frozen, run_nonce=uuid.uuid4().hex)
            with freeze_path.open("xb") as stream:
                stream.write(encoded(frozen))
                stream.flush()
                os.fsync(stream.fileno())
        journal = Journal(directory)
        if any(r["kind"] == "result" for r in journal.records):
            raise ControlError("completed/failed run is immutable; no automatic retry")
        # Even observed events cannot be replayed after controller loss. Only a
        # durably closed attempt is safe to advance past without live ownership.
        started = {r["attempt"] for r in journal.records if r["kind"] == "attempt_start"}
        finished = {r["attempt"] for r in journal.records if r["kind"] == "attempt_end"}
        if started - finished:
            journal.add("result", classification="technical_invalid", reason="uncertain interrupted attempt; no replay", oracle_status="not_evaluated")
            return journal.records[-1]
        config = frozen["config"]
        deadline = time.monotonic() + config["run_timeout"]
        try:
            for attempt, workers in enumerate(config["attempts"]):
                if attempt in finished:
                    continue
                journal.add("attempt_start", attempt=attempt)
                run_attempt(frozen, directory, journal, attempt, workers, deadline)
                journal.add("attempt_end", attempt=attempt)
            journal.add("result", classification="execution_complete", oracle_status="not_evaluated")
        except (ControlError, OSError, ValueError) as exc:
            journal.add("result", classification="technical_invalid", reason=str(exc), oracle_status="not_evaluated")
        return journal.records[-1]


def run_attempt(frozen, directory, journal, attempt, workers, deadline):
    config = frozen["config"]
    events = {e["event_id"]: e for e in frozen["schedule"] if e["attempt"] == attempt}
    state = {key: {"ready": {}, "released": set(), "phase": "waiting", "since": None} for key in events}
    owned, channels, logs = {}, {}, []
    selector = selectors.DefaultSelector()
    nonce = frozen["run_nonce"]
    provisional = []
    failure = None
    started_at = time.monotonic()
    try:
        for role, argv in workers.items():
            parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            channels[role] = parent
            env = dict(config["env"], FT_CONTROL_FD=str(child.fileno()), FT_ROLE=role,
                       FT_ATTEMPT=str(attempt), FT_RUN_NONCE=nonce,
                       FT_HANDSHAKE_TIMEOUT=str(config["handshake_timeout"]))
            log = (directory / f"attempt-{attempt}-{role}.log").open("xb")
            logs.append(log)
            try:
                process = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, pass_fds=(child.fileno(),), start_new_session=True)
                # An unreaped Popen child cannot have its PID reused. Retain its
                # pidfd before validation so a validation failure still has an
                # exact, launcher-owned cleanup handle.
                fd = os.pidfd_open(process.pid)
                provisional.append((process, fd))
                owned[role] = OwnedProcess.register(process, pidfd=fd)
                provisional.pop()
            finally:
                child.close()
            channels[role] = parent
            selector.register(parent, selectors.EVENT_READ, role)
            journal.add("process_registered", attempt=attempt, role=role, identity=owned[role].original,
                        argv=argv, env=config["env"])
        while True:
            now = time.monotonic()
            if now >= deadline:
                hit = any(s["phase"] == "released" for s in state.values())
                if hit and all(s["phase"] == "released" for s in state.values()):
                    journal.add("method_observation", outcome="timeout", attempt=attempt)
                    return
                raise ControlError("run deadline before completed injection/release")
            for key, s in state.items():
                if s["since"] is not None and s["phase"] != "released" and now - s["since"] > config["handshake_timeout"]:
                    raise ControlError(f"barrier watchdog: {key}")
            if events and not any(s["since"] is not None for s in state.values()) and now - started_at > config["handshake_timeout"]:
                raise ControlError("startup/ready deadline")
            if all(p.process.poll() is not None for p in owned.values()) and all(s["phase"] == "released" for s in state.values()):
                journal.add("method_observation", attempt=attempt,
                            exit_codes={role: p.process.returncode for role, p in owned.items()})
                return
            for selected, _ in selector.select(0.02):
                role = selected.data
                channel = channels[role]
                try:
                    message = receive(channel)
                except ChannelClosed:
                    selector.unregister(channel)
                    for event_id, event in events.items():
                        if role in event["waiters"]:
                            status = state[event_id]
                            target_done = role == event["target"] and status["phase"] in ("releasing", "released")
                            if not target_done and role not in status["released"]:
                                raise ControlError("worker disconnected before completed barrier")
                    continue
                key = message.get("event_id")
                if key not in events:
                    raise ControlError("unknown event")
                event, s = events[key], state[key]
                expected = {"event_id": key, "attempt": attempt, "run_nonce": nonce,
                            "event_nonce": event_nonce(nonce, key, attempt), "role": role,
                            "pid": owned[role].process.pid}
                if role not in event["waiters"] or any(message.get(k) != v for k, v in expected.items()):
                    raise ControlError("ready identity/nonce mismatch")
                kind = message.get("kind")
                if kind == "released":
                    if s["phase"] not in ("releasing", "released"):
                        raise ControlError("release receipt without release")
                    if role not in s["released"]:
                        s["released"].add(role)
                        journal.add("release_receipt", **expected)
                    if s["released"] == set(event["waiters"]) - {event["target"]}:
                        s["phase"] = "released"
                    continue
                if kind != "ready" or message.get("evidence") != event["evidence"]:
                    raise ControlError("semantic evidence mismatch")
                if role in s["ready"]:
                    if s["ready"][role] != message:
                        raise ControlError("conflicting duplicate ready")
                    continue
                s["ready"][role] = message
                if s["since"] is None:
                    s["since"] = now
                journal.add(**message)
                if set(s["ready"]) != set(event["waiters"]):
                    continue
                shared = {k: v for k, v in expected.items() if k not in ("role", "pid")}
                journal.add("armed", **shared)
                # Write-ahead fired is intentionally uncertain until observed.
                journal.add("fired", **shared, target=event["target"], signal=signal.SIGKILL)
                target = owned[event["target"]]
                if not target.send(signal.SIGKILL):
                    raise ControlError("target exited before signal")
                try:
                    code = target.process.wait(timeout=min(config["handshake_timeout"], max(0.001, deadline-time.monotonic())))
                except subprocess.TimeoutExpired as exc:
                    raise ControlError("signal observation timeout") from exc
                if code != -signal.SIGKILL:
                    raise ControlError("target exit did not confirm SIGKILL")
                journal.add("observed", **shared, target=event["target"], exit_code=code)
                s["phase"] = "releasing"
                for waiter in event["waiters"]:
                    if waiter == event["target"]:
                        continue
                    if owned[waiter].process.poll() is not None:
                        raise ControlError("non-target waiter died before release")
                    reply = {k: v for k, v in s["ready"][waiter].items() if k not in ("kind", "evidence")}
                    send(channels[waiter], dict(reply, kind="release"))
                    journal.add("release", **reply)
                if len(event["waiters"]) == 1:
                    s["phase"] = "released"
    except BaseException as exc:
        failure = str(exc)
        raise
    finally:
        pending_abort = {}
        for key, s in state.items():
            for role, message in s["ready"].items():
                if role in channels and role not in s["released"]:
                    reply = {k: v for k, v in message.items() if k not in ("kind", "evidence")}
                    try:
                        send(channels[role], dict(reply, kind="abort", reason=failure or "attempt closed"))
                        journal.add("abort", **reply, reason=failure or "attempt closed")
                        pending_abort[role] = reply
                    except OSError:
                        pass
        abort_deadline = time.monotonic() + min(config["cleanup_timeout"], config["handshake_timeout"])
        while pending_abort and time.monotonic() < abort_deadline:
            for selected, _ in selector.select(min(0.02, max(0, abort_deadline-time.monotonic()))):
                role = selected.data
                try:
                    receipt = receive(channels[role])
                except (ControlError, OSError, ValueError):
                    selector.unregister(channels[role])
                    pending_abort.pop(role, None)
                    continue
                expected = pending_abort.get(role)
                if expected and receipt.get("kind") == "aborted" and all(receipt.get(k) == v for k, v in expected.items()):
                    journal.add("abort_receipt", **expected)
                    pending_abort.pop(role)
            for role in list(pending_abort):
                if owned[role].process.poll() is not None:
                    journal.add("abort_worker_exit", **pending_abort.pop(role))
        for expected in pending_abort.values():
            journal.add("watchdog", **expected, reason="abort receipt deadline")
        # Closing every control endpoint triggers the independent worker watchdog.
        for channel in channels.values():
            channel.close()
        selector.close()
        cleanup_deadline = time.monotonic() + config["cleanup_timeout"]
        cleanup_errors = []
        for process, fd in provisional:
            try:
                if process.poll() is None:
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
                process.wait(timeout=max(0.001, cleanup_deadline-time.monotonic()))
            except (OSError, subprocess.TimeoutExpired) as exc:
                cleanup_errors.append(str(exc))
            finally:
                os.close(fd)
        for process in owned.values():
            try:
                process.send(signal.SIGKILL)
                process.process.wait(timeout=max(0.001, cleanup_deadline-time.monotonic()))
            except (ControlError, OSError, subprocess.TimeoutExpired) as exc:
                cleanup_errors.append(str(exc))
            finally:
                process.close()
        for log in logs:
            log.close()
        if cleanup_errors:
            journal.add("cleanup_failure", errors=cleanup_errors)
            raise ControlError("bounded cleanup failed: " + "; ".join(cleanup_errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--dry-run", "--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    frozen = preflight(args.config, args.schedule)
    if args.dry_run:
        print(json.dumps(frozen, indent=2))
        return
    if not args.run_dir:
        parser.error("--run-dir required for execution")
    result = execute(frozen, args.run_dir, args.resume)
    print(json.dumps(result))
    if result["classification"] == "technical_invalid":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
