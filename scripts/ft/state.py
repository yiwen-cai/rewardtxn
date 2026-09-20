"""R-only, single-host POSIX state contract. No GPU/backend fencing is claimed.

Only ``cpu_contract`` process registrations are currently accepted. A backend
adapter must establish genuine finalization before these receipts can describe
real training. This module never reads observer data or performs replay/ACK.
"""

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid
from dataclasses import dataclass


class StateError(RuntimeError):
    pass


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise StateError("duplicate JSON key")
        result[key] = value
    return result


def _read(path):
    try:
        return json.loads(path.read_bytes(), object_pairs_hook=_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              StateError("nonfinite JSON")))
    except (OSError, ValueError) as exc:
        raise StateError(f"cannot read {path.name}") from exc


def _path(root, relative):
    if not isinstance(relative, str) or not relative or relative.startswith("/"):
        raise StateError("expected relative path")
    if any(part in ("", ".", "..") for part in relative.split("/")):
        raise StateError("unsafe path")
    path = root
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise StateError("symlink forbidden")
    return path


def _directory(path):
    if not path.exists():
        _directory(path.parent)
        path.mkdir(exist_ok=True)
        _sync_dir(path.parent)
    if not path.is_dir() or path.is_symlink():
        raise StateError("expected real directory")


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path, value, immutable=True):
    data = _bytes(value)
    _directory(path.parent)
    temporary = path.parent / (".tmp-" + uuid.uuid4().hex)
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.is_symlink() or path.read_bytes() != data:
                    raise StateError("immutable record conflict")
        else:
            if path.is_symlink():
                raise StateError("symlink forbidden")
            os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _identity(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        return {"pid": int(pid), "start_time": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "state": fields[0]}
    except FileNotFoundError:
        return None


def process_identity(pid):
    """Capture an actual local process for CPU contract registration."""
    found = _identity(pid)
    if found is None:
        raise StateError("process missing")
    return {key: found[key] for key in ("pid", "start_time", "boot_id")}


def _exited(identity):
    actual = _identity(identity["pid"])
    return actual is None or actual["state"] == "Z" or any(
        actual[key] != identity[key] for key in ("start_time", "boot_id"))


@dataclass
class Owner:
    root: Path
    epoch: int
    nonce: str
    fd: int
    pid: int

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@contextlib.contextmanager
def _locked(owner):
    if owner.fd < 0 or owner.pid != os.getpid():
        raise StateError("inactive or inherited owner")
    fd = os.open(_path(owner.root, "mutation.lock"),
                 os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        control = _read(_path(owner.root, "control.json"))
        if (control["epoch"], control["owner_nonce"]) != (owner.epoch, owner.nonce):
            raise StateError("stale owner")
        yield control
    finally:
        os.close(fd)


def acquire_owner(run_dir, expected_epoch, prior_owner_exit_proof=None, *,
                  scope="cpu_contract", participants=(), run_nonce=None, config_sha256=None,
                  verifier_version=None):
    """Acquire exclusive ownership; prior proof must match persisted identities.

    ``participants`` registers all CPU contract workers at creation. Registration
    completeness is a caller contract, not proof about unregistered descendants.
    GPU/unknown scopes are rejected until a real adapter defines ownership.
    Verifier version is frozen for the run. Each attempt binds an explicitly
    authorized policy version; actual policy provenance/staleness is the
    adapter's responsibility, not established by this storage contract.
    """
    if scope != "cpu_contract":
        raise StateError("only CPU contract ownership is implemented")
    root = Path(os.path.abspath(run_dir))
    for path in (root, *root.parents):
        if path.is_symlink():
            raise StateError("symlink root")
    _directory(root)
    fd = os.open(_path(root, "owner.lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control_path = _path(root, "control.json")
        control = _read(control_path) if control_path.exists() else None
        if control is None:
            if expected_epoch != -1 or not run_nonce or not _digest(config_sha256):
                raise StateError("new run requires epoch -1, nonce and config hash")
            if not isinstance(verifier_version, str) or not verifier_version:
                raise StateError("new run requires frozen verifier version")
            control = {"epoch": -1, "run_nonce": run_nonce, "config_sha256": config_sha256,
                       "attempts": {}, "accepted": {}, "head": None,
                       "verifier_version": verifier_version}
        else:
            if expected_epoch != control["epoch"]:
                raise StateError("epoch CAS failed")
            if prior_owner_exit_proof != control["processes"]:
                raise StateError("exit proof must identify persisted processes")
            if not all(_exited(item) for item in control["processes"]):
                raise StateError("previous registered process still alive")
            if run_nonce is not None and run_nonce != control["run_nonce"]:
                raise StateError("run mismatch")
            if config_sha256 is not None and config_sha256 != control["config_sha256"]:
                raise StateError("config mismatch")
            if verifier_version is not None and verifier_version != control["verifier_version"]:
                raise StateError("frozen version contract mismatch")
        registered = [process_identity(os.getpid())]
        for item in participants:
            if process_identity(item["pid"]) != item:
                raise StateError("participant identity mismatch")
            if item not in registered:
                registered.append(item)
        control.update(epoch=control["epoch"] + 1, owner_nonce=uuid.uuid4().hex,
                       processes=registered, scope=scope)
        _write(control_path, control, immutable=False)
        owner = Owner(root, control["epoch"], control["owner_nonce"], fd, os.getpid())
        os.register_at_fork(after_in_child=owner.close)
        return owner
    except BaseException:
        os.close(fd)
        raise


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def authorize_attempt(owner, logical_sample, expected_attempt, new_attempt, *, expected_policy_version):
    if type(expected_policy_version) is not int or expected_policy_version < 0:
        raise StateError("explicit nonnegative policy version required")
    with _locked(owner) as control:
        old = control["attempts"].get(logical_sample)
        if ((old["attempt"] if old else None) != expected_attempt or not new_attempt
                or new_attempt == expected_attempt):
            raise StateError("attempt CAS failed")
        attempt = {"sample": logical_sample, "attempt": new_attempt,
                   "epoch": owner.epoch, "owner_nonce": owner.nonce,
                   "versions": {"policy_version": expected_policy_version,
                                "verifier_version": control["verifier_version"]}}
        control["attempts"][logical_sample] = attempt
        control["accepted"].pop(logical_sample, None)
        _write(_path(owner.root, "control.json"), control, immutable=False)
        return attempt


def _check_versions(control, attempt, payload):
    versions = attempt.get("versions", {})
    if (set(versions) != {"policy_version", "verifier_version"}
            or versions["verifier_version"] != control["verifier_version"]
            or type(versions["policy_version"]) is not int or versions["policy_version"] < 0):
        raise StateError("attempt version authorization mismatch")
    if (type(payload.get("policy_version")) is not int
            or any(payload.get(key) != value for key, value in versions.items())):
        raise StateError("result violates policy/verifier authorization")


def _check_group_versions(control, group):
    versions = set()
    for sample in group["samples"]:
        receipt = sample["receipt"]
        _check_versions(control, receipt["attempt"], receipt["payload"])
        versions.add(receipt["payload"]["verifier_version"])
    if len(versions) != 1:
        raise StateError("mixed verifier versions within logical group")


def accept_result(owner, attempt, payload_manifest):
    with _locked(owner) as control:
        if (attempt.get("epoch"), attempt.get("owner_nonce")) != (owner.epoch, owner.nonce):
            raise StateError("stale result")
        if control["attempts"].get(attempt.get("sample")) != attempt:
            raise StateError("unauthorized attempt")
        if not payload_manifest or not all(_digest(payload_manifest.get(key)) for key in
                ("response_sha256", "reward_sha256", "tensor_input_sha256")):
            raise StateError("incomplete result hashes")
        _check_versions(control, attempt, payload_manifest)
        receipt = {"attempt": attempt, "payload": payload_manifest}
        old = control["accepted"].get(attempt["sample"])
        if old is not None and old != receipt:
            raise StateError("conflicting result")
        control["accepted"][attempt["sample"]] = receipt
        _write(_path(owner.root, "control.json"), control, immutable=False)
        return receipt


def _sets(data):
    try:
        drawn, consumed = data["drawn"], data["consumed"]
        pending = data["pending"]
        keys = [item["sample"] for item in pending]
        if any(len(items) != len(set(items)) for items in (drawn, consumed, keys)):
            raise StateError("duplicate data identity")
        if set(consumed) & set(keys) or set(drawn) != set(consumed) | set(keys):
            raise StateError("cursor coverage mismatch")
        if data["cursor"] != len(drawn) or data["epoch"] < 0 or not data["shuffle_state"]:
            raise StateError("invalid data cursor")
        if not _digest(data["source_sha256"]):
            raise StateError("data source hash missing")
        for item in pending:
            if (item["action"] != "regenerate" or not item.get("prompt")
                    or item["prompt_sha256"] != _hash(_bytes(item["prompt"]))
                    or not isinstance(item.get("k"), int) or item["k"] < 1):
                raise StateError("first implementation only supports explicit pending regeneration")
        return set(consumed)
    except (KeyError, TypeError) as exc:
        raise StateError("incomplete data state") from exc


def _generation(owner, gid):
    if not isinstance(gid, str) or not re.fullmatch(r"g-[0-9a-f]{32}", gid):
        raise StateError("invalid generation ID")
    return _path(owner.root, "generations/" + gid)


def _inventory(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise StateError("symlink in snapshot")
        if path.is_dir():
            continue
        if not stat.S_ISREG(path.stat().st_mode):
            raise StateError("nonregular snapshot file")
        digest = hashlib.sha256()
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            os.fsync(stream.fileno())
            after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise StateError("snapshot changed while hashing")
        result[str(path.relative_to(directory))] = {"size": after.st_size, "sha256": digest.hexdigest()}
    if not result:
        raise StateError("empty snapshot")
    return result


def _validate_token(owner, gid):
    directory = _generation(owner, gid)
    token = _read(_path(directory, "token.json"))
    manifest = _read(_path(directory, "manifest.json"))
    if (token.get("generation") != gid or token.get("manifest_sha256") != _hash(_bytes(manifest))
            or token.get("intent_sha256") != _hash(_path(directory, "intent.json").read_bytes())
            or manifest.get("files") != _inventory(_path(directory, "checkpoint"))):
        raise StateError("corrupt token or snapshot")
    return token, manifest


def _head(owner, control):
    root = _path(owner.root, "generations")
    tokens = {}
    if root.exists():
        for directory in root.iterdir():
            if directory.is_symlink():
                raise StateError("symlink generation")
            if (directory / "token.json").exists():
                token, manifest = _validate_token(owner, directory.name)
                if token["run_nonce"] != control["run_nonce"] or manifest["config_sha256"] != control["config_sha256"]:
                    raise StateError("token run/config mismatch")
                tokens[directory.name] = (token, manifest)
    head = None
    remaining = dict(tokens)
    while remaining:
        children = [gid for gid, (token, _) in remaining.items() if token["parent"] == head]
        if len(children) != 1:
            raise StateError("forked or disconnected committed chain")
        gid = children[0]
        token, _ = remaining.pop(gid)
        head = {"generation": gid, "token_sha256": _hash(_bytes(token))}
    return head, tokens


def prepare_generation(owner, intent, data_snapshot):
    with _locked(owner) as control:
        head, tokens = _head(owner, control)
        directory = _path(owner.root, "generations")
        _directory(directory)
        for item in directory.iterdir():
            if (item / "intent.json").exists() and not (item / "token.json").exists() and not (item / "abandoned.json").exists():
                raise StateError("unresolved generation")
        if intent.get("parent") != head or intent.get("ack_capability") != "none":
            raise StateError("parent mismatch or unsupported ACK")
        if intent.get("config_sha256") != control["config_sha256"]:
            raise StateError("configuration mismatch")
        ranks = intent.get("expected_ranks", [])
        if not ranks or len(ranks) != len(set(ranks)):
            raise StateError("invalid expected ranks")
        consumed = _sets(data_snapshot)
        previous = set() if head is None else set(tokens[head["generation"]][1]["data"]["consumed"])
        if head is not None:
            old_data = tokens[head["generation"]][1]["data"]
            if (data_snapshot["source_sha256"] != old_data["source_sha256"]
                    or data_snapshot["drawn"][:len(old_data["drawn"])] != old_data["drawn"]):
                raise StateError("data lineage changed")
        samples, updates, physical = set(), set(), set()
        for update in intent.get("updates", []):
            if (update["logical_update_id"] in updates or not _digest(update["train_input_sha256"])
                    or not update.get("physical_update_id") or update["physical_update_id"] in physical):
                raise StateError("duplicate or unmapped update")
            updates.add(update["logical_update_id"])
            physical.add(update["physical_update_id"])
            for group in update["groups"]:
                _check_group_versions(control, group)
                if not group.get("prompt") or group.get("prompt_sha256") != _hash(_bytes(group["prompt"])):
                    raise StateError("group regeneration input missing")
                entries = group["samples"]
                if sorted(item["sample_index"] for item in entries) != list(range(group["k"])) or group["k"] < 1:
                    raise StateError("incomplete group")
                for entry in entries:
                    key = entry["sample"]
                    if key in samples or control["accepted"].get(key) != entry["receipt"]:
                        raise StateError("missing or duplicate accepted sample")
                    if key != f"{group['logical_group_id']}:{entry['sample_index']}":
                        raise StateError("sample/group identity mismatch")
                    if entry["receipt"]["attempt"] != control["attempts"].get(key):
                        raise StateError("sample attempt changed")
                    if entry["receipt"]["attempt"]["epoch"] != owner.epoch:
                        raise StateError("old epoch sample")
                    _check_versions(control, entry["receipt"]["attempt"], entry["receipt"]["payload"])
                    samples.add(key)
        if not updates or consumed != previous | samples or previous & samples:
            raise StateError("consumption mapping mismatch")
        prior_updates = {u["logical_update_id"] for _, manifest in tokens.values() for u in manifest["updates"]}
        if updates & prior_updates:
            raise StateError("retained logical update duplicated")
        gid = "g-" + uuid.uuid4().hex
        target = _generation(owner, gid)
        _directory(target)
        record = dict(intent, schema=1, run_nonce=control["run_nonce"], generation=gid,
                      execution_epoch=owner.epoch, execution_owner=owner.nonce, data=data_snapshot)
        _write(_path(target, "intent.json"), record)
        _sync_dir(directory)
        return gid


def record_evidence(owner, generation, evidence):
    with _locked(owner):
        directory = _generation(owner, generation)
        intent = _read(_path(directory, "intent.json"))
        if (intent["execution_epoch"], intent["execution_owner"]) != (owner.epoch, owner.nonce):
            raise StateError("cannot add evidence for old execution")
        if evidence.get("snapshot_id") != intent["data_snapshot_id"]:
            raise StateError("snapshot mismatch")
        kind = evidence.get("kind")
        if kind == "optimizer":
            expected = [u["physical_update_id"] for u in intent["updates"]]
            if evidence.get("physical_updates") != expected or evidence.get("successful") is not True or evidence.get("scheduler_applied") is not True:
                raise StateError("optimizer/scheduler evidence incomplete")
            name = "optimizer"
        elif kind == "finalize":
            if evidence.get("rank") not in intent["expected_ranks"] or evidence.get("writer_closed") is not True:
                raise StateError("rank not finalized")
            name = "rank-" + _hash(evidence["rank"].encode())
        else:
            raise StateError("unknown evidence")
        path = _path(directory, "receipts/" + name + ".json")
        if _path(directory, "token.json").exists():
            if not path.exists() or _read(path) != evidence:
                raise StateError("committed evidence is immutable")
            return evidence
        _write(path, evidence)
        return evidence


def _complete(owner, gid):
    directory = _generation(owner, gid)
    intent = _read(_path(directory, "intent.json"))
    optimizer = _read(_path(directory, "receipts/optimizer.json"))
    receipts = [optimizer]
    if (optimizer.get("kind") != "optimizer" or optimizer.get("successful") is not True
            or optimizer.get("scheduler_applied") is not True
            or optimizer.get("physical_updates") != [u["physical_update_id"] for u in intent["updates"]]):
        raise StateError("invalid optimizer receipt")
    for rank in intent["expected_ranks"]:
        receipt = _read(_path(directory, "receipts/rank-" + _hash(rank.encode()) + ".json"))
        if receipt.get("kind") != "finalize" or receipt.get("rank") != rank or receipt.get("writer_closed") is not True:
            raise StateError("invalid finalize receipt")
        receipts.append(receipt)
    if any(item["snapshot_id"] != intent["data_snapshot_id"] for item in receipts):
        raise StateError("inconsistent snapshot")
    components = intent["components"]
    required = {"model", "optimizer_master", "optimizer_moments", "optimizer_step", "scheduler",
                "rng_python", "rng_numpy", "rng_torch_cpu", "rng_device", "rng_tracker", "policy"}
    if set(components) != required:
        raise StateError("missing state component")
    files = _inventory(_path(directory, "checkpoint"))
    referenced = set()
    for mapping in components.values():
        if set(mapping) != set(intent["expected_ranks"]):
            raise StateError("component rank coverage mismatch")
        for paths in mapping.values():
            if not paths:
                raise StateError("empty component")
            for path in paths:
                _path(directory / "checkpoint", path)
                if path not in files:
                    raise StateError("component file missing")
                referenced.add(path)
    if referenced != set(files):
        raise StateError("unmapped checkpoint files")
    _sets(intent["data"])
    return intent, receipts, files


def commit_generation(owner, generation):
    with _locked(owner) as control:
        head, tokens = _head(owner, control)
        if generation in tokens:
            return tokens[generation][0]
        directory = _generation(owner, generation)
        if _path(directory, "abandoned.json").exists():
            raise StateError("abandoned generation")
        intent, receipts, files = _complete(owner, generation)
        if intent["parent"] != head:
            raise StateError("parent CAS failed")
        for update in intent["updates"]:
            for group in update["groups"]:
                _check_group_versions(control, group)
                for sample in group["samples"]:
                    if control["attempts"].get(sample["sample"]) != sample["receipt"]["attempt"]:
                        raise StateError("prepared attempt superseded")
                    _check_versions(control, sample["receipt"]["attempt"], sample["receipt"]["payload"])
        # The backend writes directly into this generation. Its finalization
        # receipt is a contract, not proof against an uncooperative writer.
        checkpoint = _path(directory, "checkpoint")
        for path in sorted(checkpoint.rglob("*"), reverse=True):
            if path.is_dir():
                _sync_dir(path)
        _sync_dir(checkpoint)
        if _inventory(checkpoint) != files:
            raise StateError("checkpoint changed after finalization")
        manifest = dict(intent, files=files, receipts=receipts)
        _write(_path(directory, "manifest.json"), manifest)
        token = {"schema": 1, "run_nonce": control["run_nonce"], "generation": generation,
                 "parent": head, "intent_sha256": _hash(_bytes(intent)), "manifest_sha256": _hash(_bytes(manifest)),
                 "execution_epoch": intent["execution_epoch"], "commit_epoch": owner.epoch,
                 "commit_owner": owner.nonce}
        _write(_path(directory, "token.json"), token)
        control["head"] = {"generation": generation, "token_sha256": _hash(_bytes(token))}
        _write(_path(owner.root, "control.json"), control, immutable=False)
        return token


def select_recovery(owner):
    with _locked(owner) as control:
        head, _ = _head(owner, control)
        root = _path(owner.root, "generations")
        candidates = [] if not root.exists() else [item.name for item in root.iterdir()
            if (item / "intent.json").exists() and not (item / "token.json").exists()
            and not (item / "abandoned.json").exists()]
        if len(candidates) > 1:
            raise StateError("multiple unresolved candidates")
        promote = None
        reason = "verified committed chain"
        for gid in candidates:
            try:
                intent, _, _ = _complete(owner, gid)
                if intent["parent"] != head:
                    raise StateError("candidate parent mismatch")
                promote = gid
            except StateError as exc:
                reason = "incomplete candidate: " + str(exc)
                _write(_path(_generation(owner, gid), "abandoned.json"),
                       {"epoch": owner.epoch, "reason": reason})
    if promote is not None:
        token = commit_generation(owner, promote)
        head = {"generation": promote, "token_sha256": _hash(_bytes(token))}
        reason = "completed durable candidate"
    # Preserve uncommitted consumption obligations in R's own intents. They are
    # recovery inputs, never evidence that replay has actually completed.
    rollback_intents = [] if not root.exists() else [str(item / "intent.json") for item in sorted(root.iterdir())
        if (item / "abandoned.json").exists() and not (item / "token.json").exists()]
    if head is None:
        return {"generation": None, "checkpoint": None, "pending": [],
                "rollback_intents": rollback_intents, "reason": reason}
    directory = _generation(owner, head["generation"])
    _, manifest = _validate_token(owner, head["generation"])
    return {"generation": head["generation"], "checkpoint": str(directory / "checkpoint"),
            "pending": manifest["data"]["pending"], "rollback_intents": rollback_intents, "reason": reason}
