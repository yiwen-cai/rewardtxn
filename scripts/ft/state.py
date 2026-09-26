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

from . import perf_probe


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
        raw = path.read_bytes()
    except OSError as exc:
        raise StateError(f"cannot read {path.name}") from exc
    return _parse(raw, path)


def _parse(raw, path):
    try:
        with perf_probe.span("state.json_parse"):
            return json.loads(raw, object_pairs_hook=_pairs,
                              parse_constant=lambda value: (_ for _ in ()).throw(
                                  StateError("nonfinite JSON")))
    except ValueError as exc:
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


# Parsed control.json per state root, keyed by its exact bytes. Accessed only
# while holding mutation.lock. Read-only sections share the cached object;
# write sections get a private parse and drop the cache entry on exit, so a
# failed or partial write is never visible in-process.
_CONTROL_CACHE = {}


@contextlib.contextmanager
def _locked(owner, *, write=False):
    if owner.fd < 0 or owner.pid != os.getpid():
        raise StateError("inactive or inherited owner")
    fd = os.open(_path(owner.root, "mutation.lock"),
                 os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    key = str(owner.root)
    try:
        with perf_probe.span("state.flock_wait"):
            fcntl.flock(fd, fcntl.LOCK_EX)
        path = _path(owner.root, "control.json")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise StateError("cannot read control.json") from exc
        cached = _CONTROL_CACHE.get(key)
        if write:
            _CONTROL_CACHE.pop(key, None)
            control = _parse(raw, path)
        elif cached is not None and cached[0] == raw:
            control = cached[1]
        else:
            control = _parse(raw, path)
            _CONTROL_CACHE[key] = (raw, control)
        if (control["epoch"], control["owner_nonce"]) != (owner.epoch, owner.nonce):
            raise StateError("stale owner")
        yield control
    except BaseException:
        _CONTROL_CACHE.pop(key, None)
        raise
    finally:
        # A reward worker can fork while another thread owns this descriptor.
        # close() alone keeps the flock alive in that child; release the shared
        # open-file-description lock when the parent's critical section ends.
        fcntl.flock(fd, fcntl.LOCK_UN)
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
    with _locked(owner, write=True) as control:
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


def authorize_attempts(owner, requests):
    """Batch form of ``authorize_attempt``: ``requests`` is a list of
    (logical_sample, expected_attempt, new_attempt, expected_policy_version).
    All CAS checks pass before one durable write; otherwise control is unchanged."""
    if not requests:
        return []
    samples = [item[0] for item in requests]
    if len(samples) != len(set(samples)):
        raise StateError("duplicate sample in batch authorization")
    for _, _, _, version in requests:
        if type(version) is not int or version < 0:
            raise StateError("explicit nonnegative policy version required")
    with _locked(owner, write=True) as control:
        attempts = []
        for logical_sample, expected_attempt, new_attempt, version in requests:
            old = control["attempts"].get(logical_sample)
            if ((old["attempt"] if old else None) != expected_attempt or not new_attempt
                    or new_attempt == expected_attempt):
                raise StateError("attempt CAS failed")
            attempts.append({"sample": logical_sample, "attempt": new_attempt,
                             "epoch": owner.epoch, "owner_nonce": owner.nonce,
                             "versions": {"policy_version": version,
                                          "verifier_version": control["verifier_version"]}})
        for attempt in attempts:
            control["attempts"][attempt["sample"]] = attempt
            control["accepted"].pop(attempt["sample"], None)
        _write(_path(owner.root, "control.json"), control, immutable=False)
        return attempts


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
    with _locked(owner, write=True) as control:
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


def _stat_identity(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_mode, st.st_nlink)


def _identities(directory):
    """Enumerate the whole tree; content is not read."""
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise StateError("symlink in snapshot")
        if path.is_dir():
            continue
        found = path.lstat()
        if not stat.S_ISREG(found.st_mode):
            raise StateError("nonregular snapshot file")
        result[str(path.relative_to(directory))] = _stat_identity(found)
    return result


def prehash(directory, *, only=None):
    """Hash files ahead of commit; reused only while their stat identity is
    unchanged. ``only`` (name predicate) restricts which files are hashed."""
    identities = {}
    with perf_probe.span("state.prehash"):
        files = _inventory(directory, identities=identities, only=only)
    return {name: (identities[name], info) for name, info in files.items()}


CONTENT_HASHED = []  # observation only: directories whose files were content-hashed

# Large-file digest format (R_PERF_REDESIGN_PLAN v2.1 section 8.3). Files of at
# least CHUNK_THRESHOLD bytes are recorded as fixed-size SHA-256 chunks bound by
# ``chunked_digest``; smaller files keep the legacy whole-file ``sha256`` entry.
# Legacy manifests (large files with ``sha256``) stay verifiable read-only; a
# chain mixing both large-file formats is rejected.
DIGEST_SCHEMA = "sha256-chunked-v1"
CHUNK_THRESHOLD = 256 * 1024 * 1024
CHUNK_BYTES = 64 * 1024 * 1024
HASH_THREADS = 8
COMMIT_CHUNKED = True  # frozen with the source; False reproduces legacy manifests


def _chunk_count(size, chunk_bytes):
    return -(-size // chunk_bytes)


def _chunked_digest(size, chunk_bytes, chunks):
    return _hash(DIGEST_SCHEMA.encode() + b"\0" + size.to_bytes(8, "big")
                 + chunk_bytes.to_bytes(8, "big") + b"".join(bytes.fromhex(c) for c in chunks))


def _hash_chunks(fd, size):
    import concurrent.futures
    chunk_bytes = CHUNK_BYTES

    def one(index):
        offset, digest = index * chunk_bytes, hashlib.sha256()
        end = min(size, offset + chunk_bytes)
        while offset < end:
            data = os.pread(fd, min(8 * 1024 * 1024, end - offset), offset)
            if not data:
                raise StateError("snapshot truncated while hashing")
            digest.update(data)
            offset += len(data)
        return digest.hexdigest()

    with concurrent.futures.ThreadPoolExecutor(HASH_THREADS) as pool:
        chunks = list(pool.map(one, range(_chunk_count(size, chunk_bytes))))
    return {"size": size, "digest_schema": DIGEST_SCHEMA, "chunk_bytes": chunk_bytes,
            "chunk_sha256": chunks, "chunked_digest": _chunked_digest(size, chunk_bytes, chunks)}


def _entry_chunked(info):
    """True for a well-formed chunked entry, False for a whole-file entry."""
    if "digest_schema" not in info:
        if set(info) != {"size", "sha256"}:
            raise StateError("malformed file digest entry")
        return False
    if (set(info) != {"size", "digest_schema", "chunk_bytes", "chunk_sha256", "chunked_digest"}
            or info["digest_schema"] != DIGEST_SCHEMA or type(info["size"]) is not int
            or info["size"] < CHUNK_THRESHOLD or info["chunk_bytes"] != CHUNK_BYTES
            or not isinstance(info["chunk_sha256"], list)
            or len(info["chunk_sha256"]) != _chunk_count(info["size"], info["chunk_bytes"])
            or not all(_digest(c) for c in info["chunk_sha256"])
            or info["chunked_digest"] != _chunked_digest(info["size"], info["chunk_bytes"], info["chunk_sha256"])):
        raise StateError("malformed chunked digest entry")
    return True


def _manifest_schema(manifest):
    """Large-file digest format of a manifest: DIGEST_SCHEMA, "legacy" or None (no large file)."""
    found = set()
    for info in manifest["files"].values():
        if _entry_chunked(info):
            found.add(DIGEST_SCHEMA)
        elif info["size"] >= CHUNK_THRESHOLD:
            found.add("legacy")
    if len(found) > 1:
        raise StateError("mixed file digest formats")
    return found.pop() if found else None


def _inventory(directory, *, cache=None, identities=None, chunked=None, only=None):
    """``chunked`` is None for a new commit (COMMIT_CHUNKED decides) or a
    name -> bool map that reproduces an existing manifest's entry formats."""
    result = {}
    hashed = False
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise StateError("symlink in snapshot")
        if path.is_dir():
            continue
        if not stat.S_ISREG(path.lstat().st_mode):
            raise StateError("nonregular snapshot file")
        name = str(path.relative_to(directory))
        if only is not None and not only(name):
            continue
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            cached = None if cache is None else cache.get(name)
            if cached is not None and cached[0] == _stat_identity(before):
                result[name] = dict(cached[1])
                if identities is not None:
                    identities[name] = cached[0]
                continue
            large = before.st_size >= CHUNK_THRESHOLD
            if (COMMIT_CHUNKED and large) if chunked is None else chunked.get(name, False):
                info = _hash_chunks(stream.fileno(), before.st_size)
            else:
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                info = {"size": before.st_size, "sha256": digest.hexdigest()}
            os.fsync(stream.fileno())
            after = os.fstat(stream.fileno())
        if _stat_identity(before) != _stat_identity(after):
            raise StateError("snapshot changed while hashing")
        result[name] = info
        hashed = True
        if identities is not None:
            identities[name] = _stat_identity(after)
    if not result:
        raise StateError("empty snapshot")
    if hashed:
        CONTENT_HASHED.append(str(directory))
    return result


def _prune_marker(directory, token, manifest):
    """Deterministic pruning record; None when this generation was never pruned."""
    path = directory / "pruned.json"
    if not path.exists():
        return None
    marker = _read(_path(directory, "pruned.json"))
    removed = marker.get("removed")
    if (marker.get("schema") != 1 or marker.get("generation") != token["generation"]
            or marker.get("manifest_sha256") != token["manifest_sha256"]
            or not isinstance(removed, dict) or not removed
            or any(manifest["files"].get(name) != info for name, info in removed.items())):
        raise StateError("invalid pruning marker")
    return marker


def _check_files(directory, token, manifest):
    """Metadata check of a committed generation: file set and sizes, no content read."""
    marker = _prune_marker(directory, token, manifest)
    removed = set() if marker is None else set(marker["removed"])
    actual = _identities(_path(directory, "checkpoint"))
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not set(actual) <= set(expected):
        raise StateError("corrupt token or snapshot")
    for name, info in expected.items():
        if name not in actual:
            if name not in removed:
                raise StateError("corrupt token or snapshot")
        elif actual[name][2] != info["size"]:
            raise StateError("corrupt token or snapshot")
    return marker


def _validate_token(owner, gid, *, content=False):
    directory = _generation(owner, gid)
    token = _read(_path(directory, "token.json"))
    manifest = _read(_path(directory, "manifest.json"))
    if (token.get("generation") != gid or token.get("manifest_sha256") != _hash(_bytes(manifest))
            or token.get("intent_sha256") != _hash(_path(directory, "intent.json").read_bytes())):
        raise StateError("corrupt token or snapshot")
    marker = _check_files(directory, token, manifest)
    if content:
        # Only a generation that recovery will actually load is content-hashed.
        chunked = {name: _entry_chunked(info) for name, info in manifest["files"].items()}
        if marker is not None or manifest["files"] != _inventory(_path(directory, "checkpoint"), chunked=chunked):
            raise StateError("corrupt token or snapshot")
    return token, manifest


def _chain(tokens):
    head, order = None, []
    remaining = dict(tokens)
    while remaining:
        children = [gid for gid, (token, _) in remaining.items() if token["parent"] == head]
        if len(children) != 1:
            raise StateError("forked or disconnected committed chain")
        gid = children[0]
        token, _ = remaining.pop(gid)
        head = {"generation": gid, "token_sha256": _hash(_bytes(token))}
        order.append(gid)
    return head, order


def _head(owner, control, *, verify_head_content=False):
    """Committed chain from tokens. Ancestors are checked by metadata only;
    their checkpoint content is never loaded by recovery."""
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
    if len({_manifest_schema(m) for _, m in tokens.values()} - {None}) > 1:
        raise StateError("mixed file digest formats in committed chain")
    head, _ = _chain(tokens)
    if verify_head_content and head is not None:
        _validate_token(owner, head["generation"], content=True)
    return head, tokens


def _unresolved(owner):
    root = _path(owner.root, "generations")
    return [] if not root.exists() else sorted(item.name for item in root.iterdir()
        if (item / "intent.json").exists() and not (item / "token.json").exists()
        and not (item / "abandoned.json").exists())


def _pending_parent(parent):
    """Pipelined (lag-1) parent form: an uncommitted predecessor bound by the
    SHA-256 of its intent.json bytes (R_PERF_PHASE2_PLAN section 1.1)."""
    return isinstance(parent, dict) and set(parent) == {"generation", "intent_sha256"}


def _parent_ok(parent, head, tokens):
    """Commit-time CAS for either parent form against the committed head."""
    if _pending_parent(parent):
        return (head is not None and head["generation"] == parent["generation"]
                and tokens[head["generation"]][0]["intent_sha256"] == parent["intent_sha256"])
    return parent == head


def prepare_generation(owner, intent, data_snapshot):
    with _locked(owner) as control:
        head, tokens = _head(owner, control)
        directory = _path(owner.root, "generations")
        _directory(directory)
        unresolved = _unresolved(owner)
        if len(unresolved) > 1:
            raise StateError("unresolved generation")
        # Base the new generation on the committed head, or (lag-1) on the one
        # uncommitted predecessor P of this execution whose parent is that head.
        base_data = None if head is None else tokens[head["generation"]][1]["data"]
        pending_updates = set()
        expected_parent = head
        if unresolved:
            gid_p = unresolved[0]
            raw = _path(_generation(owner, gid_p), "intent.json").read_bytes()
            prior = _parse(raw, Path("intent.json"))
            if (not _parent_ok(prior.get("parent"), head, tokens)
                    or (prior.get("execution_epoch"), prior.get("execution_owner")) != (owner.epoch, owner.nonce)):
                raise StateError("unresolved generation")
            expected_parent = {"generation": gid_p, "intent_sha256": _hash(raw)}
            base_data = prior["data"]
            pending_updates = {u["logical_update_id"] for u in prior["updates"]}
        if intent.get("parent") != expected_parent or intent.get("ack_capability") != "none":
            raise StateError("parent mismatch or unsupported ACK")
        if intent.get("config_sha256") != control["config_sha256"]:
            raise StateError("configuration mismatch")
        ranks = intent.get("expected_ranks", [])
        if not ranks or len(ranks) != len(set(ranks)):
            raise StateError("invalid expected ranks")
        consumed = _sets(data_snapshot)
        previous = set() if base_data is None else set(base_data["consumed"])
        if base_data is not None:
            if (data_snapshot["source_sha256"] != base_data["source_sha256"]
                    or data_snapshot["drawn"][:len(base_data["drawn"])] != base_data["drawn"]):
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
        prior_updates |= pending_updates
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


def _complete(owner, gid, *, cache=None, identities=None):
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
    files = _inventory(_path(directory, "checkpoint"), cache=cache, identities=identities)
    referenced = set()
    for mapping in components.values():
        if set(mapping) != set(intent["expected_ranks"]):
            raise StateError("component rank coverage mismatch")
        for paths in mapping.values():
            if not paths:
                raise StateError("empty component")
            for path in paths:
                _path(directory / "checkpoint", path[:-1] if path.endswith('/') else path)
                # Native DCP chooses shard filenames during save. A frozen
                # directory prefix maps its complete finalized inventory.
                if path.endswith('/'):
                    matches = {name for name in files if name.startswith(path)}
                    if not matches:
                        raise StateError("component directory empty")
                    referenced.update(matches)
                    continue
                if path not in files:
                    raise StateError("component file missing")
                referenced.add(path)
    if referenced != set(files):
        raise StateError("unmapped checkpoint files")
    _sets(intent["data"])
    return intent, receipts, files


def commit_generation(owner, generation, *, prehashed=None):
    with _locked(owner, write=True) as control:
        head, tokens = _head(owner, control)
        if generation in tokens:
            return tokens[generation][0]
        directory = _generation(owner, generation)
        if _path(directory, "abandoned.json").exists():
            raise StateError("abandoned generation")
        identities = {}
        intent, receipts, files = _complete(owner, generation, cache=prehashed, identities=identities)
        if not _parent_ok(intent["parent"], head, tokens):
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
        # Content was hashed once above; re-enumerate the whole tree and require
        # identical stat identity. Same-size rewrites within one timestamp tick
        # are a disclosed limitation (writer is already joined and receipted).
        if _identities(checkpoint) != identities:
            raise StateError("checkpoint changed after finalization")
        manifest = dict(intent, files=files, receipts=receipts)
        schemas = {_manifest_schema(m) for _, m in tokens.values()} | {_manifest_schema(manifest)}
        if len(schemas - {None}) > 1:
            raise StateError("mixed file digest formats in committed chain")
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
        head, tokens = _head(owner, control, verify_head_content=True)
        root = _path(owner.root, "generations")
        candidates = _unresolved(owner)
        intents = {gid: _read(_path(_generation(owner, gid), "intent.json")) for gid in candidates}
        # At most a lag-1 chain: c1 on the committed head (either parent form),
        # c2 pending on c1.
        on_head = [gid for gid in candidates if _parent_ok(intents[gid]["parent"], head, tokens)]
        ordered = on_head + [gid for gid in candidates if gid not in on_head]
        if len(ordered) > 2 or (len(ordered) == 2 and not (
                len(on_head) == 1 and _pending_parent(intents[ordered[1]]["parent"])
                and intents[ordered[1]]["parent"]["generation"] == ordered[0])):
            raise StateError("multiple unresolved candidates")
    promoted = []
    reason = "verified committed chain"
    failed = False
    for gid in ordered:
        own = "predecessor not promoted"
        if not failed:
            try:
                with _locked(owner):
                    identities = {}
                    _, _, files = _complete(owner, gid, identities=identities)
                prehashed = {name: (identities[name], info) for name, info in files.items()}
                token = commit_generation(owner, gid, prehashed=prehashed)
                head = {"generation": gid, "token_sha256": _hash(_bytes(token))}
                promoted.append(gid)
                reason = "completed durable candidate"
                continue
            except StateError as exc:
                failed = True
                own = reason = "incomplete candidate: " + str(exc)
        with _locked(owner):
            _write(_path(_generation(owner, gid), "abandoned.json"), {"epoch": owner.epoch, "reason": own})
    # Preserve uncommitted consumption obligations in R's own intents. They are
    # recovery inputs, never evidence that replay has actually completed.
    rollback_intents = [] if not root.exists() else [str(item / "intent.json") for item in sorted(root.iterdir())
        if (item / "abandoned.json").exists() and not (item / "token.json").exists()]
    if head is None:
        return {"generation": None, "checkpoint": None, "pending": [],
                "rollback_intents": rollback_intents, "reason": reason, "promoted": promoted}
    directory = _generation(owner, head["generation"])
    # Content was verified above (or hashed by the promotion commit).
    _, manifest = _validate_token(owner, head["generation"])
    return {"generation": head["generation"], "checkpoint": str(directory / "checkpoint"),
            "pending": manifest["data"]["pending"], "rollback_intents": rollback_intents, "reason": reason,
            "promoted": promoted}


def pin_generation(owner, gid, reason):
    """Evidence pin (FT-v1 section 9): a pinned generation is never pruned."""
    if reason not in ("recovery_loaded", "first_commit_after_recovery", "terminal"):
        raise StateError("unknown pin reason")
    with _locked(owner) as control:
        _, tokens = _head(owner, control)
        if gid not in tokens:
            raise StateError("pin requires committed generation")
        _write(_path(owner.root, "pins/" + reason + "-" + gid + ".json"),
               {"schema": 1, "generation": gid, "reason": reason})


def pinned(owner):
    root = owner.root / "pins"
    result = set()
    if root.exists():
        for path in root.iterdir():
            if path.name.startswith(".tmp-"):
                continue
            record = _read(_path(root, path.name))
            if path.name != record["reason"] + "-" + record["generation"] + ".json":
                raise StateError("invalid pin record")
            result.add(record["generation"])
    return result


def prune_generations(owner, prunable, *, keep=2):
    """Drop bulk files of committed generations older than the last ``keep``
    on the token-derived chain. The marker is deterministic and durable before
    any deletion; a crash between them is resumed without rewriting it."""
    removed_bytes = 0
    with _locked(owner) as control:
        _, tokens = _head(owner, control)
        _, order = _chain(tokens)
        keep_set = set(order[-keep:]) | pinned(owner)
        for gid in order[:-keep]:
            if gid in keep_set:
                continue
            token, manifest = tokens[gid]
            names = sorted(name for name in manifest["files"] if prunable(name))
            if not names:
                continue
            directory = _generation(owner, gid)
            checkpoint = _path(directory, "checkpoint")
            present = [name for name in names if (checkpoint / name).exists()]
            if not present and (directory / "pruned.json").exists():
                continue
            marker = {"schema": 1, "generation": gid, "manifest_sha256": token["manifest_sha256"],
                      "removed": {name: manifest["files"][name] for name in names}}
            _write(_path(directory, "pruned.json"), marker)
            for name in present:
                path = _path(checkpoint, name)
                removed_bytes += path.lstat().st_size
                path.unlink()
                _sync_dir(path.parent)
    return removed_bytes
