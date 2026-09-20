"""CPU contract files and real local processes; not backend/GPU tests."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.ft import state


HASH = "a" * 64
COMPONENTS = ("model", "optimizer_master", "optimizer_moments", "optimizer_step", "scheduler",
              "rng_python", "rng_numpy", "rng_torch_cpu", "rng_device", "rng_tracker", "policy")


def new_owner(root, **kwargs):
    kwargs.setdefault("verifier_version", "fixture-v1")
    return state.acquire_owner(root, -1, run_nonce="cpu-test", config_sha256=HASH, **kwargs)


def fixture(owner, number=0, parent=None):
    key = f"group{number}:0"
    attempt = state.authorize_attempt(owner, key, None, "attempt1", expected_policy_version=number)
    receipt = state.accept_result(owner, attempt, {
        "response_sha256": HASH, "reward_sha256": HASH, "tensor_input_sha256": HASH,
        "policy_version": number, "verifier_version": "fixture-v1"})
    consumed = []
    if parent:
        previous = json.loads((owner.root / "generations" / parent["generation"] / "manifest.json").read_text())
        consumed = previous["data"]["consumed"]
    consumed = consumed + [key]
    data = {"source_sha256": HASH, "epoch": 0, "shuffle_state": {"seed": 1},
            "cursor": len(consumed), "drawn": consumed, "consumed": consumed, "pending": []}
    prompt = {"messages": [{"role": "user", "content": f"fixture {number}"}], "label": "1"}
    intent = {"parent": parent, "ack_capability": "none", "config_sha256": HASH,
              "expected_ranks": ["actor:0"], "data_snapshot_id": f"cut{number}",
              "components": {name: {"actor:0": [name + ".bin"]} for name in COMPONENTS},
              "updates": [{"logical_update_id": f"u{number}", "physical_update_id": f"p{number}",
                           "train_input_sha256": HASH, "groups": [{"logical_group_id": f"group{number}",
                           "prompt": prompt, "prompt_sha256": hashlib.sha256(state._bytes(prompt)).hexdigest(),
                           "k": 1, "samples": [{"sample_index": 0, "sample": key, "receipt": receipt}]}]}]}
    return intent, data


def complete(owner, number=0, parent=None, finalize=True):
    intent, data = fixture(owner, number, parent)
    gid = state.prepare_generation(owner, intent, data)
    directory = owner.root / "generations" / gid / "checkpoint"
    directory.mkdir()
    for name in COMPONENTS:
        (directory / (name + ".bin")).write_bytes(name.encode() + b"x" * 10000)
    state.record_evidence(owner, gid, {"kind": "optimizer", "snapshot_id": f"cut{number}",
        "physical_updates": [f"p{number}"], "successful": True, "scheduler_applied": True})
    if finalize:
        state.record_evidence(owner, gid, {"kind": "finalize", "snapshot_id": f"cut{number}",
            "rank": "actor:0", "writer_closed": True})
    return gid


def child_prepare(root, mode, participant=None):
    source = """
import json, sys
from pathlib import Path
from tests.ft.test_state import new_owner, complete
from scripts.ft import state
root, mode, participant = sys.argv[1:]
workers = [] if participant == 'null' else [json.loads(participant)]
with new_owner(root, participants=workers) as owner:
    gid = complete(owner, finalize=mode != 'incomplete')
    if mode == 'commit':
        state.commit_generation(owner, gid)
    print(gid, flush=True)
"""
    result = subprocess.run([sys.executable, "-c", source, str(root), mode, json.dumps(participant)],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


class StateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "run"

    def owner(self):
        owner = new_owner(self.root)
        self.addCleanup(owner.close)
        return owner

    def resume(self):
        control = json.loads((self.root / "control.json").read_text())
        owner = state.acquire_owner(self.root, control["epoch"], control["processes"])
        self.addCleanup(owner.close)
        return owner

    def test_commit_is_idempotent_and_retained_chain(self):
        owner = self.owner()
        gid = complete(owner)
        token = state.commit_generation(owner, gid)
        self.assertEqual(token, state.commit_generation(owner, gid))
        parent = {"generation": gid, "token_sha256": hashlib.sha256(state._bytes(token)).hexdigest()}
        second = complete(owner, 1, parent)
        state.commit_generation(owner, second)
        self.assertEqual(state.select_recovery(owner)["generation"], second)
        self.assertFalse((owner.root / "generations" / gid / "snapshot").exists())

    def test_new_process_recovers_committed_state(self):
        gid = child_prepare(self.root, "commit")
        owner = self.resume()
        result = state.select_recovery(owner)
        self.assertEqual(result["generation"], gid)
        self.assertTrue(Path(result["checkpoint"]).is_dir())

    def test_complete_candidate_promotes_and_preserves_execution_epoch(self):
        gid = child_prepare(self.root, "complete")
        owner = self.resume()
        result = state.select_recovery(owner)
        self.assertEqual(result["generation"], gid)
        token = json.loads((owner.root / "generations" / gid / "token.json").read_text())
        self.assertEqual((token["execution_epoch"], token["commit_epoch"]), (0, 1))

    def test_incomplete_candidate_abandoned_without_false_commit(self):
        gid = child_prepare(self.root, "incomplete")
        owner = self.resume()
        recovered = state.select_recovery(owner)
        self.assertIsNone(recovered["generation"])
        self.assertEqual(len(recovered["rollback_intents"]), 1)
        self.assertTrue((owner.root / "generations" / gid / "abandoned.json").exists())
        with self.assertRaises(state.StateError):
            state.commit_generation(owner, gid)

    def test_middle_content_corruption_rejected(self):
        owner = self.owner()
        gid = complete(owner)
        state.commit_generation(owner, gid)
        path = owner.root / "generations" / gid / "checkpoint/model.bin"
        with path.open("r+b") as stream:
            stream.seek(5000)
            stream.write(b"z")
        with self.assertRaises(state.StateError):
            state.select_recovery(owner)

    def test_token_missing_file_rejected(self):
        owner = self.owner()
        gid = complete(owner)
        state.commit_generation(owner, gid)
        (owner.root / "generations" / gid / "checkpoint/optimizer_step.bin").unlink()
        with self.assertRaises(state.StateError):
            state.select_recovery(owner)

    def test_missing_rank_and_missing_component_rejected(self):
        owner = self.owner()
        gid = complete(owner, finalize=False)
        with self.assertRaises(state.StateError):
            state.commit_generation(owner, gid)
        state.record_evidence(owner, gid, {"kind": "finalize", "snapshot_id": "cut0",
            "rank": "actor:0", "writer_closed": True})
        (owner.root / "generations" / gid / "checkpoint/optimizer_master.bin").unlink()
        with self.assertRaises(state.StateError):
            state.commit_generation(owner, gid)

    def test_cursor_cannot_skip_pending_or_consumption_mapping(self):
        owner = self.owner()
        intent, data = fixture(owner)
        broken = copy.deepcopy(data)
        broken["drawn"].append("lost:0")
        broken["cursor"] += 1
        with self.assertRaises(state.StateError):
            state.prepare_generation(owner, intent, broken)
        broken = copy.deepcopy(data)
        broken["consumed"] = []
        broken["pending"] = [{"sample": "group0:0", "action": "regenerate", "prompt_sha256": HASH}]
        with self.assertRaises(state.StateError):
            state.prepare_generation(owner, intent, broken)

    def test_old_attempt_cannot_win_by_arriving_first(self):
        owner = self.owner()
        old = state.authorize_attempt(owner, "sample", None, "old", expected_policy_version=0)
        new = state.authorize_attempt(owner, "sample", "old", "new", expected_policy_version=0)
        payload = {"response_sha256": HASH, "reward_sha256": HASH, "tensor_input_sha256": HASH,
                   "policy_version": 0, "verifier_version": "fixture-v1"}
        with self.assertRaises(state.StateError):
            state.accept_result(owner, old, payload)
        receipt = state.accept_result(owner, new, payload)
        self.assertEqual(receipt, state.accept_result(owner, new, payload))
        with self.assertRaises(state.StateError):
            state.accept_result(owner, new, dict(payload, reward_sha256="b" * 64))

    def test_prepared_attempt_superseded_cannot_commit(self):
        owner = self.owner()
        gid = complete(owner)
        state.authorize_attempt(owner, "group0:0", "attempt1", "attempt2", expected_policy_version=0)
        with self.assertRaises(state.StateError):
            state.commit_generation(owner, gid)

    def test_wrong_verifier_or_policy_cannot_claim_authorized_attempt(self):
        owner = self.owner()
        attempt = state.authorize_attempt(owner, "sample", None, "a1", expected_policy_version=3)
        good = {"response_sha256": HASH, "tensor_input_sha256": HASH,
                "reward_sha256": hashlib.sha256(b"0.0").hexdigest(),
                "policy_version": 3, "verifier_version": "fixture-v1"}
        different = dict(good, verifier_version="fixture-v2",
                         reward_sha256=hashlib.sha256(b"1.0").hexdigest())
        for payload in (different, dict(good, verifier_version="fixture-v2"), dict(good, policy_version=2)):
            with self.subTest(payload=payload):
                with self.assertRaises(state.StateError):
                    state.accept_result(owner, attempt, payload)
        self.assertEqual(state.accept_result(owner, attempt, good)["payload"], good)

    def test_policy_authorization_required_and_distinct_authorized_policies_allowed(self):
        owner = self.owner()
        with self.assertRaises(TypeError):
            state.authorize_attempt(owner, "sample", None, "missing-policy")
        intent, data = fixture(owner)
        attempt = state.authorize_attempt(owner, "group0:1", None, "a1", expected_policy_version=1)
        receipt = state.accept_result(owner, attempt, {"response_sha256": HASH,
            "reward_sha256": HASH, "tensor_input_sha256": HASH,
            "policy_version": 1, "verifier_version": "fixture-v1"})
        group = intent["updates"][0]["groups"][0]
        group["k"] = 2
        group["samples"].append({"sample": "group0:1", "sample_index": 1, "receipt": receipt})
        data["drawn"] = ["group0:0", "group0:1"]
        data["consumed"] = list(data["drawn"])
        data["cursor"] = 2
        self.assertTrue(state.prepare_generation(owner, intent, data).startswith("g-"))

    def test_prepare_rechecks_receipt_declared_version(self):
        owner = self.owner()
        intent, data = fixture(owner)
        intent["updates"][0]["groups"][0]["samples"][0]["receipt"]["payload"]["verifier_version"] = "fixture-v2"
        with self.assertRaisesRegex(state.StateError, "authorization"):
            state.prepare_generation(owner, intent, data)

    def test_commit_rechecks_durable_intent_version(self):
        owner = self.owner()
        gid = complete(owner)
        path = owner.root / "generations" / gid / "intent.json"
        intent = json.loads(path.read_text())
        intent["updates"][0]["groups"][0]["samples"][0]["receipt"]["payload"]["policy_version"] = 9
        # Corrupt the durable input before publication; no token may be created.
        path.write_text(json.dumps(intent))
        with self.assertRaisesRegex(state.StateError, "authorization"):
            state.commit_generation(owner, gid)
        self.assertFalse((path.parent / "token.json").exists())

    def test_committed_evidence_only_allows_identical_existing_receipt(self):
        owner = self.owner()
        gid = complete(owner)
        state.commit_generation(owner, gid)
        directory = owner.root / "generations" / gid
        path = directory / "receipts/optimizer.json"
        receipt = json.loads(path.read_text())
        before = path.stat().st_mtime_ns
        self.assertEqual(state.record_evidence(owner, gid, receipt), receipt)
        self.assertEqual(path.stat().st_mtime_ns, before)
        with self.assertRaisesRegex(state.StateError, "immutable"):
            state.record_evidence(owner, gid, dict(receipt, unexpected="new metadata"))
        path.unlink()
        with self.assertRaisesRegex(state.StateError, "immutable"):
            state.record_evidence(owner, gid, receipt)
        self.assertFalse(path.exists())

    def test_actual_second_process_cannot_take_owner(self):
        self.owner()
        source = "from scripts.ft.state import acquire_owner; import sys; acquire_owner(sys.argv[1], 0, True)"
        result = subprocess.run([sys.executable, "-c", source, str(self.root)], capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_live_registered_worker_blocks_takeover(self):
        worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            child_prepare(self.root, "commit", state.process_identity(worker.pid))
            control = json.loads((self.root / "control.json").read_text())
            with self.assertRaises(state.StateError):
                state.acquire_owner(self.root, 0, True)
            with self.assertRaises(state.StateError):
                state.acquire_owner(self.root, 0, control["processes"])
            worker.kill()
            worker.wait(timeout=5)
            self.resume()
        finally:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=5)

    def test_closed_owner_cannot_publish(self):
        owner = self.owner()
        gid = complete(owner)
        owner.close()
        with self.assertRaises(state.StateError):
            state.commit_generation(owner, gid)

    def test_head_hint_crash_does_not_lose_token(self):
        owner = self.owner()
        gid = complete(owner)
        original = state._write
        def fail_head(path, value, immutable=True):
            if path.name == "control.json" and value.get("head"):
                raise OSError("injected head write crash")
            return original(path, value, immutable)
        with patch.object(state, "_write", side_effect=fail_head):
            with self.assertRaises(OSError):
                state.commit_generation(owner, gid)
        self.assertEqual(state.select_recovery(owner)["generation"], gid)

    def test_sigkill_before_and_after_token_publication(self):
        source = """
import os, signal, sys
from tests.ft.test_state import new_owner, complete
from scripts.ft import state
root, cut = sys.argv[1:]
owner = new_owner(root)
gid = complete(owner)
original = state._write
def interrupted(path, value, immutable=True):
    original(path, value, immutable)
    if path.name == cut:
        os.kill(os.getpid(), signal.SIGKILL)
state._write = interrupted
state.commit_generation(owner, gid)
"""
        for cut in ("manifest.json", "token.json"):
            with self.subTest(cut=cut):
                root = Path(self.temporary.name) / cut
                result = subprocess.run([sys.executable, "-c", source, str(root), cut], capture_output=True)
                self.assertEqual(result.returncode, -9, result.stderr)
                control = json.loads((root / "control.json").read_text())
                with state.acquire_owner(root, 0, control["processes"]) as owner:
                    recovered = state.select_recovery(owner)
                    self.assertIsNotNone(recovered["generation"])
                    self.assertEqual(len(list((root / "generations").glob("*/token.json"))), 1)

    def test_unsafe_paths_and_gpu_scope_rejected(self):
        for relative in ("/tmp/escape", "../escape", "a/../b", "a//b"):
            with self.assertRaises(state.StateError):
                state._path(self.root, relative)
        with self.assertRaises(state.StateError):
            new_owner(self.root, scope="gpu")
        owner = self.owner()
        gid = complete(owner)
        path = owner.root / "generations" / gid / "checkpoint/model.bin"
        path.unlink()
        path.symlink_to("optimizer_step.bin")
        with self.assertRaises(state.StateError):
            state.commit_generation(owner, gid)


if __name__ == "__main__":
    unittest.main()
