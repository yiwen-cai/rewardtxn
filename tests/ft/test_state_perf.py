"""R perf redesign D1: control cache and batch authorization contracts (CPU)."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.ft import state
from test_state import HASH, new_owner


def payload(version=0):
    return {"response_sha256": HASH, "reward_sha256": HASH, "tensor_input_sha256": HASH,
            "policy_version": version, "verifier_version": "fixture-v1"}


class ControlCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.owner = new_owner(Path(self.tmp.name) / "run")

    def tearDown(self):
        self.owner.close()
        self.tmp.cleanup()

    def read(self):
        with state._locked(self.owner) as control:
            return json.loads(json.dumps(control))

    def test_failed_accept_write_is_not_visible(self):
        attempt = state.authorize_attempt(self.owner, "g:0", None, "a1", expected_policy_version=0)
        self.read()  # warm the cache
        original = state._write
        def fail(path, value, immutable=True):
            if path.name == "control.json":
                raise OSError("injected control write failure")
            return original(path, value, immutable)
        with patch.object(state, "_write", side_effect=fail):
            with self.assertRaises(OSError):
                state.accept_result(self.owner, attempt, payload())
        self.assertNotIn("g:0", self.read()["accepted"])
        # A failed write never poisons later readers or writers.
        state.accept_result(self.owner, attempt, payload())
        self.assertIn("g:0", self.read()["accepted"])

    def test_failed_authorize_write_is_not_visible(self):
        self.read()
        with patch.object(state, "_write", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                state.authorize_attempt(self.owner, "g:0", None, "a1", expected_policy_version=0)
        self.assertNotIn("g:0", self.read()["attempts"])

    def test_same_size_external_rewrite_is_detected(self):
        before = self.read()
        path = self.owner.root / "control.json"
        raw = path.read_bytes()
        changed = raw.replace(b'"cpu_contract"', b'"cpu_contracX"')
        self.assertEqual(len(raw), len(changed))
        stat = path.stat()
        path.write_bytes(changed)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(self.read()["scope"], "cpu_contracX")
        self.assertEqual(before["scope"], "cpu_contract")

    def test_stale_owner_rejected_with_cached_content(self):
        self.read()
        path = self.owner.root / "control.json"
        control = json.loads(path.read_bytes())
        control["owner_nonce"] = "0" * 32
        path.write_bytes(json.dumps(control).encode())
        with self.assertRaises(state.StateError):
            self.read()

    def test_exception_inside_read_section_drops_cache(self):
        self.read()
        with self.assertRaises(RuntimeError):
            with state._locked(self.owner) as control:
                control["accepted"]["poison"] = {}
                raise RuntimeError("caller failure")
        self.assertNotIn("poison", self.read()["accepted"])


class BatchAuthorize(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.owner = new_owner(Path(self.tmp.name) / "run")

    def tearDown(self):
        self.owner.close()
        self.tmp.cleanup()

    def test_matches_single_authorization(self):
        attempts = state.authorize_attempts(self.owner, [("g:0", None, "a0", 3), ("g:1", None, "a1", 3)])
        with state._locked(self.owner) as control:
            self.assertEqual([control["attempts"]["g:0"], control["attempts"]["g:1"]], attempts)
        single = state.authorize_attempt(self.owner, "h:0", None, "b0", expected_policy_version=3)
        self.assertEqual(set(single), set(attempts[0]))
        self.assertEqual(single["versions"], attempts[0]["versions"])

    def test_all_or_nothing(self):
        first = state.authorize_attempt(self.owner, "g:1", None, "old", expected_policy_version=0)
        state.accept_result(self.owner, first, payload())
        raw = (self.owner.root / "control.json").read_bytes()
        with self.assertRaises(state.StateError):
            # g:1 expects no prior attempt but one exists: whole batch rejected.
            state.authorize_attempts(self.owner, [("g:0", None, "a0", 0), ("g:1", None, "a1", 0)])
        self.assertEqual((self.owner.root / "control.json").read_bytes(), raw)
        with state._locked(self.owner) as control:
            self.assertNotIn("g:0", control["attempts"])
            self.assertIn("g:1", control["accepted"])

    def test_replacement_clears_receipt_and_rejects_duplicates(self):
        first = state.authorize_attempt(self.owner, "g:0", None, "old", expected_policy_version=0)
        state.accept_result(self.owner, first, payload())
        state.authorize_attempts(self.owner, [("g:0", "old", "new", 1)])
        with state._locked(self.owner) as control:
            self.assertNotIn("g:0", control["accepted"])
            self.assertEqual(control["attempts"]["g:0"]["attempt"], "new")
        with self.assertRaises(state.StateError):
            state.authorize_attempts(self.owner, [("x:0", None, "a", 0), ("x:0", None, "b", 0)])
        with self.assertRaises(state.StateError):
            state.authorize_attempts(self.owner, [("x:0", None, "a", -1)])


if __name__ == "__main__":
    unittest.main()


class ChunkedDigest(unittest.TestCase):
    """Small thresholds so fixture files (~10 kB) use the chunked format."""

    def setUp(self):
        from test_state import complete
        self.complete = complete
        self.tmp = tempfile.TemporaryDirectory()
        self.owner = new_owner(Path(self.tmp.name) / "run")
        for name, value in (("CHUNK_THRESHOLD", 4096), ("CHUNK_BYTES", 3000)):
            patcher = patch.object(state, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.owner.close()
        self.tmp.cleanup()

    def commit(self, number, parent=None):
        gid = self.complete(self.owner, number, parent)
        token = state.commit_generation(self.owner, gid)
        return gid, {"generation": gid, "token_sha256": state._hash(state._bytes(token))}

    def manifest(self, gid):
        return json.loads((self.owner.root / "generations" / gid / "manifest.json").read_text())

    def model(self, gid):
        return self.owner.root / "generations" / gid / "checkpoint" / "model.bin"

    def test_entry_format_and_recovery(self):
        gid, _ = self.commit(0)
        info = self.manifest(gid)["files"]["model.bin"]
        raw = self.model(gid).read_bytes()
        self.assertEqual(info["digest_schema"], state.DIGEST_SCHEMA)
        self.assertEqual(info["chunk_sha256"], [state._hash(raw[i:i + 3000]) for i in range(0, len(raw), 3000)])
        self.assertEqual(state.select_recovery(self.owner)["generation"], gid)

    def tamper(self, gid, mutate):
        path = self.model(gid)
        data = bytearray(path.read_bytes())
        path.unlink()
        path.write_bytes(bytes(mutate(data)))
        with self.assertRaises(state.StateError):
            state.select_recovery(self.owner)

    def test_flip_byte_in_any_chunk_rejected(self):
        gid, _ = self.commit(0)
        def flip(data):
            data[len(data) - 2] ^= 1  # last (short) chunk
            return data
        self.tamper(gid, flip)

    def test_truncation_rejected(self):
        gid, _ = self.commit(0)
        self.tamper(gid, lambda data: data[:-1])

    def test_swapped_chunks_rejected(self):
        gid, _ = self.commit(0)
        info = dict(self.manifest(gid)["files"]["model.bin"])
        info["chunk_sha256"] = list(reversed(info["chunk_sha256"]))
        with self.assertRaises(state.StateError):
            state._entry_chunked(info)
        self.tamper(gid, lambda data: data[3000:6000] + data[:3000] + data[6000:])

    def test_legacy_manifest_verifies_and_mixed_chain_rejected(self):
        with patch.object(state, "COMMIT_CHUNKED", False):
            gid, parent = self.commit(0)
        self.assertEqual(set(self.manifest(gid)["files"]["model.bin"]), {"size", "sha256"})
        self.assertEqual(state.select_recovery(self.owner)["generation"], gid)
        second = self.complete(self.owner, 1, parent)
        with self.assertRaises(state.StateError):
            state.commit_generation(self.owner, second)

    def test_offline_checker_is_independent_and_agrees(self):
        import hashlib
        from offline_generation import check_files
        gid, _ = self.commit(0)
        directory = self.owner.root / "generations" / gid
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        token = json.loads((directory / "token.json").read_text())
        self.assertGreater(check_files(directory, token, self.manifest(gid), sha)[0], 0)
        path = self.model(gid)
        data = bytearray(path.read_bytes()); data[5] ^= 1
        path.unlink(); path.write_bytes(bytes(data))
        with self.assertRaises(AssertionError):
            check_files(directory, token, self.manifest(gid), sha)


class PipelinedParent(unittest.TestCase):
    """Phase 2 lag-1 protocol: one uncommitted predecessor P."""

    def setUp(self):
        from test_state import complete, fixture, COMPONENTS
        self.complete_fn, self.fixture, self.components = complete, fixture, COMPONENTS
        self.tmp = tempfile.TemporaryDirectory()
        self.owner = new_owner(Path(self.tmp.name) / "run")

    def tearDown(self):
        self.owner.close()
        self.tmp.cleanup()

    def gdir(self, gid):
        return self.owner.root / "generations" / gid

    def pending_parent(self, gid):
        return {"generation": gid, "intent_sha256": state._hash((self.gdir(gid) / "intent.json").read_bytes())}

    def prepare_on(self, number, base_gid, parent):
        intent, data = self.fixture(self.owner, number, None)
        base = json.loads((self.gdir(base_gid) / "intent.json").read_text())["data"]
        key = f"group{number}:0"
        consumed = base["consumed"] + [key]
        data.update(drawn=consumed, consumed=consumed, cursor=len(consumed))
        intent["parent"] = parent
        return state.prepare_generation(self.owner, intent, data)

    def finish(self, gid, number):
        directory = self.gdir(gid) / "checkpoint"
        directory.mkdir()
        for name in self.components:
            (directory / (name + ".bin")).write_bytes(name.encode() + b"x" * 1000)
        state.record_evidence(self.owner, gid, {"kind": "optimizer", "snapshot_id": f"cut{number}",
            "physical_updates": [f"p{number}"], "successful": True, "scheduler_applied": True})
        state.record_evidence(self.owner, gid, {"kind": "finalize", "snapshot_id": f"cut{number}",
            "rank": "actor:0", "writer_closed": True})

    def g0(self):
        return self.complete_fn(self.owner, 0, None, finalize=True)

    def test_prepare_and_commit_on_pending_predecessor(self):
        g0 = self.g0()
        g1 = self.prepare_on(1, g0, self.pending_parent(g0))
        self.finish(g1, 1)
        with self.assertRaises(state.StateError):  # predecessor not committed yet
            state.commit_generation(self.owner, g1)
        state.commit_generation(self.owner, g0)
        token = state.commit_generation(self.owner, g1)
        self.assertEqual(token["parent"]["generation"], g0)
        self.assertEqual(state.select_recovery(self.owner)["generation"], g1)

    def test_wrong_parent_forms_rejected(self):
        g0 = self.g0()
        with self.assertRaises(state.StateError):  # head form while P unresolved
            self.prepare_on(1, g0, None)
        bad = dict(self.pending_parent(g0), intent_sha256="0" * 64)
        with self.assertRaises(state.StateError):
            self.prepare_on(1, g0, bad)

    def test_two_unresolved_rejects_third(self):
        g0 = self.g0()
        g1 = self.prepare_on(1, g0, self.pending_parent(g0))
        with self.assertRaises(state.StateError):
            self.prepare_on(2, g1, self.pending_parent(g1))

    def test_lineage_and_duplicate_samples_checked_against_predecessor(self):
        g0 = self.g0()
        intent, data = self.fixture(self.owner, 1, None)
        intent["parent"] = self.pending_parent(g0)
        data.update(drawn=["group1:0"], consumed=["group1:0"], cursor=1)  # drops P's consumption
        with self.assertRaises(state.StateError):
            state.prepare_generation(self.owner, intent, data)

    def test_abandoned_predecessor_blocks_successor(self):
        g0 = self.g0()
        g1 = self.prepare_on(1, g0, self.pending_parent(g0))
        self.finish(g1, 1)
        (self.gdir(g0) / "checkpoint" / "model.bin").unlink()  # g0 incomplete
        result = state.select_recovery(self.owner)
        self.assertIsNone(result["generation"])
        self.assertTrue((self.gdir(g0) / "abandoned.json").exists())
        self.assertTrue((self.gdir(g1) / "abandoned.json").exists())
        self.assertEqual(len(result["rollback_intents"]), 2)

    def test_recovery_promotes_complete_predecessor_only(self):
        g0 = self.g0()
        g1 = self.prepare_on(1, g0, self.pending_parent(g0))  # intent only (F2 shape)
        result = state.select_recovery(self.owner)
        self.assertEqual((result["generation"], result["promoted"]), (g0, [g0]))
        self.assertTrue((self.gdir(g1) / "abandoned.json").exists())

    def test_recovery_promotes_both_when_complete(self):
        g0 = self.g0()
        g1 = self.prepare_on(1, g0, self.pending_parent(g0))
        self.finish(g1, 1)
        result = state.select_recovery(self.owner)
        self.assertEqual(result["promoted"], [g0, g1])
        self.assertEqual(result["generation"], g1)

    def test_recovery_complete_c1_incomplete_c2(self):
        g0 = self.g0()
        g1 = self.prepare_on(1, g0, self.pending_parent(g0))
        (self.gdir(g1) / "checkpoint").mkdir()
        result = state.select_recovery(self.owner)
        self.assertEqual(result["promoted"], [g0])
        self.assertIn("incomplete candidate", json.loads((self.gdir(g1) / "abandoned.json").read_text())["reason"])

    def test_non_chain_unresolved_pair_is_an_error(self):
        g0 = self.g0()
        other, data = self.fixture(self.owner, 1, None)
        # Forge a second unresolved intent that is not pending on g0.
        forged = self.gdir("g-" + "f" * 32)
        forged.mkdir(parents=True)
        (forged / "intent.json").write_bytes(state._bytes(dict(other, parent=None, data=data)))
        with self.assertRaises(state.StateError):
            state.select_recovery(self.owner)


class PipelinedSteadyState(PipelinedParent):
    def test_predecessor_with_pending_parent_promoted(self):
        g0 = self.g0()
        g1 = self.prepare_on(1, g0, self.pending_parent(g0))
        self.finish(g1, 1)
        state.commit_generation(self.owner, g0)  # B2 at save(1)
        g2 = self.prepare_on(2, g1, self.pending_parent(g1))
        result = state.select_recovery(self.owner)
        self.assertEqual((result["generation"], result["promoted"]), (g1, [g1]))
        self.assertTrue((self.gdir(g2) / "abandoned.json").exists())


class OfflineParentCheck(PipelinedParent):
    def test_parent_matches_both_forms(self):
        import hashlib
        from offline_generation import parent_matches
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        g0 = self.g0()
        g1 = self.prepare_on(1, g0, self.pending_parent(g0))
        self.finish(g1, 1)
        state.commit_generation(self.owner, g0)
        state.commit_generation(self.owner, g1)
        root = self.owner.root / "generations"
        for gid in (g0, g1):
            token = json.loads((root / gid / "token.json").read_text())
            manifest = json.loads((root / gid / "manifest.json").read_text())
            self.assertTrue(parent_matches(token, manifest["parent"], root, sha))
        token = json.loads((root / g1 / "token.json").read_text())
        forged = dict(json.loads((root / g1 / "manifest.json").read_text())["parent"], intent_sha256="0" * 64)
        self.assertFalse(parent_matches(token, forged, root, sha))
