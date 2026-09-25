"""R slowdown fix (R_SLOWDOWN_FIX_PLAN v2): hash counts, stat recheck, pruning, pins.

CPU contract files only; not backend/GPU tests.
"""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.ft import state
try:
    from tests.ft.test_state import complete, new_owner
except ImportError:  # run from tests/ft with bare module names
    from test_state import complete, new_owner


def bulk(name):
    return name == "model.bin"


class Counter:
    """Counts full content reads done by ``state._inventory`` (cache hits excluded)."""

    def __init__(self):
        self.files = 0
        self.original = state._inventory

    def __call__(self, directory, *, cache=None, identities=None):
        before = dict(cache or {})
        result = self.original(directory, cache=cache, identities=identities)
        self.files += sum(1 for name in result if name not in before)
        return result


class EfficiencyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "run"

    def owner(self, root=None):
        owner = new_owner(root or self.root)
        self.addCleanup(owner.close)
        return owner

    def chain(self, owner, length, prune=False):
        parent, gids = None, []
        for number in range(length):
            gid = complete(owner, number, parent)
            token = state.commit_generation(owner, gid)
            parent = {"generation": gid, "token_sha256": state._hash(state._bytes(token))}
            gids.append(gid)
            if prune:
                state.prune_generations(owner, bulk, keep=2)
        return gids

    # --- hash counts -------------------------------------------------------

    def test_commit_hashes_only_new_generation_for_any_chain_length(self):
        for length in (1, 5, 30):
            with self.subTest(length=length):
                owner = self.owner(Path(self.temporary.name) / f"len{length}")
                gids = self.chain(owner, length)
                parent = {"generation": gids[-1], "token_sha256": state._hash(state._bytes(
                    json.loads((owner.root / "generations" / gids[-1] / "token.json").read_text())))}
                gid = complete(owner, length, parent)
                counter = Counter()
                with patch.object(state, "_inventory", counter):
                    state.commit_generation(owner, gid)
                    with state._locked(owner) as control:
                        state._head(owner, control)
                # One generation of 11 component files, hashed once.
                self.assertEqual(counter.files, 11)

    def test_prehash_reused_when_unchanged(self):
        owner = self.owner()
        gid = complete(owner)
        prehashed = state.prehash(owner.root / "generations" / gid / "checkpoint")
        counter = Counter()
        with patch.object(state, "_inventory", counter):
            state.commit_generation(owner, gid, prehashed=prehashed)
        self.assertEqual(counter.files, 0)

    def test_prehash_invalidated_by_rewrite(self):
        owner = self.owner()
        gid = complete(owner)
        checkpoint = owner.root / "generations" / gid / "checkpoint"
        prehashed = state.prehash(checkpoint)
        path = checkpoint / "model.bin"
        # Size-changing rewrite. Unlink + same-size rewrite can reuse the inode
        # within one timestamp tick and keep an identical stat identity: that is
        # the disclosed L2 boundary (writer already joined), not tested here.
        data = path.read_bytes()
        path.unlink()
        path.write_bytes(data + b"y")
        token = state.commit_generation(owner, gid, prehashed=prehashed)
        manifest = json.loads((owner.root / "generations" / gid / "manifest.json").read_text())
        self.assertEqual(manifest["files"]["model.bin"]["sha256"], state._hash(path.read_bytes()))
        self.assertEqual(token["generation"], gid)

    def test_recovery_hashes_head_once(self):
        for length in (1, 5, 30):
            with self.subTest(length=length):
                # Same owner: cross-process takeover is covered by test_state.
                owner = self.owner(Path(self.temporary.name) / f"rec{length}")
                self.chain(owner, length)
                counter = Counter()
                with patch.object(state, "_inventory", counter):
                    state.select_recovery(owner)
                self.assertEqual(counter.files, 11)

    def test_candidate_promotion_hashes_candidate_once(self):
        owner = self.owner()
        gids = self.chain(owner, 5)
        parent = {"generation": gids[-1], "token_sha256": state._hash(
            (owner.root / "generations" / gids[-1] / "token.json").read_bytes())}
        candidate = complete(owner, 5, parent)
        counter = Counter()
        with patch.object(state, "_inventory", counter):
            self.assertEqual(state.select_recovery(owner)["generation"], candidate)
        self.assertEqual(counter.files, 22)  # old head + candidate, each once

    # --- corruption boundaries ---------------------------------------------

    def test_multi_generation_head_corruption_rejected(self):
        owner = self.owner()
        gids = self.chain(owner, 3)
        path = owner.root / "generations" / gids[-1] / "checkpoint/model.bin"
        with path.open("r+b") as stream:
            stream.seek(5000)
            stream.write(b"z")
        with self.assertRaises(state.StateError):
            state.select_recovery(owner)

    def test_ancestor_same_size_corruption_is_online_boundary(self):
        owner = self.owner()
        gids = self.chain(owner, 3)
        path = owner.root / "generations" / gids[0] / "checkpoint/model.bin"
        with path.open("r+b") as stream:
            stream.seek(5000)
            stream.write(b"z")
        # Disclosed limitation: recovery never loads the ancestor, so it is not
        # content-hashed online; the offline audit (full hash) must catch it.
        self.assertEqual(state.select_recovery(owner)["generation"], gids[-1])
        manifest = json.loads((owner.root / "generations" / gids[0] / "manifest.json").read_text())
        self.assertNotEqual(manifest["files"]["model.bin"]["sha256"], state._hash(path.read_bytes()))

    def test_ancestor_missing_or_resized_file_rejected(self):
        for mutation in ("missing", "resized", "extra"):
            with self.subTest(mutation=mutation):
                root = Path(self.temporary.name) / mutation
                owner = self.owner(root)
                gids = self.chain(owner, 3)
                checkpoint = owner.root / "generations" / gids[0] / "checkpoint"
                if mutation == "missing":
                    (checkpoint / "scheduler.bin").unlink()
                elif mutation == "resized":
                    with (checkpoint / "scheduler.bin").open("ab") as stream:
                        stream.write(b"!")
                else:
                    (checkpoint / "extra.bin").write_bytes(b"x")
                with self.assertRaises(state.StateError):
                    state.select_recovery(owner)

    def test_stale_control_head_does_not_hide_token_head(self):
        owner = self.owner()
        gids = self.chain(owner, 2)
        control = json.loads((owner.root / "control.json").read_text())
        control["head"] = {"generation": gids[0], "token_sha256": state._hash(
            (owner.root / "generations" / gids[0] / "token.json").read_bytes())}
        state._write(owner.root / "control.json", control, immutable=False)
        with state._locked(owner) as locked:
            head, _ = state._head(owner, locked)
        self.assertEqual(head["generation"], gids[1])

    # --- stat identity recheck (L2) ----------------------------------------

    def test_change_between_hash_and_recheck_rejected(self):
        def mutations(checkpoint):
            model = checkpoint / "model.bin"
            yield "append", lambda: model.open("ab").write(b"!")
            yield "new_file", lambda: (checkpoint / "late.bin").write_bytes(b"x")
            def replace():
                temp = checkpoint / "model.tmp"
                temp.write_bytes(model.read_bytes())
                os.replace(temp, model)
            yield "rename_replace", replace
            def truncate_rewrite():
                # Size-changing rewrite. A same-size rewrite inside one
                # timestamp tick is the disclosed L2 limitation, not tested.
                data = model.read_bytes()
                with model.open("r+b") as stream:
                    stream.truncate(0)
                    stream.write(data[:-1])
            yield "truncate_rewrite", truncate_rewrite
            yield "symlink", lambda: (checkpoint / "link.bin").symlink_to("model.bin")
        names = [name for name, _ in mutations(Path("/nonexistent"))]
        for name in names:
            with self.subTest(mutation=name):
                owner = self.owner(Path(self.temporary.name) / name)
                gid = complete(owner)
                checkpoint = owner.root / "generations" / gid / "checkpoint"
                action = dict(mutations(checkpoint))[name]
                original = state._sync_dir
                fired = []
                def sync(path):
                    if Path(path) == checkpoint and not fired:
                        fired.append(True)
                        action()
                    return original(path)
                with patch.object(state, "_sync_dir", sync):
                    with self.assertRaises(state.StateError):
                        state.commit_generation(owner, gid)
                self.assertFalse((owner.root / "generations" / gid / "token.json").exists())

    # --- pruning and pins (L3) ---------------------------------------------

    def test_prune_keeps_last_two_and_metadata(self):
        owner = self.owner()
        gids = self.chain(owner, 6, prune=True)
        for gid in gids[:-2]:
            directory = owner.root / "generations" / gid
            self.assertFalse((directory / "checkpoint/model.bin").exists())
            self.assertTrue((directory / "checkpoint/scheduler.bin").exists())
            marker = json.loads((directory / "pruned.json").read_text())
            self.assertEqual(set(marker["removed"]), {"model.bin"})
        for gid in gids[-2:]:
            self.assertTrue((owner.root / "generations" / gid / "checkpoint/model.bin").exists())
            self.assertFalse((owner.root / "generations" / gid / "pruned.json").exists())
        self.assertEqual(state.select_recovery(owner)["generation"], gids[-1])

    def test_pins_are_never_pruned(self):
        owner = self.owner()
        gids = self.chain(owner, 2)
        state.pin_generation(owner, gids[0], "recovery_loaded")
        parent = {"generation": gids[-1], "token_sha256": state._hash(
            (owner.root / "generations" / gids[-1] / "token.json").read_bytes())}
        for number in range(2, 6):
            gid = complete(owner, number, parent)
            token = state.commit_generation(owner, gid)
            parent = {"generation": gid, "token_sha256": state._hash(state._bytes(token))}
            state.prune_generations(owner, bulk, keep=2)
        self.assertTrue((owner.root / "generations" / gids[0] / "checkpoint/model.bin").exists())
        self.assertFalse((owner.root / "generations" / gids[1] / "checkpoint/model.bin").exists())
        with self.assertRaises(state.StateError):
            state.pin_generation(owner, gids[0], "unknown")

    def test_crash_between_marker_and_delete_resumes_without_rewrite(self):
        owner = self.owner()
        gids = self.chain(owner, 3)
        original_unlink = Path.unlink
        def crash(self, *args, **kwargs):
            raise OSError("injected crash before delete")
        with patch.object(Path, "unlink", crash):
            with self.assertRaises(OSError):
                state.prune_generations(owner, bulk, keep=2)
        directory = owner.root / "generations" / gids[0]
        marker = (directory / "pruned.json").read_bytes()
        self.assertTrue((directory / "checkpoint/model.bin").exists())
        # Marker present and file still there: metadata check accepts it.
        self.assertEqual(state.select_recovery(owner)["generation"], gids[-1])
        state.prune_generations(owner, bulk, keep=2)
        self.assertEqual((directory / "pruned.json").read_bytes(), marker)
        self.assertFalse((directory / "checkpoint/model.bin").exists())
        self.assertEqual(state.select_recovery(owner)["generation"], gids[-1])

    def test_missing_bulk_without_marker_rejected(self):
        owner = self.owner()
        gids = self.chain(owner, 3)
        (owner.root / "generations" / gids[0] / "checkpoint/model.bin").unlink()
        with self.assertRaises(state.StateError):
            state.select_recovery(owner)

    def test_tampered_marker_rejected(self):
        owner = self.owner()
        gids = self.chain(owner, 3, prune=True)
        path = owner.root / "generations" / gids[0] / "pruned.json"
        marker = json.loads(path.read_text())
        marker["removed"]["scheduler.bin"] = {"size": 1, "sha256": "b" * 64}
        path.unlink()
        path.write_text(json.dumps(marker))
        with self.assertRaises(state.StateError):
            state.select_recovery(owner)

    def test_prune_uses_token_chain_with_stale_control_head(self):
        owner = self.owner()
        gids = self.chain(owner, 4)
        control = json.loads((owner.root / "control.json").read_text())
        control["head"] = {"generation": gids[1], "token_sha256": state._hash(
            (owner.root / "generations" / gids[1] / "token.json").read_bytes())}
        state._write(owner.root / "control.json", control, immutable=False)
        state.prune_generations(owner, bulk, keep=2)
        present = [(owner.root / "generations" / g / "checkpoint/model.bin").exists() for g in gids]
        self.assertEqual(present, [False, False, True, True])

    def test_abandoned_and_candidate_generations_not_pruned(self):
        owner = self.owner()
        gids = self.chain(owner, 3)
        parent = {"generation": gids[-1], "token_sha256": state._hash(
            (owner.root / "generations" / gids[-1] / "token.json").read_bytes())}
        candidate = complete(owner, 3, parent, finalize=False)
        state.prune_generations(owner, bulk, keep=2)
        self.assertTrue((owner.root / "generations" / candidate / "checkpoint/model.bin").exists())
        state.select_recovery(owner)  # abandons the incomplete candidate
        self.assertTrue((owner.root / "generations" / candidate / "abandoned.json").exists())
        state.prune_generations(owner, bulk, keep=2)
        self.assertTrue((owner.root / "generations" / candidate / "checkpoint/model.bin").exists())


if __name__ == "__main__":
    unittest.main()
