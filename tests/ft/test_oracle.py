"""Hand-written observer graphs, independent of R state/replay implementation."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.ft.oracle import audit_run


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


class Graph:
    def __init__(self):
        self.files = {"dataset.json": [{"prompt": [{"role": "user", "content": "one?"}], "label": "1"}]}
        self.freeze = {"schema": 1, "run_nonce": "oracle-fixture", "evidence_level": "fixture",
                       "timeline": "controller-1", "run_start": 0, "expected_ranks": ["actor:0"],
                       "fault_id": "f1", "initial_state": "root", "target_group": "target:epoch0:row0",
                       "target_role": "reward", "k": 2, "max_policy_staleness": 1,
                       "reward_transform": "identity", "recovery_window_seconds": 900,
                       "verifier": {"name": "fixture-exact", "version": "exact-v1"},
                       "applicability": {"status": "applicable"}, "inputs": {}}
        self.evidence = {"run_nonce": "oracle-fixture", "events": [], "states": [],
                         "updates": [], "groups": [], "final_state": "root", "end_time": 30}
        self.state("root", None, [], [], 0)
        self.event("initial-root", "initial_state", 0, state="root")
        self.event("load-root", "checkpoint_loaded", 0.1, state="root", epoch=0)
        self.event("fault", "fault_observed", 2, fault_id="f1")
        self.event("ready", "role_ready", 3)

    def event(self, identifier, kind, time, **fields):
        event = {"id": identifier, "type": kind, "time": time, "run_nonce": "oracle-fixture",
                 "timeline": "controller-1", "role": "reward", "rank": 0, "pid": 100,
                 "start_time": "10", "boot_id": "boot", "cgroup": "fixture", "event_nonce": identifier,
                 "source_monotonic": time}
        event.update(fields)
        self.evidence["events"].append(event)
        return event

    def state(self, identifier, parent, updates, consumed, epoch):
        data_name = identifier + "-data.json"
        self.files[data_name] = {"drawn": list(consumed), "consumed": list(consumed), "pending": [], "cursor": len(consumed)}
        names = ("model", "optimizer_master", "optimizer_moments", "optimizer_step", "scheduler",
                 "rng_python", "rng_numpy", "rng_torch_cpu", "rng_device", "rng_tracker", "data", "policy")
        components = {}
        for name in names:
            path = data_name if name == "data" else identifier + "-" + name + ".json"
            self.files.setdefault(path, {"fixture_component": name, "state": identifier})
            components[name] = {"actor:0": [path]}
        state = {"id": identifier, "parent": parent, "updates": updates, "epoch": epoch,
                 "components": components, "data_file": data_name, "snapshot_id": identifier,
                 "persist_event": "initial-root" if parent is None else "persist-" + identifier,
                 "save_event": "save-" + identifier, "finalize_events": ["finalize-" + identifier]}
        self.evidence["states"].append(state)
        return state

    def update(self, identifier="g1", parent="root", logical=None, logical_update=None, epoch=0, at=4, loaded="root"):
        logical = logical or self.freeze["target_group"]
        physical = "physical-" + identifier
        source = self.files["dataset.json"][0]
        prompt_hash = digest(encoded(source["prompt"]))
        samples, rows = [], []
        for index in range(2):
            attempt = identifier + "-attempt"
            payload = {"prompt": source["prompt"], "label": source["label"], "prompt_sha256": prompt_hash,
                       "completion": "1", "policy_version": 0, "tokens": [1, 2],
                       "loss_mask": [0, 1], "logprobs": [0, -0.2], "prompt_ids": [1], "completion_ids": [2]}
            path = f"{identifier}-sample{index}.json"
            self.files[path] = payload
            sample = {"index": index, "attempt": attempt, "payload_file": path, "policy_version": 0,
                      "verifier_version": "exact-v1", "reward": 1.0}
            samples.append(sample)
            rows.append({"group": logical, "sample": index, "attempt": attempt,
                         "payload_sha256": digest(encoded(payload)), "reward": 1.0,
                         "tokens": list(payload["tokens"]), "loss_mask": list(payload["loss_mask"]),
                         "logprobs": list(payload["logprobs"])})
            self.event(f"auth-{identifier}-{index}", "authorize", at, group=logical, sample=index,
                       attempt=attempt, epoch=epoch, policy_version=0, verifier_version="exact-v1")
        tensor_path = identifier + "-tensor.json"
        self.files[tensor_path] = {"transform": "identity", "rows": rows}
        self.evidence["groups"].append({"id": "group-" + identifier, "logical_id": logical,
            "k": 2, "samples": samples, "prompt_sha256": prompt_hash, "source_file": "dataset.json", "source_index": 0})
        self.event("start-" + identifier, "optimizer_start", at + 1, update=physical, epoch=epoch,
                   tensor_sha256=digest(encoded(self.files[tensor_path])))
        self.event("end-" + identifier, "optimizer_end", at + 2, update=physical, epoch=epoch,
                   successful=True, scheduler_applied=True)
        self.evidence["updates"].append({"id": physical, "logical_id": logical_update or "update-" + identifier,
            "epoch": epoch, "policy_version": 0, "loaded_state": loaded,
            "start_event": "start-" + identifier, "end_event": "end-" + identifier,
            "groups": ["group-" + identifier], "tensor_file": tensor_path})
        parent_data = next(s["data_file"] for s in self.evidence["states"] if s["id"] == parent)
        consumed = self.files[parent_data]["consumed"] + [logical]
        state = self.state(identifier, parent, [physical], consumed, epoch)
        self.event(state["save_event"], "checkpoint_schedule", at + 3, state=identifier, snapshot_id=identifier)
        self.event(state["finalize_events"][0], "checkpoint_finalize", at + 4, state=identifier,
                   snapshot_id=identifier, actor_rank="actor:0")
        self.event(state["persist_event"], "checkpoint_persisted", at + 5, state=identifier, snapshot_id=identifier)
        self.evidence["final_state"] = identifier
        return state

    def write(self, directory):
        self.freeze["inputs"] = {"dataset.json": digest(encoded(self.files["dataset.json"]))}
        self.evidence["events"].sort(key=lambda event: event["time"])
        self.evidence["freeze_sha256"] = digest(encoded(self.freeze))
        artifacts = dict(self.files, **{"evidence.json": self.evidence})
        seal = {}
        for name, value in artifacts.items():
            data = encoded(value)
            (directory / name).write_bytes(data)
            seal[name] = digest(data)
        return {"freeze": copy.deepcopy(self.freeze), "observer_seal": seal}


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def audit(self, graph):
        return audit_run(self.root, graph.write(self.root))

    def test_baseline_without_token_correct_recovery(self):
        graph = Graph()
        graph.update()
        report = self.audit(graph)
        self.assertEqual(report["status"], "correct_recovered", report)
        self.assertEqual(report["rto_seconds"], 7)
        self.assertEqual(report["safety"], "pass")
        self.assertEqual(report["training_continuation"], "continued")

    def test_legal_rollback_recomputation_not_duplicate(self):
        graph = Graph()
        graph.update("lost", logical_update="same-logical")
        graph.event("reload", "checkpoint_loaded", 10, state="root", epoch=1)
        graph.update("retained", logical_update="same-logical", epoch=1, at=11)
        report = self.audit(graph)
        self.assertEqual(report["status"], "correct_recovered", report)
        self.assertEqual(report["rolled_back_updates"], ["physical-lost"])
        self.assertEqual(report["rto_seconds"], 14)

    def test_parent_consumption_still_counts_after_load(self):
        graph = Graph()
        graph.update("first")
        graph.event("reload", "checkpoint_loaded", 10, state="first", epoch=1)
        graph.update("second", parent="first", epoch=1, at=11, loaded="first")
        report = self.audit(graph)
        self.assertEqual(report["status"], "invalid_commit", report)
        self.assertTrue(any("duplicate retained consumption" in item for item in report["violations"]))

    def test_old_ancestor_reload_cannot_keep_discarded_parent(self):
        for epoch in (0, 1):
            with self.subTest(reload_epoch=epoch):
                graph = Graph()
                graph.update("first")
                graph.event("reload-old", "checkpoint_loaded", 10, state="root", epoch=epoch)
                graph.update("second", parent="first", logical="another-group", epoch=epoch, at=11, loaded="root")
                report = self.audit(graph)
                self.assertEqual(report["status"], "invalid_commit", report)
                self.assertEqual(report["safety"], "invalid_commit")
                self.assertIsNone(report["rto_seconds"])
                self.assertTrue(any("saved parent is not retained" in item for item in report["violations"]))

    def test_same_execution_continuous_saves_keep_parent(self):
        graph = Graph()
        graph.update("first")
        graph.update("second", parent="first", logical="another-group", epoch=0, at=11, loaded="root")
        report = self.audit(graph)
        self.assertEqual(report["status"], "correct_recovered", report)
        self.assertEqual(report["retained_updates"], ["physical-first", "physical-second"])

    def test_actual_parent_reload_keeps_parent_across_or_within_epoch(self):
        for epoch in (0, 1):
            with self.subTest(reload_epoch=epoch):
                graph = Graph()
                graph.update("first")
                graph.event("reload-parent", "checkpoint_loaded", 10, state="first", epoch=epoch)
                graph.update("second", parent="first", logical="another-group", epoch=epoch, at=11, loaded="first")
                report = self.audit(graph)
                self.assertEqual(report["status"], "correct_recovered", report)
                self.assertEqual(report["retained_states"], ["root", "first", "second"])

    def test_reload_between_optimizer_and_save_cannot_retain_update(self):
        graph = Graph()
        graph.update("first")
        graph.event("reload-mid-save", "checkpoint_loaded", 6.5, state="root", epoch=0)
        report = self.audit(graph)
        self.assertEqual(report["status"], "invalid_commit", report)
        self.assertTrue(any("interrupted by reload" in item for item in report["violations"]))

    def test_retained_logical_update_duplicate_with_different_groups(self):
        graph = Graph()
        graph.update("first", logical_update="duplicate")
        graph.event("reload", "checkpoint_loaded", 10, state="first", epoch=1)
        graph.update("second", parent="first", logical="different", logical_update="duplicate", epoch=1, at=11, loaded="first")
        report = self.audit(graph)
        self.assertEqual(report["status"], "invalid_commit")
        self.assertTrue(any("logical update" in item for item in report["violations"]))

    def test_rolled_back_target_does_not_end_rto(self):
        graph = Graph()
        graph.update("lost")
        graph.event("reload", "checkpoint_loaded", 10, state="root", epoch=1)
        graph.update("other", logical="unrelated", epoch=1, at=11)
        graph.evidence["end_time"] = 1000
        report = self.audit(graph)
        self.assertEqual(report["training_continuation"], "continued", report)
        self.assertEqual(report["affected_work_recovery"], "unresolved")
        self.assertEqual(report["status"], "timeout")
        self.assertIsNone(report["rto_seconds"])
        self.assertEqual(report["penalized_score"], 900)

    def test_role_must_be_ready_at_or_after_target_commit(self):
        graph = Graph()
        graph.update()
        graph.event("down-again", "role_down", 8)
        graph.event("ready-again", "role_ready", 12)
        report = self.audit(graph)
        self.assertEqual(report["rto_seconds"], 10, report)
        self.assertEqual(report["first_role_ready"], 3)

    def test_safe_drop_and_other_group_continuation_are_distinct(self):
        graph = Graph()
        graph.update(logical="other")
        graph.event("drop", "work_dropped", 10, group=graph.freeze["target_group"])
        graph.event("stop", "safe_stop", 11)
        report = self.audit(graph)
        self.assertEqual(report["safety"], "pass", report)
        self.assertEqual(report["training_continuation"], "continued")
        self.assertEqual(report["affected_work_recovery"], "safely_dropped")
        self.assertIsNone(report["rto_seconds"])

    def test_missing_optimizer_mapping_is_unverifiable(self):
        graph = Graph()
        graph.update()
        graph.evidence["updates"] = []
        report = self.audit(graph)
        self.assertEqual(report["status"], "unverifiable", report)

    def test_missing_rank_or_components_is_unverifiable(self):
        graph = Graph()
        graph.update()
        graph.freeze["expected_ranks"].append("actor:1")
        self.assertEqual(self.audit(graph)["status"], "unverifiable")

    def test_old_attempt_first_arrival_is_invalid(self):
        graph = Graph()
        graph.update()
        graph.event("new-auth", "authorize", 4.5, group=graph.freeze["target_group"], sample=0,
                    attempt="newer", epoch=0, policy_version=0, verifier_version="exact-v1")
        self.assertEqual(self.audit(graph)["status"], "invalid_commit")

    def test_incomplete_k_and_mixed_versions_are_invalid(self):
        for change in ("k", "version"):
            with self.subTest(change=change):
                graph = Graph()
                graph.update()
                group = graph.evidence["groups"][0]
                if change == "k":
                    group["samples"].pop()
                else:
                    group["samples"][0]["verifier_version"] = "wrong"
                self.assertEqual(self.audit(graph)["status"], "invalid_commit")

    def test_distinct_authorized_sample_policies_with_legal_staleness(self):
        graph = Graph()
        graph.update()
        graph.evidence["updates"][0]["policy_version"] = 1
        sample = graph.evidence["groups"][0]["samples"][1]
        sample["policy_version"] = 1
        raw = graph.files[sample["payload_file"]]
        raw["policy_version"] = 1
        graph.files["g1-tensor.json"]["rows"][1]["payload_sha256"] = digest(encoded(raw))
        for event in graph.evidence["events"]:
            if event["type"] == "authorize" and event["sample"] == 1:
                event["policy_version"] = 1
            if event["type"] == "optimizer_start":
                event["tensor_sha256"] = digest(encoded(graph.files["g1-tensor.json"]))
        report = self.audit(graph)
        self.assertEqual(report["status"], "correct_recovered", report)
        sample["policy_version"] = 2
        self.assertEqual(self.audit(graph)["status"], "invalid_commit")

    def test_fake_success_and_reward_cannot_override_authority(self):
        graph = Graph()
        graph.update()
        graph.evidence["success"] = True
        graph.evidence["groups"][0]["samples"][0].update(reward=0.0, authoritative_reward=0.0)
        report = self.audit(graph)
        self.assertEqual(report["status"], "invalid_commit")
        self.assertTrue(any("independent reward" in item for item in report["violations"]))

    def test_wrong_label_compared_to_frozen_source(self):
        graph = Graph()
        graph.update()
        graph.files["g1-sample0.json"]["label"] = "2"
        self.assertEqual(self.audit(graph)["status"], "invalid_commit")

    def test_actual_tensor_must_match_samples(self):
        graph = Graph()
        graph.update()
        graph.files["g1-tensor.json"]["rows"][0]["reward"] = 0.0
        event = next(e for e in graph.evidence["events"] if e["type"] == "optimizer_start")
        event["tensor_sha256"] = digest(encoded(graph.files["g1-tensor.json"]))
        self.assertEqual(self.audit(graph)["status"], "invalid_commit")

    def test_training_token_mask_or_logprob_change_with_same_payload_hash(self):
        for field, replacement in (("tokens", [1, 9]), ("loss_mask", [1, 1]), ("logprobs", [0, -0.9])):
            with self.subTest(field=field):
                graph = Graph()
                graph.update()
                row = graph.files["g1-tensor.json"]["rows"][0]
                original_hash = row["payload_sha256"]
                row[field] = replacement
                event = next(e for e in graph.evidence["events"] if e["type"] == "optimizer_start")
                event["tensor_sha256"] = digest(encoded(graph.files["g1-tensor.json"]))
                report = self.audit(graph)
                self.assertEqual(row["payload_sha256"], original_hash)
                self.assertEqual(report["status"], "invalid_commit", report)
                self.assertTrue(any("actual training tensor" in item for item in report["violations"]))

    def test_unsupported_transform_and_real_evidence_not_silently_certified(self):
        graph = Graph()
        graph.update()
        graph.freeze["reward_transform"] = "group-normalization"
        self.assertEqual(self.audit(graph)["status"], "unverifiable")
        graph.freeze["reward_transform"] = "identity"
        graph.freeze["evidence_level"] = "observer-normalized"
        self.assertEqual(self.audit(graph)["status"], "unverifiable")

    def test_full_hash_detects_artifact_tampering(self):
        graph = Graph()
        graph.update()
        spec = graph.write(self.root)
        path = self.root / "g1-model.json"
        path.write_bytes(path.read_bytes().replace(b"model", b"other"))
        self.assertEqual(audit_run(self.root, spec)["status"], "unverifiable")

    def test_n_a_requires_pre_run_source_and_fault_miss_is_technical(self):
        graph = Graph()
        graph.freeze["applicability"] = {"status": "not_applicable", "source": "audit:F3", "reason": "no ACK", "decided_at": -1}
        self.assertEqual(self.audit(graph)["status"], "not_applicable")
        graph.freeze["applicability"]["decided_at"] = 5
        self.assertEqual(self.audit(graph)["status"], "unverifiable")
        graph = Graph()
        graph.evidence["events"] = [e for e in graph.evidence["events"] if e["type"] != "fault_observed"]
        self.assertEqual(self.audit(graph)["status"], "technical_invalid")

    def test_verifier_name_cannot_import_arbitrary_module(self):
        graph = Graph()
        graph.update()
        graph.freeze["verifier"]["name"] = "os.system"
        self.assertEqual(self.audit(graph)["status"], "unverifiable")

    def test_implementation_does_not_import_method_decisions(self):
        path = Path(__file__).resolve().parents[2] / "scripts/ft/oracle.py"
        tree = ast.parse(path.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertFalse(any(name in entry for name in ("ft.state", "replay", "phase2_reconciler") for entry in imports))


if __name__ == "__main__":
    unittest.main()
