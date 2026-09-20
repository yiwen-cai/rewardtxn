"""Targeted fixed-driver checks, not the twelve-case acceptance batch."""

import copy
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("p2_fixed_driver", HERE / "p2_schedule_driver.py")
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)
compiler = driver.local_module("p2_schedule_cases")


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_k8_independent_control_and_training_rows(self):
        case = compiler.compile_case("F1", 0, 1)
        graph = driver.oracle_graph(driver.external_fixture(case), case["id"])
        for group in graph.evidence["groups"]:
            self.assertEqual(group["k"], 8)
            self.assertEqual([sample["index"] for sample in group["samples"]], list(range(8)))
        self.assertEqual(len(graph.files["target-tensor.json"]["rows"]), 8)
        report = driver.audit_variant(self.root, "control", graph)
        self.assertEqual(report["status"], "correct_recovered", report)
        self.assertEqual(report["evidence_level"], "fixture")

    def test_safe_drop_and_physical_rollback_are_distinct(self):
        for cell, boundary, interleave in (("F1", 0, 1), ("F2", 4, 0)):
            case = compiler.compile_case(cell, boundary, interleave)
            path = self.root / cell
            path.mkdir()
            driver.write_json(path / "external-fixture.json", driver.external_fixture(case))
            reports = driver.oracle_lane(path, case["id"])
            if cell == "F1":
                self.assertEqual(reports["primary"]["affected_work_recovery"], "safely_dropped")
                self.assertEqual(reports["counterexample"]["status"], "invalid_commit")
            else:
                self.assertEqual(reports["primary"]["rolled_back_updates"], ["physical-target"])
                self.assertIn("physical-recomputed", reports["primary"]["retained_updates"])

    def test_unsupported_case_is_not_executed(self):
        case = compiler.compile_case("F4", 1, 6)
        output = self.root / "unsupported"
        result = driver.run_case(case, output)
        self.assertEqual(result["status"], "not_executed")
        self.assertFalse((output / "state-run").exists())
        self.assertEqual(list(output.glob("*-operations.jsonl")), [])

    def test_api_negative_case_has_real_rejection_receipt(self):
        case = compiler.compile_case("F1", 0, 1)
        output = self.root / "partial-group"
        result = driver.run_case(case, output, timeout=25)
        self.assertEqual(result["status"], "passed", result)
        event = driver.check_reference(output, result["must_reach"]["target_boundary_reached"])
        rejection = driver.check_reference(output, event["rejection"])
        self.assertEqual(rejection["result"], "rejected")
        self.assertIn("incomplete group", rejection["reason"])
        self.assertEqual(result["must_reach"]["shared_input_and_arrival_verified"]["actual_bootstrap_order"], list(reversed(range(8))))
        frozen = json.loads((output / "external-fixture.json").read_text())
        self.assertEqual(json.loads((output / "oracle-primary/target-sample0.json").read_text()), frozen["groups"]["target"][0])

    def test_race_manifest_cut_and_fresh_process_recovery(self):
        case = compiler.compile_case("F2", 9, 5)
        output = self.root / "race-cut"
        result = driver.run_case(case, output, timeout=25)
        self.assertEqual(result["status"], "passed", result)
        reached = result["must_reach"]
        self.assertEqual(reached["confirmed_sigkill"]["returncode"], -9)
        initial = reached["confirmed_sigkill"]["identity"]
        recovered = driver.check_reference(output, reached["fresh_process_recovery"])
        self.assertNotEqual(initial, recovered["identity"])
        self.assertEqual(len(reached["two_live_owner_contenders"]["identities"]), 2)

    def test_wrong_reward_and_missing_evidence_are_separate_probes(self):
        driver.write_json(self.root / "external-fixture.json", driver.external_fixture(compiler.compile_case("X1", 9, 8)))
        reports = driver.oracle_lane(self.root, "X1.b09.i08")
        self.assertEqual(reports["healthy"]["status"], "correct_recovered")
        self.assertEqual(reports["primary"]["status"], "invalid_commit")
        self.assertEqual(reports["mutation"]["status"], "unverifiable")
        self.assertTrue(any("independent reward" in item for item in reports["primary"]["violations"]))

    def test_missing_must_reach_cannot_pass(self):
        case = copy.deepcopy(compiler.compile_case("F1", 0, 1))
        case["must_reach"].append("nonexistent_real_cut")
        result = driver.run_case(case, self.root / "missing-receipt", timeout=25)
        self.assertEqual(result["status"], "harness_failure")
        self.assertIn("nonexistent_real_cut", result["missing_requirements"])

    def test_stage_one_support_set_is_exact_and_keeps_representatives(self):
        self.assertEqual(len(driver.STAGE_ONE), 88)
        self.assertEqual(len(driver.SUPPORTED), 408)
        self.assertEqual(len(driver.STAGE_THREE), 72)
        self.assertEqual(len(driver.STAGE_TWO), 96)
        self.assertTrue(set(driver.REPRESENTATIVES) <= set(driver.SUPPORTED))
        self.assertTrue(all(compiler.compile_case(name.split(".")[0], int(name.split(".")[1][1:]), int(name.split(".")[2][1:]))["execution_status"] == "executable" for name in driver.SUPPORTED))

    def test_positive_commit_and_same_evidence_replay_are_real(self):
        case = compiler.compile_case("X1", 0, 2)
        path = self.root / "positive"
        result = driver.run_case(case, path, timeout=25)
        self.assertEqual(result["status"], "passed", result)
        self.assertTrue(result["boundary_checks"]["target_committed"])
        events = [json.loads(line) for line in (path / "owner0-operations.jsonl").read_text().splitlines()]
        repeats = [event for event in events if event["operation"] == "state.record_evidence.identical"]
        self.assertEqual(len(repeats), 2)
        target = next(event["generation"] for event in events if event["operation"] == "state.prepare_generation" and event["logical_group"] == "target")
        (path / "state-run/generations" / target / "token.json").unlink()
        with self.assertRaisesRegex(driver.ContractFailure, "actual target commit/token"):
            driver.check_stage_one(path, case, events)
        without_rank = [event for event in events if not (event["operation"] == "state.record_evidence" and event["generation"] == target and event["evidence"].get("rank") == "actor:1")]
        token = next(event["token"] for event in events if event["operation"] == "state.commit_generation" and event["generation"] == target)
        driver.write_json(path / "state-run/generations" / target / "token.json", token)
        with self.assertRaisesRegex(driver.ContractFailure, "rank finalize"):
            driver.check_stage_one(path, case, without_rank)
        without_optimizer = [event for event in events if not (event["operation"] == "state.record_evidence" and event["generation"] == target and event["evidence"]["kind"] == "optimizer")]
        with self.assertRaisesRegex(driver.ContractFailure, "optimizer receipt"):
            driver.check_stage_one(path, case, without_optimizer)

    def test_target_reverse_order_uses_actual_arrivals(self):
        case = compiler.compile_case("F1", 3, 1)
        result = driver.run_case(case, self.root / "reverse", timeout=25)
        self.assertEqual(result["status"], "passed", result)
        observed = result["must_reach"]["shared_input_and_arrival_verified"]
        self.assertEqual(observed["actual_target_order"], [3, 2, 1, 0])
        self.assertEqual(observed["actual_bootstrap_order"], list(range(8)))

    def test_method_payload_missing_tokens_and_observer_mutation_separate(self):
        case = compiler.compile_case("F1", 7, 8)
        path = self.root / "missing-tokens"
        result = driver.run_case(case, path, timeout=25)
        self.assertEqual(result["status"], "passed", result)
        self.assertNotIn("tokens", json.loads((path / "method-target-7.json").read_text()))
        self.assertEqual(result["oracle_model_lane"]["healthy"]["status"], "correct_recovered")
        primary = json.loads((path / "oracle-primary-report.json").read_text())
        modifier = json.loads((path / "oracle-missing-report.json").read_text())
        self.assertTrue(any("tokens" in item for item in primary["missing_evidence"]))
        self.assertTrue(any("missing optimizer_start" in item for item in modifier["missing_evidence"]))

    def test_missing_completion_suffix_is_rejected_before_claiming_pass(self):
        case = copy.deepcopy(compiler.compile_case("X1", 0, 0))
        index = next(index for index, op in enumerate(case["operations"]) if op["call"] == "state.prepare_generation")
        case["operations"] = case["operations"][:index + 1]
        result = driver.run_case(case, self.root / "omitted-completion")
        self.assertEqual(result["status"], "harness_failure", result)
        self.assertIn("prepare/commit completion", result["reason"])
        self.assertNotIn("bootstrap_committed", result["must_reach"])

    def storage_case(self, cell, boundary, interleave):
        case = compiler.compile_case(cell, boundary, interleave)
        path = self.root / case["id"]
        result = driver.run_case(case, path, timeout=25)
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["must_reach"]["confirmed_sigkill"]["returncode"], -9)
        cut = json.loads((path / "storage-cut.json").read_text())
        recovery = driver.check_reference(path, result["must_reach"]["fresh_process_recovery"])
        self.assertNotEqual(cut["writer_identity"], recovery["identity"])
        self.assertTrue(cut["observed_before_sigkill"])
        return path, result, cut

    def test_storage_before_intent_preserves_no_invented_rollback(self):
        path, result, cut = self.storage_case("F2", 0, 4)
        self.assertIsNone(cut["intent"])
        self.assertEqual(cut["artifacts"], {})
        self.assertTrue(result["boundary_checks"]["no_target_intent"])
        self.assertEqual(result["state_process_lane"]["recovery"]["rollback_intents"], [])

    def test_storage_before_start_and_started_cuts_are_distinct(self):
        _, _, before = self.storage_case("F2", 2, 1)
        _, _, started = self.storage_case("F2", 3, 0)
        self.assertTrue(before["before_start_marker"])
        self.assertEqual(before["optimizer_starts"], 0)
        self.assertEqual(started["optimizer_starts"], 1)
        self.assertEqual(started["optimizer_ends"], 0)
        self.assertEqual(started["receipts"], {})

    def test_scheduler_false_is_rejected_before_real_kill(self):
        _, result, cut = self.storage_case("F2", 6, 2)
        self.assertTrue(cut["scheduler_rejected"])
        self.assertEqual(cut["receipts"], {})
        self.assertEqual(cut["optimizer_starts"], 0)
        self.assertEqual(len(result["boundary_checks"]["rollback_intents"]), 1)

    def test_all_rank_files_without_any_finalize_then_corruption(self):
        path, result, cut = self.storage_case("F2", 8, 9)
        self.assertEqual(cut["saved_ranks"], ["actor:0", "actor:1"])
        self.assertEqual(cut["finalized_ranks"], [])
        self.assertEqual(list(cut["receipts"]), ["optimizer.json"])
        self.assertEqual(len([name for name in cut["artifacts"] if name.startswith("checkpoint/")]), 22)
        self.assertFalse(cut["manifest_present"])
        self.assertFalse(cut["token_present"])
        self.assertTrue((path / "state-corruption.json").exists())
        self.assertEqual(result["oracle_model_lane"]["mutation"]["status"], "unverifiable")

    def test_complete_candidate_preserves_execution_epoch_and_probe(self):
        path, result, cut = self.storage_case("F2", 9, 3)
        self.assertTrue(cut["manifest_present"])
        self.assertFalse(cut["token_present"])
        self.assertTrue(result["boundary_checks"]["candidate_promoted"])
        token = json.loads((path / "state-run/generations" / cut["generation"] / "token.json").read_text())
        self.assertEqual(token["execution_epoch"], 0)
        self.assertEqual(token["commit_epoch"], 1)
        self.assertEqual(result["state_process_lane"]["recovery"]["pending"][0]["sample"], "probe:0")

    def test_visible_token_cut_and_independent_missing_event_probe(self):
        _, result, cut = self.storage_case("F3", 1, 8)
        self.assertTrue(cut["manifest_present"])
        self.assertTrue(cut["token_present"])
        self.assertTrue(result["boundary_checks"]["visible_token_verified"])
        self.assertFalse(result["boundary_checks"]["candidate_promoted"])
        self.assertEqual(result["oracle_model_lane"]["mutation"]["status"], "unverifiable")

    def x1_case(self, boundary, interleave=0):
        case = compiler.compile_case("X1", boundary, interleave)
        path = self.root / case["id"]
        result = driver.run_case(case, path, timeout=25)
        self.assertEqual(result["status"], "passed", result)
        report = json.loads((path / "oracle-primary-report.json").read_text())
        return path, result, report

    def test_x1_version_label_alone_rejected_with_same_reward(self):
        path, result, report = self.x1_case(2, 1)
        self.assertTrue(result["boundary_checks"]["unauthorized_result_rejected"])
        self.assertTrue(any("stale/unauthorized sample" in item for item in report["violations"]))
        operations = [json.loads(line) for line in (path / "owner0-operations.jsonl").read_text().splitlines()]
        change = next(event for event in operations if event["operation"] == "fixture.change_declared_verifier")
        self.assertEqual(change["before_reward"], change["after_reward"])

    def test_x1_changed_label_and_independent_missing_evidence(self):
        _, result, report = self.x1_case(3, 8)
        self.assertTrue(any("frozen source" in item for item in report["violations"]))
        self.assertEqual(result["oracle_model_lane"]["mutation"]["status"], "unverifiable")

    def test_x1_changed_response_keeps_original_seal(self):
        path, _, report = self.x1_case(4, 9)
        self.assertEqual(report["status"], "unverifiable")
        spec = json.loads((path / "oracle-primary/spec.json").read_text())
        self.assertNotEqual(driver.digest((path / "oracle-primary/target-sample0.json").read_bytes()),
                            spec["observer_seal"]["target-sample0.json"])
        self.assertTrue(any("hash mismatch" in item for item in report["missing_evidence"]))

    def test_x1_foreign_group_reward_has_real_source(self):
        path, _, report = self.x1_case(6, 3)
        source = json.loads((path / "method-other-0.json").read_text())
        self.assertEqual((source["completion"], source["label"]), ("2", "1"))
        self.assertTrue(any("independent reward mismatch" in item for item in report["violations"]))
        other = json.loads((path / "oracle-primary/other-reward-source.json").read_text())
        self.assertEqual(other, source)

    def test_x1_unauthorized_policy_and_missing_authority(self):
        _, result, report = self.x1_case(7, 4)
        self.assertTrue(result["boundary_checks"]["unauthorized_result_rejected"])
        self.assertTrue(any("policy staleness violation" in item for item in report["violations"]))
        _, result, report = self.x1_case(8, 8)
        self.assertEqual(report["status"], "unverifiable")
        self.assertTrue(any("completion" in item for item in report["missing_evidence"]))
        self.assertEqual(result["oracle_model_lane"]["mutation"]["status"], "unverifiable")

    def test_same_group_distinct_authorized_policies_are_legal(self):
        case = compiler.compile_case("X1", 7, 0)
        inputs = driver.external_fixture(case)
        inputs["groups"]["target"][0]["policy_version"] = 1
        path = self.root / "legal-policy"
        path.mkdir()
        driver.write_json(path / "case.json", case)
        driver.write_json(path / "external-fixture.json", inputs)
        worker = driver.Worker(path, "legal-owner")
        with driver.state.acquire_owner(path / "state-run", -1, run_nonce="legal-policy", config_sha256=driver.digest(b"p2-cpu-driver"), verifier_version="exact-v1") as owner:
            worker.owner = owner
            with contextlib.redirect_stdout(io.StringIO()):
                worker.make_bootstrap()
            for index in range(8):
                worker.generate("target", index)
                key = f"target:{index}"
                worker.authorize(key, policy=1 if index == 0 else 0)
                worker.accept(key)
            generation = worker.prepare("target")
            worker.optimizer(generation, "target")
            worker.checkpoint(generation, "target")
            driver.state.commit_generation(owner, generation)
        report = driver.audit_variant(path, "legal-policy", driver.legal_policy_graph(inputs))
        self.assertEqual(report["status"], "correct_recovered", report)
        self.assertTrue((path / "state-run/generations" / generation / "token.json").exists())

    def test_existing_case_directory_never_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "result.json"
        marker.write_text('{"original":true}')
        with self.assertRaises(FileExistsError):
            driver.run_case(compiler.compile_case("F1", 0, 1), output)
        self.assertEqual(json.loads(marker.read_text()), {"original": True})


if __name__ == "__main__":
    unittest.main()
