"""Static schedule checks. No state API, processes, GPU, or case driver runs."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
MODULE_SPEC = importlib.util.spec_from_file_location("p2_static_schedule_cases", HERE / "p2_schedule_cases.py")
cases = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(cases)


class ScheduleManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = cases.compile_manifest()

    def test_exact_grid_counts_and_no_completed_claims(self):
        counts = cases.validate_manifest(self.manifest)
        self.assertEqual(counts, {"executable": 408, "limitation_only": 8, "pending_interface": 184})
        expected_executable = {"F1": 80, "F2": 80, "F3": 16, "F4": 80, "X1": 72, "X2": 80}
        for cell in cases.CELLS:
            rows = [row for row in self.manifest["cases"] if row["cell"] == cell]
            self.assertEqual(len(rows), 100)
            self.assertEqual({(row["boundary"], row["interleaving"]) for row in rows},
                             {(b, i) for b in range(10) for i in range(10)})
            self.assertEqual(sum(row["execution_status"] == "executable" for row in rows), expected_executable[cell])
        self.assertTrue(all(row["completed"] is False for row in self.manifest["cases"]))

    def test_classification_comes_from_required_operations(self):
        for row in self.manifest["cases"]:
            requirements = sorted({op["requires_interface"] for op in row["operations"] if "requires_interface" in op})
            self.assertEqual(row["required_interfaces"], requirements)
            if requirements:
                self.assertEqual(row["execution_status"], "pending_interface")
            elif row["cell"] == "F3" and row["boundary"] == 9:
                self.assertEqual(row["execution_status"], "limitation_only")
            else:
                self.assertEqual(row["execution_status"], "executable")
            if row["interleaving"] in (6, 7):
                self.assertEqual(row["execution_status"], "pending_interface")

    def test_committed_manifest_is_exact_reproducible_output(self):
        path = HERE / "fixtures/p2_schedules.json"
        self.assertEqual(path.read_bytes(), cases.canonical(self.manifest) + b"\n")
        self.assertEqual(cases.canonical(cases.compile_manifest()), cases.canonical(self.manifest))
        cases.validate_manifest(json.loads(path.read_bytes()))

    def test_unique_without_case_ids_and_alpha_renamed_handles(self):
        signatures = {cases.sha(cases.semantic_content(row)) for row in self.manifest["cases"]}
        self.assertEqual(len(signatures), 600)
        row = copy.deepcopy(self.manifest["cases"][0])
        reference = cases.sha(cases.semantic_content(row))
        row.update(id="different-label", cell="untrusted-label", expected={"fake": "pass"})
        for operation in row["operations"]:
            arguments = operation["arguments"]
            for key in ("group", "logical_sample"):
                if key in arguments:
                    arguments[key] = arguments[key].replace("target", "renamed-target").replace("bootstrap", "renamed-bootstrap")
            for key in ("generation", "parent"):
                if key in arguments:
                    arguments[key] = "renamed-" + arguments[key]
        self.assertEqual(cases.sha(cases.semantic_content(row)), reference)

    def test_duplicate_trace_cannot_hide_behind_new_id_or_expected(self):
        manifest = copy.deepcopy(self.manifest)
        first, second = manifest["cases"][:2]
        second["operations"] = copy.deepcopy(first["operations"])
        second["cut"] = copy.deepcopy(first["cut"])
        second["expected"] = {"a_new_label": "cannot_fix_duplicate"}
        second["semantic_sha256"] = cases.sha(cases.semantic_content(second))
        with self.assertRaisesRegex(cases.ManifestError, "duplicate semantic"):
            cases.validate_manifest(manifest)

    def test_tamper_rejected_even_with_recomputed_hash(self):
        manifest = copy.deepcopy(self.manifest)
        row = manifest["cases"][0]
        generated = next(op for op in row["operations"] if op["call"] == "fixture.generate_sample")
        generated["arguments"]["completion"] = "tampered"
        row["semantic_sha256"] = cases.sha(cases.semantic_content(row))
        with self.assertRaisesRegex(cases.ManifestError, "compiled contract"):
            cases.validate_manifest(manifest)

    def test_expected_and_status_tampering_rejected(self):
        for field, value in (("execution_status", "passed"), ("expected", {"primary": "success"}), ("completed", True)):
            with self.subTest(field=field):
                manifest = copy.deepcopy(self.manifest)
                manifest["cases"][0][field] = value
                with self.assertRaises(cases.ManifestError):
                    cases.validate_manifest(manifest)

    def test_boundary_and_interleaving_change_actual_operations(self):
        for cell in cases.CELLS:
            for fixed in range(10):
                boundary_signatures = [cases.sha(cases.semantic_content(cases.compile_case(cell, b, fixed))) for b in range(10)]
                interference_signatures = [cases.sha(cases.semantic_content(cases.compile_case(cell, fixed, i))) for i in range(10)]
                self.assertEqual(len(set(boundary_signatures)), 10)
                self.assertEqual(len(set(interference_signatures)), 10)

    def test_cut_and_dependency_tamper_rejected(self):
        for mutation in ("cut", "dependency"):
            manifest = copy.deepcopy(self.manifest)
            row = next(row for row in manifest["cases"] if row["cut"])
            if mutation == "cut":
                row["cut"]["call_offset"] = len(row["operations"]) + 1
            else:
                row["operations"][1]["depends_on"] = [9]
            row["semantic_sha256"] = cases.sha(cases.semantic_content(row))
            with self.assertRaises(cases.ManifestError):
                cases.validate_manifest(manifest)

    def test_business_depth_requirements_and_no_placeholders(self):
        for row in self.manifest["cases"]:
            self.assertIn("bootstrap_committed", row["must_reach"])
            self.assertIn("target_boundary_reached", row["must_reach"])
            self.assertTrue(row["operations"])
            if row["cut"]:
                self.assertIn("confirmed_sigkill", row["must_reach"])
            if row["interleaving"] in (8, 9):
                self.assertIn("unmodified_control_audited", row["must_reach"])
            text = cases.canonical(row).decode()
            for placeholder in ("TODO", "<placeholder>", "...具体", "<full sha256>"):
                self.assertNotIn(placeholder, text)

    def test_wrong_version_reaches_actual_accept_arguments(self):
        for boundary_index in (1, 2):
            row = cases.compile_case("X1", boundary_index, 0)
            accepted = [op for op in row["operations"] if op["call"] == "state.accept_result"
                        and op["arguments"]["logical_sample"] == "target:0"]
            self.assertEqual(accepted[0]["arguments"]["verifier_version"], "exact-v2")
            self.assertEqual(accepted[0]["arguments"]["reward"], 0.0 if boundary_index == 1 else 1.0)
        row = cases.compile_case("F4", 6, 0)
        authorizations = [op["arguments"]["logical_sample"] for op in row["operations"]
                          if op["call"] == "state.authorize_attempt"]
        self.assertEqual(authorizations, [f"target:{index}" for index in range(8)])

    def test_hand_written_prepare_only_cannot_claim_committed_target(self):
        incomplete = {"expected": {"primary": {"state": "committed_target"}}, "operations": [
            {"call": "state.prepare_generation", "arguments": {"generation": "target-state"}}]}
        with self.assertRaisesRegex(cases.ManifestError, "prepare/commit completion"):
            cases.validate_completion(incomplete)
        incomplete["operations"].append({"call": "state.commit_generation", "arguments": {"generation": "target-state"}})
        with self.assertRaisesRegex(cases.ManifestError, "optimizer completion"):
            cases.validate_completion(incomplete)

    def test_v2_changes_only_ten_x1_positive_cases(self):
        archive = HERE.parents[1] / "docs/experiments/rewardtxn-ft-20260916/p2_evidence/schedule-v1"
        old = json.loads((archive / "p2_schedules.json").read_text())
        changed = []
        for previous, current in zip(old["cases"], self.manifest["cases"]):
            if previous != current:
                changed.append(current["id"])
                self.assertNotEqual(previous["semantic_sha256"], current["semantic_sha256"])
                self.assertEqual(previous["expected"], current["expected"])
        self.assertEqual(changed, [f"X1.b00.i{index:02d}" for index in range(10)])

    def test_duplicate_mapping_is_not_masked_by_second_prepare(self):
        row = cases.compile_case("X2", 8, 0)
        operations = row["operations"]
        prepares = [index for index, op in enumerate(operations) if op["call"] == "state.prepare_generation"]
        mutation = next(index for index, op in enumerate(operations) if op["call"] == "fixture.duplicate_consumption_mapping")
        self.assertEqual(len(prepares), 1)
        self.assertLess(mutation, prepares[0])


if __name__ == "__main__":
    unittest.main()
