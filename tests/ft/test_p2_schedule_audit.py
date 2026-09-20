"""Post-audit invariance/sensitivity tests; never import production validators."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("independent_schedule_audit", HERE / "p2_schedule_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
EVIDENCE = HERE.parents[1] / "docs/experiments/rewardtxn-ft-20260916/p2_evidence"


def save(path, value):
    path.write_bytes(audit.canonical(value))


def reidentify(directory):
    """Consistent opaque renaming + real digest cascades, never expected labels."""
    generations = list((directory / "state-run/generations").iterdir())
    renames = {path.name: "renamed-generation-" + str(index) for index, path in enumerate(generations)}
    def visit(value, key=""):
        if isinstance(value, list):
            return [visit(item, key) for item in value]
        if isinstance(value, dict):
            return {name: visit(item, name) for name, item in value.items()}
        if key in ("pid",):
            return value + 10000
        if key in ("time", "monotonic", "source_monotonic", "run_start", "end_time"):
            return value + 10000
        if key in ("boot_id", "start_time", "owner_nonce", "execution_owner", "commit_owner", "event_nonce", "run_nonce", "case_id"):
            return "unrelated-" + str(value)
        if key == "case_semantic_sha256":
            return "b" * 64
        if key == "boundary":
            return "unrelated-display-tag"
        if isinstance(value, str):
            for before, after in renames.items():
                value = value.replace(before, after)
        return value
    for path in directory.rglob("*.json"):
        original = json.loads(path.read_text())
        updated = visit(original)
        if original != updated:
            save(path, updated)
    for path in directory.glob("*-operations.jsonl"):
        rows = [visit(json.loads(line)) for line in path.read_text().splitlines()]
        path.write_bytes(b"\n".join(audit.canonical(row) for row in rows) + b"\n")
    for path in generations:
        path.rename(path.parent / renames[path.name])
    inputs = json.loads((directory / "external-fixture.json").read_text())
    generation_dirs = list((directory / "state-run/generations").iterdir())
    generation_dirs.sort(key=lambda path: json.loads((path / "intent.json").read_text())["parent"] is not None)
    for path in generation_dirs:
        intent = json.loads((path / "intent.json").read_text())
        intent["data"]["source_sha256"] = audit.sha(audit.canonical(inputs))
        if intent["parent"]:
            parent = directory / "state-run/generations" / intent["parent"]["generation"] / "token.json"
            intent["parent"]["token_sha256"] = audit.sha(audit.canonical(json.loads(parent.read_text())))
        for update in intent["updates"]:
            update["train_input_sha256"] = audit.sha(audit.canonical([entry for group in update["groups"] for entry in group["samples"]]))
        save(path / "intent.json", intent)
        if (path / "manifest.json").exists():
            manifest = json.loads((path / "manifest.json").read_text())
            manifest = dict(intent, files=manifest["files"], receipts=manifest["receipts"])
            save(path / "manifest.json", manifest)
            token = json.loads((path / "token.json").read_text())
            token.update(parent=intent["parent"], intent_sha256=audit.sha(audit.canonical(intent)), manifest_sha256=audit.sha(audit.canonical(manifest)))
            save(path / "token.json", token)
    cut_path = directory / "storage-cut.json"
    if cut_path.exists():
        cut = json.loads(cut_path.read_text())
        if cut["intent"] is not None:
            cut["intent"] = json.loads((directory / "state-run/generations" / cut["generation"] / "intent.json").read_text())
        save(cut_path, cut)
    for path in directory.glob("*-operations.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["operation"] == "state.commit_generation":
                row["token"] = json.loads((directory / "state-run/generations" / row["generation"] / "token.json").read_text())
        path.write_bytes(b"\n".join(audit.canonical(row) for row in rows) + b"\n")
    for folder in directory.glob("oracle-*"):
        if not folder.is_dir():
            continue
        spec = json.loads((folder / "spec.json").read_text())
        for name in spec["freeze"]["inputs"]:
            spec["freeze"]["inputs"][name] = audit.sha((folder / name).read_bytes())
        spec["freeze"]["external_fixture_sha256"] = audit.sha(audit.canonical(json.loads((folder / "external-fixture.json").read_text())))
        evidence = json.loads((folder / "evidence.json").read_text())
        evidence["freeze_sha256"] = audit.sha(audit.canonical(spec["freeze"]))
        save(folder / "evidence.json", evidence)
        for name in spec["observer_seal"]:
            spec["observer_seal"][name] = audit.sha((folder / name).read_bytes())
        save(folder / "spec.json", spec)


class ScheduleAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = EVIDENCE / "stage-two-r2/F2.b01.i00"

    def copied(self, name="copy"):
        return Path(shutil.copytree(self.source, self.root / name))

    def test_pid_time_generation_nonce_path_and_case_digest_cascade_are_irrelevant(self):
        before = audit.normalize_case(self.source)
        target = self.copied()
        reidentify(target)
        after = audit.normalize_case(target)
        self.assertNotEqual(json.loads((self.source / "external-fixture.json").read_text())["case_semantic_sha256"],
                            json.loads((target / "external-fixture.json").read_text())["case_semantic_sha256"])
        self.assertEqual(before["signatures"], after["signatures"], audit.first_difference(before["normalized"], after["normalized"]))
        self.assertNotEqual(before["read_sha256"], after["read_sha256"])

    def test_actual_api_state_equivalence_is_not_hidden_by_marker_or_case_labels(self):
        first = audit.normalize_case(self.source)
        second = audit.normalize_case(EVIDENCE / "stage-two-r2/F2.b02.i00")
        for lane in ("S", "P", "O", "core_joint"):
            self.assertEqual(first["signatures"][lane], second["signatures"][lane])
        self.assertNotEqual(first["signatures"]["markers"], second["signatures"]["markers"])

    def test_true_generation_arrival_order_changes_signature(self):
        target = self.copied()
        path = target / "owner0-operations.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        positions = [index for index, row in enumerate(rows) if row["operation"] == "fixture.generate_sample" and row["group"] == "target"]
        a, b = positions[:2]
        rows[a], rows[b] = rows[b], rows[a]
        for index, row in enumerate(rows):
            row["sequence"] = index
        path.write_bytes(b"\n".join(audit.canonical(row) for row in rows) + b"\n")
        self.assertNotEqual(audit.normalize_case(self.source)["signatures"]["S"], audit.normalize_case(target)["signatures"]["S"])

    def test_missing_payload_field_and_version_difference_are_not_dropped(self):
        raw = {"prompt": "p", "label": "1", "completion": "1", "tokens": [1, 2], "loss_mask": [0, 1],
               "logprobs": [0, -.2], "policy_version": 0}
        missing = copy.deepcopy(raw)
        del missing["loss_mask"]
        changed = dict(raw, policy_version=1)
        signatures = {audit.sha(audit.canonical(audit.raw_payload(value))) for value in (raw, missing, changed)}
        self.assertEqual(len(signatures), 3)

    def test_actual_rank_mapping_changes_oracle_signature(self):
        target = self.copied()
        path = target / "oracle-primary/evidence.json"
        value = json.loads(path.read_text())
        del value["states"][0]["components"]["model"]["actor:1"]
        save(path, value)
        first = audit.normalize_case(self.source)
        second = audit.normalize_case(target)
        self.assertNotEqual(first["signatures"]["O"], second["signatures"]["O"])

    def test_unknown_event_field_is_an_explicit_parse_gap(self):
        target = self.copied()
        path = target / "owner0-operations.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["new_semantic_input"] = "never silently discard"
        path.write_bytes(b"\n".join(audit.canonical(row) for row in rows) + b"\n")
        with self.assertRaises(audit.Unsupported):
            audit.normalize_case(target)

    def test_report_equivalence_requires_investigation_not_functional_failure(self):
        run = self.root / "run"
        run.mkdir()
        shutil.copytree(self.source, run / "left")
        shutil.copytree(EVIDENCE / "stage-two-r2/F2.b02.i00", run / "right")
        save(run / "inventory.json", {"source_sha256": {"driver": "provenance-only"}})
        report = audit.audit_runs([run], self.root / "audit")
        self.assertEqual(report["status"], "investigation_required")
        self.assertEqual(report["counts"]["distinct_labels_parsed"], 2)
        collision = next(item for item in report["investigations"] if item["kind"] == "cross_case_core_equivalence_requires_investigation")
        self.assertEqual(collision["functional_verdict"], "not_inferred")
        self.assertNotEqual(collision["marker_differences"][0]["difference"], None)

    def copied_source(self, relative, name):
        return Path(shutil.copytree(EVIDENCE / relative, self.root / name))

    def test_new_scope_opaque_identity_cascade_invariant(self):
        source = EVIDENCE / 'stage-four-r1/X2.b07.i00'
        before = audit.normalize_case(source)
        copied = self.copied_source('stage-four-r1/X2.b07.i00', 'foreign')
        reidentify(copied)
        after = audit.normalize_case(copied)
        self.assertEqual(before['signatures'], after['signatures'], audit.first_difference(before['normalized'], after['normalized']))

    def test_exit_takeover_and_corruption_checker_are_bound_to_real_actors(self):
        source = EVIDENCE / 'stage-four-r1/X2.b04.i09'
        normalized = audit.normalize_case(source)['normalized']
        roles = {p['role']: p for p in normalized['P']}
        takeover = next(r for r in roles['scope-recovery']['receipts'] if r['type'] == 'scope_recovered')
        self.assertEqual(takeover['epoch'], 1)
        self.assertEqual(takeover['prior_registered'], [{'role': 'writer', 'exit_code': 0}])
        self.assertTrue(takeover['new_process'])
        self.assertEqual(roles['corruption-check']['final_owner_binding']['epoch'], 2)
        self.assertEqual(roles['corruption-check']['final_owner_binding']['registered_roles'], ['corruption-check'])
        target = self.copied_source('stage-four-r1/X2.b04.i09', 'takeover')
        messages = json.loads((target / 'parent-receipts.json').read_text())
        next(m for m in messages if m['type'] == 'scope_recovered')['prior_processes'][0]['pid'] += 1000000
        save(target / 'parent-receipts.json', messages)
        with self.assertRaises(audit.Unsupported):
            audit.normalize_case(target)

    def test_foreign_root_and_sample_mismatch_have_material_core_evidence(self):
        foreign = audit.normalize_case(EVIDENCE / 'stage-four-r1/X2.b07.i00')['normalized']['S']
        event = next(e for a in foreign['actors'] for e in a['events'] if e['operation'] == 'fixture.foreign_run_result')
        self.assertTrue(event['distinct_run'])
        self.assertTrue(event['recorded_acceptance_matches'])
        source = EVIDENCE / 'stage-four-r1/X2.b06.i00'
        before = audit.normalize_case(source)
        target = self.copied_source('stage-four-r1/X2.b06.i00', 'mismatch')
        path = target / 'owner0-operations.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        event = next(e for e in rows if e['operation'] == 'fixture.submit_scope')
        event['submitted']['sample'] = event['authorized']['sample']
        path.write_bytes(b'\n'.join(audit.canonical(e) for e in rows) + b'\n')
        self.assertNotEqual(before['signatures']['S'], audit.normalize_case(target)['signatures']['S'])

    def test_f4_duplicate_raw_and_technical_invalid_not_labeled_away(self):
        first = audit.normalize_case(EVIDENCE / 'full-408-r1-F4/F4.b02.i08')
        second = audit.normalize_case(EVIDENCE / 'full-408-r1-F4/F4.b03.i08')
        self.assertNotEqual(first['signatures']['S'], second['signatures']['S'])
        event = next(e for a in second['normalized']['S']['actors'] for e in a['events'] if e['operation'] == 'fixture.generate_sample_again')
        self.assertTrue(event['previously_generated'])
        self.assertTrue(event['same_payload'])
        missed = audit.normalize_case(EVIDENCE / 'full-408-r1-F4/F4.b09.i08')
        invalid = [e for g in missed['normalized']['O'] for e in g['events'] if e['type'] == 'technical_invalid']
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0]['reason'], 'technical_invalid_missed_window')
        self.assertTrue(any('fault_window_contract' in g for g in missed['normalized']['O']))

    def test_lost_retry_normative_events_do_not_invent_processes(self):
        for b in (7, 8):
            source = EVIDENCE / f'full-408-r1-F4/F4.b{b:02d}.i00'
            normalized = audit.normalize_case(source)['normalized']
            self.assertEqual([p['role'] for p in normalized['P']], ['writer'])
            for graph in normalized['O']:
                self.assertFalse(any(e['type'] in ('worker_lost_model', 'native_retry_model') for e in graph['events']))
            self.assertTrue(any('normative_events' in claim for claim in normalized['markers']['oracle_claims']))
        source = EVIDENCE / 'full-408-r1-F4/F4.b07.i00'
        before = audit.normalize_case(source)
        changed = self.copied_source('full-408-r1-F4/F4.b07.i00', 'marker-only')
        path = changed / 'owner0-operations.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        next(e for e in rows if e['operation'] == 'fixture.worker_lost')['samples'] = [2]
        path.write_bytes(b'\n'.join(audit.canonical(e) for e in rows) + b'\n')
        for folder in changed.glob('oracle-*'):
            if not folder.is_dir():
                continue
            evidence = json.loads((folder / 'evidence.json').read_text())
            for event in evidence['events']:
                if event['type'] == 'worker_lost_model':
                    event['samples'] = [2]
            save(folder / 'evidence.json', evidence)
            spec = json.loads((folder / 'spec.json').read_text())
            spec['observer_seal']['evidence.json'] = audit.sha((folder / 'evidence.json').read_bytes())
            save(folder / 'spec.json', spec)
        after = audit.normalize_case(changed)
        self.assertEqual(before['signatures']['core_joint'], after['signatures']['core_joint'])
        self.assertNotEqual(before['signatures']['markers'], after['signatures']['markers'])
        target = self.copied_source('full-408-r1-F4/F4.b08.i00', 'unknown')
        path = target / 'owner0-operations.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        next(e for e in rows if e['operation'] == 'fixture.native_retry_model')['new_hidden_scope'] = 'opaque'
        path.write_bytes(b'\n'.join(audit.canonical(e) for e in rows) + b'\n')
        with self.assertRaises(audit.Unsupported):
            audit.normalize_case(target)


if __name__ == "__main__":
    unittest.main()
