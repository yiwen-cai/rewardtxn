"""Compile declarative test plans only; this file executes no state or processes.

The call records document fixed calls for a future test driver, not an executable
DSL. Arguments use stable fixture handles such as g0/current: the future driver
must construct actual Owner/receipt/intent objects and hashes from those inputs.
They are not direct **kwargs for production functions. No driver exists here.
Run directly to avoid an unrelated installed ``tests`` package.
"""

import argparse
import hashlib
import json
from pathlib import Path


CELLS = ("F1", "F2", "F3", "F4", "X1", "X2")
COMPONENTS = ("model", "optimizer_master", "optimizer_moments", "optimizer_step", "scheduler",
              "rng_python", "rng_numpy", "rng_torch_cpu", "rng_device", "rng_tracker", "policy")
PENDING_INTERFACES = {"injection_journal", "second_fault_oracle", "durable_ack", "training_normalization"}


class ManifestError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def call(name, **arguments):
    return {"call": name, "arguments": arguments}


def requirement(name, operation, **arguments):
    return dict(call(operation, **arguments), requires_interface=name)


def samples(count=8):
    return [call("fixture.generate_sample", group="target", sample=i,
                 prompt=[{"role": "user", "content": "Return the integer one."}],
                 label="1", completion="1", tokens=[11, 21 + i], loss_mask=[0, 1], logprobs=[0.0, -0.25])
            for i in range(count)]


def results(count=8, version="exact-v1", start=0):
    operations = []
    for index in range(start, start + count):
        operations.extend([
            call("state.authorize_attempt", logical_sample=f"target:{index}", expected_attempt=None,
                 new_attempt="current", expected_policy_version=0),
            call("fixture.score_sample", group="target", sample=index, function="fixture-exact", verifier_version=version),
            call("state.accept_result", logical_sample=f"target:{index}", attempt="current",
                 policy_version=0, verifier_version=version, reward=1.0)])
    return operations


def prepare():
    return call("state.prepare_generation", generation="target-state", parent="g0", group="target",
                k=8, expected_samples=list(range(8)), expected_ranks=["actor:0", "actor:1"],
                data_snapshot="target-cut", ack_capability="none", logical_update="target-update")


def optimizer():
    return [call("fixture.optimizer_start", physical_update="target-physical", logical_update="target-update"),
            call("fixture.optimizer_end", physical_update="target-physical", successful=True),
            call("state.record_evidence", generation="target-state", kind="optimizer",
                 snapshot_id="target-cut", physical_updates=["target-physical"], successful=True, scheduler_applied=True)]


def checkpoint(ranks=("actor:0", "actor:1"), finalize=True):
    operations = []
    for rank in ranks:
        for component in COMPONENTS:
            operations.append(call("fixture.write_component", generation="target-state", rank=rank,
                                   component=component, byte_length=32768, fill_byte=65))
        if finalize:
            operations.append(call("state.record_evidence", generation="target-state", kind="finalize",
                                   snapshot_id="target-cut", rank=rank, writer_closed=True))
    return operations


def commit():
    return call("state.commit_generation", generation="target-state")


def boundary(cell, index):
    """Hand-written boundary operations and expected rules, never method output."""
    generated = samples()
    accepted = generated + results()
    prepared = accepted + [prepare()]
    updated = prepared + optimizer()
    complete = updated + checkpoint()
    good = {"state": "committed_target", "oracle_control": "correct_recovered", "oracle_counterexample": "invalid_commit"}
    cut = None
    if cell == "F1":
        if index < 7:
            ops = samples(index + 1) + results(index + 1) + [prepare()]
            expected = dict(good, state="reject_incomplete_group", oracle_control="safe_drop")
        elif index == 7:
            ops = generated + [call("fixture.remove_method_payload", group="target", sample=7, field="tokens")] + results() + [prepare()]
            expected = dict(good, state="not_claimed_payload_bytes_outside_state", oracle_control="unverifiable_missing_payload")
        elif index == 8:
            ops = generated + [call("fixture.remove_observer_payload_field", group="target", sample=7, field="logprobs")] + results() + [prepare()]
            expected = dict(good, state="not_claimed_payload_bytes_outside_state", oracle_control="unverifiable_missing_logprobs")
        else:
            ops = generated + results(7) + [call("fixture.assert_expected_sample", group="target", sample=7, reward_event_count=0), prepare()]
            expected = dict(good, state="reject_incomplete_group", oracle_control="safe_drop_missing_reward")
    elif cell == "F2":
        sequences = [accepted, prepared, prepared + [call("fixture.before_optimizer_start", update="target-physical")],
                     prepared + optimizer()[:1], prepared + optimizer()[:2], updated,
                     prepared + [call("state.record_evidence", generation="target-state", kind="optimizer",
                         snapshot_id="target-cut", physical_updates=["target-physical"], successful=True, scheduler_applied=False)],
                     updated + checkpoint(("actor:0",)), updated + checkpoint(finalize=False), complete + [commit()]]
        ops = sequences[index]
        cut = {"call_offset": len(ops) - 1, "when": "after_return", "signal": "SIGKILL"}
        if index == 0:
            cut["when"] = "after_last_accept_before_prepare"
        if index == 6:
            cut["when"] = "after_expected_rejection"
        if index == 9:
            cut["when"] = "after_manifest_fsync_before_token_link"
        expected = dict(good, state="promote_complete_candidate" if index == 9 else "recover_parent_with_uncommitted_obligations",
                        oracle_control="legal_rollback_recompute")
        if index == 0:
            expected["state"] = "recover_parent_no_target_intent"
    elif cell == "F3":
        if index == 0:
            ops = updated + checkpoint(("actor:0",)) + [commit()]
            cut = {"call_offset": len(ops) - 1, "when": "after_missing_rank_rejection_before_token", "signal": "SIGKILL"}
            expected = dict(good, state="recover_parent_with_uncommitted_obligations", oracle_control="legal_rollback_recompute")
        elif index == 1:
            ops = complete + [commit()]
            cut = {"call_offset": len(ops) - 1, "when": "after_token_link_before_directory_fsync", "signal": "SIGKILL"}
            expected = dict(good, state="verify_visible_complete_token_after_process_exit")
        elif index < 9:
            actions = {
                2: [requirement("durable_ack", "adapter.pause_before_ack", generation="target-state")],
                3: [requirement("durable_ack", "adapter.send_ack", generation="target-state", drop_request=True)],
                4: [requirement("durable_ack", "adapter.send_ack", generation="target-state", drop_receipt=True)],
                5: [requirement("durable_ack", "adapter.send_ack", generation="target-state", repeat=2)],
                6: [requirement("durable_ack", "adapter.send_ack", generation="target-state", epoch="previous")],
                7: [requirement("durable_ack", "adapter.send_ack", generation="g0", consuming="target-state")],
                8: [requirement("durable_ack", "adapter.send_ack", generation="target-state"),
                    requirement("durable_ack", "adapter.redeliver_consumed_work", group="target")],
            }
            ops = complete + [commit()] + actions[index]
            expected = dict(good, state="ack_protocol_contract_pending", oracle_control="ack_audit_pending")
        else:
            ops = prepared + [call("fixture.check_ack_capability", actual="none"),
                              call("oracle.audit_run", applicability="not_applicable", source="A-F3-no-ACK", decided_before_run=True)]
            expected = {"state": "no_ack_interface", "oracle_control": "not_applicable", "oracle_counterexample": "not_claimed"}
    elif cell == "F4":
        common = [call("fixture.reward_start", group="target", sample=i) for i in range(4)]
        scenarios = [generated + common + [call("fixture.observe_reward_in_flight", group="target", sample=3)],
                     generated + common + [call("fixture.reward_end", group="target", sample=i, reward=1.0) for i in range(4)],
                     samples(7) + common + results(7) + [prepare()],
                     samples(7) + [call("fixture.generate_sample_again", group="target", sample=6)] + common + results(7) + [prepare()],
                     generated + [call("fixture.reward_start", group="target", sample=0)],
                     generated + [call("fixture.reward_end", group="target", sample=0, reward=1.0),
                                  call("fixture.before_accept_result", group="target", sample=0)],
                     generated + results(7) + results(1, "exact-v2", start=7),
                     generated + [call("fixture.worker_lost", worker="reward-worker", samples=[2, 3])],
                     generated + [call("fixture.native_retry_model", samples=list(range(8)), preserve_responses=True, retry_count=1)],
                     generated + results() + [call("fixture.check_fault_window", executing_sample=None, completed_samples=list(range(8)))]]
        ops = scenarios[index]
        expected = dict(good, state="result_admission_contract_only", oracle_control="correct_recovered_model")
        if index in (2, 3):
            expected.update(state="reject_incomplete_group", oracle_control="technical_invalid_unmet_window")
        elif index == 6:
            expected.update(state="reject_unauthorized_verifier", oracle_control="invalid_commit_if_retained")
        elif index == 9:
            expected["oracle_control"] = "technical_invalid_missed_window"
    elif cell == "X1":
        mutations = [call("fixture.compute_reward", function="fixture-exact", completion="1", label="1"),
            call("fixture.compute_reward", function="one-minus-fixture-exact", completion="1", label="1", declared_verifier="exact-v2"),
            call("fixture.change_declared_verifier", group="target", sample=0, verifier="exact-v2", keep_reward=True),
            call("fixture.change_observer_label", group="target", sample=0, label="2", keep_frozen_source=True),
            call("fixture.change_response_bytes", group="target", sample=0, completion="2", preserve_observer_seal=True),
            requirement("training_normalization", "adapter.transform_training_rewards", scaling=10.0, bias=-0.5, normalization="group"),
            call("fixture.replace_reward_from_other_group", destination="target:0", source="other:0", source_completion="2", source_label="1"),
            call("fixture.change_sample_policy", group="target", sample=0, declared_policy=4, authorized_policy=0, trainer_policy=1, max_staleness=1),
            call("fixture.remove_authority_payload", group="target", sample=0, field="completion"),
            call("fixture.forge_method_success", success=True, authoritative_reward=0.0, actual_reward=0.0, completion="1", label="1")]
        result_operations = results()
        for operation in result_operations:
            arguments = operation["arguments"]
            if operation["call"] == "fixture.score_sample" and arguments["sample"] == 0 and index == 1:
                arguments.update(function="one-minus-fixture-exact", verifier_version="exact-v2")
            if operation["call"] == "state.accept_result" and arguments["logical_sample"] == "target:0":
                if index in (1, 2):
                    arguments["verifier_version"] = "exact-v2"
                if index in (1, 6, 9):
                    arguments["reward"] = 0.0
                if index == 7:
                    arguments["policy_version"] = 4
        ops = generated + [mutations[index]] + result_operations + [prepare()]
        if index == 0:
            ops += optimizer() + checkpoint() + [commit()]
        expected = dict(good)
        if index in (1, 2, 7):
            expected.update(state="reject_unauthorized_version", oracle_control="invalid_commit_if_retained")
        elif index == 5:
            expected.update(state="not_claimed", oracle_control="normalization_contract_pending")
        elif index == 8:
            expected.update(state="not_claimed", oracle_control="unverifiable_missing_authority")
        elif index == 4:
            expected.update(state="not_claimed_reward_semantics", oracle_control="unverifiable_response_fullhash_mismatch")
        elif index in (3, 6, 9):
            expected.update(state="not_claimed_reward_semantics", oracle_control="invalid_commit")
    else:
        base = samples() + results()
        tail = [
            [call("state.accept_result", logical_sample="target:0", attempt="current", identical_to_previous=True)],
            [call("state.accept_result", logical_sample="target:0", attempt="current", reward=0.0)],
            [call("state.authorize_attempt", logical_sample="target:0", expected_attempt="current", new_attempt="new", expected_policy_version=0),
             call("state.accept_result", logical_sample="target:0", attempt="current", reward=1.0),
             call("state.accept_result", logical_sample="target:0", attempt="new", reward=1.0)],
            [call("state.authorize_attempt", logical_sample="target:0", expected_attempt="current", new_attempt="new", expected_policy_version=0),
             call("state.accept_result", logical_sample="target:0", attempt="new", reward=1.0),
             call("state.accept_result", logical_sample="target:0", attempt="current", reward=1.0)],
            [call("fixture.exit_registered_owner", wait_for_exit=True), call("state.acquire_owner", expected_epoch=0, prove_registered_exit=True),
             call("state.authorize_attempt", logical_sample="target:0", expected_attempt="current", new_attempt="recovered", expected_policy_version=1),
             call("state.accept_result", logical_sample="target:0", attempt="recovered", policy_version=1, reward=1.0)],
            [call("state.accept_result", logical_sample="target:0", attempt="never-authorized", reward=1.0)],
            [call("state.accept_result", logical_sample="other:0", attempt="current", use_authorization_from="target:0")],
            [call("fixture.submit_other_run_result", source_run="foreign", destination_run="run", logical_sample="target:0")],
            [call("fixture.duplicate_consumption_mapping", logical_sample="target:0", copies=2), prepare()],
            [call("fixture.close_owner_then_submit", api="state.commit_generation", generation="g0")],
        ][index]
        ops = base + tail
        expected = dict(good, state="idempotent" if index == 0 else "accept_current_epoch" if index == 4 else "reject_stale_conflicting_or_unmapped_result",
                        oracle_control="legal_current_result" if index in (0, 4) else "invalid_commit_if_retained")
    return ops, cut, expected


def modifier(index):
    if index in (0, 1):
        return [], []
    if index == 2:
        return [call("state.record_evidence", generation="g0", kind="optimizer", identical_to_existing=True, repeat=2)], []
    if index in (3, 4):
        pre = [call("state.authorize_attempt", logical_sample="probe:0", expected_attempt=None, new_attempt="old", expected_policy_version=0),
               call("state.authorize_attempt", logical_sample="probe:0", expected_attempt="old", new_attempt="new", expected_policy_version=0)]
        order = ("new", "old") if index == 3 else ("old", "new")
        pre += [call("state.accept_result", logical_sample="probe:0", attempt=attempt, reward=1.0) for attempt in order]
        pre += [call("fixture.track_unconsumed_probe", logical_sample="probe:0", pending_action="regenerate", prompt="Return one.")]
        return pre, []
    if index == 5:
        return [call("fixture.race_owner_acquire", contenders=2, simultaneous_release=True,
                     winner_holds_until_loser_reports=True, winner_continues_target=True)], []
    if index == 6:
        return [requirement("injection_journal", "controller.persist_fired", event="primary-fault"),
                requirement("injection_journal", "controller.crash_before_observed", signal="SIGKILL"),
                requirement("injection_journal", "controller.recover_without_duplicate_injection", event="primary-fault")], []
    if index == 7:
        return [], [requirement("second_fault_oracle", "controller.inject_second_frozen_fault", event="second-fault", target="second-target"),
                    requirement("second_fault_oracle", "oracle.audit_multiple_targets", fault_ids=["primary-fault", "second-fault"])]
    if index == 8:
        return [], [call("oracle.audit_run", variant="complete_independent_control"),
                    call("fixture.remove_observer_event", event_type="optimizer_start", update="bootstrap-physical", reseal_normalized_evidence=True),
                    call("oracle.audit_run", variant="missing_optimizer_start")]
    return [], [call("oracle.audit_run", variant="complete_independent_control"),
                call("fixture.mutate_component_byte", generation="g0", rank="actor:0", component="model", offset=16384,
                     value=66, preserve_size=True, preserve_old_digest=True),
                call("state.select_recovery", expect_corruption_rejection=True),
                call("fixture.mutate_observer_byte", component="model", offset=16384, value=66, preserve_old_seal=True),
                call("oracle.audit_run", variant="fullhash_mismatch")]


def semantic_content(case):
    """Exclude labels, IDs and expected answers; runtime identities are absent.

    Operation dependencies are canonical integer positions. Plan handles have
    fixed roles (g0/target), never case-specific names, random PIDs or seeds.
    """
    namespaces = {}
    identifiers = {
        "generation": "generation", "parent": "generation", "consuming": "generation",
        "group": "group", "logical_sample": "sample", "use_authorization_from": "sample",
        "source": "sample", "destination": "sample",
        "attempt": "attempt", "expected_attempt": "attempt", "new_attempt": "attempt",
        "logical_update": "logical_update", "physical_update": "physical_update", "update": "physical_update",
        "physical_updates": "physical_update", "data_snapshot": "snapshot", "snapshot_id": "snapshot",
        "run": "run", "source_run": "run", "destination_run": "run", "worker": "worker",
        "event": "event", "fault_ids": "event", "target": "group",
    }

    def identity(namespace, value):
        if value is None:
            return value
        if isinstance(value, list):
            return [identity(namespace, item) for item in value]
        if namespace == "sample" and ":" in value:
            group, index = value.rsplit(":", 1)
            return identity("group", group) + ":" + index
        mapping = namespaces.setdefault(namespace, {})
        if value not in mapping:
            mapping[value] = f"{namespace}{len(mapping)}"
        return mapping[value]

    operations = []
    for operation in case["operations"]:
        arguments = {}
        for key, value in sorted(operation["arguments"].items()):
            if key in ("variant", "expect_corruption_rejection"):
                continue
            arguments[key] = identity(identifiers[key], value) if key in identifiers else value
        operations.append({"call": operation["call"], "arguments": arguments,
                           "depends_on": operation["depends_on"]})
    return {"operations": operations, "cut": case["cut"]}


def compile_case(cell, boundary_index, interleaving):
    target, cut, expected = boundary(cell, boundary_index)
    before, after = modifier(interleaving)
    bootstrap_order = list(range(8))
    if interleaving == 1:
        positions = [i for i, operation in enumerate(target) if operation["call"] == "fixture.generate_sample"]
        if len(positions) > 1:
            reordered = [target[i] for i in reversed(positions)]
            for position, operation in zip(positions, reordered):
                target[position] = operation
        else:
            bootstrap_order.reverse()
    prefix = [call("state.acquire_owner", run="run", expected_epoch=-1, scope="cpu_contract", verifier_version="exact-v1"),
              call("fixture.create_committed_parent", generation="g0", group="bootstrap", k=8,
                   arrival_order=bootstrap_order, expected_ranks=["actor:0", "actor:1"],
                   components=list(COMPONENTS), component_bytes=32768, component_fill_byte=65,
                   prompt="Return one.", label="1", completion="1", logical_update="bootstrap-update",
                   physical_update="bootstrap-physical")]
    if interleaving == 5:
        prefix[0] = before.pop(0)
    raw = prefix + before + target + after
    operations = [dict(operation, depends_on=[] if i == 0 else [i - 1]) for i, operation in enumerate(raw)]
    if cut is not None:
        cut["call_offset"] += len(prefix) + len(before)
    requirements = sorted({op["requires_interface"] for op in operations if "requires_interface" in op})
    status = "pending_interface" if requirements else "limitation_only" if cell == "F3" and boundary_index == 9 else "executable"
    needed = ["bootstrap_committed", "target_boundary_reached", "independent_oracle_graph_built"]
    if cut:
        needed += ["exact_cut_ready", "confirmed_sigkill", "fresh_process_recovery"]
    if interleaving == 5:
        needed += ["two_live_owner_contenders", "one_winner_one_rejection", "winner_continues_target"]
    if interleaving in (8, 9):
        needed += ["unmodified_control_audited", "specific_mutation_audited"]
    case = {"id": f"{cell}.b{boundary_index:02d}.i{interleaving:02d}", "cell": cell,
            "boundary": boundary_index, "interleaving": interleaving, "execution_status": status,
            "required_interfaces": requirements, "operations": operations, "cut": cut,
            "expected": {"primary": expected, "modifier": ("identity", "reordered_independent_arrivals", "idempotent_duplicate",
                "late_old_rejected", "old_first_rejected", "exclusive_owner", "journal_pending", "second_fault_pending",
                "specific_missing_evidence", "fullhash_corruption_rejected")[interleaving]},
            "must_reach": needed, "evidence_scope": "cpu_contract_and_independent_identity_reward_model",
            "completed": False}
    case["semantic_sha256"] = sha(semantic_content(case))
    return case


def compile_manifest():
    cases = [compile_case(cell, boundary_index, interleaving) for cell in CELLS
             for boundary_index in range(10) for interleaving in range(10)]
    counts = {key: sum(case["execution_status"] == key for case in cases)
              for key in ("executable", "limitation_only", "pending_interface")}
    return {"schema": 1, "schedule_revision": 2, "purpose": "static_cpu_schedule_inventory_not_execution_results", "seed": 20260920,
            "compiler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "counts": counts, "cases": cases}


def validate_completion(case):
    """Check positive target closure directly, independently of canonical equality."""
    primary = case.get("expected", {}).get("primary", {})
    if not isinstance(primary, dict) or primary.get("state") != "committed_target":
        return
    operations = case["operations"]
    target = [(index, op) for index, op in enumerate(operations)
              if op["arguments"].get("generation") == "target-state"]
    prepares = [index for index, op in target if op["call"] == "state.prepare_generation"]
    commits = [index for index, op in target if op["call"] == "state.commit_generation"]
    if len(prepares) != 1 or len(commits) != 1 or prepares[0] >= commits[0]:
        raise ManifestError("committed_target lacks ordered prepare/commit completion")
    middle = [op for index, op in target if prepares[0] < index < commits[0]]
    optimizer_receipts = [op["arguments"] for op in middle if op["call"] == "state.record_evidence"
                          and op["arguments"].get("kind") == "optimizer"]
    if not any(item.get("successful") is True and item.get("scheduler_applied") is True for item in optimizer_receipts):
        raise ManifestError("committed_target lacks optimizer completion receipt")
    for rank in ("actor:0", "actor:1"):
        writes = {op["arguments"]["component"] for op in middle if op["call"] == "fixture.write_component"
                  and op["arguments"].get("rank") == rank}
        finalized = any(op["call"] == "state.record_evidence" and op["arguments"].get("kind") == "finalize"
                        and op["arguments"].get("rank") == rank and op["arguments"].get("writer_closed") is True for op in middle)
        if writes != set(COMPONENTS) or not finalized:
            raise ManifestError("committed_target lacks complete rank files/finalize")


def validate_manifest(manifest):
    expected = compile_manifest()
    if manifest.get("schema") != 1 or len(manifest.get("cases", [])) != 600:
        raise ManifestError("exactly 600 schema-1 cases required")
    seen_ids, seen_semantics = set(), set()
    for case in manifest["cases"]:
        if case["id"] in seen_ids:
            raise ManifestError("duplicate case ID")
        seen_ids.add(case["id"])
        fingerprint = sha(semantic_content(case))
        if fingerprint in seen_semantics:
            raise ManifestError("duplicate semantic trace, regardless of labels")
        seen_semantics.add(fingerprint)
        if fingerprint != case["semantic_sha256"]:
            raise ManifestError("semantic hash mismatch")
        for index, operation in enumerate(case["operations"]):
            if operation["depends_on"] != ([] if index == 0 else [index - 1]):
                raise ManifestError("dependency order changed")
        if case["cut"] and not 0 <= case["cut"]["call_offset"] < len(case["operations"]):
            raise ManifestError("cut refers to absent call")
        if not set(case["required_interfaces"]) <= PENDING_INTERFACES:
            raise ManifestError("unknown required interface")
    for case in manifest["cases"]:
        validate_completion(case)
    if canonical(manifest) != canonical(expected):
        raise ManifestError("manifest differs from hand-written compiled contract")
    return dict(manifest["counts"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-manifest", type=Path)
    action.add_argument("--check-manifest", type=Path)
    args = parser.parse_args()
    manifest = compile_manifest() if args.write_manifest else json.loads(args.check_manifest.read_text())
    counts = validate_manifest(manifest)
    if args.write_manifest:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_bytes(canonical(manifest) + b"\n")
    print(json.dumps({"declared": 600, **counts, "executed": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
