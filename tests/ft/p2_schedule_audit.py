"""Read-only, stdlib post-audit of saved CPU schedules; no functional verdicts."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


class Unsupported(ValueError):
    pass


def shape(value, allowed, required=""):
    if not isinstance(value, dict) or set(value) - set(allowed.split()) or set(required.split()) - set(value):
        raise Unsupported("unsupported/missing fields: " + repr(value if not isinstance(value, dict) else sorted(value)))
    return value


class Names:
    def __init__(self):
        self.maps = {}

    def get(self, kind, value):
        if value is None:
            return None
        table = self.maps.setdefault(kind, {})
        if value not in table:
            table[value] = f"{kind}{len(table)}"
        return table[value]


COMPONENTS = {"model", "optimizer_master", "optimizer_moments", "optimizer_step", "scheduler", "rng_python", "rng_numpy", "rng_torch_cpu", "rng_device", "rng_tracker", "policy"}


def component_paths(value, oracle=False):
    if not isinstance(value, dict) or set(value) - (COMPONENTS | ({"data"} if oracle else set())):
        raise Unsupported("unknown component schema")
    for component, ranks in value.items():
        if not isinstance(ranks, dict):
            raise Unsupported("invalid rank mapping")
        for rank, paths in ranks.items():
            if not re.fullmatch(r"actor:[0-9]+", rank) or not isinstance(paths, list):
                raise Unsupported("unknown rank mapping")
            if not oracle and any(not re.fullmatch(r"actor-[0-9]+/(?:" + "|".join(COMPONENTS) + r")\.bin", path) for path in paths):
                raise Unsupported("unknown component path")


def reason(value):
    exact = {"corrupt token or snapshot", "inactive or inherited owner", "incomplete group", "optimizer/scheduler evidence incomplete",
             "result violates policy/verifier authorization", "unauthorized attempt", "conflicting result", "stale result", "verified committed chain", "completed durable candidate"}
    suffix = value.removeprefix("incomplete candidate: ")
    if value in exact or suffix == "cannot read optimizer.json" or any(suffix == "cannot read rank-" + sha(rank.encode()) + ".json" for rank in ("actor:0", "actor:1")):
        return value
    raise Unsupported("unknown rejection reason")


RAW_KEYS = "prompt label completion tokens loss_mask logprobs policy_version prompt_ids completion_ids prompt_sha256"


def raw_payload(value):
    shape(value, RAW_KEYS)
    result = {key: item for key, item in value.items() if key != "prompt_sha256"}
    if "prompt_sha256" in value:
        result["prompt_digest_matches"] = value["prompt_sha256"] == sha(canonical(value.get("prompt")))
    return result


COMMON = "operation sequence identity monotonic"
# No arbitrary event/field passthrough; unknown schemas become investigation items.
EVENT_FIELDS = {
    "state.acquire_owner": "epoch result", "fixture.generate_sample": "group sample payload_file payload_sha256 evidence_level",
    "state.authorize_attempt": "sample attempt", "state.accept_result": "sample attempt result reward",
    "state.prepare_generation": "generation logical_group intent_file", "state.record_evidence": "generation evidence",
    "state.record_evidence.identical": "generation evidence repeat result", "state.commit_generation": "generation token",
    "state.select_recovery": "decision", "fixture.score_sample": "sample function value source declared_verifier",
    "fixture.write_component": "generation rank component relative_path size sha256",
    "fixture.compute_reward": "function actual_value completion label exact_value declared_verifier sample",
    "fixture.submitted_declaration": "sample payload authorized", "fixture.scheduler_evidence_attempt": "generation evidence",
    "fixture.assert_idempotent_generation": "before after", "fixture.remove_method_payload": "sample field before_sha256 after_sha256 payload_file",
    "fixture.change_response_bytes": "sample field value before_sha256 after_sha256 payload_file seal_file",
    "fixture.remove_authority_payload": "sample field file sha256 before after", "fixture.change_observer_label": "sample field file sha256 before after",
    "fixture.replace_reward_from_other_group": "destination source source_file source_sha256 source_reward target_exact_reward",
    "fixture.change_declared_verifier": "sample verifier before_reward after_reward",
    "fixture.change_sample_policy": "sample declared_policy authorized_policy trainer_policy max_staleness",
    "fixture.forge_method_success": "sample success authoritative_reward actual_reward correct_reward",
    "fixture.optimizer_start": "physical_update evidence_level", "fixture.optimizer_end": "physical_update successful evidence_level",
    "fixture.before_optimizer_start": "physical_update", "fixture.reward_start": "sample evidence_level",
    "fixture.track_unconsumed_probe": "sample pending_action prompt",
    "target_boundary": "accepted bad_sample commit_receipt ends executing_sample field generated generation limitation rejection result starts when old_attempt receipt_equal evidence_level",
    "cut_ready": "generation when",
    "fixture.submit_scope": "submitted authorized payload",
    "fixture.foreign_run_result": "root attempt receipt",
    "fixture.duplicate_consumption_mapping": "intent data",
    "fixture.recovered_policy_payload": "payload_file before_sha256 after_sha256 policy_version limitation",
    "fixture.generate_sample_again": "group sample payload_file payload_sha256 unique_new_sample",
    "fixture.reward_end": "sample reward function",
    "fixture.observe_reward_in_flight": "sample evidence_level",
    "fixture.before_accept_result": "sample",
    "fixture.check_fault_window": "executing_sample completed_samples unique_generated classification evidence_level",
    "fixture.worker_lost": "worker samples evidence_level",
    "fixture.retry_score": "sample reward function evidence_level",
    "fixture.native_retry_model": "before after retry_count preserve_responses evidence_level",
}
REJECTS = {"state.accept_result.wrong_verifier", "state.accept_result.wrong_policy", "state.accept_result.old_probe",
           "state.accept_result.old_target_first", "state.prepare_generation.after_rejected_reward",
           "state.prepare_generation.missing_reward", "state.prepare_generation.incomplete",
           "state.commit_generation.missing_rank", "state.commit_generation.closed_owner", "state.select_recovery.corrupt",
           "state.record_evidence.bad_scheduler", "state.accept_result.conflict", "state.accept_result.unmapped",
           "state.accept_result.foreign", "state.prepare_generation.duplicate"}
for _name in REJECTS:
    EVENT_FIELDS[_name] = "reason result"
MARKERS = {"fixture.optimizer_start", "fixture.optimizer_end", "fixture.before_optimizer_start", "fixture.reward_start",
           "fixture.track_unconsumed_probe", "target_boundary", "cut_ready", "fixture.forge_method_success",
           "fixture.observe_reward_in_flight", "fixture.before_accept_result", "fixture.worker_lost", "fixture.native_retry_model", "fixture.check_fault_window"}


def window_contract(value):
    keys = "required_unique_generated required_executing_sample required_incomplete_rewards scope planned_unique_generated planned_all_rewards_completed preclassified"
    shape(value, keys, keys)
    if value["scope"] != "CPU normative events only" or value["preclassified"] not in (
            "technical_invalid_unmet_window", "technical_invalid_missed_window", "model_contract_only"):
        raise Unsupported("unknown F4 window contract")
    return dict(value)


class CaseReader:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.names = Names()
        self.read_hashes = {}
        self.inputs = self.read("external-fixture.json")
        shape(self.inputs, "case_semantic_sha256 groups bootstrap_arrival_order target_arrival_order expansion", "groups")
        # Planned arrival_order and case digest never enter a signature.
        self.source = sorted([[raw_payload(raw) for raw in rows] for rows in self.inputs["groups"].values()], key=canonical)
        self.raw_samples = {}
        self.original_samples = {}
        self.intent_cache = {}
        self.binary_pool = None
        self.policy_mutations = {}

    def path(self, relative):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise Unsupported("unsafe artifact reference")
        target = self.root / path
        if any(item.is_symlink() for item in [target, *target.parents] if item != self.root.parent and self.root in (item, *item.parents)):
            raise Unsupported("symlink artifact")
        return target

    def bytes(self, relative):
        value = self.path(relative).read_bytes()
        self.read_hashes[str(relative)] = sha(value)
        return value

    def read(self, relative):
        return json.loads(self.bytes(relative))

    def binary_details(self, details):
        shape(details, "sha256 size", "sha256 size")
        if self.binary_pool is None:
            self.binary_pool = set()
            for path in self.root.glob("state-run/generations/*/checkpoint/actor-*/*.bin"):
                blob = self.bytes(str(path.relative_to(self.root)))
                self.binary_pool.add((sha(blob), len(blob)))
        if (details["sha256"], details["size"]) not in self.binary_pool:
            raise Unsupported("binary digest lacks resolvable actual content")
        return details

    def sample(self, value):
        if isinstance(value, int):
            return value
        group, index = value.rsplit(":", 1)
        return [self.names.get("group", group), int(index)]

    def attempt(self, value):
        shape(value, "sample attempt epoch owner_nonce versions", "sample attempt epoch owner_nonce versions")
        shape(value["versions"], "policy_version verifier_version", "policy_version verifier_version")
        return {"sample": self.sample(value["sample"]), "attempt": self.names.get("attempt", value["attempt"]),
                "epoch": value["epoch"], "owner": self.names.get("owner", value["owner_nonce"]), "versions": value["versions"]}

    def payload(self, value, sample, reward=None):
        shape(value, "policy_version verifier_version response_sha256 reward_sha256 tensor_input_sha256",
              "policy_version verifier_version response_sha256 reward_sha256 tensor_input_sha256")
        if sample not in self.raw_samples:
            raise Unsupported("unresolved payload sample")
        raw = self.raw_samples[sample]
        result = {key: value[key] for key in ("policy_version", "verifier_version")}
        result.update(raw=raw_payload(raw), response_matches=value["response_sha256"] == sha(canonical(raw)))
        if reward is None:
            candidates = [score for score in (0.0, 1.0) if sha(canonical(score)) == value["reward_sha256"]]
            if len(candidates) != 1:
                raise Unsupported("unresolved reward digest")
            reward = candidates[0]
        result.update(reward=reward, reward_matches=value["reward_sha256"] == sha(canonical(reward)),
                      tensor_matches=value["tensor_input_sha256"] == sha(canonical({"raw": raw, "reward": reward})))
        return result

    def receipt(self, value):
        shape(value, "attempt payload", "attempt payload")
        return {"attempt": self.attempt(value["attempt"]),
                "payload": self.payload(value["payload"], value["attempt"]["sample"])}

    def evidence(self, value):
        shape(value, "kind snapshot_id physical_updates successful scheduler_applied rank writer_closed", "kind snapshot_id")
        result = dict(value)
        result["snapshot_id"] = self.names.get("snapshot", value["snapshot_id"])
        if "physical_updates" in value:
            result["physical_updates"] = [self.names.get("update", name) for name in value["physical_updates"]]
        return result

    def parent(self, value):
        if value is None:
            return None
        shape(value, "generation token_sha256", "generation token_sha256")
        raw = self.read(f"state-run/generations/{value['generation']}/token.json")
        return {"generation": self.names.get("generation", value["generation"]),
                "token_reference_matches": sha(canonical(raw)) == value["token_sha256"]}

    def intent(self, generation, supplied=None):
        if supplied is None and generation in self.intent_cache:
            return self.intent_cache[generation]
        value = supplied if supplied is not None else self.read(f"state-run/generations/{generation}/intent.json")
        shape(value, "ack_capability components config_sha256 data data_snapshot_id execution_epoch execution_owner expected_ranks generation parent run_nonce schema updates",
              "data generation parent updates components expected_ranks execution_epoch")
        if value["config_sha256"] != sha(b"p2-cpu-driver"):
            raise Unsupported("unknown fixture configuration")
        data = value["data"]
        shape(data, "consumed cursor drawn epoch pending shuffle_state source_sha256", "consumed cursor drawn epoch pending source_sha256")
        component_paths(value["components"])
        shape(data["shuffle_state"], "seed", "seed")
        result = {"generation": self.names.get("generation", generation), "parent": self.parent(value["parent"]),
                  "ack_capability": value["ack_capability"], "schema": value["schema"], "execution_epoch": value["execution_epoch"],
                  "owner": self.names.get("owner", value["execution_owner"]), "ranks": value["expected_ranks"],
                  "snapshot": self.names.get("snapshot", value["data_snapshot_id"]), "components": value["components"],
                  "data": {"consumed": [self.sample(x) for x in data["consumed"]], "drawn": [self.sample(x) for x in data["drawn"]],
                           "cursor": data["cursor"], "epoch": data["epoch"], "pending": self.pending(data["pending"]),
                           "shuffle_state": data["shuffle_state"], "source": self.source,
                           "source_reference_matches": data["source_sha256"] == sha(canonical(self.inputs))}, "updates": []}
        for update in value["updates"]:
            shape(update, "groups logical_update_id physical_update_id train_input_sha256", "groups logical_update_id physical_update_id train_input_sha256")
            groups = []
            all_entries = []
            for group in update["groups"]:
                shape(group, "k logical_group_id prompt prompt_sha256 samples", "k logical_group_id prompt prompt_sha256 samples")
                entries = []
                for entry in group["samples"]:
                    shape(entry, "receipt sample sample_index", "receipt sample sample_index")
                    entries.append({"receipt": self.receipt(entry["receipt"]), "sample": self.sample(entry["sample"]), "index": entry["sample_index"]})
                all_entries.extend(group["samples"])
                groups.append({"group": self.names.get("group", group["logical_group_id"]), "k": group["k"], "prompt": group["prompt"],
                               "prompt_digest_matches": sha(canonical(group["prompt"])) == group["prompt_sha256"], "samples": entries})
            result["updates"].append({"groups": groups, "logical": self.names.get("logical-update", update["logical_update_id"]),
                                      "physical": self.names.get("update", update["physical_update_id"]),
                                      "train_reference_matches": update["train_input_sha256"] == sha(canonical(all_entries))})
        if supplied is None:
            self.intent_cache[generation] = result
        return result

    def pending(self, values):
        result = []
        for value in values:
            shape(value, "sample action prompt prompt_sha256 k", "sample action prompt prompt_sha256 k")
            result.append({"sample": self.sample(value["sample"]), "action": value["action"], "prompt": value["prompt"], "k": value["k"],
                           "prompt_digest_matches": value["prompt_sha256"] == sha(canonical(value["prompt"]))})
        return result

    def token(self, value):
        shape(value, "schema run_nonce generation parent intent_sha256 manifest_sha256 execution_epoch commit_epoch commit_owner",
              "generation parent intent_sha256 manifest_sha256 execution_epoch commit_epoch commit_owner")
        generation = value["generation"]
        intent = self.read(f"state-run/generations/{generation}/intent.json")
        manifest = self.read(f"state-run/generations/{generation}/manifest.json")
        shape(manifest, "ack_capability components config_sha256 data data_snapshot_id execution_epoch execution_owner expected_ranks generation parent run_nonce schema updates files receipts", "files receipts")
        if {key: item for key, item in manifest.items() if key not in ("files", "receipts")} != intent:
            raise Unsupported("manifest differs from captured intent")
        # Raw hashes are checked as relationships, never fed into signatures.
        return {"generation": self.names.get("generation", generation), "parent": self.parent(value["parent"]),
                "intent_matches": value["intent_sha256"] == sha(canonical(intent)),
                "manifest_matches": value["manifest_sha256"] == sha(canonical(manifest)),
                "execution_epoch": value["execution_epoch"], "commit_epoch": value["commit_epoch"],
                "commit_owner": self.names.get("owner", value["commit_owner"]), "intent": self.intent(generation),
                "receipts": [self.evidence(x) for x in manifest["receipts"]],
                "files": {name: self.binary_details(details) for name, details in manifest["files"].items()}}

    def decision(self, value):
        shape(value, "checkpoint generation pending reason rollback_intents", "checkpoint generation pending reason rollback_intents")
        def generation_from_path(path):
            match = re.search(r"/generations/([^/]+)/", path)
            if not match:
                raise Unsupported("unresolved generation path")
            return self.names.get("generation", match.group(1))
        return {"generation": self.names.get("generation", value["generation"]),
                "checkpoint_generation": generation_from_path(value["checkpoint"]) if value["checkpoint"] else None,
                "pending": self.pending(value["pending"]), "reason": reason(value["reason"]),
                "rollback_generations": [generation_from_path(path) for path in value["rollback_intents"]]}

    def event(self, event):
        op = event["operation"]
        if op not in EVENT_FIELDS:
            raise Unsupported("unknown operation: " + op)
        shape(event, COMMON + " " + EVENT_FIELDS[op], COMMON)
        shape(event["identity"], "boot_id pid start_time", "boot_id pid start_time")
        result = {"operation": ".".join(op.split(".")[:2]) if op in REJECTS or op == "state.record_evidence.identical" else op}
        if op in REJECTS:
            result.update(result=event["result"], reason=reason(event["reason"]))
        elif op == "state.acquire_owner":
            result.update({key: event[key] for key in ("epoch", "result") if key in event})
        elif op == "fixture.generate_sample":
            sample = f"{event['group']}:{event['sample']}"
            raw = self.read(event["payload_file"])
            original = self.inputs["groups"][event["group"]][event["sample"]]
            historical = sample in self.policy_mutations
            if historical:
                mutation = self.policy_mutations[sample]
                if event["payload_sha256"] != mutation["before_sha256"] or raw.get("policy_version") != mutation["policy_version"]:
                    raise Unsupported("policy mutation lacks historical binding")
                raw = original
            self.raw_samples[sample] = raw
            self.original_samples[sample] = original
            result.update(sample=self.sample(sample), raw=raw_payload(raw), original=raw_payload(original),
                          raw_matches_record=(sha(json.dumps(raw, sort_keys=True, indent=2, allow_nan=False).encode())
                              if historical else sha(self.bytes(event["payload_file"]))) == event["payload_sha256"])
        elif op == "fixture.submit_scope":
            result.update(submitted=self.attempt(event["submitted"]), authorized=self.attempt(event["authorized"]))
            if "payload" in event:
                result["payload"] = self.payload(event["payload"], event["authorized"]["sample"])
        elif op == "fixture.foreign_run_result":
            foreign = self.read(event["root"] + "/control.json")
            local = self.read("state-run/control.json")
            fields = "epoch run_nonce config_sha256 attempts accepted head verifier_version owner_nonce processes scope"
            shape(foreign, fields, fields)
            if (foreign["head"] is not None or foreign["scope"] != "cpu_contract"
                    or set(foreign["attempts"]) != {event["attempt"]["sample"]}
                    or set(foreign["accepted"]) != {event["attempt"]["sample"]}):
                raise Unsupported("unknown foreign control contents")
            if foreign["config_sha256"] != sha(b"p2-cpu-driver"):
                raise Unsupported("unknown foreign fixture config")
            for identity in foreign["processes"]:
                shape(identity, "pid boot_id start_time", "pid boot_id start_time")
            result.update(attempt=self.attempt(event["attempt"]), receipt=self.receipt(event["receipt"]),
                foreign_epoch=foreign["epoch"], frozen_verifier=foreign["verifier_version"],
                distinct_run=foreign["run_nonce"] != local["run_nonce"],
                distinct_owner=foreign["owner_nonce"] != local["owner_nonce"],
                recorded_attempt_matches=foreign["attempts"].get(event["attempt"]["sample"]) == event["attempt"],
                recorded_acceptance_matches=foreign["accepted"].get(event["attempt"]["sample"]) == event["receipt"],
                same_registered_process=foreign["processes"] == [event["identity"]])
        elif op == "fixture.duplicate_consumption_mapping":
            supplied = dict(event["intent"])
            shape(supplied, "parent ack_capability config_sha256 expected_ranks data_snapshot_id components updates",
                  "parent ack_capability config_sha256 expected_ranks data_snapshot_id components updates")
            owner = supplied["updates"][0]["groups"][0]["samples"][0]["receipt"]["attempt"]
            supplied.update(data=event["data"], generation="rejected-submission", schema=1,
                execution_epoch=owner["epoch"], execution_owner=owner["owner_nonce"])
            result["submitted_intent"] = self.intent("rejected-submission", supplied)
        elif op == "fixture.recovered_policy_payload":
            raw = self.read(event["payload_file"])
            sample = next((key for key, value in self.policy_mutations.items() if value["payload_file"] == event["payload_file"]), None)
            if sample is None:
                raise Unsupported("unbound recovered policy payload")
            result.update(sample=self.sample(sample), before=raw_payload(self.raw_samples[sample]), after=raw_payload(raw),
                policy_version=event["policy_version"], after_matches_file=sha(self.bytes(event["payload_file"])) == event["after_sha256"],
                changed=event["before_sha256"] != event["after_sha256"])
            self.raw_samples[sample] = raw
        elif op == "fixture.generate_sample_again":
            sample = f"{event['group']}:{event['sample']}"
            raw = self.read(event["payload_file"])
            result.update(sample=self.sample(sample), raw=raw_payload(raw),
                previously_generated=sample in self.raw_samples, same_payload=raw == self.raw_samples.get(sample),
                matches_record=sha(self.bytes(event["payload_file"])) == event["payload_sha256"],
                unique_new_sample=event["unique_new_sample"])
        elif op == "fixture.native_retry_model":
            if set(event["before"]) != set(event["after"]) or set(event["before"]) != {str(i) for i in range(8)}:
                raise Unsupported("unknown retry file set")
            result.update(retry_count=event["retry_count"], preserve_responses=event["preserve_responses"],
                files=[{"sample": int(i), "unchanged": event["before"][i] == event["after"][i],
                    "after_matches_file": event["after"][i] == sha(self.bytes(f"method-target-{i}.json"))}
                    for i in sorted(event["before"])])
        elif op == "state.authorize_attempt":
            result.update(sample=self.sample(event["sample"]), attempt=self.attempt(event["attempt"]))
        elif op == "state.accept_result":
            result.update(sample=self.sample(event["sample"]), attempt=self.attempt(event["attempt"]), receipt=self.receipt(event["result"]), reward=event["reward"])
        elif op == "state.prepare_generation":
            result.update(intent=self.intent(event["generation"]))
        elif op.startswith("state.record_evidence"):
            result.update(generation=self.names.get("generation", event["generation"]), evidence=self.evidence(event["evidence"]))
            if "result" in event:
                result["returned"] = self.evidence(event["result"])
        elif op == "state.commit_generation":
            result["token"] = self.token(event["token"])
        elif op == "state.select_recovery":
            result["decision"] = self.decision(event["decision"])
        elif op == "fixture.write_component":
            result.update(generation=self.names.get("generation", event["generation"]), rank=event["rank"], component=event["component"],
                          declared=self.binary_details({"size": event["size"], "sha256": event["sha256"]}),
                          actual_binary_sha256=sha(self.bytes(event["relative_path"])))
        elif op == "fixture.submitted_declaration":
            result.update(authorized=self.attempt(event["authorized"]), payload=self.payload(event["payload"], event["sample"]))
        elif op == "fixture.scheduler_evidence_attempt":
            result.update(generation=self.names.get("generation", event["generation"]), evidence=self.evidence(event["evidence"]))
        elif op == "fixture.assert_idempotent_generation":
            if set(event["before"]) != set(event["after"]):
                result["file_sets_equal"] = False
            else:
                result["file_sets_equal"] = True
            # File names retain component/rank identity; JSON digest cascades are relationship-only.
            result["files"] = sorted(event["before"])
            result["unchanged"] = {name: value == event["after"].get(name) for name, value in event["before"].items()}
        elif op in ("fixture.remove_method_payload", "fixture.change_response_bytes"):
            result.update(sample=event["sample"], field=event["field"], actual=raw_payload(self.read(event["payload_file"])),
                          before_after_equal=event["before_sha256"] == event["after_sha256"],
                          after_matches_file=event["after_sha256"] == sha(self.bytes(event["payload_file"])))
            if "value" in event:
                result["value"] = event["value"]
            if "seal_file" in event:
                seal = self.read(event["seal_file"])
                result["old_seal_preserved"] = seal[event["payload_file"]] == event["before_sha256"]
        elif op in ("fixture.remove_authority_payload", "fixture.change_observer_label"):
            result.update(sample=event["sample"], field=event["field"], before=raw_payload(event["before"]), after=raw_payload(event["after"]),
                          actual=raw_payload(self.read(event["file"])), matches_record=sha(self.bytes(event["file"])) == event["sha256"])
        elif op == "fixture.replace_reward_from_other_group":
            result.update(destination=self.sample(event["destination"]), source=self.sample(event["source"]),
                          source_raw=raw_payload(self.read(event["source_file"])), source_reward=event["source_reward"],
                          target_exact_reward=event["target_exact_reward"], source_matches=sha(self.bytes(event["source_file"])) == event["source_sha256"])
        else:
            # Explicit remaining flat schema; no digest/path/identity is allowed here.
            for key in EVENT_FIELDS[op].split():
                if key not in event or key in ("when", "result", "limitation", "evidence_level", "commit_receipt", "rejection"):
                    continue
                value = event[key]
                if key == "old_attempt":
                    value = self.attempt(value)
                elif key == "worker":
                    value = self.names.get("worker", value)
                elif key == "generation":
                    value = self.names.get("generation", value)
                elif key == "physical_update":
                    value = self.names.get("update", value)
                elif key in ("sample", "source") and isinstance(value, str):
                    value = self.sample(value)
                result[key] = value
        return result

    def storage(self):
        path = self.root / "storage-cut.json"
        if not path.exists():
            return None
        value = self.read(path.name)
        shape(value, "artifacts before_start_marker declared_last_call finalized_ranks generation intent manifest_present missing_rank_rejected observed_before_sigkill observed_cut optimizer_ends optimizer_starts raw_receipt_sequences receipts saved_ranks scheduler_rejected token_present writer_identity",
              "artifacts generation intent receipts manifest_present token_present observed_before_sigkill")
        if value["intent"] is not None and value["intent"] != self.read(f"state-run/generations/{value['generation']}/intent.json"):
            raise Unsupported("cut intent no longer matches immutable saved intent")
        files = []
        for name, details in sorted(value["artifacts"].items()):
            shape(details, "sha256 size", "sha256 size")
            if name.startswith("checkpoint/"):
                files.append([name, self.binary_details(details)])
            elif name.startswith(".tmp-"):
                files.append(["temporary-file", {"present": True}])
            elif name in ("intent.json", "manifest.json", "token.json") or name.startswith("receipts/"):
                files.append([name, {"present": True}])
            else:
                raise Unsupported("unknown cut artifact: " + name)
        return {"generation": self.names.get("generation", value["generation"]), "files": files,
                "intent": self.intent(value["generation"]) if value["intent"] is not None else None,
                "receipts": sorted([self.evidence(item) for item in value["receipts"].values()], key=canonical),
                "manifest_present": value["manifest_present"], "token_present": value["token_present"],
                "captured_before_kill": value["observed_before_sigkill"]}


O_META = "id type time source_monotonic boot_id pid start_time cgroup event_nonce rank role run_nonce timeline"
O_EVENTS = {
    "initial_state": "state", "checkpoint_loaded": "state epoch", "authorize": "group sample attempt epoch policy_version verifier_version",
    "optimizer_start": "update epoch tensor_sha256", "optimizer_end": "update epoch successful scheduler_applied",
    "checkpoint_schedule": "state snapshot_id", "checkpoint_finalize": "state snapshot_id actor_rank",
    "checkpoint_persisted": "state snapshot_id", "generation_done": "group sample", "reward_done": "group sample",
    "reward_start": "group sample", "fault_observed": "fault_id boundary condition target_group",
    "technical_invalid": "origin reason unique_generated completed_samples evidence_level",
    "worker_lost_model": "samples scope", "native_retry_model": "samples retry_count preserve_responses scope",
    "role_ready": "boundary target_group", "work_dropped": "group", "safe_stop": "",
}


def normalize_oracle(reader, folder):
    prefix = folder + "/"
    spec = reader.read(prefix + "spec.json")
    shape(spec, "freeze observer_seal", "freeze observer_seal")
    freeze = spec["freeze"]
    shape(freeze, "schema run_nonce evidence_level timeline run_start expected_ranks fault_id initial_state target_group target_role k max_policy_staleness reward_transform recovery_window_seconds verifier applicability inputs external_fixture_sha256 fault_model fault_window_contract_sha256",
          "schema expected_ranks initial_state target_group target_role k max_policy_staleness reward_transform verifier inputs")
    if freeze.get("schema") != 1 or freeze.get("evidence_level") != "fixture" or freeze.get("reward_transform") != "identity":
        raise Unsupported("oracle fixture schema/evidence level/transform unsupported")
    shape(freeze["verifier"], "name version", "name version")
    shape(freeze["applicability"], "status", "status")
    if freeze["applicability"]["status"] != "applicable":
        raise Unsupported("non-applicable oracle graph outside saved business scope")
    if "fault_model" in freeze:
        shape(freeze["fault_model"], "case_id kind role target_group", "kind role target_group")
    evidence = reader.read(prefix + "evidence.json")
    shape(evidence, "run_nonce events states updates groups final_state end_time freeze_sha256 success authoritative_reward reward_substitution",
          "events states updates groups final_state freeze_sha256")
    normative_events = []
    for event in evidence["events"]:
        if event["type"] in ("worker_lost_model", "native_retry_model"):
            shape(event, O_META + " " + O_EVENTS[event["type"]], O_META)
            expected_scope = "no_actual_worker_loss" if event["type"] == "worker_lost_model" else "no_actual_native_retry"
            if event["scope"] != expected_scope:
                raise Unsupported("unknown model-only scope")
            normative_events.append({"type": event["type"],
                "at_core_event_position": sum(e["type"] not in ("worker_lost_model", "native_retry_model")
                    for e in evidence["events"][:evidence["events"].index(event)]), **{key: event[key] for key in O_EVENTS[event["type"]].split()}})
    # These new model-only events do not invent processes or alter core time slots.
    evidence["events"] = [event for event in evidence["events"] if event["type"] not in ("worker_lost_model", "native_retry_model")]
    names = Names()
    for state in evidence["states"]:
        names.get("state", state["id"])
    for update in evidence["updates"]:
        names.get("update", update["id"])
    for group in evidence["groups"]:
        names.get("group", group["id"])
        names.get("logical-group", group["logical_id"])
    for event in evidence["events"]:
        names.get("event", event["id"])
    seal = spec["observer_seal"]
    blobs, file_nodes = {}, {}
    for name, expected in seal.items():
        data = reader.bytes(prefix + name)
        blobs[name] = (json.loads(data), sha(data) == expected)
    times = sorted({event["time"] for event in evidence["events"]})
    slots = {value: index for index, value in enumerate(times)}

    def file_ref(name, kind):
        identifier = names.get("file", name)
        if identifier in file_nodes:
            if file_nodes[identifier]["kind"] != kind:
                raise Unsupported("conflicting artifact types")
            return identifier
        if name not in blobs:
            file_nodes[identifier] = {"kind": kind, "missing": True}
            return identifier
        raw, matches = blobs[name]
        if kind == "payload":
            content = raw_payload(raw)
        elif kind == "data":
            shape(raw, "consumed cursor drawn pending", "consumed cursor drawn pending")
            content = {"consumed": [names.get("logical-group", x) for x in raw["consumed"]],
                       "drawn": [names.get("logical-group", x) for x in raw["drawn"]],
                       "pending": raw["pending"], "cursor": raw["cursor"]}
            if raw["pending"]:
                raise Unsupported("oracle pending schema not yet supported")
        elif kind == "component":
            shape(raw, "fixture_component state fixture_bytes")
            content = dict(raw)
            if "state" in content:
                content["state"] = names.get("state", content["state"])
            if "fixture_bytes" in content:
                content["fixture_bytes"] = {"length": len(raw["fixture_bytes"]), "sha256": sha(raw["fixture_bytes"].encode())}
        elif kind == "tensor":
            shape(raw, "transform rows", "transform rows")
            rows = []
            for row in raw["rows"]:
                shape(row, "group sample attempt payload_sha256 reward tokens loss_mask logprobs", "group sample attempt payload_sha256 reward")
                candidates = [item for group in evidence["groups"] if group["logical_id"] == row["group"]
                              for item in group["samples"] if item["index"] == row["sample"] and item["attempt"] == row["attempt"]]
                bound = candidates[0]["payload_file"] if len(candidates) == 1 else None
                rows.append({"group": names.get("logical-group", row["group"]), "sample": row["sample"],
                             "attempt": names.get("attempt", row["attempt"]), "reward": row["reward"],
                             "tokens": row.get("tokens"), "loss_mask": row.get("loss_mask"), "logprobs": row.get("logprobs"),
                             "payload": file_ref(bound, "payload") if bound else None,
                             "payload_binding": None if not bound else row["payload_sha256"] == seal.get(bound)})
            content = {"transform": raw["transform"], "rows": rows}
        else:
            raise Unsupported("unknown oracle artifact kind")
        file_nodes[identifier] = {"kind": kind, "content": content, "seal_matches": matches}
        return identifier

    states = []
    for state in evidence["states"]:
        shape(state, "components data_file epoch finalize_events id parent persist_event save_event snapshot_id updates",
              "components data_file epoch finalize_events id parent persist_event save_event snapshot_id updates")
        component_paths(state["components"], oracle=True)
        components = {component: {rank: [file_ref(name, "data" if component == "data" else "component") for name in paths]
                                 for rank, paths in sorted(ranks.items())} for component, ranks in sorted(state["components"].items())}
        states.append({"id": names.get("state", state["id"]), "parent": names.get("state", state["parent"]), "epoch": state["epoch"],
                       "snapshot": names.get("snapshot", state["snapshot_id"]), "components": components,
                       "data": file_ref(state["data_file"], "data"), "updates": [names.get("update", x) for x in state["updates"]],
                       "finalize_events": [names.get("event", x) for x in state["finalize_events"]],
                       "persist_event": names.get("event", state["persist_event"]), "save_event": names.get("event", state["save_event"])})
    groups = []
    for group in evidence["groups"]:
        shape(group, "id k logical_id prompt_sha256 samples source_file source_index", "id k logical_id prompt_sha256 samples source_file source_index")
        source = blobs[group["source_file"]][0][group["source_index"]]
        shape(source, "prompt label", "prompt label")
        samples = []
        for sample in group["samples"]:
            shape(sample, "attempt index payload_file policy_version reward verifier_version", "attempt index payload_file policy_version reward verifier_version")
            samples.append({"index": sample["index"], "attempt": names.get("attempt", sample["attempt"]),
                            "payload": file_ref(sample["payload_file"], "payload"), "policy_version": sample["policy_version"],
                            "verifier_version": sample["verifier_version"], "reward": sample["reward"]})
        groups.append({"id": names.get("group", group["id"]), "logical": names.get("logical-group", group["logical_id"]),
                       "k": group["k"], "source": source, "source_index": group["source_index"], "samples": samples,
                       "prompt_matches": group["prompt_sha256"] == sha(canonical(source["prompt"]))})
    updates = []
    for update in evidence["updates"]:
        shape(update, "end_event epoch groups id loaded_state logical_id policy_version start_event tensor_file",
              "end_event epoch groups id loaded_state logical_id policy_version start_event tensor_file")
        updates.append({"id": names.get("update", update["id"]), "logical": names.get("logical-update", update["logical_id"]),
                        "epoch": update["epoch"], "policy_version": update["policy_version"], "loaded": names.get("state", update["loaded_state"]),
                        "groups": [names.get("group", x) for x in update["groups"]], "start": names.get("event", update["start_event"]),
                        "end": names.get("event", update["end_event"]), "tensor": file_ref(update["tensor_file"], "tensor")})
    events = []
    for event in evidence["events"]:
        kind = event["type"]
        if kind not in O_EVENTS:
            raise Unsupported("unknown oracle event: " + kind)
        shape(event, O_META + " " + O_EVENTS[kind], O_META)
        if kind == "technical_invalid" and (event["reason"] not in ("technical_invalid_unmet_window", "technical_invalid_missed_window")
                or event["origin"] != "controller" or event["evidence_level"] != "normative_only"):
            raise Unsupported("unknown technical window classification")
        item = {"id": names.get("event", event["id"]), "type": kind, "time_order": slots[event["time"]],
                "role": event["role"], "rank": event["rank"],
                "process": names.get("process", (event["pid"], event["boot_id"], event["start_time"])),
                "event_nonce": names.get("event-nonce", event["event_nonce"]),
                "timeline": names.get("timeline", event["timeline"]),
                "run_reference_matches": event["run_nonce"] == evidence["run_nonce"] == freeze["run_nonce"]}
        for key in O_EVENTS[kind].split():
            if key not in event or key == "boundary":
                continue
            value = event[key]
            if key in ("state", "update", "attempt", "snapshot_id", "fault_id"):
                value = names.get({"snapshot_id": "snapshot", "fault_id": "fault"}.get(key, key), value)
            elif key in ("group", "target_group"):
                value = names.get("logical-group", value)
            elif key == "condition":
                shape(value, "boundary generation_done reward_done missing_reward reward_start reward_end reward_started executing_sample scope")
                if "scope" in value and value["scope"] != "normative_only_not_GPU_reachability":
                    raise Unsupported("unknown fault model scope")
                value = {k: v for k, v in value.items() if k != "boundary"}
            elif key == "tensor_sha256":
                match = next((u for u in evidence["updates"] if u["id"] == event["update"]), None)
                value = None if match is None else value == sha(reader.bytes(prefix + match["tensor_file"]))
                key = "tensor_reference_matches"
            item[key] = value
        events.append(item)
    normalized_inputs = []
    for name, expected in freeze["inputs"].items():
        value, _ = blobs[name]
        if name == "external-fixture.json":
            shape(value, "case_semantic_sha256 groups bootstrap_arrival_order target_arrival_order expansion", "groups")
            content = sorted([[raw_payload(x) for x in rows] for rows in value["groups"].values()], key=canonical)
        elif name == "dataset.json":
            for row in value:
                shape(row, "prompt label", "prompt label")
            content = value
        else:
            raise Unsupported("unknown frozen input file")
        normalized_inputs.append({"kind": name, "content": content, "reference_matches": expected == sha(reader.bytes(prefix + name))})
    faults = [event for event in evidence["events"] if event["type"] == "fault_observed" and event.get("fault_id") == freeze["fault_id"]]
    window = None
    if len(faults) == 1:
        at = faults[0]["time"]
        window = {"events_after_fault": [event["time"] >= at for event in evidence["events"]],
                  "events_within_window": [0 <= event["time"] - at <= freeze["recovery_window_seconds"] for event in evidence["events"]],
                  "end_beyond_window": evidence["end_time"] - at >= freeze["recovery_window_seconds"]}
    result = {"states": states, "updates": updates, "groups": groups, "events": events, "artifacts": file_nodes,
              "window_relations": window, "recovery_window": freeze["recovery_window_seconds"],
              "final": names.get("state", evidence["final_state"]), "initial": names.get("state", freeze["initial_state"]),
              "target": names.get("logical-group", freeze["target_group"]), "role": freeze["target_role"],
              "ranks": freeze["expected_ranks"], "k": freeze["k"], "staleness": freeze["max_policy_staleness"],
              "transform": freeze["reward_transform"], "verifier": freeze["verifier"], "inputs": sorted(normalized_inputs, key=canonical),
              "freeze_reference_matches": evidence["freeze_sha256"] == sha(canonical(freeze)),
              "evidence_seal_matches": blobs["evidence.json"][1]}
    if "reward_substitution" in evidence:
        sub = evidence["reward_substitution"]
        shape(sub, "destination source_file source_sha256 source_reward", "destination source_file source_sha256 source_reward")
        source = blobs[sub["source_file"]][0]
        result["reward_substitution"] = {"destination_index": int(sub["destination"].rsplit(":", 1)[1]), "source": raw_payload(source),
                                         "reward": sub["source_reward"], "source_matches": sha(reader.bytes(prefix + sub["source_file"])) == sub["source_sha256"]}
    if "fault_window_contract_sha256" in freeze:
        raw = reader.read(prefix + "f4-window-contract.json")
        result["fault_window_contract"] = window_contract(raw)
        result["fault_window_reference_matches"] = freeze["fault_window_contract_sha256"] == sha(canonical(raw))
    claims = {key: evidence[key] for key in ("success", "authoritative_reward") if key in evidence}
    if normative_events:
        claims["normative_events"] = normative_events
    return result, claims


def normalize_case(directory):
    reader = CaseReader(directory)
    launches = reader.read("launches.json")
    actors = []
    filenames = {}
    for launch in launches:
        shape(launch, "argv identity pid returncode_before_cleanup", "argv identity pid returncode_before_cleanup")
        argv = launch["argv"]
        if len(argv) != 10 or argv[2] != "--worker" or argv[4] != "--case" or argv[6] != "--output" or argv[8] != "--role":
            raise Unsupported("unknown worker launch schema")
        filename = argv[9] + "-operations.jsonl"
        events = [json.loads(line) for line in reader.bytes(filename).decode().splitlines()]
        if not events:
            raise Unsupported("empty actor trace")
        for index, event in enumerate(events):
            if event.get("sequence") != index or event.get("identity") != launch["identity"]:
                raise Unsupported("broken actor identity/local order")
        mode = argv[3]
        if mode == "initial":
            first = events[0]
            if first["operation"] != "state.acquire_owner":
                raise Unsupported("initial actor lacks acquire")
            role = "loser" if first.get("result") == "lock_contended" else "writer"
        elif mode == "resume":
            role = "recovery"
        elif mode == "scope-resume":
            role = "scope-recovery"
        elif mode == "corrupt-check":
            role = "corruption-check"
        else:
            raise Unsupported("unknown process mode")
        if launch["pid"] != launch["identity"]["pid"]:
            raise Unsupported("launch PID does not bind identity")
        actors.append({"role": role, "events": events, "filename": filename, "launch": launch})
    ordering = {"writer": 0, "loser": 1, "recovery": 2, "scope-recovery": 2, "corruption-check": 3}
    actors.sort(key=lambda item: ordering[item["role"]])
    if len({item["role"] for item in actors}) != len(actors):
        raise Unsupported("multiple actors of same role need explicit schema")
    for actor in actors:
        for event in actor["events"]:
            if event["operation"] == "fixture.recovered_policy_payload":
                if event["payload_file"] != "method-target-0.json":
                    raise Unsupported("unknown recovered payload mapping")
                reader.policy_mutations["target:0"] = event
    state_events, markers, processes = [], [], []
    identities = {}
    for actor in actors:
        role = actor["role"]
        identities[canonical(actor["launch"]["identity"])] = role
        core, model, positions = [], [], []
        for event in actor["events"]:
            positions.append(len(core))
            normalized = reader.event(event)
            if event["operation"] in MARKERS:
                model.append({"at_api_position": len(core), "event": normalized})
            else:
                core.append(normalized)
        filenames[actor["filename"]] = (role, actor["events"], positions)
        state_events.append({"actor": role, "events": core})
        markers.append({"actor": role, "events": model})
        processes.append({"role": role, "exit_code": actor["launch"]["returncode_before_cleanup"], "receipts": []})
    process_lookup = {item["role"]: item for item in processes}
    parent = reader.read("parent-receipts.json")
    for message in parent:
        shape(message, "identity mode type receipt generation token_file result when decision rejection bad_sample generated accepted missing_sample starts ends executing_sample limitation commit_receipt field old_attempt receipt_equal evidence_level epoch prior_processes",
              "identity type")
        identity = canonical(message["identity"])
        if identity not in identities:
            raise Unsupported("parent references unknown process")
        item = {"type": message["type"]}
        if "receipt" in message:
            ref = message["receipt"]
            shape(ref, "file sequence", "file sequence")
            if ref["file"] not in filenames:
                raise Unsupported("missing parent receipt file")
            role, events, positions = filenames[ref["file"]]
            index = ref["sequence"]
            if not isinstance(index, int) or not 0 <= index < len(events) or role != identities[identity]:
                raise Unsupported("unbound parent receipt")
            item["api_position"] = positions[index]
            item["observed_operation"] = events[index]["operation"].split(".")[:2]
        if "prior_processes" in message:
            if message["type"] != "scope_recovered":
                raise Unsupported("exit evidence on unknown message")
            prior = []
            for registered in message["prior_processes"]:
                shape(registered, "pid boot_id start_time", "pid boot_id start_time")
                role = identities.get(canonical(registered))
                if role is None:
                    raise Unsupported("exit evidence references unobserved actor")
                prior.append({"role": role, "exit_code": process_lookup[role]["exit_code"]})
            item.update(epoch=message["epoch"], prior_registered=prior,
                new_process=all(canonical(i) != identity for i in message["prior_processes"]))
        if "decision" in message:
            item["decision"] = reader.decision(message["decision"])
        process_lookup[identities[identity]]["receipts"].append(item)
    if "corruption-check" in process_lookup and ("scope-recovery" in process_lookup or any(
            event["operation"] == "state.commit_generation.closed_owner" for actor in actors for event in actor["events"])):
        control = reader.read("state-run/control.json")
        fields = "epoch run_nonce config_sha256 attempts accepted head verifier_version owner_nonce processes scope"
        shape(control, fields, fields)
        registered = []
        for ident in control["processes"]:
            shape(ident, "pid boot_id start_time", "pid boot_id start_time")
            role = identities.get(canonical(ident))
            if role is None:
                raise Unsupported("final control process has no launch evidence")
            registered.append(role)
        process_lookup["corruption-check"]["final_owner_binding"] = {"epoch": control["epoch"], "registered_roles": registered,
            "scope": control["scope"], "prior_actor_exit_codes": {r: process_lookup[r]["exit_code"]
                for r in ("writer", "scope-recovery") if r in process_lookup}}
    oracle = []
    oracle_claims = []
    # Directory labels are provenance, not a way to manufacture a distinct signature.
    for folder in sorted(path.name for path in reader.root.glob("oracle-*") if path.is_dir()):
        if not (reader.root / folder / "spec.json").exists():
            raise Unsupported("oracle directory missing frozen spec")
        normalized, claims = normalize_oracle(reader, folder)
        oracle.append(normalized)
        if claims:
            oracle_claims.append(claims)
    if not oracle:
        raise Unsupported("no independent oracle evidence")
    lanes = {"S": {"actors": state_events, "cut_files": reader.storage()}, "P": processes,
             "O": sorted(oracle, key=canonical), "markers": {"actors": markers, "oracle_claims": sorted(oracle_claims, key=canonical)}}
    if (reader.root / "f4-window-contract.json").exists():
        lanes["markers"]["window_contract"] = window_contract(reader.read("f4-window-contract.json"))
    signatures = {key: sha(canonical(value)) for key, value in lanes.items()}
    signatures["core_joint"] = sha(canonical({key: lanes[key] for key in ("S", "P", "O")}))
    signatures["with_markers"] = sha(canonical(lanes))
    return {"schema": 1, "status": "parsed_not_a_uniqueness_verdict", "signatures": signatures, "normalized": lanes,
            "read_sha256": dict(sorted(reader.read_hashes.items())),
            "limitations": ["Recorded operations only; no GPU or functional revalidation", "Parent start/continue commands were not separately journaled",
                            "Normative markers cannot establish different actual API/state executions"]}


def first_difference(left, right, path=""):
    if type(left) is not type(right):
        return {"path": path, "left": left, "right": right}
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return {"path": path + "/" + key, "left_present": key in left, "right_present": key in right}
            result = first_difference(left[key], right[key], path + "/" + key)
            if result:
                return result
    elif isinstance(left, list):
        if len(left) != len(right):
            return {"path": path, "left_length": len(left), "right_length": len(right)}
        for index, (a, b) in enumerate(zip(left, right)):
            result = first_difference(a, b, path + "/" + str(index))
            if result:
                return result
    elif left != right:
        return {"path": path, "left": left, "right": right}
    return None


def audit_runs(runs, output):
    output = Path(output).resolve()
    if any(Path(run).resolve() == output or Path(run).resolve() in output.parents for run in runs):
        raise ValueError("audit output must not alter an input evidence tree")
    output.mkdir(parents=True, exist_ok=False)
    records, issues, selected, normalized = [], [], {}, {}
    for run in runs:
        run = Path(run)
        inventory = json.loads((run / "inventory.json").read_text())
        provenance = {"run": str(run.resolve()), "inventory_sha256": sha((run / "inventory.json").read_bytes()),
                      "recorded_source_sha256": inventory.get("source_sha256", {})}
        for directory in sorted(path for path in run.iterdir() if path.is_dir()):
            # Labels are used ONLY to link reports and identify reruns, after normalization.
            label = directory.name
            record = {"label": label, "source": str(directory.resolve()), "provenance": provenance}
            try:
                audit = normalize_case(directory)
                number = len(records)
                artifact = f"normalized-{number:04d}.json"
                (output / artifact).write_bytes(canonical(audit) + b"\n")
                record.update(status=audit["status"], signatures=audit["signatures"], artifact=artifact)
                normalized[number] = audit["normalized"]
                if label in selected:
                    previous = selected[label]
                    if records[previous]["signatures"] != record["signatures"]:
                        issues.append({"kind": "rerun_signature_disagreement", "label": label, "records": [previous, number],
                                       "first_difference": first_difference(normalized[previous], audit["normalized"])})
                else:
                    selected[label] = number
            except (Unsupported, OSError, KeyError, ValueError, TypeError, IndexError, AttributeError) as exc:
                record.update(status="unverifiable_normalization", reason=f"{type(exc).__name__}: {exc}")
                issues.append({"kind": "unverifiable_normalization", "record": len(records), "label": label, "reason": record["reason"]})
            records.append(record)
    collisions = {}
    for lane in ("S", "P", "O", "core_joint", "with_markers"):
        groups = {}
        for label, number in selected.items():
            groups.setdefault(records[number]["signatures"][lane], []).append(number)
        collisions[lane] = [numbers for numbers in groups.values() if len(numbers) > 1]
    for numbers in collisions["core_joint"]:
        reference = numbers[0]
        issues.append({"kind": "cross_case_core_equivalence_requires_investigation", "records": numbers,
                       "labels": [records[index]["label"] for index in numbers],
                       "marker_differences": [{"record": index, "difference": first_difference(normalized[reference]["markers"], normalized[index]["markers"])}
                                              for index in numbers[1:]],
                       "functional_verdict": "not_inferred"})
    incomplete = any(row["status"] == "unverifiable_normalization" for row in records)
    report = {"schema": 1, "status": "audit_incomplete" if incomplete else "investigation_required" if issues else "no_collisions_observed_in_parsed_scope",
              "normalization_completeness": "incomplete" if incomplete else "complete_for_requested_inputs",
              "requested_runs": [str(Path(run).resolve()) for run in runs], "records": records,
              "counts": {"observations": len(records), "distinct_labels_observed": len({row['label'] for row in records}),
                         "distinct_labels_parsed": len(selected), "unverifiable_observations": sum(row["status"] == "unverifiable_normalization" for row in records)},
              "collisions": collisions, "investigations": issues,
              "functional_passed_count": "not_recomputed", "signature_is_not_functional_safety": True}
    (output / "audit.json").write_bytes(canonical(report) + b"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_runs(args.run, args.output)
    print(json.dumps({"status": report["status"], **report["counts"], "investigations": len(report["investigations"])}))
    # Investigations are findings, not program failures or automatic functional failures.


if __name__ == "__main__":
    main()
