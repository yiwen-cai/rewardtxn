"""Bounded CPU contract cases; no general manifest interpreter or GPU use."""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ft import state
from scripts.ft.oracle import audit_run


REPRESENTATIVES = ("F1.b00.i01", "F1.b09.i04", "F2.b04.i00", "F2.b09.i05", "F3.b00.i00", "F3.b01.i09",
             "F4.b00.i00", "F4.b06.i03", "X1.b01.i00", "X1.b09.i08", "X2.b02.i04", "X2.b09.i05")
STAGE_ONE = tuple(f"{cell}.b{boundary:02d}.i{interleave:02d}"
                  for cell, boundaries in (("F1", range(10)), ("X1", (0,)))
                  for boundary in boundaries for interleave in (0, 1, 2, 3, 4, 5, 8, 9))
STAGE_TWO = tuple(f"{cell}.b{boundary:02d}.i{interleave:02d}"
                  for cell, boundaries in (("F2", range(10)), ("F3", range(2)))
                  for boundary in boundaries for interleave in (0, 1, 2, 3, 4, 5, 8, 9))
STAGE_THREE = tuple(f"X1.b{boundary:02d}.i{interleave:02d}"
                    for boundary in (0, 1, 2, 3, 4, 6, 7, 8, 9)
                    for interleave in (0, 1, 2, 3, 4, 5, 8, 9))
STAGE_FOUR = tuple(f"X2.b{b:02d}.i{i:02d}" for b in range(10) for i in (0, 1, 2, 3, 4, 5, 8, 9))
STAGE_FIVE = tuple(f"F4.b{b:02d}.i{i:02d}" for b in range(10) for i in (0, 1, 2, 3, 4, 5, 8, 9))
SUPPORTED = tuple(dict.fromkeys(REPRESENTATIVES + STAGE_ONE + STAGE_TWO + STAGE_THREE + STAGE_FOUR + STAGE_FIVE))
COMPONENTS = ("model", "optimizer_master", "optimizer_moments", "optimizer_step", "scheduler",
              "rng_python", "rng_numpy", "rng_torch_cpu", "rng_device", "rng_tracker", "policy")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def local_module(name):
    spec = importlib.util.spec_from_file_location("p2_driver_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def external_fixture(case):
    """Freeze external inputs before either lane runs, never use method output."""
    parent = next(op["arguments"] for op in case["operations"] if op["call"] == "fixture.create_committed_parent")
    target = [op["arguments"] for op in case["operations"] if op["call"] == "fixture.generate_sample"]
    if not target:
        raise HarnessError("supported case lacks explicit target generation inputs")
    explicit = {item["sample"]: item for item in target}
    groups = {}
    for name, prompt, label in (("bootstrap", parent["prompt"], parent["label"]),
                                ("target", target[0]["prompt"], target[0]["label"]),
                                ("probe", "Return one.", "1")):
        entries = []
        for index in range(1 if name == "probe" else 8):
            item = explicit.get(index) if name == "target" else None
            tokens = item["tokens"] if item else [11, 21 + index]
            entries.append({"prompt": prompt, "label": label, "completion": item["completion"] if item else "1",
                "tokens": tokens, "loss_mask": item["loss_mask"] if item else [0, 1],
                "logprobs": item["logprobs"] if item else [0.0, -0.25], "policy_version": 0,
                "prompt_ids": tokens[:1], "completion_ids": tokens[1:], "prompt_sha256": digest(encoded(prompt))})
        groups[name] = entries
    if case["cell"] == "X1" and case["boundary"] == 6:
        source = next(op["arguments"] for op in case["operations"] if op["call"] == "fixture.replace_reward_from_other_group")
        raw = copy.deepcopy(groups["target"][0])
        raw.update(completion=source["source_completion"], label=source["source_label"])
        groups["other"] = [raw]
    return {"case_semantic_sha256": case["semantic_sha256"], "groups": groups,
            "bootstrap_arrival_order": parent["arrival_order"],
            "target_arrival_order": [item["sample"] for item in target],
            "expansion": "bootstrap/probe and ungenerated target indices freeze CPU token values [11,21+index], mask [0,1], logprobs [0,-0.25]"}


class HarnessError(RuntimeError):
    pass


class ContractFailure(RuntimeError):
    pass


class Worker:
    def __init__(self, directory, role):
        self.directory = directory
        self.role = role
        self.identity = state.process_identity(os.getpid())
        self.trace = directory / (role + "-operations.jsonl")
        self.sequence = 0
        self.receipts = {}
        self.attempts = {}
        self.owner = None
        self.parent = None
        self.bootstrap = None
        self.inputs = json.loads((directory / "external-fixture.json").read_text())
        self.case = json.loads((directory / "case.json").read_text())

    def record(self, operation, **fields):
        event = {"sequence": self.sequence, "operation": operation, "identity": self.identity,
                 "monotonic": time.monotonic(), **fields}
        with self.trace.open("a", encoding="utf-8") as stream:
            stream.write(encoded(event).decode() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.sequence += 1
        return {"file": self.trace.name, "sequence": event["sequence"]}

    def send(self, kind, **fields):
        print(json.dumps({"type": kind, "identity": self.identity, **fields}), flush=True)

    def command(self, expected):
        line = sys.stdin.readline()
        if not line or json.loads(line)["command"] != expected:
            raise HarnessError("parent protocol disconnected or wrong command")

    def rejected(self, operation, function, *args, **kwargs):
        try:
            function(*args, **kwargs)
        except state.StateError as exc:
            expected = {
                "state.prepare_generation.incomplete": "incomplete group",
                "state.prepare_generation.missing_reward": "incomplete group",
                "state.prepare_generation.after_rejected_reward": "incomplete group",
                "state.accept_result.wrong_verifier": "result violates policy/verifier authorization",
                "state.accept_result.wrong_policy": "result violates policy/verifier authorization",
                "state.accept_result.old_probe": "unauthorized attempt",
                "state.accept_result.old_target_first": "unauthorized attempt",
                "state.commit_generation.missing_rank": "cannot read rank-",
                "state.record_evidence.bad_scheduler": "optimizer/scheduler evidence incomplete",
                "state.commit_generation.closed_owner": "inactive or inherited owner",
                "state.select_recovery.corrupt": "corrupt token or snapshot",
                "state.accept_result.conflict": "conflicting result",
                "state.accept_result.unmapped": "unauthorized attempt",
                "state.accept_result.foreign": "stale result",
                "state.prepare_generation.duplicate": "incomplete group",
            }[operation]
            if expected not in str(exc):
                raise ContractFailure(f"wrong rejection boundary for {operation}: {exc}") from exc
            return self.record(operation, result="rejected", reason=str(exc))
        raise ContractFailure("expected API rejection did not occur: " + operation)

    def generate(self, group, index):
        payload = copy.deepcopy(self.inputs["groups"][group][index])
        path = self.directory / f"method-{group}-{index}.json"
        write_json(path, payload)
        self.record("fixture.generate_sample", group=group, sample=index, payload_file=path.name,
                    payload_sha256=digest(path.read_bytes()), evidence_level="cpu_fixture")
        return payload

    def authorize(self, key, name="current", previous=None, policy=0):
        attempt = state.authorize_attempt(self.owner, key, previous, name, expected_policy_version=policy)
        self.attempts[key] = attempt
        self.record("state.authorize_attempt", sample=key, attempt=attempt)
        return attempt

    def payload(self, key, reward=1.0, verifier="exact-v1"):
        group, index = key.rsplit(":", 1)
        raw = json.loads((self.directory / f"method-{group}-{index}.json").read_text())
        return {"response_sha256": digest(encoded(raw)), "reward_sha256": digest(encoded(reward)),
                "tensor_input_sha256": digest(encoded({"raw": raw, "reward": reward})),
                "policy_version": raw["policy_version"], "verifier_version": verifier}

    def accept(self, key, attempt=None, reward=1.0, verifier="exact-v1"):
        attempt = attempt or self.attempts[key]
        result = state.accept_result(self.owner, attempt, self.payload(key, reward, verifier))
        self.receipts[key] = result
        self.record("state.accept_result", sample=key, attempt=attempt, result=result, reward=reward)
        return result

    def group(self, name, count=8, reverse=False, accepted=None, remove_tokens=False, compute_positive=False):
        order = self.inputs["target_arrival_order"] if name == "target" else list(range(count))
        if reverse:
            order.reverse()
        for index in order:
            self.generate(name, index)
        if remove_tokens:
            path = self.directory / "method-target-7.json"
            before = digest(path.read_bytes())
            payload = json.loads(path.read_text())
            del payload["tokens"]
            with path.open("w") as stream:
                json.dump(payload, stream, sort_keys=True, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            self.record("fixture.remove_method_payload", sample=7, field="tokens", before_sha256=before,
                        after_sha256=digest(path.read_bytes()), payload_file=path.name)
        if compute_positive:
            raw = json.loads((self.directory / "method-target-0.json").read_text())
            value = float(raw["completion"].strip() == raw["label"].strip())
            self.record("fixture.compute_reward", function="fixture-exact", completion=raw["completion"], label=raw["label"], actual_value=value)
        for index in range(count if accepted is None else accepted):
            key = f"{name}:{index}"
            self.authorize(key)
            raw = json.loads((self.directory / f"method-{name}-{index}.json").read_text())
            value = float(raw["completion"].strip() == raw["label"].strip())
            self.record("fixture.score_sample", sample=key, function="fixture-exact", value=value)
            self.accept(key, reward=value)

    def intent(self, name):
        entries = [{"sample_index": index, "sample": f"{name}:{index}", "receipt": self.receipts[f"{name}:{index}"]}
                   for index in range(8) if f"{name}:{index}" in self.receipts]
        previous = [] if name == "bootstrap" else [f"bootstrap:{index}" for index in range(8)]
        current = [f"{name}:{index}" for index in range(8)]
        pending = []
        if "probe:0" in self.receipts:
            probe_prompt = self.inputs["groups"]["probe"][0]["prompt"]
            pending = [{"sample": "probe:0", "action": "regenerate", "prompt": probe_prompt,
                        "prompt_sha256": digest(encoded(probe_prompt)), "k": 1}]
        prompt = {key: self.inputs["groups"][name][0][key] for key in ("prompt", "label")}
        data = {"source_sha256": digest(encoded(self.inputs)), "epoch": 0, "shuffle_state": {"seed": 20260920},
                "drawn": previous + current + [item["sample"] for item in pending],
                "consumed": previous + current, "pending": pending,
                "cursor": len(previous + current) + len(pending)}
        intent = {"parent": self.parent, "ack_capability": "none", "config_sha256": digest(b"p2-cpu-driver"),
                  "expected_ranks": ["actor:0", "actor:1"], "data_snapshot_id": name + "-cut",
                  "components": {component: {rank: [rank.replace(":", "-") + "/" + component + ".bin"]
                                   for rank in ("actor:0", "actor:1")} for component in COMPONENTS},
                  "updates": [{"logical_update_id": name + "-update", "physical_update_id": name + "-physical",
                               "train_input_sha256": digest(encoded(entries)), "groups": [{"logical_group_id": name,
                               "prompt": prompt, "prompt_sha256": digest(encoded(prompt)), "k": 8, "samples": entries}]}]}
        return intent, data

    def prepare(self, name):
        intent, data = self.intent(name)
        generation = state.prepare_generation(self.owner, intent, data)
        self.record("state.prepare_generation", generation=generation, logical_group=name,
                    intent_file=str(Path("state-run/generations") / generation / "intent.json"))
        return generation

    def optimizer(self, generation, name, receipt=True):
        self.record("fixture.optimizer_start", physical_update=name + "-physical", evidence_level="cpu_fixture")
        self.record("fixture.optimizer_end", physical_update=name + "-physical", successful=True,
                    evidence_level="cpu_fixture_no_training_backend")
        if receipt:
            result = state.record_evidence(self.owner, generation, {"kind": "optimizer", "snapshot_id": name + "-cut",
                "physical_updates": [name + "-physical"], "successful": True, "scheduler_applied": True})
            self.record("state.record_evidence", generation=generation, evidence=result)

    def checkpoint(self, generation, name, ranks=("actor:0", "actor:1"), finalize=True):
        directory = self.owner.root / "generations" / generation / "checkpoint"
        for rank in ranks:
            target = directory / rank.replace(":", "-")
            target.mkdir(parents=True)
            for component in COMPONENTS:
                path = target / (component + ".bin")
                with path.open("xb") as stream:
                    stream.write(b"A" * 32768)
                    stream.flush()
                    os.fsync(stream.fileno())
                self.record("fixture.write_component", generation=generation, rank=rank, component=component,
                            relative_path=str(path.relative_to(self.directory)), size=32768, sha256=digest(path.read_bytes()))
            if finalize:
                result = state.record_evidence(self.owner, generation, {"kind": "finalize", "snapshot_id": name + "-cut",
                                                                       "rank": rank, "writer_closed": True})
                self.record("state.record_evidence", generation=generation, evidence=result)

    def make_bootstrap(self, reverse=False):
        self.group("bootstrap", reverse=reverse)
        generation = self.prepare("bootstrap")
        self.optimizer(generation, "bootstrap")
        self.checkpoint(generation, "bootstrap")
        token = state.commit_generation(self.owner, generation)
        self.parent = {"generation": generation, "token_sha256": digest(encoded(token))}
        self.bootstrap = generation
        reference = self.record("state.commit_generation", generation=generation, token=token)
        self.send("bootstrap", generation=generation, receipt=reference,
                  token_file=str(Path("state-run/generations") / generation / "token.json"))

    def probe(self, old_first):
        self.generate("probe", 0)
        old = self.authorize("probe:0", "old")
        self.authorize("probe:0", "new", "old")
        if not old_first:
            self.accept("probe:0")
        self.rejected("state.accept_result.old_probe", state.accept_result, self.owner, old, self.payload("probe:0"))
        if old_first:
            self.accept("probe:0")
        self.record("fixture.track_unconsumed_probe", sample="probe:0", pending_action="regenerate",
                    prompt=self.inputs["groups"]["probe"][0]["prompt"])

    def boundary(self, result, **fields):
        reference = self.record("target_boundary", result=result, **fields)
        self.send("boundary", result=result, receipt=reference, **fields)

    def cut(self, generation, when):
        self.boundary("exact_cut_reached", generation=generation, when=when)
        reference = self.record("cut_ready", generation=generation, when=when)
        self.send("cut", generation=generation, when=when, receipt=reference)
        self.command("never_release_sigkill_required")
        raise HarnessError("fault cut unexpectedly released")

    def duplicate_bootstrap_evidence(self):
        directory = self.owner.root / "generations" / self.bootstrap
        before = {str(path.relative_to(directory)): digest(path.read_bytes()) for path in directory.rglob("*") if path.is_file()}
        evidence = {"kind": "optimizer", "snapshot_id": "bootstrap-cut", "physical_updates": ["bootstrap-physical"],
                    "successful": True, "scheduler_applied": True}
        for index in range(2):
            result = state.record_evidence(self.owner, self.bootstrap, evidence)
            self.record("state.record_evidence.identical", generation=self.bootstrap, repeat=index, evidence=evidence, result=result)
        after = {str(path.relative_to(directory)): digest(path.read_bytes()) for path in directory.rglob("*") if path.is_file()}
        if before != after:
            raise ContractFailure("identical committed evidence changed generation bytes")
        self.record("fixture.assert_idempotent_generation", before=before, after=after)

    def target_f1(self, boundary):
        count = boundary + 1 if boundary < 7 else 8
        accepted = 7 if boundary == 9 else count
        self.group("target", count=count, accepted=accepted, remove_tokens=boundary == 7)
        if boundary in (7, 8):
            generation = self.prepare("target")
            self.boundary("payload_semantics_outside_state", generation=generation, field="tokens" if boundary == 7 else "logprobs")
        else:
            operation = "state.prepare_generation.missing_reward" if boundary == 9 else "state.prepare_generation.incomplete"
            reference = self.rejected(operation, state.prepare_generation, self.owner, *self.intent("target"))
            self.boundary("missing_reward_detected_from_expected_k" if boundary == 9 else "incomplete_group_rejected",
                          rejection=reference, generated=count, accepted=accepted)

    def target_x1_positive(self):
        self.group("target", compute_positive=True)
        generation = self.prepare("target")
        self.optimizer(generation, "target")
        self.checkpoint(generation, "target")
        token = state.commit_generation(self.owner, generation)
        reference = self.record("state.commit_generation", generation=generation, token=token)
        self.boundary("committed_target", generation=generation, commit_receipt=reference)

    def target_storage(self, cell, boundary):
        self.group("target")
        if cell == "F2" and boundary == 0:
            self.cut(None, "after_last_accept_before_prepare")
        generation = self.prepare("target")
        if cell == "F2":
            if boundary == 1:
                self.cut(generation, "after_prepare")
            if boundary == 2:
                self.record("fixture.before_optimizer_start", physical_update="target-physical")
                self.cut(generation, "before_optimizer_start")
            if boundary == 3:
                self.record("fixture.optimizer_start", physical_update="target-physical", evidence_level="cpu_fixture")
                self.cut(generation, "after_optimizer_start_before_end")
            if boundary == 6:
                evidence = {"kind": "optimizer", "snapshot_id": "target-cut", "physical_updates": ["target-physical"],
                            "successful": True, "scheduler_applied": False}
                self.record("fixture.scheduler_evidence_attempt", generation=generation, evidence=evidence)
                self.rejected("state.record_evidence.bad_scheduler", state.record_evidence, self.owner, generation, evidence)
                self.cut(generation, "after_scheduler_evidence_rejection")
        self.optimizer(generation, "target", receipt=not (cell == "F2" and boundary == 4))
        if cell == "F2":
            if boundary == 4:
                self.cut(generation, "after_optimizer_success_before_receipt")
            if boundary == 5:
                self.cut(generation, "after_optimizer_receipt")
            if boundary == 7:
                self.checkpoint(generation, "target", ranks=("actor:0",))
                self.cut(generation, "after_single_rank_finalize")
            if boundary == 8:
                self.checkpoint(generation, "target", finalize=False)
                self.cut(generation, "after_all_rank_files_without_finalize")
        if cell == "F3" and boundary == 0:
            self.checkpoint(generation, "target", ranks=("actor:0",))
            self.rejected("state.commit_generation.missing_rank", state.commit_generation, self.owner, generation)
            self.cut(generation, "after_missing_rank_rejection_before_token")
        self.checkpoint(generation, "target")
        if cell == "F2" and boundary == 9:
            original = state._write
            def interrupted(path, value, immutable=True):
                original(path, value, immutable)
                if path.name == "manifest.json" and path.parent.name == generation:
                    self.cut(generation, "after_manifest_fsync_before_token_link")
            state._write = interrupted
        elif cell == "F3" and boundary == 1:
            original = state.os.link
            def interrupted(source, target, *args, **kwargs):
                result = original(source, target, *args, **kwargs)
                if Path(target).name == "token.json" and Path(target).parent.name == generation:
                    self.cut(generation, "after_token_link_before_directory_fsync")
                return result
            state.os.link = interrupted
        else:
            raise HarnessError("unimplemented storage boundary")
        state.commit_generation(self.owner, generation)
        raise HarnessError("commit returned without requested cut")

    def target_x1_negative(self, boundary):
        for index in self.inputs["target_arrival_order"]:
            self.generate("target", index)
        raw = json.loads((self.directory / "method-target-0.json").read_text())
        exact = float(raw["completion"].strip() == raw["label"].strip())
        value, verifier = exact, "exact-v1"
        if boundary == 1:
            value, verifier = 1.0 - exact, "exact-v2"
            self.record("fixture.compute_reward", function="one-minus-fixture-exact", exact_value=exact,
                        actual_value=value, declared_verifier=verifier, sample=0)
        elif boundary == 2:
            verifier = "exact-v2"
            self.record("fixture.change_declared_verifier", sample=0, verifier=verifier, before_reward=exact, after_reward=value)
        elif boundary in (3, 8):
            authority = copy.deepcopy(raw)
            if boundary == 3:
                authority["label"] = "2"
            else:
                del authority["completion"]
            path = self.directory / "method-observer-authority-0.json"
            write_json(path, authority)
            self.record("fixture.change_observer_label" if boundary == 3 else "fixture.remove_authority_payload",
                        sample=0, field="label" if boundary == 3 else "completion", file=path.name,
                        sha256=digest(path.read_bytes()), before=raw, after=authority)
        elif boundary == 4:
            path = self.directory / "method-target-0.json"
            sealed = digest(path.read_bytes())
            write_json(self.directory / "method-response-seal.json", {path.name: sealed})
            changed = dict(raw, completion="2")
            with path.open("w") as stream:
                json.dump(changed, stream, sort_keys=True, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            self.record("fixture.change_response_bytes", sample=0, field="completion", value="2",
                        before_sha256=sealed, after_sha256=digest(path.read_bytes()), payload_file=path.name,
                        seal_file="method-response-seal.json")
        elif boundary == 6:
            other = self.generate("other", 0)
            value = float(other["completion"].strip() == other["label"].strip())
            self.record("fixture.replace_reward_from_other_group", destination="target:0", source="other:0",
                        source_file="method-other-0.json", source_sha256=digest((self.directory / "method-other-0.json").read_bytes()),
                        source_reward=value, target_exact_reward=exact)
        elif boundary == 7:
            self.record("fixture.change_sample_policy", sample=0, declared_policy=4, authorized_policy=0,
                        trainer_policy=1, max_staleness=1)
        elif boundary == 9:
            value = 0.0
            self.record("fixture.forge_method_success", success=True, authoritative_reward=value, actual_reward=value,
                        correct_reward=exact, sample=0)
        else:
            raise HarnessError("unsupported X1 negative boundary")
        for index in range(8):
            key = f"target:{index}"
            self.authorize(key)
            score = value if index == 0 else 1.0
            self.record("fixture.score_sample", sample=key, function="one-minus-fixture-exact" if boundary == 1 and index == 0 else "fixture-exact",
                        value=score, source="other:0" if boundary == 6 and index == 0 else key)
            if index == 0 and boundary in (1, 2, 7):
                payload = self.payload(key, score, verifier)
                if boundary == 7:
                    payload["policy_version"] = 4
                operation = "state.accept_result.wrong_policy" if boundary == 7 else "state.accept_result.wrong_verifier"
                self.record("fixture.submitted_declaration", sample=key, payload=payload, authorized=self.attempts[key])
                self.rejected(operation, state.accept_result, self.owner, self.attempts[key], payload)
            else:
                self.accept(key, reward=score)
        if boundary in (1, 2, 7):
            rejection = self.rejected("state.prepare_generation.after_rejected_reward", state.prepare_generation,
                                      self.owner, *self.intent("target"))
            self.boundary("unauthorized_version_rejected", rejection=rejection, bad_sample=0)
        else:
            generation = self.prepare("target")
            self.boundary("state_does_not_check_reward_semantics", generation=generation)

    def target_f4(self, boundary):
        for index in self.inputs["target_arrival_order"]:
            self.generate("target", index)
        if boundary == 3:
            raw = copy.deepcopy(self.inputs["groups"]["target"][6])
            path = self.directory / "method-target-6-repeat.json"
            write_json(path, raw)
            self.record("fixture.generate_sample_again", group="target", sample=6,
                payload_file=path.name, payload_sha256=digest(path.read_bytes()), unique_new_sample=False)
        if boundary in (0, 1, 2, 3, 4):
            for index in range(1 if boundary == 4 else 4):
                self.record("fixture.reward_start", sample=index, evidence_level="normative_events_not_real_RLVR")
        if boundary in (1, 5):
            for index in range(4 if boundary == 1 else 1):
                raw = json.loads((self.directory / f"method-target-{index}.json").read_text())
                value = float(raw["completion"].strip() == raw["label"].strip())
                self.record("fixture.reward_end", sample=index, reward=value, function="fixture-exact")
        if boundary == 0:
            self.record("fixture.observe_reward_in_flight", sample=3, evidence_level="normative_only")
        elif boundary in (2, 3, 6, 9):
            for index in range(7 if boundary in (2, 3) else 8):
                key = f"target:{index}"
                self.authorize(key)
                raw = json.loads((self.directory / f"method-target-{index}.json").read_text())
                value = float(raw["completion"].strip() == raw["label"].strip())
                self.record("fixture.score_sample", sample=key, function="fixture-exact", value=value,
                    declared_verifier="exact-v2" if boundary == 6 and index == 7 else "exact-v1")
                if boundary == 6 and index == 7:
                    reference = self.rejected("state.accept_result.wrong_verifier", state.accept_result,
                        self.owner, self.attempts[key], self.payload(key, value, "exact-v2"))
                else:
                    self.accept(key, reward=value)
            if boundary in (2, 3):
                reference = self.rejected("state.prepare_generation.incomplete", state.prepare_generation,
                    self.owner, *self.intent("target"))
            elif boundary == 9:
                self.record("fixture.check_fault_window", executing_sample=None, completed_samples=list(range(8)),
                    unique_generated=8, classification="technical_invalid", evidence_level="normative_only")
            if boundary != 9:
                self.boundary("f4_admission_rejection", rejection=reference)
                return
        elif boundary == 5:
            self.record("fixture.before_accept_result", sample=0)
        elif boundary == 7:
            self.record("fixture.worker_lost", worker="reward-worker", samples=[2, 3], evidence_level="model_only_no_process_loss")
        elif boundary == 8:
            before = {str(i): digest((self.directory / f"method-target-{i}.json").read_bytes()) for i in range(8)}
            for index in range(8):
                raw = json.loads((self.directory / f"method-target-{index}.json").read_text())
                self.record("fixture.retry_score", sample=index, reward=float(raw["completion"].strip() == raw["label"].strip()),
                    function="fixture-exact", evidence_level="model_only_no_native_retry")
            after = {str(i): digest((self.directory / f"method-target-{i}.json").read_bytes()) for i in range(8)}
            if before != after:
                raise ContractFailure("retry model changed response bytes")
            self.record("fixture.native_retry_model", before=before, after=after, retry_count=1,
                preserve_responses=True, evidence_level="model_only_no_native_retry")
        self.boundary("f4_normative_events", evidence_level="no_natural_GPU_window_claim")

    def target_x2(self, boundary):
        self.group("target")
        old = copy.deepcopy(self.attempts["target:0"])
        if boundary == 0:
            first = copy.deepcopy(self.receipts["target:0"])
            second = self.accept("target:0")
            if first != second:
                raise ContractFailure("identical delivery changed receipt")
            self.boundary("identical_receipt", receipt_equal=True)
            return
        if boundary == 4:
            self.boundary("registered_owner_will_exit", old_attempt=old)
            return
        if boundary in (2, 3):
            self.authorize("target:0", "new", "current")
            if boundary == 3:
                self.accept("target:0")
            reference = self.rejected("state.accept_result.old_target_first", state.accept_result,
                self.owner, old, self.payload("target:0"))
            if boundary == 2:
                self.accept("target:0")
        elif boundary == 1:
            self.record("fixture.submit_scope", submitted=old, authorized=old, payload=self.payload("target:0", 0.0))
            reference = self.rejected("state.accept_result.conflict", state.accept_result,
                self.owner, old, self.payload("target:0", 0.0))
        elif boundary in (5, 6):
            altered = copy.deepcopy(old)
            altered["attempt" if boundary == 5 else "sample"] = "never-authorized" if boundary == 5 else "other:0"
            self.record("fixture.submit_scope", submitted=altered, authorized=old)
            reference = self.rejected("state.accept_result.unmapped", state.accept_result,
                self.owner, altered, self.payload("target:0"))
        elif boundary == 7:
            with state.acquire_owner(self.directory / "foreign-run", -1, run_nonce="foreign",
                    config_sha256=digest(b"p2-cpu-driver"), verifier_version="exact-v1") as foreign:
                attempt = state.authorize_attempt(foreign, "target:0", None, "current", expected_policy_version=0)
                receipt = state.accept_result(foreign, attempt, self.payload("target:0"))
                self.record("fixture.foreign_run_result", root="foreign-run", attempt=attempt, receipt=receipt)
                reference = self.rejected("state.accept_result.foreign", state.accept_result,
                    self.owner, attempt, receipt["payload"])
        elif boundary == 8:
            intent, data = self.intent("target")
            entries = intent["updates"][0]["groups"][0]["samples"]
            entries.append(copy.deepcopy(entries[0]))
            self.record("fixture.duplicate_consumption_mapping", intent=intent, data=data)
            reference = self.rejected("state.prepare_generation.duplicate", state.prepare_generation,
                self.owner, intent, data)
        elif boundary == 9:
            self.owner.close()
            reference = self.rejected("state.commit_generation.closed_owner", state.commit_generation,
                self.owner, self.bootstrap)
        else:
            raise HarnessError("unknown X2 boundary")
        self.boundary("scope_rejected", rejection=reference)

    def target(self, case_id):
        if case_id.endswith("i04"):
            self.probe(old_first=True)
        elif case_id.endswith("i03"):
            self.probe(old_first=False)
        if case_id.startswith("F1."):
            self.target_f1(self.case["boundary"])
        elif case_id.startswith("X1.b00."):
            self.target_x1_positive()
        elif case_id.startswith("X1."):
            self.target_x1_negative(self.case["boundary"])
        elif case_id.startswith(("F2.", "F3.")):
            self.target_storage(self.case["cell"], self.case["boundary"])
        elif case_id == "F4.b00.i00":
            for index in range(8):
                self.generate("target", index)
            for index in range(4):
                self.record("fixture.reward_start", sample=index, evidence_level="normative_events_not_real_RLVR")
            self.boundary("f4_normative_window_only", generated=8, starts=[0, 1, 2, 3], ends=[], executing_sample=3,
                          limitation="no_real_reward_timing_or_natural_reachability_claim")
        elif case_id == "F4.b06.i03":
            for index in range(8):
                self.generate("target", index)
            bad = 7
            for index in range(8):
                key = f"target:{index}"
                self.authorize(key)
                if index == bad:
                    exact = float("1".strip() == "1".strip())
                    value = exact
                    self.record("fixture.compute_reward", function="fixture-exact",
                                exact_value=exact, actual_value=value, declared_verifier="exact-v2", sample=index)
                    reference = self.rejected("state.accept_result.wrong_verifier", state.accept_result,
                        self.owner, self.attempts[key], self.payload(key, value, "exact-v2"))
                else:
                    self.accept(key)
            self.boundary("unauthorized_verifier_rejected", rejection=reference, bad_sample=bad)
        elif case_id.startswith("F4."):
            self.target_f4(self.case["boundary"])
        elif case_id == "X2.b02.i04":
            self.group("target")
            old = self.attempts["target:0"]
            self.authorize("target:0", "new", "current")
            reference = self.rejected("state.accept_result.old_target_first", state.accept_result, self.owner, old, self.payload("target:0"))
            self.accept("target:0")
            self.boundary("old_attempt_first_rejected_new_accepted", rejection=reference)
        elif case_id == "X2.b09.i05":
            self.group("target")
            self.owner.close()
            reference = self.rejected("state.commit_generation.closed_owner", state.commit_generation, self.owner, self.bootstrap)
            self.boundary("closed_owner_delayed_commit_rejected", rejection=reference)
        elif case_id.startswith("X2."):
            self.target_x2(self.case["boundary"])
        else:
            raise HarnessError("worker received unsupported case")


def worker_main(directory, case_id, role, mode):
    worker = Worker(directory, role)
    worker.send("ready", mode=mode)
    worker.command("start")
    root = directory / "state-run"
    if mode in ("resume", "corrupt-check", "scope-resume"):
        control = json.loads((root / "control.json").read_text())
        with state.acquire_owner(root, control["epoch"], control["processes"]) as owner:
            worker.owner = owner
            if mode == "scope-resume":
                attempt = worker.authorize("target:0", "recovered", "current", policy=1)
                raw_path = directory / "method-target-0.json"
                raw = json.loads(raw_path.read_text())
                before = digest(raw_path.read_bytes())
                raw["policy_version"] = 1
                with raw_path.open("w") as stream:
                    json.dump(raw, stream, sort_keys=True, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
                worker.record("fixture.recovered_policy_payload", payload_file=raw_path.name,
                    before_sha256=before, after_sha256=digest(raw_path.read_bytes()), policy_version=1,
                    limitation="CPU fixture policy declaration, no actual model provenance")
                payload = worker.payload("target:0")
                receipt = state.accept_result(owner, attempt, payload)
                reference = worker.record("state.accept_result", sample="target:0", attempt=attempt, result=receipt, reward=1.0)
                worker.send("scope_recovered", epoch=owner.epoch, receipt=reference, prior_processes=control["processes"])
            elif mode == "corrupt-check":
                reference = worker.rejected("state.select_recovery.corrupt", state.select_recovery, owner)
                worker.send("corruption_rejected", receipt=reference)
            else:
                decision = state.select_recovery(owner)
                reference = worker.record("state.select_recovery", decision=decision)
                worker.send("recovered", decision=decision, receipt=reference)
        return
    try:
        worker.owner = state.acquire_owner(root, -1, run_nonce=case_id, config_sha256=digest(b"p2-cpu-driver"), verifier_version="exact-v1")
    except BlockingIOError:
        reference = worker.record("state.acquire_owner", result="lock_contended")
        worker.send("owner_rejected", receipt=reference)
        return
    worker.send("acquired", receipt=worker.record("state.acquire_owner", epoch=worker.owner.epoch))
    worker.command("continue")
    try:
        worker.make_bootstrap(reverse=worker.inputs["bootstrap_arrival_order"] == list(reversed(range(8))))
        if case_id.endswith("i02"):
            worker.duplicate_bootstrap_evidence()
        worker.target(case_id)
        worker.send("completed")
    finally:
        worker.owner.close()


class Child:
    def __init__(self, directory, case_id, role, mode):
        self.role = role
        self.log = (directory / (role + "-stderr.log")).open("x")
        self.process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", mode,
            "--case", case_id, "--output", str(directory), "--role", role],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        self.identity = None
        self.buffer = b""

    def send(self, command):
        self.process.stdin.write((json.dumps({"command": command}) + "\n").encode())
        self.process.stdin.flush()

    def receive(self, deadline):
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise HarnessError("worker deadline exceeded")
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise HarnessError("worker exited before expected receipt")
            self.buffer += data
        line, self.buffer = self.buffer.split(b"\n", 1)
        message = json.loads(line)
        if self.identity is None:
            if message["type"] != "ready" or message["identity"] != state.process_identity(self.process.pid):
                raise HarnessError("worker identity mismatch")
            self.identity = message["identity"]
        elif message["identity"] != self.identity:
            raise HarnessError("worker identity changed")
        return message

    def wait(self, deadline, expected=0):
        result = self.process.wait(timeout=max(0.01, deadline - time.monotonic()))
        if result != expected:
            raise HarnessError(f"worker exit {result}, expected {expected}")
        return result

    def cleanup(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()


def check_reference(directory, reference, operation=None):
    rows = [json.loads(line) for line in (directory / reference["file"]).read_text().splitlines()]
    event = rows[reference["sequence"]]
    if event["sequence"] != reference["sequence"] or operation is not None and event["operation"] != operation:
        raise HarnessError("operation receipt mismatch")
    return event


def oracle_graph(inputs, case_id, scenario="continuous"):
    """Independent existing hand-written graph, expanded to eight actual rows."""
    graph = local_module("test_oracle").Graph()
    graph.freeze["k"] = 8
    graph.freeze["expected_ranks"] = ["actor:0", "actor:1"]
    graph.files["external-fixture.json"] = inputs
    graph.freeze["external_fixture_sha256"] = digest(encoded(inputs))
    role = "rollout" if case_id.startswith("F1") else "actor" if case_id.startswith(("F2", "F3")) else "reward"
    graph.freeze["target_role"] = role
    graph.freeze["fault_model"] = {"case_id": case_id, "role": role, "target_group": graph.freeze["target_group"],
                                   "kind": "normative_boundary_not_actual_GPU_hit"}
    for event in graph.evidence["events"]:
        if event["type"] == "fault_observed":
            event["time"] = 10
        elif event["type"] == "role_ready":
            event["time"] = 11
        if event["type"] in ("fault_observed", "role_ready"):
            event["role"] = role
            event["boundary"] = case_id.split(".")[1]
            event["target_group"] = graph.freeze["target_group"]
    graph.update("bootstrap", logical="bootstrap", at=1)
    graph.update("target", parent="bootstrap", at=12)
    if scenario == "rollback":
        graph.evidence["states"] = [saved for saved in graph.evidence["states"] if saved["id"] != "target"]
        graph.evidence["events"] = [event for event in graph.evidence["events"]
            if event["id"] not in ("save-target", "finalize-target", "persist-target")]
        graph.event("reload-bootstrap", "checkpoint_loaded", 18, state="bootstrap", epoch=1)
        graph.update("recomputed", parent="bootstrap", logical_update="update-target", epoch=1, at=20, loaded="bootstrap")
        for event in graph.evidence["events"]:
            if event["type"] == "fault_observed":
                event["time"] = 14.5
            elif event["type"] == "role_ready":
                event["time"] = 19
    elif scenario == "token-reload":
        graph.event("reload-target", "checkpoint_loaded", 18, state="target", epoch=1)
        graph.update("after-reload", parent="target", logical="post-recovery-other", epoch=1, at=20, loaded="target")
        for event in graph.evidence["events"]:
            if event["type"] == "fault_observed":
                event["time"] = 16.5
            elif event["type"] == "role_ready":
                event["time"] = 19
    graph.files["dataset.json"] = [{key: inputs["groups"][name][0][key] for key in ("prompt", "label")}
                                   for name in ("bootstrap", "target")]
    for group in graph.evidence["groups"]:
        suffix = group["id"][len("group-"):]
        external_group = "bootstrap" if suffix == "bootstrap" else "target"
        group["source_index"] = 0 if external_group == "bootstrap" else 1
        group["prompt_sha256"] = inputs["groups"][external_group][0]["prompt_sha256"]
        base_authorization = next(event for event in graph.evidence["events"] if event["id"] == f"auth-{suffix}-0")
        original = copy.deepcopy(group["samples"][0])
        group["k"] = 8
        group["samples"] = []
        rows = []
        for index in range(8):
            sample = copy.deepcopy(original)
            raw = copy.deepcopy(inputs["groups"][external_group][index])
            path = f"{suffix}-sample{index}.json"
            graph.files[path] = raw
            sample.update(index=index, payload_file=path)
            group["samples"].append(sample)
            rows.append({"group": group["logical_id"], "sample": index, "attempt": sample["attempt"],
                "payload_sha256": digest(encoded(raw)), "reward": 1.0, "tokens": list(raw["tokens"]),
                "loss_mask": list(raw["loss_mask"]), "logprobs": list(raw["logprobs"])})
            if index >= 2:
                graph.event(f"auth-{suffix}-{index}", "authorize", base_authorization["time"],
                    group=group["logical_id"], sample=index, attempt=sample["attempt"], epoch=base_authorization["epoch"],
                    policy_version=0, verifier_version="exact-v1")
        graph.files[suffix + "-tensor.json"]["rows"] = rows
        for event in graph.evidence["events"]:
            if event["id"] == "start-" + suffix:
                event["tensor_sha256"] = digest(encoded(graph.files[suffix + "-tensor.json"]))
    for saved in graph.evidence["states"]:
        for mapping in saved["components"].values():
            mapping["actor:1"] = list(mapping["actor:0"])
        if saved["parent"] is not None:
            first = next(event for event in graph.evidence["events"] if event["id"] == saved["finalize_events"][0])
            second = copy.deepcopy(first)
            second.update(id=first["id"] + "-rank1", actor_rank="actor:1", event_nonce=first["event_nonce"] + "-rank1")
            graph.evidence["events"].append(second)
            saved["finalize_events"].append(second["id"])
    if scenario == "safe-drop":
        graph.evidence["final_state"] = "bootstrap"
        graph.evidence["states"] = [saved for saved in graph.evidence["states"] if saved["id"] != "target"]
        graph.evidence["updates"] = [update for update in graph.evidence["updates"] if update["id"] != "physical-target"]
        graph.evidence["events"] = [event for event in graph.evidence["events"] if not event["id"].endswith("target")
                                   and event["id"] != "finalize-target-rank1"]
        graph.event("drop-target", "work_dropped", 14, group=graph.freeze["target_group"])
        graph.event("stop", "safe_stop", 15)
    return graph


def add_boundary_model(graph, case_id):
    """Explicit fixture progress before the modeled fault, no real-time claim."""
    fault = next(event for event in graph.evidence["events"] if event["type"] == "fault_observed")
    boundary = int(case_id.split(".")[1][1:])
    generated = boundary + 1 if case_id.startswith("F1") and boundary < 7 else 8
    if case_id.startswith("F4.") and boundary in (2, 3):
        generated = 7
    for index in range(generated):
        graph.event(f"model-generation-{index}", "generation_done", fault["time"] - 2 + index * .05,
                    group=graph.freeze["target_group"], sample=index)
    if case_id == "F4.b00.i00":
        for index in range(4):
            graph.event(f"model-reward-start-{index}", "reward_start", fault["time"] - 1 + index * .05,
                        group=graph.freeze["target_group"], sample=index)
        fault["condition"] = {"generation_done": 8, "reward_started": [0, 1, 2, 3], "reward_done": [], "executing_sample": 3}
    elif case_id.startswith("F4."):
        starts = list(range(4)) if boundary in (0, 1, 2, 3) else [0] if boundary == 4 else []
        ends = list(range(4)) if boundary == 1 else [0] if boundary == 5 else list(range(8)) if boundary == 9 else []
        if boundary == 3:
            graph.event("model-generation-repeat", "generation_done", fault["time"] - 1.6,
                        group=graph.freeze["target_group"], sample=6)
        for index in starts:
            graph.event(f"model-reward-start-{index}", "reward_start", fault["time"] - 1 + index * .05,
                        group=graph.freeze["target_group"], sample=index)
        for index in ends:
            graph.event(f"model-reward-end-{index}", "reward_done", fault["time"] - .5 + index * .02,
                        group=graph.freeze["target_group"], sample=index)
        if boundary == 7:
            graph.event("model-worker-lost", "worker_lost_model", fault["time"] - .2,
                        samples=[2, 3], scope="no_actual_worker_loss")
        elif boundary == 8:
            graph.event("model-native-retry", "native_retry_model", fault["time"] - .2,
                        samples=list(range(8)), retry_count=1, preserve_responses=True, scope="no_actual_native_retry")
        fault["condition"] = {"generation_done": generated, "reward_started": starts, "reward_done": ends,
            "executing_sample": 3 if boundary == 0 else 0 if boundary == 4 else None,
            "scope": "normative_only_not_GPU_reachability"}
    elif case_id.startswith("F1.b09."):
        for index in range(7):
            graph.event(f"model-reward-end-{index}", "reward_done", fault["time"] - 1 + index * .05,
                        group=graph.freeze["target_group"], sample=index)
        fault["condition"] = {"generation_done": 8, "reward_done": list(range(7)), "missing_reward": 7}
    else:
        fault["condition"] = {"generation_done": generated, "boundary": case_id.split(".")[1]}
    return graph


def write_graph(graph, target):
    graph.freeze["inputs"] = {name: digest(encoded(graph.files[name])) for name in ("dataset.json", "external-fixture.json")}
    graph.evidence["events"].sort(key=lambda event: event["time"])
    graph.evidence["freeze_sha256"] = digest(encoded(graph.freeze))
    seal = {}
    for name, value in dict(graph.files, **{"evidence.json": graph.evidence}).items():
        data = encoded(value)
        (target / name).write_bytes(data)
        seal[name] = digest(data)
    return {"freeze": copy.deepcopy(graph.freeze), "observer_seal": seal}


def audit_variant(directory, name, graph, corrupt_response=False):
    target = directory / ("oracle-" + name)
    target.mkdir()
    spec = write_graph(graph, target)
    write_json(target / "spec.json", spec)
    if corrupt_response:
        path = target / "target-sample0.json"
        raw = json.loads(path.read_text())
        before = digest(path.read_bytes())
        raw["completion"] = "2"
        path.write_bytes(encoded(raw))
        write_json(target / "response-mutation.json", {"file": path.name, "before_sha256": before,
                   "after_sha256": digest(path.read_bytes()), "field": "completion", "value": "2", "old_seal_preserved": True})
    report = audit_run(target, spec)
    write_json(directory / ("oracle-" + name + "-report.json"), report)
    return report


def mutate_x1_graph(graph, inputs, boundary):
    """Independent declared faults from external fixtures; never method output."""
    sample = graph.evidence["groups"][-1]["samples"][0]
    raw = graph.files["target-sample0.json"]
    row = graph.files["target-tensor.json"]["rows"][0]
    if boundary in (1, 2):
        sample["verifier_version"] = "exact-v2"
        if boundary == 1:
            sample["reward"] = row["reward"] = 1.0 - float(raw["completion"].strip() == raw["label"].strip())
    elif boundary == 3:
        raw["label"] = "2"
    elif boundary == 4:
        return "unverifiable"  # audit_variant changes actual bytes only after freezing the seal.
    elif boundary == 6:
        source = copy.deepcopy(inputs["groups"]["other"][0])
        graph.files["other-reward-source.json"] = source
        reward = float(source["completion"].strip() == source["label"].strip())
        sample["reward"] = row["reward"] = reward
        graph.evidence["reward_substitution"] = {"destination": "target:0", "source_file": "other-reward-source.json",
                                                 "source_sha256": digest(encoded(source)), "source_reward": reward}
    elif boundary == 7:
        sample["policy_version"] = 4
        next(update for update in graph.evidence["updates"] if update["id"] == "physical-target")["policy_version"] = 1
    elif boundary == 8:
        del raw["completion"]
        return "unverifiable"
    elif boundary == 9:
        sample["reward"] = row["reward"] = 0.0
        graph.evidence["success"] = True
        graph.evidence["authoritative_reward"] = 0.0
    else:
        raise HarnessError("unsupported X1 oracle boundary")
    row["payload_sha256"] = digest(encoded(raw))
    next(event for event in graph.evidence["events"] if event["id"] == "start-target")["tensor_sha256"] = digest(encoded(graph.files["target-tensor.json"]))
    return "invalid_commit"


def check_x1_oracle_report(boundary, report):
    expected = {1: "independent reward mismatch", 2: "stale/unauthorized sample",
                3: "payload prompt/label differs from frozen source", 4: "hash mismatch",
                6: "independent reward mismatch", 7: "policy staleness violation",
                8: "completion", 9: "independent reward mismatch"}[boundary]
    messages = report["missing_evidence"] if boundary in (4, 8) else report["violations"]
    if not any(expected in message for message in messages):
        raise ContractFailure("X1 oracle did not reach prescribed fault: " + expected)
    if boundary in (1, 7) and not any("stale/unauthorized sample" in message for message in report["violations"]):
        raise ContractFailure("X1 declared version was not checked against authorization")


def legal_policy_graph(inputs):
    """Auxiliary positive contract: two authorized policies within staleness=1."""
    graph = oracle_graph(inputs, "X1.b07.i00")
    group = graph.evidence["groups"][-1]
    group["samples"][0]["policy_version"] = 1
    graph.files["target-sample0.json"]["policy_version"] = 1
    next(event for event in graph.evidence["events"] if event["id"] == "auth-target-0")["policy_version"] = 1
    next(update for update in graph.evidence["updates"] if update["id"] == "physical-target")["policy_version"] = 1
    graph.files["target-tensor.json"]["rows"][0]["payload_sha256"] = digest(encoded(graph.files["target-sample0.json"]))
    next(event for event in graph.evidence["events"] if event["id"] == "start-target")["tensor_sha256"] = digest(encoded(graph.files["target-tensor.json"]))
    return graph


def oracle_lane(directory, case_id):
    inputs = json.loads((directory / "external-fixture.json").read_text())
    graph = add_boundary_model(oracle_graph(inputs, case_id), case_id)
    healthy = audit_variant(directory, "healthy", graph)
    if healthy["status"] != "correct_recovered":
        raise HarnessError("independent complete K8 control did not pass: " + repr(healthy))
    healthy_graph = copy.deepcopy(graph)
    boundary = int(case_id.split(".")[1][1:])
    target = graph.evidence["groups"][-1]
    expected = "correct_recovered"
    if case_id.startswith("F1"):
        invalid = copy.deepcopy(graph)
        invalid.evidence["groups"][-1]["samples"] = invalid.evidence["groups"][-1]["samples"][:boundary + 1 if boundary < 7 else 7]
        counterexample = audit_variant(directory, "invalid-retained-k", invalid)
        if counterexample["status"] != "invalid_commit":
            raise ContractFailure("incomplete retained K was not rejected")
        if boundary in (7, 8):
            field = "tokens" if boundary == 7 else "logprobs"
            del graph.files["target-sample7.json"][field]
            expected = "unverifiable"
        else:
            graph = add_boundary_model(oracle_graph(inputs, case_id, "safe-drop"), case_id)
            expected = "safe_stop"
    elif case_id.startswith("F2") or case_id.startswith("F3.b00."):
        graph = add_boundary_model(oracle_graph(inputs, case_id, "rollback"), case_id)
    elif case_id.startswith("F3.b01."):
        graph = add_boundary_model(oracle_graph(inputs, case_id, "token-reload"), case_id)
    elif case_id == "F4.b06.i03":
        target["samples"][7]["verifier_version"] = "exact-v2"
        expected = "invalid_commit"
    elif case_id.startswith("F4."):
        if boundary == 6:
            target["samples"][7]["verifier_version"] = "exact-v2"
            expected = "invalid_commit"
        elif boundary in (2, 3, 9):
            contract = json.loads((directory / "f4-window-contract.json").read_text())
            graph.files["f4-window-contract.json"] = contract
            graph.freeze["fault_window_contract_sha256"] = digest(encoded(contract))
            graph.event("window-invalid", "technical_invalid", 10, origin="controller",
                reason=contract["preclassified"], unique_generated=7 if boundary in (2, 3) else 8,
                completed_samples=list(range(8)) if boundary == 9 else [], evidence_level="normative_only")
            expected = "technical_invalid"
    elif case_id.startswith("X1.") and boundary != 0:
        expected = mutate_x1_graph(graph, inputs, boundary)
    elif case_id == "X2.b02.i04":
        graph.event("newer-authority", "authorize", 12.5, group=target["logical_id"], sample=0,
                    attempt="newer", epoch=0, policy_version=0, verifier_version="exact-v1")
        expected = "invalid_commit"
    elif case_id == "X2.b09.i05":
        graph.event("new-epoch-authority", "authorize", 12.5, group=target["logical_id"], sample=0,
                    attempt=target["samples"][0]["attempt"], epoch=1, policy_version=0, verifier_version="exact-v1")
        expected = "invalid_commit"
    elif case_id.startswith("X2."):
        if boundary == 0:
            pass
        elif boundary == 4:
            graph = oracle_graph(inputs, case_id, "rollback")
            group = graph.evidence["groups"][-1]
            group["samples"][0]["policy_version"] = 1
            graph.files["recomputed-sample0.json"]["policy_version"] = 1
            next(e for e in graph.evidence["events"] if e["id"] == "auth-recomputed-0")["policy_version"] = 1
            next(u for u in graph.evidence["updates"] if u["id"] == "physical-recomputed")["policy_version"] = 1
            graph.files["recomputed-tensor.json"]["rows"][0]["payload_sha256"] = digest(encoded(graph.files["recomputed-sample0.json"]))
            next(e for e in graph.evidence["events"] if e["id"] == "start-recomputed")["tensor_sha256"] = digest(encoded(graph.files["recomputed-tensor.json"]))
        elif boundary in (2, 3, 9):
            graph.event("replacement-authority", "authorize", 12.5, group=target["logical_id"], sample=0,
                attempt="replacement", epoch=1 if boundary == 9 else 0, policy_version=0, verifier_version="exact-v1")
            expected = "invalid_commit"
        elif boundary == 1:
            target["samples"][0]["reward"] = 0.0
            expected = "invalid_commit"
        elif boundary in (5, 6, 7):
            target["samples"][0]["attempt"] = "never-authorized" if boundary == 5 else "other-sample-attempt" if boundary == 6 else "foreign-run-attempt"
            expected = "invalid_commit"
        elif boundary == 8:
            target["samples"].append(copy.deepcopy(target["samples"][0]))
            expected = "invalid_commit"
    primary = audit_variant(directory, "primary", graph, corrupt_response=case_id.startswith("X1.b04."))
    if primary["status"] != expected:
        raise ContractFailure(f"independent oracle expected {expected}, observed {primary}")
    if case_id.startswith("F1") and boundary not in (7, 8) and (primary["affected_work_recovery"] != "safely_dropped" or primary["safety"] != "pass"):
        raise ContractFailure("F1 did not preserve safe-drop semantics")
    if case_id.startswith("F1") and boundary in (7, 8):
        field = "tokens" if boundary == 7 else "logprobs"
        if not any(field in item for item in primary["missing_evidence"]):
            raise ContractFailure("missing-payload primary did not identify exact field")
    if case_id.startswith("F2") or case_id.startswith("F3.b00."):
        if primary["rolled_back_updates"] != ["physical-target"] or "physical-recomputed" not in primary["retained_updates"]:
            raise ContractFailure("legal rollback/recompute path was not actually audited")
    if case_id.startswith("X1.") and boundary != 0:
        check_x1_oracle_report(boundary, primary)
    reports = {"healthy": healthy, "primary": primary}
    if case_id.startswith("F1"):
        reports["counterexample"] = counterexample
    if "counterexample" not in reports:
        if primary["status"] == "invalid_commit":
            reports["counterexample"] = primary
        else:
            invalid = copy.deepcopy(healthy_graph if primary["status"] in ("unverifiable", "technical_invalid") else graph)
            if case_id.startswith("F2") or case_id.startswith("F3.b00."):
                final = next(saved for saved in invalid.evidence["states"] if saved["id"] == invalid.evidence["final_state"])
                final["updates"].insert(0, "physical-target")
            else:
                invalid.evidence["groups"][-1]["samples"].pop()
            counterexample = audit_variant(directory, "invalid-retained", invalid)
            if counterexample["status"] != "invalid_commit":
                raise ContractFailure("manifest invalid-retention counterexample did not reject")
            reports["counterexample"] = counterexample
    if case_id.endswith("i08"):
        graph = copy.deepcopy(healthy_graph)
        graph.evidence["events"] = [event for event in graph.evidence["events"] if event["id"] != "start-bootstrap"]
        mutated = audit_variant(directory, "missing", graph)
        if mutated["status"] != "unverifiable" or not any("missing optimizer_start" in item for item in mutated["missing_evidence"]):
            raise HarnessError("specific missing bootstrap optimizer evidence was not detected")
        reports["mutation"] = mutated
    elif case_id.endswith("i09"):
        target_dir = directory / "oracle-corrupt"
        target_dir.mkdir()
        graph.files["root-model.json"] = {"fixture_bytes": "A" * 32768}
        spec = write_graph(graph, target_dir)
        write_json(target_dir / "spec.json", spec)
        path = target_dir / "root-model.json"
        with path.open("r+b") as stream:
            stream.seek(16384)
            stream.write(b"B")
        mutated = audit_run(target_dir, spec)
        write_json(directory / "oracle-corrupt-report.json", mutated)
        if mutated["status"] != "unverifiable" or not any("hash mismatch" in item for item in mutated["missing_evidence"]):
            raise HarnessError("observer fullhash corruption not detected")
        reports["mutation"] = mutated
    return reports


def check_f4_boundary(directory, case, operations):
    b = case["boundary"]
    generated = [e["sample"] for e in operations if e["operation"] == "fixture.generate_sample" and e["group"] == "target"]
    unique = set(generated)
    if unique != set(range(7 if b in (2, 3) else 8)):
        raise ContractFailure("F4 unique generation set wrong")
    repeats = [e for e in operations if e["operation"] == "fixture.generate_sample_again"]
    if len(repeats) != int(b == 3):
        raise ContractFailure("F4 repeated generation count wrong")
    if repeats:
        repeat = repeats[0]
        raw = (directory / repeat["payload_file"]).read_bytes()
        if repeat["sample"] != 6 or digest(raw) != repeat["payload_sha256"] or json.loads(raw) != json.loads((directory / "method-target-6.json").read_text()):
            raise ContractFailure("F4 duplicate is not same logical sample/bytes")
    started = [e["sample"] for e in operations if e["operation"] == "fixture.reward_start"]
    ended = [e["sample"] for e in operations if e["operation"] == "fixture.reward_end"]
    if started != (list(range(4)) if b in (0, 1, 2, 3) else [0] if b == 4 else []):
        raise ContractFailure("F4 reward-start sequence differs")
    if ended != (list(range(4)) if b == 1 else [0] if b == 5 else []):
        raise ContractFailure("F4 reward-end sequence differs")
    for event in operations:
        if event["operation"] in ("fixture.reward_end", "fixture.retry_score"):
            raw = json.loads((directory / f"method-target-{event['sample']}.json").read_text())
            if event["reward"] != float(raw["completion"].strip() == raw["label"].strip()):
                raise ContractFailure("F4 actual fixture score incorrect")
    accepted = [e for e in operations if e["operation"] == "state.accept_result" and e["sample"].startswith("target:")]
    count = 7 if b in (2, 3, 6) else 8 if b == 9 else 0
    if [e["sample"] for e in accepted] != [f"target:{i}" for i in range(count)]:
        raise ContractFailure("F4 accepted result set differs")
    if b in (2, 3, 6):
        op = "state.accept_result.wrong_verifier" if b == 6 else "state.prepare_generation.incomplete"
        if len([e for e in operations if e["operation"] == op and e.get("result") == "rejected"]) != 1:
            raise ContractFailure("F4 admission rejection missing")
    if b == 5 and not any(e["operation"] == "fixture.before_accept_result" and e["sample"] == 0 for e in operations):
        raise ContractFailure("F4 before-admission gap missing")
    if b == 7:
        lost = [e for e in operations if e["operation"] == "fixture.worker_lost"]
        if len(lost) != 1 or lost[0]["samples"] != [2, 3] or lost[0]["evidence_level"] != "model_only_no_process_loss":
            raise ContractFailure("F4 lost-worker model missing")
    if b == 8:
        retry = [e for e in operations if e["operation"] == "fixture.native_retry_model"]
        scores = [e for e in operations if e["operation"] == "fixture.retry_score"]
        if len(retry) != 1 or [e["sample"] for e in scores] != list(range(8)) or retry[0]["before"] != retry[0]["after"] or retry[0]["retry_count"] != 1:
            raise ContractFailure("F4 retry model evidence missing")
    if b == 9:
        checks = [e for e in operations if e["operation"] == "fixture.check_fault_window"]
        if len(checks) != 1 or checks[0]["completed_samples"] != list(range(8)) or checks[0]["executing_sample"] is not None:
            raise ContractFailure("F4 missed-window evidence missing")
    contract = json.loads((directory / "f4-window-contract.json").read_text())
    classification = "technical_invalid_unmet_window" if len(unique) < 8 else "technical_invalid_missed_window" if b == 9 else "model_contract_only"
    if contract["preclassified"] != classification:
        raise ContractFailure("F4 pre-run window contract contradicted")
    return {"unique_generated": len(unique), "generation_events": len(generated) + len(repeats),
            "classification": classification, "scope": "CPU model; no actual native retry or F4 GPU fault"}


def check_x2_boundary(directory, case, operations):
    b = case["boundary"]
    admitted = [e for e in operations if e["operation"] == "state.accept_result" and e["sample"].startswith("target:")]
    if [e["sample"] for e in admitted[:8]] != [f"target:{i}" for i in range(8)]:
        raise ContractFailure("X2 baseline admission missing")
    rejection_names = {1: "state.accept_result.conflict", 2: "state.accept_result.old_target_first",
        3: "state.accept_result.old_target_first", 5: "state.accept_result.unmapped", 6: "state.accept_result.unmapped",
        7: "state.accept_result.foreign", 8: "state.prepare_generation.duplicate", 9: "state.commit_generation.closed_owner"}
    if b == 0:
        if len(admitted) != 9 or admitted[-1]["result"] != admitted[0]["result"]:
            raise ContractFailure("X2 duplicate not idempotent")
    elif b != 4:
        rejects = [e for e in operations if e["operation"] == rejection_names[b]]
        if len(rejects) != 1 or rejects[0]["result"] != "rejected":
            raise ContractFailure("X2 exact rejection absent")
        if b in (2, 3):
            if len(admitted) != 9 or admitted[-1]["attempt"]["attempt"] != "new" or ((rejects[0]["sequence"] < admitted[-1]["sequence"]) != (b == 2)):
                raise ContractFailure("X2 old/new ordering differs")
        if b == 7:
            foreign = json.loads((directory / "foreign-run/control.json").read_text())
            local = json.loads((directory / "state-run/control.json").read_text())
            if foreign["owner_nonce"] == local["owner_nonce"] or foreign["run_nonce"] == local["run_nonce"] or "target:0" not in foreign["accepted"]:
                raise ContractFailure("X2 foreign root/result not real")
        if b == 6:
            submitted = next(e for e in operations if e["operation"] == "fixture.submit_scope")
            if submitted["submitted"]["sample"] != "other:0" or submitted["authorized"]["sample"] != "target:0":
                raise ContractFailure("X2 sample mismatch absent")
    return {"baseline_admitted": 8, "boundary": b, "scope": "actual CPU state API; O independently models invalid retention"}


def check_stage_one(directory, case, operations):
    """Actual boundary evidence, not a claimed worker success label."""
    if case["id"] not in STAGE_ONE:
        return {}
    boundary = case["boundary"]
    accepted = [event for event in operations if event["operation"] == "state.accept_result" and event["sample"].startswith("target:")]
    count = boundary + 1 if case["cell"] == "F1" and boundary < 7 else 7 if case["cell"] == "F1" and boundary == 9 else 8
    if [event["sample"] for event in accepted] != [f"target:{index}" for index in range(count)]:
        raise ContractFailure("target admitted sample set/order differs from boundary")
    prepared = [event for event in operations if event["operation"] == "state.prepare_generation" and event["logical_group"] == "target"]
    requires_prepare = case["cell"] == "X1" or boundary in (7, 8)
    if len(prepared) != int(requires_prepare):
        raise ContractFailure("target prepare depth differs from boundary")
    if not requires_prepare:
        suffix = "missing_reward" if boundary == 9 else "incomplete"
        rejected = [event for event in operations if event["operation"] == "state.prepare_generation." + suffix]
        if len(rejected) != 1 or rejected[0]["result"] != "rejected" or "incomplete group" not in rejected[0]["reason"]:
            raise ContractFailure("target incomplete K rejection absent")
    else:
        intent = json.loads((directory / prepared[0]["intent_file"]).read_text())
        data = intent["data"]
        if case["interleaving"] in (3, 4) and not any(item["sample"] == "probe:0" and item["action"] == "regenerate" for item in data["pending"]):
            raise ContractFailure("accepted unconsumed probe lost pending obligation")
        if case["cell"] == "X1":
            computed = [event for event in operations if event["operation"] == "fixture.compute_reward"]
            if len(computed) != 1 or computed[0]["function"] != "fixture-exact" or computed[0]["actual_value"] != 1.0:
                raise ContractFailure("positive fixture exact reward was not computed")
            generation = prepared[0]["generation"]
            commits = [event for event in operations if event["operation"] == "state.commit_generation" and event["generation"] == generation]
            token_path = directory / "state-run/generations" / generation / "token.json"
            if len(commits) != 1 or not token_path.exists() or json.loads(token_path.read_text()) != commits[0]["token"]:
                raise ContractFailure("committed_target missing actual target commit/token")
            receipts = [event["evidence"] for event in operations if event["operation"] == "state.record_evidence" and event["generation"] == generation]
            if not any(item["kind"] == "optimizer" and item["successful"] and item["scheduler_applied"] for item in receipts):
                raise ContractFailure("committed_target missing successful optimizer receipt")
            if {item["rank"] for item in receipts if item["kind"] == "finalize"} != set(intent["expected_ranks"]):
                raise ContractFailure("committed_target missing rank finalize receipts")
            files = [event for event in operations if event["operation"] == "fixture.write_component" and event["generation"] == generation]
            if {(event["rank"], event["component"]) for event in files} != {(rank, component) for rank in intent["expected_ranks"] for component in COMPONENTS}:
                raise ContractFailure("committed_target incomplete component mapping")
            for event in files:
                if digest((directory / event["relative_path"]).read_bytes()) != event["sha256"]:
                    raise ContractFailure("committed_target component does not match write receipt")
    check_modifiers(case, operations)
    return {"admitted_target_samples": count, "target_prepared": requires_prepare,
            "target_committed": case["cell"] == "X1", "modifier": case["expected"]["modifier"],
            "receipt_sequences": [event["sequence"] for event in operations]}


def check_x1_boundary(directory, case, operations, inputs):
    boundary = case["boundary"]
    rejected_version = boundary in (1, 2, 7)
    accepted = [event for event in operations if event["operation"] == "state.accept_result" and event["sample"].startswith("target:")]
    expected_indices = list(range(1 if rejected_version else 0, 8))
    if [event["sample"] for event in accepted] != [f"target:{index}" for index in expected_indices]:
        raise ContractFailure("X1 actual admitted sample set differs")
    expected_reward = 0.0 if boundary in (6, 9) else 1.0
    if any(event["reward"] != (expected_reward if event["sample"] == "target:0" else 1.0) for event in accepted):
        raise ContractFailure("X1 did not submit intended reward values")
    prepared = [event for event in operations if event["operation"] == "state.prepare_generation" and event["logical_group"] == "target"]
    if len(prepared) != int(not rejected_version):
        raise ContractFailure("X1 target preparation depth differs")
    if rejected_version:
        operation = "state.accept_result.wrong_policy" if boundary == 7 else "state.accept_result.wrong_verifier"
        rejected = [event for event in operations if event["operation"] == operation]
        submissions = [event for event in operations if event["operation"] == "fixture.submitted_declaration"]
        incomplete = [event for event in operations if event["operation"] == "state.prepare_generation.after_rejected_reward"]
        if len(rejected) != 1 or len(submissions) != 1 or len(incomplete) != 1 or "incomplete group" not in incomplete[0]["reason"]:
            raise ContractFailure("X1 authorization rejection did not preserve incomplete group")
        submitted = submissions[0]
        payload = submitted["payload"]
        if submitted["authorized"]["versions"] != {"policy_version": 0, "verifier_version": "exact-v1"}:
            raise ContractFailure("X1 expected authorization was not frozen")
        if (payload["policy_version"], payload["verifier_version"]) != ((4, "exact-v1") if boundary == 7 else (0, "exact-v2")):
            raise ContractFailure("X1 did not submit intended unauthorized declaration")
        if payload["reward_sha256"] != digest(encoded(0.0 if boundary == 1 else 1.0)):
            raise ContractFailure("X1 unauthorized result carried wrong test score")
    if boundary == 1:
        scored = [event for event in operations if event["operation"] == "fixture.compute_reward"]
        if len(scored) != 1 or (scored[0]["exact_value"], scored[0]["actual_value"]) != (1.0, 0.0):
            raise ContractFailure("alternate verifier did not compute distinct score")
    elif boundary == 2:
        changed = [event for event in operations if event["operation"] == "fixture.change_declared_verifier"]
        if len(changed) != 1 or (changed[0]["before_reward"], changed[0]["after_reward"]) != (1.0, 1.0):
            raise ContractFailure("label-only verifier mutation changed reward")
    elif boundary in (3, 8):
        name = "fixture.change_observer_label" if boundary == 3 else "fixture.remove_authority_payload"
        changed = [event for event in operations if event["operation"] == name]
        if len(changed) != 1:
            raise ContractFailure("X1 authority mutation missing")
        event = changed[0]
        source = inputs["groups"]["target"][0]
        expected = copy.deepcopy(source)
        if boundary == 3:
            expected["label"] = "2"
        else:
            del expected["completion"]
        path = directory / event["file"]
        if event["before"] != source or event["after"] != expected or json.loads(path.read_text()) != expected or digest(path.read_bytes()) != event["sha256"]:
            raise ContractFailure("X1 authority mutation did not preserve frozen source")
    elif boundary == 6:
        substitutions = [event for event in operations if event["operation"] == "fixture.replace_reward_from_other_group"]
        if len(substitutions) != 1:
            raise ContractFailure("actual foreign reward source absent")
        event = substitutions[0]
        raw = json.loads((directory / event["source_file"]).read_text())
        value = float(raw["completion"].strip() == raw["label"].strip())
        if (event["destination"], event["source"], value, event["source_reward"], event["target_exact_reward"]) != ("target:0", "other:0", 0.0, 0.0, 1.0):
            raise ContractFailure("foreign score provenance does not match substitution")
        if digest((directory / event["source_file"]).read_bytes()) != event["source_sha256"]:
            raise ContractFailure("foreign score source hash differs")
    elif boundary == 9:
        forged = [event for event in operations if event["operation"] == "fixture.forge_method_success"]
        if len(forged) != 1 or (forged[0]["success"], forged[0]["authoritative_reward"], forged[0]["actual_reward"], forged[0]["correct_reward"]) != (True, 0.0, 0.0, 1.0):
            raise ContractFailure("forged method success fixture absent")
    if prepared and case["interleaving"] in (3, 4):
        intent = json.loads((directory / prepared[0]["intent_file"]).read_text())
        if [item["sample"] for item in intent["data"]["pending"]] != ["probe:0"]:
            raise ContractFailure("X1 unconsumed probe obligation lost")
    check_modifiers(case, operations)
    return {"boundary": boundary, "accepted_indices": expected_indices, "target_prepared": bool(prepared),
            "unauthorized_result_rejected": rejected_version, "reward_semantics_not_claimed_by_state": True,
            "receipt_sequences": [event["sequence"] for event in operations]}


def check_modifiers(case, operations):
    if case["interleaving"] == 2:
        repeated = [event for event in operations if event["operation"] == "state.record_evidence.identical"]
        snapshots = [event for event in operations if event["operation"] == "fixture.assert_idempotent_generation"]
        if len(repeated) != 2 or any(event["evidence"] != event["result"] for event in repeated) or len(snapshots) != 1 or snapshots[0]["before"] != snapshots[0]["after"]:
            raise ContractFailure("committed evidence replay not proven idempotent")
    if case["interleaving"] in (3, 4):
        arrivals = [event["operation"] for event in operations if
                    event["operation"] == "state.accept_result.old_probe" or
                    event["operation"] == "state.accept_result" and event["sample"] == "probe:0"]
        expected = ["state.accept_result", "state.accept_result.old_probe"]
        if case["interleaving"] == 4:
            expected.reverse()
        if arrivals != expected:
            raise ContractFailure("probe actual arrival order differs from interleaving")


def inspect_storage_cut(directory, case, message, operations):
    """Read the blocked writer's real files before SIGKILL or recovery changes them."""
    cell, boundary = case["cell"], case["boundary"]
    generation = message["generation"]
    no_intent = cell == "F2" and boundary == 0
    if (generation is None) != no_intent:
        raise ContractFailure("wrong intent presence at cut")
    root = directory / "state-run/generations"
    intents = {path.parent.name: json.loads(path.read_text()) for path in root.glob("*/intent.json")}
    target_intents = {name: intent for name, intent in intents.items() if intent["data_snapshot_id"] == "target-cut"}
    if set(target_intents) != (set() if no_intent else {generation}):
        raise ContractFailure("unexpected target intents at cut")
    artifacts = {} if no_intent else {str(path.relative_to(root / generation)): {
        "sha256": digest(path.read_bytes()), "size": path.stat().st_size}
        for path in (root / generation).rglob("*") if path.is_file()}
    receipts = {} if no_intent else {path.name: json.loads(path.read_text()) for path in (root / generation / "receipts").glob("*.json")}
    intent = target_intents.get(generation)
    starts = [event for event in operations if event["operation"] == "fixture.optimizer_start" and event["physical_update"] == "target-physical"]
    ends = [event for event in operations if event["operation"] == "fixture.optimizer_end" and event["physical_update"] == "target-physical"]
    # Each row fixes start/end/optimizer-receipt, saved ranks, finalized ranks, manifest, token.
    both = ("actor:0", "actor:1")
    rows = ((0, 0, False, (), (), False, False),
            (0, 0, False, (), (), False, False),
            (0, 0, False, (), (), False, False),
            (1, 0, False, (), (), False, False),
            (1, 1, False, (), (), False, False),
            (1, 1, True, (), (), False, False),
            (0, 0, False, (), (), False, False),
            (1, 1, True, ("actor:0",), ("actor:0",), False, False),
            (1, 1, True, both, (), False, False),
            (1, 1, True, both, both, True, False))
    expected = rows[boundary] if cell == "F2" else (
        (1, 1, True, ("actor:0",), ("actor:0",), False, False) if boundary == 0
        else (1, 1, True, both, both, True, True))
    nstart, nend, optimizer, ranks, finalized, manifest, token = expected
    if (len(starts), len(ends)) != (nstart, nend) or any(event["successful"] is not True for event in ends):
        raise ContractFailure("optimizer progress does not match exact cut")
    expected_receipts = ({"optimizer.json"} if optimizer else set()) | {"rank-" + digest(rank.encode()) + ".json" for rank in finalized}
    if set(receipts) != expected_receipts:
        raise ContractFailure("optimizer/rank receipt presence differs from cut")
    if optimizer and receipts["optimizer.json"] != {"kind": "optimizer", "snapshot_id": "target-cut",
            "physical_updates": ["target-physical"], "successful": True, "scheduler_applied": True}:
        raise ContractFailure("wrong successful optimizer receipt at cut")
    for rank in finalized:
        if receipts["rank-" + digest(rank.encode()) + ".json"] != {"kind": "finalize", "snapshot_id": "target-cut", "rank": rank, "writer_closed": True}:
            raise ContractFailure("wrong rank finalize receipt at cut")
    expected_components = {"checkpoint/" + rank.replace(":", "-") + "/" + component + ".bin"
                           for rank in ranks for component in COMPONENTS}
    actual_components = {name for name in artifacts if name.startswith("checkpoint/")}
    if actual_components != expected_components or any(artifacts[name] != {"sha256": digest(b"A" * 32768), "size": 32768} for name in actual_components):
        raise ContractFailure("actual component coverage/bytes differ from cut")
    if (("manifest.json" in artifacts), ("token.json" in artifacts)) != (manifest, token):
        raise ContractFailure("manifest/token presence differs from cut")
    before_start = [event for event in operations if event["operation"] == "fixture.before_optimizer_start"]
    scheduler_rejections = [event for event in operations if event["operation"] == "state.record_evidence.bad_scheduler"]
    missing_rank = [event for event in operations if event["operation"] == "state.commit_generation.missing_rank"]
    if len(before_start) != int(cell == "F2" and boundary == 2):
        raise ContractFailure("before-optimizer marker does not match cut")
    scheduler_attempts = [event for event in operations if event["operation"] == "fixture.scheduler_evidence_attempt"]
    if len(scheduler_attempts) != int(cell == "F2" and boundary == 6) or any(event["evidence"] != {
            "kind": "optimizer", "snapshot_id": "target-cut", "physical_updates": ["target-physical"],
            "successful": True, "scheduler_applied": False} for event in scheduler_attempts):
        raise ContractFailure("wrong scheduler evidence submitted before cut")
    if len(scheduler_rejections) != int(cell == "F2" and boundary == 6) or any("optimizer/scheduler evidence incomplete" not in event["reason"] for event in scheduler_rejections):
        raise ContractFailure("scheduler rejection not demonstrated at cut")
    if len(missing_rank) != int(cell == "F3" and boundary == 0) or any("cannot read rank-" not in event["reason"] for event in missing_rank):
        raise ContractFailure("missing-rank rejection not demonstrated at cut")
    accepted = [event for event in operations if event["operation"] == "state.accept_result" and event["sample"].startswith("target:")]
    if [event["sample"] for event in accepted] != [f"target:{index}" for index in range(8)]:
        raise ContractFailure("storage target did not admit complete K8")
    check_modifiers(case, operations)
    if intent:
        expected_consumed = [f"{group}:{index}" for group in ("bootstrap", "target") for index in range(8)]
        if intent["data"]["consumed"] != expected_consumed:
            raise ContractFailure("target cut consumption obligations differ")
        expected_pending = ["probe:0"] if case["interleaving"] in (3, 4) else []
        if [item["sample"] for item in intent["data"]["pending"]] != expected_pending:
            raise ContractFailure("cut lost unconsumed probe obligation")
    declared_call = case["operations"][case["cut"]["call_offset"]]
    return {"generation": generation, "intent": intent, "artifacts": artifacts, "receipts": receipts,
            "optimizer_starts": len(starts), "optimizer_ends": len(ends), "saved_ranks": list(ranks),
            "finalized_ranks": list(finalized), "manifest_present": manifest, "token_present": token,
            "before_start_marker": bool(before_start), "scheduler_rejected": bool(scheduler_rejections),
            "missing_rank_rejected": bool(missing_rank), "declared_last_call": declared_call,
            "observed_cut": message["when"], "observed_before_sigkill": True,
            "writer_identity": message["identity"], "raw_receipt_sequences": [event["sequence"] for event in operations]}


def token_head(root):
    """Head of the committed chain derived from tokens, not control.json."""
    tokens = {}
    for item in (root / "generations").iterdir():
        if (item / "token.json").exists():
            tokens[item.name] = json.loads((item / "token.json").read_text())
    parents = {t["parent"]["generation"] for t in tokens.values() if t["parent"] is not None}
    heads = [gid for gid in tokens if gid not in parents]
    if len(heads) != 1:
        raise HarnessError("no unique committed head for corruption")
    return heads[0]


def check_storage_recovery(directory, case, cut_snapshot, bootstrap, decision, receipt):
    promote = case["cell"] == "F2" and case["boundary"] == 9
    visible = case["cell"] == "F3" and case["boundary"] == 1
    no_intent = case["cell"] == "F2" and case["boundary"] == 0
    target = cut_snapshot["generation"]
    expected = target if promote or visible else bootstrap["generation"]
    if decision["generation"] != expected:
        raise ContractFailure("selected wrong recovery generation")
    obligations = [] if promote or visible or no_intent else [str(directory / "state-run/generations" / target / "intent.json")]
    if decision["rollback_intents"] != obligations:
        raise ContractFailure("recovery rollback obligations differ from cut")
    if promote and decision["reason"] != "completed durable candidate":
        raise ContractFailure("complete candidate was not durably promoted")
    if not (promote or visible or no_intent) and not decision["reason"].startswith("incomplete candidate:"):
        raise ContractFailure("incomplete generation was not explicitly rejected")
    token = json.loads((directory / "state-run/generations" / expected / "token.json").read_text())
    if token["generation"] != expected:
        raise ContractFailure("recovery token does not exist for selected generation")
    if promote or visible:
        if token["execution_epoch"] != cut_snapshot["intent"]["execution_epoch"]:
            raise ContractFailure("recovery changed target execution epoch")
        if decision["pending"] != cut_snapshot["intent"]["data"]["pending"]:
            raise ContractFailure("retained generation lost pending obligations")
    if target:
        path = directory / "state-run/generations" / target / "intent.json"
        if digest(path.read_bytes()) != cut_snapshot["artifacts"]["intent.json"]["sha256"]:
            raise ContractFailure("recovery altered immutable consumption intent")
    return {"snapshot": "storage-cut.json", "selected_generation": expected,
            "rollback_intents": obligations, "recovery_receipt": receipt,
            "original_execution_epoch": None if no_intent else cut_snapshot["intent"]["execution_epoch"],
            "no_target_intent": no_intent, "candidate_promoted": promote, "visible_token_verified": visible,
            "recovery_scope": "no target intent exists; accepted results are not claimed replayable" if no_intent else "immutable intent obligations and selected generation only; no actual optimizer replay"}


def run_case(case, directory, timeout=30):
    directory = Path(directory).absolute()
    directory.mkdir(parents=True, exist_ok=False)
    result = {"case_id": case["id"], "semantic_sha256": case["semantic_sha256"], "status": "not_executed",
              "must_reach": {}, "state_process_lane": {}, "oracle_model_lane": {}, "missing_requirements": [],
              "limitations": ["CPU component fixtures, no training backend", "oracle times are simulated, not measured RTO",
                              "no AReaL, GPU, P1 injection journal or real ACK tested"]}
    write_json(directory / "case.json", case)
    if case["id"] not in SUPPORTED or case["execution_status"] != "executable":
        result["reason"] = "case outside implemented boundary combinations or missing interface"
        write_json(directory / "result.json", result)
        return result
    inputs = external_fixture(case)
    write_json(directory / "external-fixture.json", inputs)
    if case["cell"] == "F4":
        write_json(directory / "f4-window-contract.json", {
            "required_unique_generated": 8, "required_executing_sample": True,
            "required_incomplete_rewards": True, "scope": "CPU normative events only",
            "planned_unique_generated": 7 if case["boundary"] in (2, 3) else 8,
            "planned_all_rewards_completed": case["boundary"] == 9,
            "preclassified": "technical_invalid_unmet_window" if case["boundary"] in (2, 3)
                else "technical_invalid_missed_window" if case["boundary"] == 9 else "model_contract_only"})
    deadline = time.monotonic() + timeout
    children, messages = [], []
    reached = result["must_reach"]
    def spawn(role, mode="initial"):
        child = Child(directory, case["id"], role, mode)
        children.append(child)
        messages.append(child.receive(deadline))
        return child
    try:
        local_module("p2_schedule_cases").validate_completion(case)
        count = 2 if case["id"].endswith("i05") else 1
        initial = [spawn("owner" + str(index)) for index in range(count)]
        if count == 2:
            if not all(child.process.poll() is None for child in initial):
                raise HarnessError("owner contenders not simultaneously alive")
            reached["two_live_owner_contenders"] = {"identities": [child.identity for child in initial]}
        for child in initial:
            child.send("start")
        answers = [(child, child.receive(deadline)) for child in initial]
        messages.extend(message for _, message in answers)
        winners = [child for child, message in answers if message["type"] == "acquired"]
        losers = [child for child, message in answers if message["type"] == "owner_rejected"]
        if len(winners) != 1 or len(losers) != count - 1:
            raise HarnessError("owner exclusivity failed")
        winner = winners[0]
        for loser in losers:
            loser.wait(deadline)
        if count == 2:
            reached["one_winner_one_rejection"] = {"messages": answers[0][1:]+answers[1][1:]}
        winner.send("continue")
        bootstrap = None
        cut = None
        while True:
            message = winner.receive(deadline)
            messages.append(message)
            if message["type"] == "bootstrap":
                bootstrap = message
                check_reference(directory, message["receipt"], "state.commit_generation")
                token = json.loads((directory / message["token_file"]).read_text())
                if token["generation"] != message["generation"]:
                    raise HarnessError("bootstrap token mismatch")
                reached["bootstrap_committed"] = message["receipt"]
                if count == 2:
                    reached["winner_continues_target"] = {"identity": winner.identity, "bootstrap": message["receipt"]}
            elif message["type"] == "boundary":
                check_reference(directory, message["receipt"], "target_boundary")
                reached["target_boundary_reached"] = message["receipt"]
                result["state_process_lane"]["boundary"] = message["result"]
            elif message["type"] == "cut":
                check_reference(directory, message["receipt"], "cut_ready")
                cut = message
                reached["exact_cut_ready"] = message["receipt"]
                if state.process_identity(winner.process.pid) != winner.identity:
                    raise HarnessError("signal target identity changed")
                before_kill = [json.loads(line) for line in (directory / (winner.role + "-operations.jsonl")).read_text().splitlines()]
                cut_snapshot = inspect_storage_cut(directory, case, message, before_kill)
                write_json(directory / "storage-cut.json", cut_snapshot)
                reached["storage_cut_verified"] = {"file": "storage-cut.json", "sha256": digest((directory / "storage-cut.json").read_bytes())}
                winner.process.kill()
                code = winner.wait(deadline, expected=-signal.SIGKILL)
                reached["confirmed_sigkill"] = {"identity": winner.identity, "returncode": code}
                break
            elif message["type"] == "completed":
                winner.wait(deadline)
                break
            else:
                raise HarnessError("unexpected worker message")
        operations = [json.loads(line) for line in (directory / (winner.role + "-operations.jsonl")).read_text().splitlines()]
        generated = [event for event in operations if event["operation"] == "fixture.generate_sample"]
        order = [event["sample"] for event in generated if event["group"] == "bootstrap"]
        if order != inputs["bootstrap_arrival_order"]:
            raise HarnessError("actual bootstrap generation order differs from frozen manifest")
        target_order = [event["sample"] for event in generated if event["group"] == "target"]
        if target_order != inputs["target_arrival_order"]:
            raise HarnessError("actual target generation order differs from frozen manifest")
        for event in generated:
            path = directory / event["payload_file"]
            expected_raw = copy.deepcopy(inputs["groups"][event["group"]][event["sample"]])
            expected_hash = event["payload_sha256"]
            if case["cell"] == "F1" and case["boundary"] == 7 and event["group"] == "target" and event["sample"] == 7:
                mutations = [item for item in operations if item["operation"] == "fixture.remove_method_payload"]
                if len(mutations) != 1 or mutations[0]["before_sha256"] != expected_hash or mutations[0]["field"] != "tokens":
                    raise HarnessError("missing exact method payload mutation receipt")
                expected_hash = mutations[0]["after_sha256"]
                del expected_raw["tokens"]
            if case["cell"] == "X1" and case["boundary"] == 4 and event["group"] == "target" and event["sample"] == 0:
                mutations = [item for item in operations if item["operation"] == "fixture.change_response_bytes"]
                if len(mutations) != 1 or mutations[0]["before_sha256"] != expected_hash:
                    raise HarnessError("missing precise response mutation")
                seal = json.loads((directory / mutations[0]["seal_file"]).read_text())
                if seal[path.name] != expected_hash:
                    raise HarnessError("response seal was silently updated")
                expected_hash = mutations[0]["after_sha256"]
                expected_raw["completion"] = "2"
            if digest(path.read_bytes()) != expected_hash or json.loads(path.read_text()) != expected_raw:
                raise HarnessError("method payload differs from independent frozen input/mutation")
        result["boundary_checks"] = check_stage_one(directory, case, operations)
        if case["cell"] == "X1" and case["boundary"] != 0:
            result["boundary_checks"] = check_x1_boundary(directory, case, operations, inputs)
        if case["cell"] == "F4":
            result["boundary_checks"] = check_f4_boundary(directory, case, operations)
        if case["cell"] == "X2":
            result["boundary_checks"] = check_x2_boundary(directory, case, operations)
        reached["shared_input_and_arrival_verified"] = {"fixture": "external-fixture.json", "actual_bootstrap_order": order, "actual_target_order": target_order,
            "receipts": [{"file": winner.role + "-operations.jsonl", "sequence": event["sequence"]} for event in generated]}
        if case["cut"] is not None and cut is None:
            raise HarnessError("declared cut not reached")
        if cut is not None:
            resumed = spawn("resume", "resume")
            resumed.send("start")
            message = resumed.receive(deadline)
            messages.append(message)
            resumed.wait(deadline)
            if message["type"] != "recovered" or resumed.identity == winner.identity:
                raise HarnessError("fresh process recovery absent")
            check_reference(directory, message["receipt"], "state.select_recovery")
            result["boundary_checks"] = check_storage_recovery(directory, case, cut_snapshot, bootstrap,
                                                                   message["decision"], message["receipt"])
            reached["fresh_process_recovery"] = message["receipt"]
            result["state_process_lane"]["recovery"] = message["decision"]
        if case["cell"] == "X2" and case["boundary"] == 4:
            if winner.process.poll() != 0:
                raise HarnessError("registered owner has not exited")
            resumed = spawn("scope-resume", "scope-resume")
            resumed.send("start")
            message = resumed.receive(deadline)
            messages.append(message)
            resumed.wait(deadline)
            accepted = check_reference(directory, message["receipt"], "state.accept_result")
            if (message["type"] != "scope_recovered" or message["epoch"] != 1
                    or winner.identity not in message["prior_processes"]
                    or accepted["attempt"]["versions"]["policy_version"] != 1
                    or resumed.identity == winner.identity):
                raise ContractFailure("real exited-owner epoch takeover missing")
            reached["exited_owner_new_epoch_acceptance"] = message
        if case["id"].endswith("i09"):
            control_report = audit_variant(directory, "pre-state-mutation-control", oracle_graph(inputs, case["id"], "token-reload" if case["id"].startswith("F3.b01.") else "continuous"))
            if control_report["status"] != "correct_recovered":
                raise HarnessError("unmodified control failed before state corruption")
            # Recovery content-hashes only the generation it loads (the
            # token-derived head); ancestors are metadata-checked online and
            # content-checked offline. Corrupt the generation recovery loads.
            path = directory / "state-run/generations" / token_head(directory / "state-run") / "checkpoint/actor-0/model.bin"
            before = digest(path.read_bytes())
            with path.open("r+b") as stream:
                stream.seek(16384)
                stream.write(b"B")
                stream.flush()
                os.fsync(stream.fileno())
            write_json(directory / "state-corruption.json", {"file": str(path.relative_to(directory)), "offset": 16384,
                "before_sha256": before, "after_sha256": digest(path.read_bytes()), "size": path.stat().st_size})
            checker = spawn("corrupt-check", "corrupt-check")
            checker.send("start")
            message = checker.receive(deadline)
            messages.append(message)
            checker.wait(deadline)
            if message["type"] != "corruption_rejected":
                raise HarnessError("state corruption was accepted")
            check_reference(directory, message["receipt"], "state.select_recovery.corrupt")
        if time.monotonic() >= deadline:
            raise HarnessError("case deadline before oracle")
        reports = oracle_lane(directory, case["id"])
        result["oracle_model_lane"] = {name: {"status": report["status"], "safety": report["safety"]} for name, report in reports.items()}
        reached["independent_oracle_graph_built"] = {"spec": "oracle-healthy/spec.json", "k": 8, "ranks": 2}
        if case["id"].endswith(("i08", "i09")):
            reached["unmodified_control_audited"] = {"report": "oracle-healthy-report.json" if case["id"].endswith("i08")
                else "oracle-pre-state-mutation-control-report.json", "status": reports["healthy"]["status"]}
            reached["specific_mutation_audited"] = {"report": "oracle-missing-report.json" if case["id"].endswith("i08") else "oracle-corrupt-report.json"}
        result["missing_requirements"] = [name for name in case["must_reach"] if name not in reached]
        if result["missing_requirements"]:
            raise HarnessError("must_reach missing: " + repr(result["missing_requirements"]))
        if time.monotonic() >= deadline:
            raise HarnessError("case deadline exceeded")
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed" if isinstance(exc, ContractFailure) else "harness_failure",
                      reason=f"{type(exc).__name__}: {exc}")
    finally:
        write_json(directory / "launches.json", [{"argv": child.process.args, "pid": child.process.pid,
                   "identity": child.identity, "returncode_before_cleanup": child.process.poll()} for child in children])
        for child in children:
            child.cleanup()
        write_json(directory / "parent-receipts.json", messages)
        write_json(directory / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "fixtures/p2_schedules.json")
    parser.add_argument("--case", action="append")
    parser.add_argument("--representative", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case-timeout", type=float, default=30)
    parser.add_argument("--total-timeout", type=float, default=240)
    parser.add_argument("--worker", choices=("initial", "resume", "corrupt-check", "scope-resume"), help=argparse.SUPPRESS)
    parser.add_argument("--role", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker_main(args.output, args.case[0], args.role, args.worker)
        return
    if bool(args.case) == bool(args.representative) or args.case_timeout <= 0 or args.total_timeout <= 0:
        parser.error("choose --case or --representative, with positive deadlines")
    compiler = local_module("p2_schedule_cases")
    manifest = json.loads(args.manifest.read_text())
    compiler.validate_manifest(manifest)
    selected = list(REPRESENTATIVES) if args.representative else args.case
    lookup = {case["id"]: case for case in manifest["cases"]}
    if len(set(selected)) != len(selected) or any(name not in lookup for name in selected):
        parser.error("case IDs must be unique and present in the manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    mount = {"type": "unknown", "mountpoint": None}
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        point = Path(fields[4])
        if point == args.output.resolve() or point in args.output.resolve().parents:
            if mount["mountpoint"] is None or len(str(point)) > len(mount["mountpoint"]):
                mount = {"type": fields[fields.index("-") + 1], "mountpoint": str(point)}
    write_json(args.output / "inventory.json", {"declared": 600, "static_counts": manifest["counts"],
        "driver_supported": list(SUPPORTED), "requested": selected, "python": sys.version,
        "filesystem": mount, "case_timeout_seconds": args.case_timeout, "total_timeout_seconds": args.total_timeout,
        "manifest_sha256": digest(args.manifest.read_bytes()), "source_sha256": {
            name: digest(path.read_bytes()) for name, path in (("driver", Path(__file__)),
                ("state", ROOT / "scripts/ft/state.py"), ("oracle", ROOT / "scripts/ft/oracle.py"),
                ("oracle_fixture", HERE / "test_oracle.py"))}})
    deadline = time.monotonic() + args.total_timeout
    results = []
    for name in selected:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            result = {"case_id": name, "status": "not_executed", "reason": "total harness deadline reached"}
            target = args.output / name
            target.mkdir()
            write_json(target / "result.json", result)
        else:
            result = run_case(lookup[name], args.output / name, min(args.case_timeout, remaining))
        results.append(result)
    coverage = {"declared": 600, "requested": len(results), "executed": sum(r["status"] != "not_executed" for r in results),
                "passed": sum(r["status"] == "passed" for r in results),
                "failed": sum(r["status"] == "failed" for r in results),
                "harness_failure": sum(r["status"] == "harness_failure" for r in results),
                "not_executed": sum(r["status"] == "not_executed" for r in results),
                "pending_interface": manifest["counts"]["pending_interface"], "limitation_only": manifest["counts"]["limitation_only"],
                "lanes": {"state_started": sum(bool(r.get("state_process_lane")) for r in results),
                          "oracle_model_audited": sum(bool(r.get("oracle_model_lane")) for r in results),
                          "confirmed_sigkill": sum("confirmed_sigkill" in r.get("must_reach", {}) for r in results),
                          "owner_race_observed": sum("one_winner_one_rejection" in r.get("must_reach", {}) for r in results)},
                "case_results": [{"case_id": r["case_id"], "status": r["status"]} for r in results]}
    write_json(args.output / "coverage.json", coverage)
    print(json.dumps(coverage, sort_keys=True))
    if coverage["harness_failure"] or coverage["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
