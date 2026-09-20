"""Real AsyncRewardWrapper CPU pool probe; injection is before official scoring.

No pool/retry implementation is replaced. This is not a rollout, F4, or training
recovery probe. Observation files are never read to decide a reward or retry.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time
import uuid

from .descendants import Client, snapshot
from .namespace_run import atomic_json


EVENT_ID = "reward-entry-0"
EVIDENCE = {"boundary": "official_gsm8k_before_call"}


def record(stream, kind, **fields):
    stream.write(json.dumps(dict(kind=kind, monotonic_ns=time.monotonic_ns(), **fields), sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def controlled_gsm8k_reward(prompt, completions, prompt_ids, completion_ids, answer,
                            injection_event=None):
    """Pickleable callable actually invoked by the unmodified official pool."""
    from areal.reward.gsm8k import gsm8k_reward_fn

    invocation = uuid.uuid4().hex
    directory = Path(os.environ["FT_CONTROL_SOCKET"]).parent
    with (directory / f"reward-{os.getpid()}-{invocation}.jsonl").open("x") as stream:
        record(stream, "pool_callable_entered", invocation=invocation, identity=snapshot(os.getpid()))
        client = Client("reward", event_id=injection_event)
        try:
            record(stream, "control_registered", incarnation=client.incarnation, injection=client.injection)
            if client.injection is not None and client.injection["status"] == "pending":
                # The killed worker has entered the actual pool callable, but has
                # not executed gsm8k_reward_fn. Do not label this scoring midway.
                record(stream, "injection_boundary", event_id=injection_event, evidence=EVIDENCE)
                client.ready(injection_event, EVIDENCE)
                client.wait_release(injection_event)
            record(stream, "official_score_start")
            score = gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer=answer)
            record(stream, "official_score_returned", score=score)
            return score
        finally:
            client.close()


def memory_observation(directory):
    """Report per-process rusage and only a proven private container cgroup peak."""
    result = {"process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "reaped_children_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
              "cgroup_peak_bytes": None, "cgroup_version": None, "cgroup_scope": None}
    try:
        info = json.loads((directory / "inspect_created.json").read_text())
        if info["HostConfig"].get("CgroupnsMode") != "private":
            result["reason"] = "private cgroup namespace not established by host inspect"
            return result
        membership = [line.split(":", 2) for line in Path("/proc/self/cgroup").read_text().splitlines()]
        version, fs_name, filename = None, None, None
        if ["0", "", "/"] in membership:
            version, fs_name, filename = 2, "cgroup2", "memory.peak"
        elif any("memory" in controllers.split(",") and path == "/" for _, controllers, path in membership):
            version, fs_name, filename = 1, "cgroup", "memory.max_usage_in_bytes"
        if version is None:
            result["reason"] = "process is not at the proven private cgroup root"
            return result
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            before, after = line.split(" - ", 1)
            fields, fs = before.split(), after.split()
            if fs[0] == fs_name and fields[3] == "/" and (version == 2 or "memory" in fs[2].split(",")):
                peak = Path(fields[4]) / filename
                result.update(cgroup_peak_bytes=int(peak.read_text()), cgroup_version=version,
                              cgroup_scope="private container cgroup root", cgroup_peak_path=str(peak))
                return result
        result["reason"] = "matching private memory cgroup mount unavailable"
    except (OSError, ValueError, KeyError, IndexError) as exc:
        result["reason"] = f"container-scoped peak unavailable: {type(exc).__name__}"
    return result


def provenance():
    import areal.api.reward_api as reward_api
    import areal.reward as reward_utils
    import areal.reward.gsm8k as gsm8k
    from . import descendants, namespace_run

    files = [Path(inspect.getsourcefile(module)) for module in
             (reward_api, reward_utils, gsm8k, descendants, namespace_run)]
    files.append(Path(__file__))
    versions = {}
    for package in ("math-verify", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
            "python": sys.version, "multiprocessing_start_method": multiprocessing.get_start_method(),
            "package_versions": versions}


async def probe(inject, max_retries):
    from areal.api.reward_api import AsyncRewardWrapper
    # Import by stable project module name even when this entrypoint is __main__.
    from scripts.ft.reward_pool_probe import controlled_gsm8k_reward

    wrapper = AsyncRewardWrapper(controlled_gsm8k_reward, max_workers=1,
                                 max_retries=max_retries, timeout_seconds=15)
    return await wrapper("What is 2 + 2?", r"The answer is \boxed{4}.", [], [], answer="4",
                         injection_event=EVENT_ID if inject else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault", action="store_true")
    parser.add_argument("--max-retries", type=int, choices=range(4), default=1)
    args = parser.parse_args()
    directory = Path(os.environ["FT_CONTROL_SOCKET"]).parent
    result = {"scope": "real CPU reward pool only; not F4 or training recovery", "trainer_identity": snapshot(os.getpid()),
              "fault_requested": args.fault, "max_workers": 1, "max_retries": args.max_retries,
              "timeout_seconds": 15, "expected_score": 1.0, "input": {"prompt": "What is 2 + 2?", "completion": r"The answer is \boxed{4}.", "answer": "4"}}
    atomic_json(directory / "probe_started.json", result)
    try:
        result.update(provenance())
        result.update(status="returned", score=asyncio.run(probe(args.fault, args.max_retries)))
    except BaseException as exc:
        result.update(status="raised", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result["memory"] = memory_observation(directory)
        atomic_json(directory / "probe_result.json", result)
    if result["score"] != result["expected_score"]:
        raise RuntimeError(f"official score differs from fixed arithmetic reference: {result['score']}")


if __name__ == "__main__":
    main()
