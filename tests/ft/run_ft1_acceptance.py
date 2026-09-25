"""Post-container FT1 acceptance: check_ft1_fault → chain → input → load → finalize.

Writes new artifacts only. Never mutates events.jsonl or exitcode.
On missing prerequisites, records acceptance-status.json and stops without inventing FV.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VERIFY_TIMEOUT_SECONDS = 900
IMAGE = "sha256:c0573bb8412a8db753d44b02c9e04504d2392c54f019707f58acf26dd68d2469"


def write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_host(script: str, *args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{REPO / 'third_party' / 'areal'}:{REPO / 'tests' / 'ft'}"
    return subprocess.run(
        [sys.executable, str(REPO / script), *args],
        cwd=str(REPO),
        env=env,
        text=True,
        capture_output=True,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fresh_dir(path: Path) -> Path:
    if not path.exists():
        return path
    return path.with_name(f"{path.name}-rerun-{time.strftime('%Y%m%d-%H%M%S')}")


def docker_verify(kind: str, evidence: Path, out: Path, arm: str, gpu_uuid: str | None) -> dict:
    out.mkdir(parents=True)
    script_src = REPO / "tests" / "ft" / (
        "check_ft1_input_audit.py" if kind == "input" else "check_ft1_load.py"
    )
    verify = out / "verify.py"
    shutil.copyfile(script_src, verify)
    write(
        out / "provenance.json",
        {
            "input": evidence.name,
            "mode": kind,
            "arm": arm,
            "script_sha256": sha256(script_src),
            "generated_by": "tests/ft/run_ft1_acceptance.py",
            "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        },
    )
    args = [
        "docker",
        "create",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user",
        "1028:1029",
        "--cpus=4",
        "--memory=8g" if kind == "input" else "--memory=32g",
        "--pids-limit=256",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=256m",
        "--workdir",
        "/workspace",
        "--mount",
        f"type=bind,src={REPO},dst=/workspace,readonly",
        "--mount",
        f"type=bind,src={out},dst=/output",
        "--mount",
        f"type=bind,src={evidence},dst=/input,readonly",
        "--env",
        "PYTHONPATH=/workspace:/workspace/third_party/areal:/workspace/tests/ft",
        "--env",
        "HOME=/tmp",
        "--env",
        "USER=cpu",
        "--env",
        "LOGNAME=cpu",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "FT_RLVR_TOKENIZER=/workspace/models/Qwen2.5-0.5B-Instruct",
        "--env",
        "FT_RLVR_REPLAY_CPU=1",
        "--env",
        "FT_BATCH_IDENTITY_CPU=1",
    ]
    cmd = ["/output/verify.py", "/input", "/output"]
    if kind == "load":
        if not gpu_uuid:
            raise RuntimeError("load acceptance requires an idle GPU UUID")
        args.extend(
            [
                "--gpus",
                f"device={gpu_uuid}",
                "--shm-size=4g",
                "--env",
                f"FT1_GPU_UUID={gpu_uuid}",
                "--env",
                "CUDA_HOME=/usr/local/cuda",
                "--env",
                "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
                "--env",
                "CUDA_VISIBLE_DEVICES=0",
                "--env",
                "WORLD_SIZE=1",
                "--env",
                "RANK=0",
                "--env",
                "LOCAL_RANK=0",
                "--env",
                "MASTER_ADDR=127.0.0.1",
                "--env",
                "MASTER_PORT=29631",
                "--env",
                "AREAL_SPMD_MODE=1",
                "--env",
                "CUDA_DEVICE_MAX_CONNECTIONS=1",
                "--env",
                "USER=caiyiwen",
                "--env",
                "LOGNAME=caiyiwen",
                "--env",
                "AREAL_CACHE_DIR=/tmp/areal",
                "--env",
                "OMP_NUM_THREADS=4",
            ]
        )
        cmd = ["/output/verify.py", "--input", "/input", "--output", "/output", "--arm", arm]
    args.extend(["--entrypoint", "/opt/.venv/bin/python", IMAGE, *cmd])
    write(out / "launch.json", args)
    cid = subprocess.check_output(args, text=True).strip()
    (out / "container.id").write_text(cid + "\n")
    subprocess.run(["docker", "start", cid], check=True, capture_output=True)
    try:
        exitcode = subprocess.check_output(
            ["docker", "wait", cid], text=True, timeout=VERIFY_TIMEOUT_SECONDS
        ).strip()
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
        (out / "exitcode").write_text("timeout\n")
        raise RuntimeError(f"{kind} verify exceeded {VERIFY_TIMEOUT_SECONDS}s; container removed")
    (out / "exitcode").write_text(exitcode + "\n")
    logs = subprocess.run(["docker", "logs", cid], text=True, capture_output=True)
    (out / "verify.log").write_text(logs.stdout + logs.stderr)
    subprocess.run(["docker", "rm", cid], check=True, capture_output=True)
    return {"kind": kind, "exitcode": int(exitcode), "output": str(out)}


def accept(evidence: Path, *, gpu_uuid: str | None = None, run_docker: bool = True) -> dict:
    evidence = evidence.resolve()
    case = json.loads((evidence / "ft1-case.json").read_text())
    arm = case["arm"]
    status = {
        "evidence": str(evidence),
        "arm": arm,
        "scenario": case["scenario"],
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "immutable": ["events.jsonl", "exitcode"],
        "steps": {},
    }

    no_fault = case["scenario"] == "no_fault"
    if no_fault:
        from check_ft1_smoke import verify as verify_smoke
        try:
            smoke = verify_smoke(evidence)
            write(evidence / "smoke-acceptance.json", smoke)
            status["steps"]["smoke"] = smoke
        except Exception as exc:  # noqa: BLE001
            status["result"] = "blocked_at_smoke"
            status["blocker"] = str(exc)
            write(evidence / "acceptance-status.json", status)
            return status

    else:
        fault_out = evidence / "fault-verification.json"
        proc = run_host("tests/ft/check_ft1_fault.py", str(evidence), str(fault_out))
        status["steps"]["check_ft1_fault"] = {
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-500:],
            "stderr_tail": proc.stderr[-500:],
            "path": str(fault_out) if fault_out.exists() else None,
        }
        if proc.returncode != 0 or not fault_out.exists():
            status["result"] = "blocked_at_fault_check"
            write(evidence / "acceptance-status.json", status)
            return status

        fault = json.loads(fault_out.read_text())
        if not fault.get("valid_hit"):
            status["result"] = "fault_not_valid_hit"
            write(evidence / "acceptance-status.json", status)
            return status

    if not (evidence / "final-native-state.json").exists():
        status["result"] = "blocked_missing_final_native_state"
        status["blocker"] = (
            "final-native-state.json missing; typically native recover failed after fault. "
            "Cannot run chain/load/finalize without inventing terminal state."
        )
        write(evidence / "acceptance-status.json", status)
        return status

    chain_out = evidence / "chain-verification.json"
    proc = run_host("tests/ft/check_ft1_chain.py", str(evidence), str(chain_out))
    status["steps"]["check_ft1_chain"] = {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-500:],
        "stderr_tail": proc.stderr[-500:],
        "path": str(chain_out) if chain_out.exists() else None,
    }
    if proc.returncode != 0 or not chain_out.exists():
        status["result"] = "blocked_at_chain"
        write(evidence / "acceptance-status.json", status)
        return status

    if not run_docker:
        status["result"] = "host_checks_ok_docker_skipped"
        write(evidence / "acceptance-status.json", status)
        return status

    # Never reuse an earlier attempt's output dir (stale draw-copy made load fail).
    input_dir = fresh_dir(evidence.parent / f"ft1-input-{evidence.name}")
    load_dir = fresh_dir(evidence.parent / f"ft1-load-{evidence.name}")
    try:
        status["steps"]["input"] = docker_verify("input", evidence, input_dir, arm, None)
    except Exception as exc:  # noqa: BLE001
        status["steps"]["input"] = {"error": str(exc)}
        status["result"] = "blocked_at_input"
        write(evidence / "acceptance-status.json", status)
        return status
    if status["steps"]["input"]["exitcode"] != 0:
        status["result"] = "blocked_at_input"
        write(evidence / "acceptance-status.json", status)
        return status

    try:
        status["steps"]["load"] = docker_verify("load", evidence, load_dir, arm, gpu_uuid)
    except Exception as exc:  # noqa: BLE001
        status["steps"]["load"] = {"error": str(exc)}
        status["result"] = "blocked_at_load"
        write(evidence / "acceptance-status.json", status)
        return status
    if status["steps"]["load"]["exitcode"] != 0:
        status["result"] = "blocked_at_load"
        write(evidence / "acceptance-status.json", status)
        return status

    if no_fault:
        write(evidence / "functional-verification.json", {
            "classification": "no_fault_verified", "arm": arm,
            "safety_verified_for_retained_chain": True,
            "full_native_reload_verified": True,
            "scope": "ten-step no-fault input, chain and independent full-state load",
        })
        status["result"] = "functional_verification_written"
        status["finished_local"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        write(evidence / "acceptance-status.json", status)
        return status

    proc = run_host(
        "tests/ft/finalize_ft1_fault.py",
        str(evidence),
        str(input_dir),
        str(load_dir),
    )
    fv = evidence / "functional-verification.json"
    status["steps"]["finalize"] = {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-500:],
        "stderr_tail": proc.stderr[-500:],
        "path": str(fv) if fv.exists() else None,
    }
    status["result"] = (
        "functional_verification_written"
        if fv.exists() and proc.returncode == 0
        else "blocked_at_finalize"
    )
    status["finished_local"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    write(evidence / "acceptance-status.json", status)
    return status


def main(argv: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence")
    parser.add_argument("--gpu-uuid", default=None)
    parser.add_argument("--skip-docker", action="store_true")
    args = parser.parse_args(argv)
    report = accept(Path(args.evidence), gpu_uuid=args.gpu_uuid, run_docker=not args.skip_docker)
    print(json.dumps(report, indent=2))
    if report.get("result") != "functional_verification_written":
        raise SystemExit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
