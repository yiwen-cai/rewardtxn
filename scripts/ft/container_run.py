"""Closed CPU/four-GPU Docker supervisor; mutations use saved full resource IDs."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

from .namespace_run import atomic_json, read_config


def docker(argv, timeout=10):
    result = subprocess.run(["docker", *argv], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"docker {argv[0]} failed ({result.returncode}): {result.stderr.strip()}")
    return (result.stdout + result.stderr if argv[0] == "logs" else result.stdout).strip()


def inspect_owned(cid, nonce):
    if not re.fullmatch(r"[0-9a-f]{64}", cid):
        raise RuntimeError("full container ID required")
    info = json.loads(docker(["inspect", cid]))[0]
    if info["Id"] != cid or info["Config"].get("Labels", {}).get("rewardtxn.ft.nonce") != nonce:
        raise RuntimeError("container ownership mismatch")
    return info


def check_containment(info, profile="cpu", gpu_uuids=()):
    from .native_gpu import PROFILE, validate_uuids
    if profile not in ("cpu", PROFILE):
        raise RuntimeError("unknown execution profile")
    requests = info["HostConfig"].get("DeviceRequests") or []
    if profile == "cpu" and requests:
        raise RuntimeError("CPU profile rejects GPU devices")
    if profile == PROFILE:
        validate_uuids(gpu_uuids)
        if (len(requests) != 1 or requests[0].get("Count") not in (0, None)
                or set(requests[0].get("DeviceIDs") or []) != set(gpu_uuids)
                or len(requests[0].get("DeviceIDs") or []) != 4):
            raise RuntimeError("GPU device request mismatch")
    host = info["HostConfig"]
    if (host.get("PidMode", "") != "" or host.get("Privileged") or host.get("Init")
            or host.get("Devices")
            or not host.get("ReadonlyRootfs") or host.get("RestartPolicy", {}).get("Name") not in ("no", "")
            or "ALL" not in host.get("CapDrop", []) or not info["Config"]["User"].split(":")[0].isdigit()
            or int(info["Config"]["User"].split(":")[0]) == 0):
        raise RuntimeError("Docker containment configuration rejected")
    if not any(option in ("no-new-privileges", "no-new-privileges:true") for option in host.get("SecurityOpt", [])):
        raise RuntimeError("no-new-privileges required")


def process_record(pid):
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_time": stat[19], "pid_namespace": os.stat(f"/proc/{pid}/ns/pid").st_ino,
            "cgroup": Path(f"/proc/{pid}/cgroup").read_text()}


def supervise(config_path, source, output, image, python="python", startup_timeout=15, cleanup_timeout=3,
              profile="cpu", gpu_uuids=()):
    """No custom recovery, arbitrary Docker options, or container name lookup."""
    from .native_gpu import PROFILE, validate_uuids, idle_snapshot, provenance, validate_config
    config, digest = read_config(config_path, profile)
    if profile == PROFILE:
        validate_uuids(gpu_uuids)
        validate_config(config)
        if python != "/opt/.venv/bin/python" or config["argv"] != [python, "-m", "scripts.ft.native_gpu"]:
            raise ValueError("GPU profile requires the frozen native bootstrap")
    elif gpu_uuids:
        raise ValueError("CPU profile cannot select GPUs")
    if os.getuid() == 0:
        raise ValueError("run the supervisor as a non-root user")
    source, output = Path(source).resolve(), Path(output).resolve()
    if not (source / "scripts/ft/namespace_run.py").is_file():
        raise ValueError("source directory does not contain the namespace controller")
    if any("," in str(path) or "\n" in str(path) for path in (source, output)):
        raise ValueError("Docker bind mount paths may not contain comma/newline")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, output / "config.json")
    nonce = uuid.uuid4().hex
    host_pidns = os.stat("/proc/self/ns/pid").st_ino
    counter = 0
    atomic_json(output / "host_lease.json", {"nonce": nonce, "counter": counter})
    create = ["create", "--cidfile", str(output / "container.id"), "--label", f"rewardtxn.ft.nonce={nonce}", "--restart=no",
              "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
              "--user", f"{os.getuid()}:{os.getgid()}", "--cpus=2", "--memory=2g", "--pids-limit=128",
              "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m", "--workdir", "/workspace",
              "--mount", f"type=bind,src={source},dst=/workspace,readonly",
              "--mount", f"type=bind,src={output},dst=/output",
              "--env", "PYTHONPATH=/workspace", "--env", "PYTHONDONTWRITEBYTECODE=1",
              "--entrypoint", python, image, "-m", "scripts.ft.namespace_run",
              "--config", "/output/config.json", "--output", "/output", "--nonce", nonce,
              "--host-pidns", str(host_pidns)]
    if profile == PROFILE:
        # Mutable GPU/network arguments are confined to this exact profile.
        for name in ("areal", "name_resolve"):
            (output / name).mkdir()
        create[create.index("--cpus=2")] = "--cpus=32"
        create[create.index("--memory=2g")] = "--memory=128g"
        create[create.index("--pids-limit=128")] = "--pids-limit=4096"
        index = create.index("--tmpfs")
        (output / "tmp").mkdir()
        create[index:index + 2] = ["--mount", f"type=bind,src={output / 'tmp'},dst=/tmp", "--shm-size=16g"]
        index = create.index("--workdir")
        create[index:index] = ["--gpus", '"device=' + ','.join(gpu_uuids) + '"']
        create.extend(["--profile", profile])
        atomic_json(output / "gpu-uuids.json", list(gpu_uuids))
        atomic_json(output / "source-sha256.json", provenance(source))
        shutil.copyfile(source / "docs/experiments/rewardtxn-ft-20260916/native-trainer.yaml", output / "native-trainer.yaml")
    atomic_json(output / "launch.json", {"argv": ["docker", *create], "config_sha256": digest,
                                          "nonce": nonce, "host_pid_namespace": host_pidns})
    cid, init, failure = None, None, None
    network_id = None
    clean = False
    created_at = time.monotonic()
    try:
        if profile == PROFILE:
            idle_snapshot(gpu_uuids, output / "gpu-idle-before-create.json")
            candidate = docker(["network", "create", "--internal", "--label", f"rewardtxn.ft.nonce={nonce}", f"rtx-ft-{nonce}"])
            if not re.fullmatch(r"[0-9a-f]{64}", candidate):
                raise RuntimeError("full network ID required")
            network_id = candidate
            atomic_json(output / "network-id.json", {"id": network_id, "nonce": nonce})
            network = json.loads(docker(["network", "inspect", network_id]))[0]
            if network["Id"] != network_id or not network["Internal"] or network.get("Labels", {}).get("rewardtxn.ft.nonce") != nonce:
                raise RuntimeError("network containment mismatch")
            atomic_json(output / "network-created.json", network)
            create[create.index("--network=none")] = "--network=" + network_id
            atomic_json(output / "launch.json", {"argv": ["docker", *create], "config_sha256": digest, "nonce": nonce, "profile": profile, "host_pid_namespace": host_pidns})
        # Capture exact ID before any start; failed starts remain addressable.
        cid = docker(create)
        if not re.fullmatch(r"[0-9a-f]{64}", cid):
            raise RuntimeError("docker create did not return a full CID")
        if (output / "container.id").read_text().strip() != cid:
            raise RuntimeError("docker CID receipt mismatch")
        info = inspect_owned(cid, nonce)
        check_containment(info, profile, gpu_uuids)
        atomic_json(output / "inspect_created.json", info)
        if profile == PROFILE:
            idle_snapshot(gpu_uuids, output / "gpu-idle-before-start.json")
        docker(["start", cid])
        info = inspect_owned(cid, nonce)
        if info["State"]["Running"]:
            init = process_record(info["State"]["Pid"])
            if init["pid_namespace"] == host_pidns:
                raise RuntimeError("container shares host PID namespace")
            atomic_json(output / "init_identity.json", init)
        atomic_json(output / "inspect_started.json", info)
        last_heartbeat, last_change = None, time.monotonic()
        startup_deadline = time.monotonic() + startup_timeout
        while info["State"]["Running"]:
            now = time.monotonic()
            counter += 1
            atomic_json(output / "host_lease.json", {"nonce": nonce, "counter": counter})
            path = output / "controller_heartbeat.json"
            if path.exists():
                heartbeat = json.loads(path.read_text())
                if heartbeat.get("nonce") != nonce or type(heartbeat.get("counter")) is not int:
                    raise RuntimeError("invalid controller heartbeat")
                value = heartbeat["counter"]
                if value != last_heartbeat:
                    if last_heartbeat is not None and value < last_heartbeat:
                        raise RuntimeError("controller heartbeat regressed")
                    last_heartbeat, last_change = value, now
                elif now - last_change > config["timeouts"]["lease"]:
                    raise RuntimeError("controller heartbeat expired")
            elif now >= startup_deadline:
                raise RuntimeError("controller startup deadline")
            if now - created_at > startup_timeout + config["timeouts"]["run"] + config["timeouts"]["lease"]:
                raise RuntimeError("host absolute run deadline")
            time.sleep(0.05)
            info = inspect_owned(cid, nonce)
    except BaseException as exc:
        failure = str(exc)
    finally:
        cleanup_errors = []
        if cid is None and (output / "container.id").exists():
            cid = (output / "container.id").read_text().strip()
        if cid is not None and re.fullmatch(r"[0-9a-f]{64}", cid):
            try:
                info = inspect_owned(cid, nonce)
                atomic_json(output / "inspect_before_cleanup.json", info)
                if info["State"]["Running"]:
                    try:
                        docker(["stop", "--time", str(max(1, int(cleanup_timeout))), cid], timeout=cleanup_timeout + 3)
                    except (RuntimeError, subprocess.TimeoutExpired) as exc:
                        cleanup_errors.append(str(exc))
                    info = inspect_owned(cid, nonce)
                    if info["State"]["Running"]:
                        docker(["kill", "--signal=KILL", cid], timeout=cleanup_timeout + 3)
                    end = time.monotonic() + cleanup_timeout
                    while info["State"]["Running"] and time.monotonic() < end:
                        info = inspect_owned(cid, nonce)
                        time.sleep(0.05)
                atomic_json(output / "inspect_final.json", info)
                clean = not info["State"]["Running"]
                if failure is None and info["State"].get("ExitCode") != 0:
                    failure = f'controller exited {info["State"].get("ExitCode")}'
                if init is not None:
                    try:
                        clean = clean and process_record(init["pid"]) != init
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                (output / "container.log").write_text(docker(["logs", cid]))
                if not clean:
                    raise RuntimeError("container/init cleanup not confirmed")
                # Removal is explicit-ID only and follows preserved final inspect/logs.
                docker(["rm", cid])
            except BaseException as exc:
                cleanup_errors.append(str(exc))
        if profile == PROFILE and network_id is None:
            try:
                ids = docker(["network", "ls", "--no-trunc", "--filter", f"label=rewardtxn.ft.nonce={nonce}", "--format", "{{.ID}}"]).splitlines()
                if len(ids) == 1 and re.fullmatch(r"[0-9a-f]{64}", ids[0]):
                    network_id = ids[0]
                    atomic_json(output / "network-id.json", {"id": network_id, "nonce": nonce, "recovered_after_create_failure": True})
                elif ids:
                    raise RuntimeError("ambiguous network create receipt")
            except BaseException as exc:
                cleanup_errors.append(str(exc))
        if network_id is not None:
            try:
                network = json.loads(docker(["network", "inspect", network_id]))[0]
                if network["Id"] != network_id or network.get("Labels", {}).get("rewardtxn.ft.nonce") != nonce:
                    raise RuntimeError("network cleanup ownership mismatch")
                atomic_json(output / "network-final.json", network)
                docker(["network", "rm", network_id])
                atomic_json(output / "network-cleanup.json", {"id": network_id, "removed": True})
            except BaseException as exc:
                cleanup_errors.append(str(exc))
        result = {"nonce": nonce, "container_id": cid, "failure": failure,
                  "cleanup_confirmed": clean, "cleanup_errors": cleanup_errors,
                  "duration_seconds": time.monotonic() - created_at,
                  "scope": profile + "; no oracle classification", "network_id": network_id}
        atomic_json(output / "supervisor_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--profile", default="cpu", choices=["cpu", "native-trainer-4gpu"])
    parser.add_argument("--gpu-uuid", action="append", default=[])
    args = parser.parse_args()
    result = supervise(args.config, args.source, args.run_dir, args.image, args.python, profile=args.profile, gpu_uuids=args.gpu_uuid)
    print(json.dumps(result))
    raise SystemExit(0 if result["cleanup_confirmed"] and not result["failure"] and not result["cleanup_errors"] else 2)


if __name__ == "__main__":
    main()
