#!/usr/bin/env python3
"""资源健康门禁 (Resource Gate)

启动门禁 (check): 实验开始前检查存储与显存, 不健康则拒绝启动 (exit 2)。
运行监控 (watch): 训练期间周期采样内存/磁盘/显存, 写 JSONL 健康曲线。

阈值 (环境变量, 单位 GB):
  RTX_GATE_MIN_PUB_GB     /public 可用空间下限 (默认 200)
  RTX_GATE_MIN_ROOT_GB    / 根盘可用空间下限 (默认 100; Ray spill/scratch 在根盘)
  RTX_GATE_MIN_GPU_FREE_GB 目标 GPU 单卡空闲显存下限 (默认 30)
  RTX_GATE_MAX_GPU_UTIL   目标 GPU 利用率上限百分比 (默认 10)
  RTX_GATE_ALLOW_COMPUTE_PROCESSES 是否允许目标卡已有 compute 进程 (默认 0)
  RTX_GATE_INTERVAL       watch 采样间隔秒 (默认 60)

用法:
  resource_gate.py check --gpus '<gpus>' [--out <json>]
  resource_gate.py watch --dir <logdir> [--interval 60] [--gpus '<gpus>']
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


def _gb(kib_or_none) -> Optional[float]:
    """nvidia-smi MiB / df GiB 数值统一为 GiB。"""
    if kib_or_none is None:
        return None
    return round(float(kib_or_none) / 1024.0, 1)


def _nvidia_gpus() -> List[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip() or "nvidia-smi query failed")
        rows = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            rows.append({
                "index": int(parts[0]),
                "uuid": parts[1],
                "memory_total_gb": _gb(float(parts[2])),
                "memory_used_gb": _gb(float(parts[3])),
                "memory_free_gb": _gb(float(parts[4])),
                "utilization_gpu": float(parts[5]),
            })
        return rows
    except Exception as exc:  # pragma: no cover - nvidia-smi 不可用时
        return [{"error": str(exc)}]


def _nvidia_compute_processes() -> List[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        # nvidia-smi returns an empty successful response when no process exists.
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip() or "compute process query failed")
        rows = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",", 3)]
            if len(parts) != 4:
                continue
            rows.append({
                "gpu_uuid": parts[0], "pid": int(parts[1]),
                "process_name": parts[2], "used_gpu_memory_mib": float(parts[3]),
            })
        return rows
    except Exception as exc:  # pragma: no cover - nvidia-smi 不可用时
        return [{"error": str(exc)}]


def _parse_gpus(gpus_arg: str) -> List[int]:
    """解析 RTX_GPUS 形式: '"device=1,2,3,4"' / 'device=1,2,3' / 'all' / '1,2'。"""
    raw = gpus_arg.strip().strip('"').strip("'")
    if raw in ("", "all"):
        return [g["index"] for g in _nvidia_gpus() if "index" in g]
    if raw.startswith("device="):
        raw = raw[len("device="):]
    parts = [part.strip() for part in raw.split(",")]
    if not parts or any(not part.isdigit() for part in parts):
        return []
    return [int(part) for part in parts]


def _disk_avail_gb(mount: str) -> Optional[float]:
    try:
        out = subprocess.run(["df", "-BG", mount], capture_output=True, text=True, timeout=15)
        line = out.stdout.strip().splitlines()[1]
        # Filesystem 1G-blocks Used Available Use% Mounted on
        return float(line.split()[3].rstrip("G"))
    except Exception:
        return None


def _read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _cpu_numa_provenance() -> dict:
    model = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.partition(":")[2].strip()
                break
    except OSError:
        pass
    numa = []
    for node in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        numa.append({"node": node.name, "cpulist": _read_text(str(node / "cpulist"))})
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return {
        "model": model,
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": affinity,
        "numa_nodes": numa,
        "numa_balancing": _read_text("/proc/sys/kernel/numa_balancing"),
    }


def _io_provenance(path: str) -> dict:
    record = {"path": path, "available_gb": _disk_avail_gb(path)}
    try:
        out = subprocess.run(
            ["findmnt", "--json", "--target", path, "--output", "SOURCE,TARGET,FSTYPE,OPTIONS"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            filesystems = json.loads(out.stdout).get("filesystems", [])
            if filesystems:
                record.update(filesystems[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return record


def _system_provenance(scratch_path: str) -> dict:
    paths = []
    for candidate in ("/public", "/", scratch_path):
        if candidate and candidate not in paths:
            paths.append(candidate)
    return {"cpu_numa": _cpu_numa_provenance(), "io": [_io_provenance(p) for p in paths]}


def _gate_check(gpus_arg: str, scratch_path: str = "/tmp") -> Tuple[dict, bool]:
    min_pub = int(os.environ.get("RTX_GATE_MIN_PUB_GB", "200"))
    min_root = int(os.environ.get("RTX_GATE_MIN_ROOT_GB", "100"))
    min_gpu_free = int(os.environ.get("RTX_GATE_MIN_GPU_FREE_GB", "30"))
    max_gpu_util = float(os.environ.get("RTX_GATE_MAX_GPU_UTIL", "10"))
    allow_compute = os.environ.get("RTX_GATE_ALLOW_COMPUTE_PROCESSES", "0") == "1"

    pub = _disk_avail_gb("/public")
    root = _disk_avail_gb("/")
    gpus = _nvidia_gpus()
    processes = _nvidia_compute_processes()
    targets = _parse_gpus(gpus_arg)
    target_rows = [g for g in gpus if g.get("index") in targets]
    target_uuids = {g.get("uuid") for g in target_rows}

    checks = {
        "storage_public": {
            "avail_gb": pub, "min_gb": min_pub,
            "pass": pub is not None and pub >= min_pub,
        },
        "storage_root": {
            "avail_gb": root, "min_gb": min_root,
            "pass": root is not None and root >= min_root,
        },
    }
    gpu_detail = []
    memory_ok = (
        bool(targets) and len(targets) == len(set(targets))
        and len(target_rows) == len(targets)
    )
    utilization_ok = memory_ok
    if memory_ok:
        for g in target_rows:
            free = g.get("memory_free_gb")
            utilization = g.get("utilization_gpu")
            memory_pass = free is not None and free >= min_gpu_free
            utilization_pass = utilization is not None and utilization <= max_gpu_util
            memory_ok = memory_ok and memory_pass
            utilization_ok = utilization_ok and utilization_pass
            gpu_detail.append({
                "index": g["index"], "uuid": g.get("uuid"), "free_gb": free,
                "used_gb": g.get("memory_used_gb"), "utilization": utilization,
                "memory_pass": memory_pass, "utilization_pass": utilization_pass,
            })
    else:
        gpu_detail.append({"error": "target GPUs missing, unknown, or duplicated: {!r}".format(gpus_arg)})
    target_processes = [p for p in processes if p.get("gpu_uuid") in target_uuids]
    process_query_ok = not any("error" in p for p in processes)
    process_ok = process_query_ok and (allow_compute or not target_processes)
    checks["gpu_memory"] = {"min_free_gb": min_gpu_free, "gpus": gpu_detail, "pass": memory_ok}
    checks["gpu_utilization"] = {"max_percent": max_gpu_util, "gpus": gpu_detail, "pass": utilization_ok}
    checks["gpu_compute_processes"] = {
        "allow_existing": allow_compute, "processes": target_processes,
        "query_errors": [p for p in processes if "error" in p], "pass": process_ok,
    }

    ok = all(c["pass"] for c in checks.values())
    result = {
        "ts": time.time(),
        "pass": ok,
        "thresholds": {
            "min_pub_gb": min_pub, "min_root_gb": min_root,
            "min_gpu_free_gb": min_gpu_free, "max_gpu_util_percent": max_gpu_util,
            "allow_compute_processes": allow_compute,
        },
        "checks": checks,
        "provenance": _system_provenance(scratch_path),
        "note": "所有检查通过才允许启动实验; RTX_SKIP_GATE=1 可跳过(不推荐)",
    }
    return result, ok


def _watch(logdir: str, interval: int, gpus_arg: str, scratch_path: str) -> None:
    Path(logdir).mkdir(parents=True, exist_ok=True)
    out_path = Path(logdir) / "resource_health.jsonl"
    targets = _parse_gpus(gpus_arg)
    print(f"[resource-gate] watch -> {out_path} interval={interval}s", flush=True)
    while True:
        try:
            with open("/proc/meminfo") as f:
                mem = {}
                for line in f:
                    k, _, v = line.partition(":")
                    mem[k] = int(v.strip().split()[0])  # kB
            rec = {
                "ts": time.time(),
                "mem_total_gb": round(mem["MemTotal"] / 1024 / 1024, 1),
                "mem_used_gb": round((mem["MemTotal"] - mem["MemAvailable"]) / 1024 / 1024, 1),
                "mem_avail_gb": round(mem["MemAvailable"] / 1024 / 1024, 1),
                "public_avail_gb": _disk_avail_gb("/public"),
                "root_avail_gb": _disk_avail_gb("/"),
                "gpus": [g for g in _nvidia_gpus() if g.get("index") in targets],
                "compute_processes": _nvidia_compute_processes(),
                "provenance": _system_provenance(scratch_path),
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as exc:  # pragma: no cover - 采样失败不中断
            print(f"[resource-gate] watch error: {exc}", flush=True)
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="启动前门禁检查")
    p_check.add_argument("--gpus", default=os.environ.get("RTX_GPUS", ""))
    p_check.add_argument("--out", default=None, help="结果写入的 JSON 路径")
    p_check.add_argument("--scratch-path", default=os.environ.get("RTX_LOCAL_SCRATCH", "/tmp"))

    p_watch = sub.add_parser("watch", help="训练期间健康采样")
    p_watch.add_argument("--dir", required=True)
    p_watch.add_argument("--interval", type=int, default=int(os.environ.get("RTX_GATE_INTERVAL", "60")))
    p_watch.add_argument("--gpus", default=os.environ.get("RTX_GPUS", ""))
    p_watch.add_argument("--scratch-path", default=os.environ.get("RTX_LOCAL_SCRATCH", "/tmp"))

    args = parser.parse_args()
    if args.command == "check":
        result, ok = _gate_check(args.gpus, args.scratch_path)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps(result, ensure_ascii=False))
        return 0 if ok else 2
    _watch(args.dir, args.interval, args.gpus, args.scratch_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
