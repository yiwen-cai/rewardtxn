#!/usr/bin/env python3
"""资源健康门禁 (Resource Gate)

启动门禁 (check): 实验开始前检查存储与显存, 不健康则拒绝启动 (exit 2)。
运行监控 (watch): 训练期间周期采样内存/磁盘/显存, 写 JSONL 健康曲线。

阈值 (环境变量, 单位 GB):
  RTX_GATE_MIN_PUB_GB     /public 可用空间下限 (默认 200)
  RTX_GATE_MIN_ROOT_GB    / 根盘可用空间下限 (默认 100; Ray spill/scratch 在根盘)
  RTX_GATE_MIN_GPU_FREE_GB 目标 GPU 单卡空闲显存下限 (默认 30)
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


def _gb(kib_or_none) -> float | None:
    """nvidia-smi MiB / df GiB 数值统一为 GiB。"""
    if kib_or_none is None:
        return None
    return round(float(kib_or_none) / 1024.0, 1)


def _nvidia_gpus() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        rows = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            rows.append({
                "index": int(parts[0]),
                "memory_total_gb": _gb(float(parts[1])),
                "memory_used_gb": _gb(float(parts[2])),
                "memory_free_gb": _gb(float(parts[3])),
                "utilization_gpu": float(parts[4]),
            })
        return rows
    except Exception as exc:  # pragma: no cover - nvidia-smi 不可用时
        return [{"error": str(exc)}]


def _parse_gpus(gpus_arg: str) -> list[int]:
    """解析 RTX_GPUS 形式: '"device=1,2,3,4"' / 'device=1,2,3' / 'all' / '1,2'。"""
    raw = gpus_arg.strip().strip('"').strip("'")
    if raw in ("", "all"):
        return [g["index"] for g in _nvidia_gpus() if "index" in g]
    if raw.startswith("device="):
        raw = raw[len("device="):]
    idx = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx.append(int(part))
    return idx


def _disk_avail_gb(mount: str) -> float | None:
    try:
        out = subprocess.run(["df", "-BG", mount], capture_output=True, text=True, timeout=15)
        line = out.stdout.strip().splitlines()[1]
        # Filesystem 1G-blocks Used Available Use% Mounted on
        return float(line.split()[3].rstrip("G"))
    except Exception:
        return None


def _gate_check(gpus_arg: str) -> tuple[dict, bool]:
    min_pub = int(os.environ.get("RTX_GATE_MIN_PUB_GB", "200"))
    min_root = int(os.environ.get("RTX_GATE_MIN_ROOT_GB", "100"))
    min_gpu_free = int(os.environ.get("RTX_GATE_MIN_GPU_FREE_GB", "30"))

    pub = _disk_avail_gb("/public")
    root = _disk_avail_gb("/")
    gpus = _nvidia_gpus()
    targets = _parse_gpus(gpus_arg)
    target_rows = [g for g in gpus if g.get("index") in targets]

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
    gpu_ok = True
    if targets:
        for g in target_rows:
            free = g.get("memory_free_gb")
            ok = free is not None and free >= min_gpu_free
            gpu_ok = gpu_ok and ok
            gpu_detail.append({"index": g["index"], "free_gb": free, "used_gb": g.get("memory_used_gb"),
                               "utilization": g.get("utilization_gpu"), "pass": ok})
    else:
        gpu_ok = False
        gpu_detail.append({"error": f"no target gpus parsed from '{gpus_arg}'"})
    checks["gpu_memory"] = {"min_free_gb": min_gpu_free, "gpus": gpu_detail, "pass": gpu_ok}

    ok = all(c["pass"] for c in checks.values())
    result = {
        "ts": time.time(),
        "pass": ok,
        "thresholds": {"min_pub_gb": min_pub, "min_root_gb": min_root, "min_gpu_free_gb": min_gpu_free},
        "checks": checks,
        "note": "所有检查通过才允许启动实验; RTX_SKIP_GATE=1 可跳过(不推荐)",
    }
    return result, ok


def _watch(logdir: str, interval: int, gpus_arg: str) -> None:
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
    p_check.add_argument("--gpus", default=os.environ.get("RTX_GPUS", "device=0,1,2,5"))
    p_check.add_argument("--out", default=None, help="结果写入的 JSON 路径")

    p_watch = sub.add_parser("watch", help="训练期间健康采样")
    p_watch.add_argument("--dir", required=True)
    p_watch.add_argument("--interval", type=int, default=int(os.environ.get("RTX_GATE_INTERVAL", "60")))
    p_watch.add_argument("--gpus", default=os.environ.get("RTX_GPUS", ""))

    args = parser.parse_args()
    if args.command == "check":
        result, ok = _gate_check(args.gpus)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps(result, ensure_ascii=False))
        return 0 if ok else 2
    _watch(args.dir, args.interval, args.gpus)
    return 0


if __name__ == "__main__":
    sys.exit(main())
