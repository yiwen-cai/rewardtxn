#!/usr/bin/env python3
"""
Day 3: TransferQueue Q0/Q1 崩溃窗口实证 (文档 7 节 Day 3)

场景 A (Q0): consumer 调 get_meta (fetch 模式 -> controller 立即 mark_consumed)
             -> 未取数据即被 SIGKILL (模拟 Learner 崩溃)
             -> 重启 consumer 尝试重新 get_meta 同一批 -> 观察是否静默丢失
场景 B (Q1 对照): 完整消费 get_meta -> get_data -> clear_samples 后的状态
场景 C (B3 对照): 重复投递 (同一批 put 两次) 的去重行为 vs 场景 A 的丢失

判定指标: "队列显示已消费, 但 Optimizer 从未执行" 的静默丢失是否存在
用法: .venv-tq/bin/python scripts/day3_tq_crash_probe.py [exp_id]
产物: runs/{exp_id}/tq_crash_probe.json
"""
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import ray

os.environ.setdefault("RAY_DEDUP_LOGS", "0")

BASE = Path("/public/home/caiyiwen/rewardtxn")
EXP_ID = sys.argv[1] if len(sys.argv) > 1 else "p1-tq-Q0Q1-s42-20260825"
RUN_DIR = BASE / "runs" / EXP_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

PARTITION = "day3_partition"
TASK = "learner"
N_SAMPLES = 32  # 模拟一个 global batch

CONSUMER_CODE = textwrap.dedent("""
    import json, os, sys
    os.environ.setdefault("RAY_DEDUP_LOGS", "0")
    import ray
    import torch
    from tensordict import TensorDict
    import transfer_queue as tq

    mode = sys.argv[1]          # meta_only | full_consume | retry
    task_name = sys.argv[2]
    partition = sys.argv[3]
    batch = int(sys.argv[4])
    kill_after_meta = sys.argv[5] == "kill"

    ray.init(address="auto", namespace="rtx-day3", ignore_reinit_error=True)
    tq.init()
    client = tq.get_client()
    meta = client.get_meta(
        data_fields=["trajectory_id", "input_ids", "attention_mask", "reward"],
        batch_size=batch, partition_id=partition, task_name=task_name,
    )
    idxs = list(meta.global_indexes) if meta.size > 0 else []
    print(json.dumps({"mode": mode, "got_indexes": idxs, "n": len(idxs)}, ensure_ascii=False))
    sys.stdout.flush()
    if kill_after_meta:
        os.kill(os.getpid(), 9)  # 模拟 Learner 崩溃: get_meta 后未取数据即挂
    if mode == "full_consume":
        got = client.get_data(meta)
        client.clear_samples(meta)
        print(json.dumps({"mode": mode, "consumed": True, "n": len(idxs)}, ensure_ascii=False))
    client.close()
    ray.shutdown()
""")


def run_consumer(*args):
    r = subprocess.run(
        [sys.executable, "-c", CONSUMER_CODE, *args],
        capture_output=True, text=True, timeout=120,
    )
    out = [l for l in r.stdout.strip().splitlines() if l.startswith("{")]
    return json.loads(out[-1]) if out else {"error": r.stderr[-500:]}


def main():
    ray.init(namespace="rtx-day3", ignore_reinit_error=True)
    tq.init()
    client = tq.get_client()

    results = {"exp_id": EXP_ID, "partition": PARTITION, "batch_size": N_SAMPLES, "scenarios": {}}

    # ---------- producer: 写一批数据 (模拟一个 step 的组 batch) ----------
    traj_ids = torch.arange(N_SAMPLES, dtype=torch.int64)
    data = TensorDict({
        "trajectory_id": traj_ids,
        "input_ids": torch.randint(0, 100, (N_SAMPLES, 16)),
        "attention_mask": torch.ones(N_SAMPLES, 16),
        "reward": torch.rand(N_SAMPLES),
    }, batch_size=N_SAMPLES)
    client.put(data=data, partition_id=PARTITION)
    results["produced"] = N_SAMPLES

    # ---------- 场景 A: Q0 - get_meta 后崩溃, 重启后重取 ----------
    a1 = run_consumer("meta_only", TASK, PARTITION, str(N_SAMPLES), "kill")
    time.sleep(3)  # 等被 kill 的进程完全退出
    a2 = run_consumer("retry", TASK, PARTITION, str(N_SAMPLES), "no-kill")
    results["scenarios"]["A_Q0_getmeta_crash"] = {
        "first_get_meta": a1,
        "restart_get_meta": a2,
        "silent_loss": (len(a1.get("got_indexes", [])) > 0 and len(a2.get("got_indexes", [])) == 0),
        "verdict": ("Q0 静默丢失确认: get_meta 标记消费后崩溃, 重启无法重取该批数据"
                    if len(a1.get("got_indexes", [])) > 0 and len(a2.get("got_indexes", [])) == 0
                    else "未观察到丢失"),
    }

    # ---------- 场景 B: 正常完整消费对照 ----------
    client.put(data=data, partition_id=PARTITION)
    b1 = run_consumer("full_consume", "learner_b", PARTITION, str(N_SAMPLES), "no-kill")
    b2 = run_consumer("retry", "learner_b", PARTITION, str(N_SAMPLES), "no-kill")
    results["scenarios"]["B_normal_consume"] = {
        "full_consume": b1,
        "after_consume_retry": b2,
        "note": "正常消费(clear)后重取为空属预期; 崩溃场景 A 与此的关键区别是 clear 未执行",
    }

    # ---------- 场景 C: B3 对照 - 重复投递去重 ----------
    client.put(data=data, partition_id=PARTITION)
    c1 = run_consumer("meta_only", "learner_c", PARTITION, str(N_SAMPLES), "no-kill")
    # 同批重复 put (模拟 at-least-once 重投)
    client.put(data=data, partition_id=PARTITION)
    c2 = run_consumer("meta_only", "learner_c", PARTITION, str(N_SAMPLES), "no-kill")
    results["scenarios"]["C_dup_delivery"] = {
        "first_get": c1, "after_redelivery_get": c2,
        "note": "B3 幂等去重语义: 按 task 消费状态去重, 但无法覆盖 '已标记消费未训练' 窗口 (场景 A)",
    }

    out = RUN_DIR / "tq_crash_probe.json"
    json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    client.close()
    ray.shutdown()


if __name__ == "__main__":
    import torch  # noqa: E402
    import transfer_queue as tq  # noqa: E402
    from tensordict import TensorDict  # noqa: E402
    main()
