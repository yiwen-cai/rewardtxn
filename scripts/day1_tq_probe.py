#!/usr/bin/env python3
"""
Day 1 TransferQueue 最小实例: 模拟 Group->Queue->Learner 正常流转 (Day 3 Q0/Q1 的对照基线)

语义 (对齐 Day 1 训练): 每步 = 4 组 (U=4) x 8 轨迹 (K=8) = 32 样本
  producer: 按组写入 32 样本 batch (带 trajectory_id)
  consumer: get_meta -> get_data -> 完整性校验 -> clear_samples (消费完成)
验证: 数据完整性 / 消费状态 / 流转吞吐
输出: runs/{exp_id}/tq_probe.json
用法: .venv-tq/bin/python scripts/day1_tq_probe.py [exp_id]
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAY_DEDUP_LOGS", "0")

import ray  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

import transfer_queue as tq  # noqa: E402

BASE = Path("/public/home/caiyiwen/rewardtxn")
EXP_ID = sys.argv[1] if len(sys.argv) > 1 else "p1-slime-B0-K8-s42-20260824"
RUN_DIR = BASE / "runs" / EXP_ID
K, U = 8, 4
N_GROUPS = 8  # 模拟 2 个 step 的组流转

def main():
    ray.init(namespace="rtx-day1-tq", ignore_reinit_error=True)
    tq.init()

    client = tq.get_client()
    partition = "day1_partition"
    results = {"exp_id": EXP_ID, "groups": [], "n_samples_total": 0, "ok": True}

    t0 = time.time()
    for g in range(N_GROUPS):
        # ---- producer: 一组 = K 条轨迹 x 4 个特征字段 ----
        n = K
        traj_ids = torch.arange(g * n, (g + 1) * n, dtype=torch.int64)
        data = TensorDict(
            {
                "trajectory_id": traj_ids,
                "input_ids": torch.randint(0, 100, (n, 16)),
                "attention_mask": torch.ones(n, 16),
                "reward": torch.rand(n),
            },
            batch_size=n,
        )
        client.put(data=data, partition_id=partition)

        # ---- consumer: get_meta -> get_data -> 校验 -> clear ----
        meta = client.get_meta(
            data_fields=["trajectory_id", "input_ids", "attention_mask", "reward"],
            batch_size=n,
            partition_id=partition,
            task_name=f"day1_task_g{g}",
        )
        got = client.get_data(meta)
        assert len(meta.global_indexes) == n, f"组 {g}: 元数据样本数不对"
        got_ids = torch.cat([t.view(-1) for t in got["trajectory_id"]])
        assert torch.equal(got_ids, traj_ids), f"组 {g}: TrajectoryID 不一致"
        for k in ["input_ids", "attention_mask", "reward"]:
            assert len(got[k]) == n, f"组 {g}: 字段 {k} 样本数不对"
        client.clear_samples(meta)  # 消费完成
        results["groups"].append({"group": g, "K": n, "consumed": True, "verified": True})
        results["n_samples_total"] += n

    elapsed = time.time() - t0
    results["elapsed_s"] = round(elapsed, 3)
    results["throughput_groups_per_s"] = round(N_GROUPS / elapsed, 3)
    results["throughput_samples_per_s"] = round(results["n_samples_total"] / elapsed, 3)

    # 消费状态复核 (polling mode 下状态可能异步, 仅作参考)
    try:
        status = client.get_consumption_status(partition_id=partition)
        results["consumption_status"] = status
    except Exception as e:  # noqa: BLE001
        results["consumption_status_error"] = str(e)

    out = RUN_DIR / "tq_probe.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"TransferQueue probe: {results['n_samples_total']} samples / {N_GROUPS} groups OK")
    print(f"elapsed {elapsed:.2f}s, {results['throughput_samples_per_s']} samples/s")
    print(f"written: {out}")
    client.close()
    ray.shutdown()

if __name__ == "__main__":
    main()
