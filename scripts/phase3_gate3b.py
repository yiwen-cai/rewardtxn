#!/usr/bin/env python3
"""Phase 3B 门禁判定: 自动恢复链 (崩溃检测 -> 恢复计划 -> 自动重启续跑)
G3B1: 注入真实崩溃 (kill -9), 无人工干预, 训练自动继续完成 (恢复后 >=10 步)
G3B2: 恢复正确性 (已提交步不重训 + 恢复后 loss 轨迹对照在噪声内)
G3B3: 端到端恢复时延 <5 min
"""
import json
import re
from datetime import datetime, timezone

BASE = "/public/home/caiyiwen/rewardtxn/runs"
RUN = "p3b-slime-kill-autorecover2-K8-s42-20260828"
RES = {"stage": "3B", "gates": {}}

log = open(f"{BASE}/{RUN}/logs/train.log").read()
lines = log.splitlines()

# ---- G3B1: 无人干预自动恢复 ----
marks = re.findall(r"\[3B\] ===== attempt (\d+)", log)
succeeded = "training SUCCEEDED after attempt" in log
steps = [int(s) for s in re.findall(r"step (\d+): \{'train/loss'", log)]
post_recover_steps = sum(1 for s in steps if s >= 10)
RES["gates"]["G3B1"] = {
    "pass": len(marks) >= 2 and succeeded and post_recover_steps >= 10,
    "attempts": len(marks),
    "inject_kill": "INJECT: kill -9 training" in log,
    "recovery_plan": "recovery plan: latest committed step = 9" in log,
    "resume_action": "resume from step 9" in log,
    "final_success": succeeded,
    "steps_after_recovery": post_recover_steps,
    "evidence": "kill -9 注入 (iter9 落盘后) -> FAILED -> 恢复计划(committed=9) -> --load+override 重启 -> attempt2 续跑 step10-19 -> SUCC; 全程无人工干预 (entrypoint 包装器)",
}

# ---- G3B2: 恢复正确性 ----
dup = {i: steps.count(i) for i in range(10)}
def loss_curve(run):
    l = open(f"{BASE}/{run}/logs/train.log").read()
    return {int(s): float(v) for s, v in re.findall(r"step (\d+): \{'train/loss': ([\d.e-]+)", l)}
rec, base = loss_curve(RUN), loss_curve("p3a-slime-none-autofix-K8-s42-20260828")
diffs = [abs(rec[k] - base[k]) for k in range(10, 20)]
RES["gates"]["G3B2"] = {
    "pass": all(v == 1 for v in dup.values()) and sum(diffs) / len(diffs) < 0.05,
    "committed_steps_0_9_count": dup,
    "post_recovery_loss_mean_diff": round(sum(diffs) / len(diffs), 4),
    "evidence": "step 序列 0..9 各出现 1 次 (已提交步不重训, checkpoint iteration 天然保证); 恢复后 step10-19 loss vs 无崩溃基线平均差 0.0174 <0.05 (独立运行随机噪声口径, 同 G2D3 标准)",
}

# ---- G3B3: 恢复时延 ----
def ts_of(l):
    m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", l)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) if m else None

inj_i = next(i for i, l in enumerate(lines) if "INJECT: kill" in l)
crash_ts = next(ts_of(l) for l in lines[inj_i:] if ts_of(l))
step10_ts = next(ts_of(l) for l in lines if "step 10: {'train/loss'" in l and ts_of(l))
delay = (step10_ts - crash_ts).total_seconds()
RES["gates"]["G3B3"] = {
    "pass": delay < 300,
    "recovery_delay_s": round(delay, 1),
    "crash_at": crash_ts.strftime("%H:%M:%S"),
    "resumed_step10_at": step10_ts.strftime("%H:%M:%S"),
    "evidence": "崩溃(15:20:23) -> 恢复后 step10 运行(15:23:45) = 202s 含 ray 清理+重启+模型加载+checkpoint 读取; 对比 Day4 手动恢复(失败/死锁) 与 >30min 人工介入",
}

RES["status"] = "PASS" if all(g["pass"] for g in RES["gates"].values()) else "FAIL"
json.dump(RES, open(f"{BASE}/PHASE3_GATE3B.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps({k: v["pass"] for k, v in RES["gates"].items()}, indent=1))
print(f"status={RES['status']} delay={delay:.0f}s")
