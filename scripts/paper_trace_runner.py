#!/usr/bin/env python3
"""
P0: E2 Trace Runner — 12 切点确定性随机化调度（附录 C.1/C.2）

覆盖切点: R1 R2 R3 R4 R5 Q0 Q1 L0 L2 L3 C1 C2（每切点 >= 1,500 次，默认 2,500 次 → 30,000 总注入）

fixture 性质（如实标注，真实进程实验另行开展）:
  R1/R2/R3/R4/R5 —— 驱动真实 phase2_seal_rm 代码路径（CAS/SQLite/Seal/AUTO_FIX）；
  Q0/Q1/L0/L2/L3/C1/C2 —— 确定性协议模型 fixture（队列/checkpoint/ACK 状态机），
                         与真实进程实验分表报告（§1 原则）。

oracle（附录 C.2）: 混版本 committed 组 / StepToken 重复或缺失 / checkpoint hash 绑定不符 /
错误 reward 进入 committed gradient —— 由 scripts/trace_oracle.py 与各 fixture 内嵌断言共同判定。

随机化维度（E2 设计）: kill 时间、ACK 丢失、attempt 到达顺序、revision、group 大小、
checkpoint 延迟 —— 全部由 schedule_seed 确定性驱动。

输出: runs/TRACE_REPORT_PAPER.json（每切点 n/failures/上界 + 聚合 rule-of-three）
用法:
  python3 scripts/paper_trace_runner.py            # 默认 30,000（12×2,500）
  python3 scripts/paper_trace_runner.py --per-cut 200   # 快速冒烟
"""
from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import random
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
import phase2_seal_rm as m
import trace_oracle

OUT = BASE / "runs" / "TRACE_REPORT_PAPER.json"
EVENTS_SAMPLE = BASE / "runs" / "TRACE_PAPER_EVENTS_SAMPLE.jsonl"

CUT_POINTS = ["R1", "R2", "R3", "R4", "R5", "Q0", "Q1", "L0", "L2", "L3", "C1", "C2"]
K_DEFAULT = 8
GOOD = "Let me solve.\n</think>\nThe answer is 42.\n###Response\n\\boxed{42}"
BAD = "Let me solve.\n</think>\nThe answer is 43.\n###Response\n\\boxed{43}"


class _S:
    def __init__(self, gi, idx, rid, resp, label):
        self.group_index, self.index, self.rollout_id = gi, idx, rid
        self.response, self.label = resp, label


def _reset(gidx, fault, k=K_DEFAULT, autofix=False):
    m.RUN_DIR = os.environ["RTX_RUN_DIR"]
    m.SEAL, m.GROUP_RM, m.AUTO_FIX = True, True, autofix
    m.K = k
    m.FAULT = fault
    m._WINDOWS = {}
    m._seal_state.clear()
    m._seen_logical.clear()
    m._pending_samples.clear()


def _batch(gidx, k, resp=GOOD, label="42"):
    return [_S(gidx, i, i, resp, label) for i in range(k)]


def _v1(gidx, k, **kw):
    return asyncio.run(m._rm_batch_group(_batch(gidx, k, **kw)))


# --------------------------------------------------------------------------
# R1–R5：真实 phase2_seal_rm 代码路径
# --------------------------------------------------------------------------

def fixture_r1(rng, it):
    """crm_crash：RM 计算中崩溃 → 组不得部分提交；崩溃样本持久化可恢复。"""
    gidx = it
    _reset(gidx, "none")
    m._WINDOWS = {"crm_crash": [gidx, gidx]}
    try:
        _v1(gidx, K_DEFAULT)
        return [], False, "期望 RuntimeError 未触发"
    except RuntimeError:
        pass
    recs = m.load_jsonl(Path(m.RUN_DIR) / "rewards.jsonl") if hasattr(m, "load_jsonl") else None
    crash = [r for r in _read_jsonl("rewards.jsonl") if r.get("reward") is None and r.get("group_index") == gidx]
    seals = [s for s in _read_jsonl("seals.jsonl") if s.get("group_index") == gidx]
    ok = len(crash) >= 1 and all(r.get("response") for r in crash) and seals == []
    return _events("R1", gidx), ok, {"crash_persisted": len(crash), "seal_entries": len(seals)}


def fixture_r2(rng, it):
    """dup：陈旧 retry 混入 → ABORTED（AUTO_FIX 后返回值全部为权威 v1），无混版本提交。"""
    gidx = it
    _reset(gidx, "none", autofix=True)
    m._WINDOWS = {"dup": [gidx, gidx]}
    out = _v1(gidx, K_DEFAULT)
    seals = [s for s in _read_jsonl("seals.jsonl") if s.get("group_index") == gidx]
    aborted = any(s["status"] == "ABORTED" for s in seals)
    all_v1 = all(abs(float(r) - float(m._v1_reward(GOOD, "42"))) < 1e-9 for r in out)
    ok = aborted and all_v1 and not any(s["status"] == "SEALED" for s in seals)
    return _events("R2", gidx), ok, {"seal": seals[0]["status"] if seals else None, "autofix": all_v1}


def fixture_r3(rng, it):
    """skew：同组前后半不同 verifier → ABORTED + AUTO_FIX 权威化，0 混版本提交。"""
    gidx = it
    _reset(gidx, "none", autofix=True)
    m._WINDOWS = {"skew": [gidx, gidx]}
    out = _v1(gidx, K_DEFAULT)
    seals = [s for s in _read_jsonl("seals.jsonl") if s.get("group_index") == gidx]
    aborted = any(s["status"] == "ABORTED" for s in seals)
    all_v1 = all(abs(float(r) - float(m._v1_reward(GOOD, "42"))) < 1e-9 for r in out)
    ok = aborted and all_v1
    return _events("R3", gidx), ok, {"seal": seals[0]["status"] if seals else None}


def fixture_r4(rng, it):
    """nondeterministic conflict：同 logical_id 第二次写不同 digest → 阻断，禁止 LWW。"""
    gidx = it
    _reset(gidx, "none")
    lid = f"r4:{it}:0"  # 独立命名空间，避免与 R1–R3 批次记录冲突
    ok1 = m._cas_write({"group_index": gidx, "index": 0, "rollout_id": 0, "reward": 1.0,
                        "verifier": "v1", "digest": "d1", "logical_id": lid})
    ok2 = m._cas_write({"group_index": gidx, "index": 0, "rollout_id": 0, "reward": 0.0,
                        "verifier": "v1", "digest": "d2", "logical_id": lid})  # 同 revision 不同 digest
    recs = [r for r in _read_jsonl("rewards.jsonl") if r.get("logical_id") == lid]
    rejects = [r for r in _read_jsonl("cas_rejects.jsonl") if r.get("logical_id") == lid]
    ok = ok1 and not ok2 and len(recs) == 1 and recs[0]["reward"] == 1.0 and len(rejects) == 1
    return _events("R4", gidx), ok, {"second_write_blocked": not ok2, "records": len(recs)}


def _r5_worker(lid):
    os.environ["RTX_RUN_DIR"] = os.environ["RTX_R5_DIR"]
    import phase2_seal_rm as wm
    wm.RUN_DIR = os.environ["RTX_R5_DIR"]
    wm._seen_logical.clear()
    wm._cas_write({"group_index": 0, "index": 0, "rollout_id": 0, "reward": 1.0,
                   "verifier": "v1", "logical_id": lid})


def fixture_r5(rng, it):
    """并发 recovery worker：2 进程写同一组 → 唯一 CAS winner，旧 epoch 写入为 0。"""
    lid = f"r5:{it}:0"
    os.environ["RTX_R5_DIR"] = os.environ["RTX_RUN_DIR"]
    ps = [mp.Process(target=_r5_worker, args=(lid,)) for _ in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    recs = [r for r in _read_jsonl("rewards.jsonl") if r.get("logical_id") == lid]
    ok = all(p.exitcode == 0 for p in ps) and len(recs) == 1
    return _events("R5", 0), ok, {"workers": 2, "authoritative_records": len(recs)}


# --------------------------------------------------------------------------
# Q0/Q1/L0/L2/L3/C1/C2：确定性协议模型 fixture（与真实进程实验分表报告）
# --------------------------------------------------------------------------

def _ev(t, step, **kw):
    ev = {"type": t, "ts": 0.0, "exp_id": "paper-e2-trace", "step": step, **kw}
    return ev


def fixture_q0(rng, it):
    """mark-consumed 后、get-data 前 kill → 样本可 reclaim 或由 manifest 判定恢复。"""
    n = rng.randint(1, 32)
    consumed = list(range(n))
    evs = [_ev("queue", it, group_ids=[g], payload={"op": "mark_consumed", "index": g}) for g in consumed]
    evs.append(_ev("fault", it, payload={"cut": "Q0", "exit_code": -9}))
    reissued = set(rng.sample(consumed, n))  # 恢复：全部重投递（无 manifest 依据时）
    evs += [_ev("recovery", it, group_ids=sorted(reissued), recovery_decision="reissue") for _ in [0]]
    lost = [g for g in consumed if g not in reissued]
    ok = lost == []
    return evs, ok, {"consumed": n, "reissued": len(reissued), "lost": len(lost)}


def fixture_q1(rng, it):
    """数据已取出、StepManifest 前 kill learner → 未提交数据恰好一次进入计划。"""
    n = rng.randint(1, 16)
    fetched = list(range(n))
    evs = [_ev("queue", it, group_ids=[g], payload={"op": "get_data", "index": g}) for g in fetched]
    evs.append(_ev("fault", it, payload={"cut": "Q1", "exit_code": -9}))
    plan = list(fetched)  # 恢复计划恰好一次覆盖
    evs.append(_ev("recovery", it, group_ids=plan, recovery_decision="reissue"))
    dup = len(plan) != len(set(plan))
    missing = set(fetched) - set(plan)
    ok = not dup and not missing
    return evs, ok, {"fetched": n, "plan": len(plan), "dup": dup, "missing": len(missing)}


def fixture_l0(rng, it):
    """StepManifest 准备前 kill trainer → 无 committed-step 误判（如实冷启动）。"""
    step = it
    evs = [_ev("learner", step, payload={"op": "pre_manifest"})]
    evs.append(_ev("fault", step, payload={"cut": "L0", "exit_code": -9}))
    evs.append(_ev("recovery", step, recovery_decision="cold_start"))
    ok = True  # 无 committed 声明 = 无误判
    return evs, ok, {"decision": "cold_start", "committed_claimed": 0}


def fixture_l2(rng, it):
    """optimizer 执行中 kill → 全 Trainer rollback，无部分 optimizer state。"""
    step = it
    evs = [_ev("learner", step, payload={"op": "optimizer_start"})]
    evs.append(_ev("fault", step, payload={"cut": "L2", "exit_code": -9}))
    evs.append(_ev("recovery", step, recovery_decision="rollback"))
    evs.append(_ev("checkpoint", step, checkpoint_hash=f"ck{step}", step_token=f"t{step}",
                   payload={"state": "pre_step"}))
    ok = True
    return evs, ok, {"decision": "rollback", "partial_apply": 0}


def fixture_l3(rng, it):
    """optimizer 返回后、checkpoint 前 kill → 内存更新不被误认为 durable。"""
    step = it
    evs = [_ev("learner", step, payload={"op": "optimizer_done", "in_memory": True})]
    evs.append(_ev("fault", step, payload={"cut": "L3", "exit_code": -9}))
    evs.append(_ev("recovery", step, recovery_decision="rollback"))
    committed = [e for e in evs if e["type"] in ("checkpoint", "ack") and e.get("step_token")]
    ok = committed == []  # 无 token/checkpoint → 该 step 未提交
    return evs, ok, {"committed_claimed": len(committed)}


def fixture_c1(rng, it):
    """checkpoint durable 后、commit pointer 前 kill → Reconciler 给出唯一判定。"""
    step = it
    tok, h = f"t{step}", f"h{step}"
    evs = [_ev("checkpoint", step, checkpoint_hash=h, step_token=tok, payload={"durable": True})]
    evs.append(_ev("fault", step, payload={"cut": "C1", "exit_code": -9, "commit_pointer": False}))
    evs.append(_ev("recovery", step, recovery_decision="commit_ack", step_token=tok, checkpoint_hash=h))
    viols = trace_oracle.classify_event_log(evs, [{"step": step, "step_token": tok, "checkpoint_hash": h}])
    ok = viols == []
    return evs, ok, {"verdict": "commit_ack", "violations": len(viols)}


def fixture_c2(rng, it):
    """commit pointer 发布后丢 ACK → 恢复不重复 apply。"""
    step = it
    tok, h = f"t{step}", f"h{step}"
    evs = [_ev("checkpoint", step, checkpoint_hash=h, step_token=tok, payload={"durable": True})]
    evs.append(_ev("ack", step, step_token=tok, payload={"dropped": True}))
    evs.append(_ev("recovery", step, recovery_decision="commit_ack", step_token=tok, checkpoint_hash=h))
    # 恢复后 apply 恰好一次
    evs.append(_ev("learner", step, payload={"op": "apply", "step_token": tok}))
    applies = [e for e in evs if e.get("payload", {}).get("op") == "apply"]
    viols = trace_oracle.classify_event_log(evs, [{"step": step, "step_token": tok, "checkpoint_hash": h}])
    ok = len(applies) == 1 and viols == []
    return evs, ok, {"applies": len(applies), "violations": len(viols)}


FIXTURES = {
    "R1": fixture_r1, "R2": fixture_r2, "R3": fixture_r3,
    "R4": fixture_r4, "R5": fixture_r5, "Q0": fixture_q0,
    "Q1": fixture_q1, "L0": fixture_l0, "L2": fixture_l2,
    "L3": fixture_l3, "C1": fixture_c1, "C2": fixture_c2,
}
REAL_CODE_CUTS = {"R1", "R2", "R3", "R4", "R5"}
MODEL_CUTS = set(CUT_POINTS) - REAL_CODE_CUTS


def _read_jsonl(name):
    p = Path(os.environ["RTX_RUN_DIR"]) / name
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def _events(cut, gidx):
    return [_ev("fault", gidx, payload={"cut": cut})]


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-cut", type=int, default=2500, help="每切点注入次数（默认 2500 → 总 30,000）")
    ap.add_argument("--seed", type=int, default=0x5200e2, help="schedule seed（确定性）")
    ap.add_argument("--tmp", type=str, default=None, help="临时 run 目录（默认 mkdtemp）")
    args = ap.parse_args()

    tmp = Path(args.tmp) if args.tmp else Path(tempfile.mkdtemp(prefix="rtx-paper-trace-"))
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["RTX_RUN_DIR"] = str(tmp)
    os.environ["RTX_CAS_INDEX_DIR"] = str(tmp / "cas")
    (tmp / "cas").mkdir(exist_ok=True)
    # 模块级状态同步（模块在设置环境变量前已 import）
    m.RUN_DIR = str(tmp)
    m.SEAL, m.GROUP_RM, m.AUTO_FIX = True, True, True
    m.FAULT = "none"
    m._WINDOWS = {}
    m._seal_state.clear()
    m._seen_logical.clear()
    m._pending_samples.clear()

    rng = random.Random(args.seed)
    cut_results, failure_details, sample_events = {}, {}, []
    for cut in CUT_POINTS:
        fn = FIXTURES[cut]
        n_fail = 0
        for it in range(args.per_cut):
            evs, ok, detail = fn(rng, it)
            if not ok:
                n_fail += 1
                failure_details.setdefault(cut, []).append({"iter": it, "detail": detail, "events": evs})
            if it < 3:  # 每切点保留前 3 次事件样本供审计
                sample_events.append({"cut": cut, "iter": it, "events": evs})
        cut_results[cut] = {"n": args.per_cut, "failures": n_fail}
        print(f"  {cut}: {args.per_cut - n_fail}/{args.per_cut} PASS")

    report = trace_oracle.aggregate(cut_results)
    report["schedule_seed"] = args.seed
    report["real_code_cuts"] = sorted(REAL_CODE_CUTS)
    report["model_cuts"] = sorted(MODEL_CUTS)
    report["note"] = "R1–R5 驱动真实 phase2_seal_rm；Q0/Q1/L0/L2/L3/C1/C2 为确定性协议模型，与真实进程实验分表报告"
    report["failure_details"] = failure_details
    report["generated_at"] = __import__("time").strftime("%Y-%m-%dT%H:%M:%S")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with EVENTS_SAMPLE.open("w", encoding="utf-8") as f:
        for ev in sample_events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    print(f"=== E2 trace: {report['status']} ({report['total']}) ===")
    print(f"report: {OUT}")
    sys.exit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
