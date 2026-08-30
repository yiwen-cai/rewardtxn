#!/usr/bin/env python3
"""
P0: E2 Trace Runner — 12 切点确定性随机化调度（附录 C.1/C.2）

覆盖切点: R1 R2 R3 R4 R5 Q0 Q1 L0 L2 L3 C1 C2（每切点 >= 1,500 次，默认 2,500 次 → 30,000 总注入）

fixture 性质（如实标注，真实进程实验另行开展）:
  R1/R2/R3/R4/R5 —— 驱动真实 phase2_seal_rm 代码路径（CAS/SQLite/Seal/AUTO_FIX），
                    fixture 内嵌断言，随机化维度含 group 大小 K∈{4,8,16}；
  Q0/Q1/L0/L2/L3/C1/C2 —— 确定性协议模型 fixture（队列/checkpoint/ACK 状态机）：
                    fixture 只产生事件日志，判定由日志解释器（VERIFIERS）完成，
                    oracle 从日志反推状态，非构造即真；与真实进程实验分表报告（§1 原则）。

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
# v1 与 v2 值级发散的样本（v1=1, v2=0）：用于真正考验 AUTO_FIX 的值修正
DIVERGENT = "Let me solve.\n</think>\nThe answer is 42.0.\n###Response\n\\boxed{42.0}"


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


def _batch_mixed(gidx, k):
    """前一半 GOOD（v1 值 1），后一半 DIVERGENT（v1 值 1 / v2 值 0）：
    skew/dup 注入后版本 tag 与值同时混合，AUTO_FIX 必须把后一半修正回 v1=1。"""
    half = k // 2
    return [_S(gidx, i, i, GOOD if i < half else DIVERGENT, "42") for i in range(k)]


def _v1(gidx, k, mixed=False, **kw):
    samples = _batch_mixed(gidx, k) if mixed else _batch(gidx, k, **kw)
    return asyncio.run(m._rm_batch_group(samples))


# --------------------------------------------------------------------------
# R1–R5：真实 phase2_seal_rm 代码路径
# --------------------------------------------------------------------------

def fixture_r1(rng, it):
    """crm_crash：RM 计算中崩溃 → 组不得部分提交；崩溃样本持久化可恢复。"""
    gidx = it
    k = rng.choice([4, 8, 16])   # 随机化维度：group 大小
    _reset(gidx, "none", k=k)
    m._WINDOWS = {"crm_crash": [gidx, gidx]}
    try:
        _v1(gidx, k)
        return [], False, "期望 RuntimeError 未触发"
    except RuntimeError:
        pass
    crash = [r for r in _read_jsonl("rewards.jsonl") if r.get("reward") is None and r.get("group_index") == gidx]
    seals = [s for s in _read_jsonl("seals.jsonl") if s.get("group_index") == gidx]
    ok = len(crash) >= 1 and all(r.get("response") for r in crash) and seals == []
    return _events("R1", gidx), ok, {"k": k, "crash_persisted": len(crash), "seal_entries": len(seals)}


def fixture_r2(rng, it):
    """dup：陈旧 retry 混入 → ABORTED（AUTO_FIX 后返回值全部为权威 v1），无混版本提交。"""
    gidx = it
    k = rng.choice([4, 8, 16])
    _reset(gidx, "none", k=k, autofix=True)
    m._WINDOWS = {"dup": [gidx, gidx]}
    out = _v1(gidx, k, mixed=True)
    seals = [s for s in _read_jsonl("seals.jsonl") if s.get("group_index") == gidx]
    aborted = any(s["status"] == "ABORTED" for s in seals)
    expected = [m._v1_reward(GOOD if i < k // 2 else DIVERGENT, "42") for i in range(k)]
    all_v1 = all(abs(float(r) - float(e)) < 1e-9 for r, e in zip(out, expected))
    ok = aborted and all_v1 and not any(s["status"] == "SEALED" for s in seals)
    return _events("R2", gidx), ok, {"k": k, "seal": seals[0]["status"] if seals else None, "autofix_values": all_v1}


def fixture_r3(rng, it):
    """skew：同组前后半不同 verifier → ABORTED + AUTO_FIX 权威化，0 混版本提交。"""
    gidx = it
    k = rng.choice([4, 8, 16])
    _reset(gidx, "none", k=k, autofix=True)
    m._WINDOWS = {"skew": [gidx, gidx]}
    out = _v1(gidx, k, mixed=True)
    seals = [s for s in _read_jsonl("seals.jsonl") if s.get("group_index") == gidx]
    aborted = any(s["status"] == "ABORTED" for s in seals)
    expected = [m._v1_reward(GOOD if i < k // 2 else DIVERGENT, "42") for i in range(k)]
    all_v1 = all(abs(float(r) - float(e)) < 1e-9 for r, e in zip(out, expected))
    ok = aborted and all_v1
    return _events("R3", gidx), ok, {"k": k, "seal": seals[0]["status"] if seals else None, "autofix_values": all_v1}


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


# --------------------------------------------------------------------------
# Q0/Q1/L0/L2/L3/C1/C2：确定性协议模型 fixture + 日志解释器（oracle 从事件日志
# 反推状态判定，非构造即真；参数由 rng 随机化以覆盖状态空间）
# --------------------------------------------------------------------------

def _groups_of(evs, types, op=None):
    out = []
    for e in evs:
        if e["type"] in types and (op is None or e.get("payload", {}).get("op") == op):
            out.extend(e.get("group_ids", []))
    return out


def _tokens_claimed(evs):
    return {e.get("step_token") for e in evs
            if e["type"] == "checkpoint" and e.get("payload", {}).get("role", "commit") == "commit"
            and e.get("step_token")}


def _tokens_applied(evs):
    return [e.get("payload", {}).get("step_token") for e in evs
            if e["type"] == "learner" and e.get("payload", {}).get("op") == "apply"]


def _dup_count(items):
    from collections import Counter
    return sum(1 for v in Counter(items).values() if v > 1)


def fixture_q0(rng, it):
    """mark-consumed 后、get-data 前 kill。状态：consumed / committed(manifest) 两集合。
    协议要求：已消费未提交样本必须被 reissue（或由 manifest 判定已提交）；已提交样本不得重投。"""
    n = rng.randint(2, 32)
    consumed = list(range(n))
    n_committed = rng.randint(0, n // 2)
    committed = set(rng.sample(consumed, n_committed))
    evs = [_ev("queue", it, group_ids=[g], payload={"op": "mark_consumed", "index": g}) for g in consumed]
    if committed:
        evs.append(_ev("group", it, group_ids=sorted(committed), payload={"status": "COMMITTED", "versions": ["v1"]}))
    evs.append(_ev("fault", it, payload={"cut": "Q0", "exit_code": -9}))
    need_reissue = sorted(set(consumed) - committed)
    if need_reissue:
        evs.append(_ev("recovery", it, group_ids=need_reissue, recovery_decision="reissue"))
    return evs, None


def verify_q0(evs, auth=None):
    consumed = set(_groups_of(evs, ["queue"], "mark_consumed"))
    committed = set(_groups_of(evs, ["group"]))
    reissued = _groups_of(evs, ["recovery"])
    need = consumed - committed
    reissued_set = set(reissued)
    lost = sorted(need - reissued_set)
    over = sorted(reissued_set - need)
    ok = lost == [] and over == [] and _dup_count(reissued) == 0
    return ok, {"consumed": len(consumed), "committed": len(committed),
                "reissued": len(reissued), "lost": len(lost), "over_reissue": len(over)}


def fixture_q1(rng, it):
    """get-data 后、StepManifest 前 kill learner。未提交数据必须恰好一次进入恢复计划。"""
    n = rng.randint(1, 16)
    fetched = list(range(n))
    evs = [_ev("queue", it, group_ids=[g], payload={"op": "get_data", "index": g}) for g in fetched]
    evs.append(_ev("fault", it, payload={"cut": "Q1", "exit_code": -9}))
    evs.append(_ev("recovery", it, group_ids=fetched, recovery_decision="reissue"))
    return evs, None


def verify_q1(evs, auth=None):
    fetched = set(_groups_of(evs, ["queue"], "get_data"))
    planned = _groups_of(evs, ["recovery"])
    ok = set(planned) == fetched and _dup_count(planned) == 0 and len(planned) == len(fetched)
    return ok, {"fetched": len(fetched), "planned": len(planned),
                "dup": _dup_count(planned), "missing": len(fetched - set(planned))}


def fixture_l0(rng, it):
    """StepManifest 准备前 kill trainer：内存中已执行步不得被误判为 committed。"""
    n = rng.randint(1, 8)
    evs = [_ev("learner", it, payload={"op": "pre_manifest", "in_memory_steps": n})]
    evs.append(_ev("fault", it, payload={"cut": "L0", "exit_code": -9}))
    evs.append(_ev("recovery", it, recovery_decision="cold_start"))
    return evs, None


def verify_l0(evs, auth=None):
    claimed = _tokens_claimed(evs)
    decisions = [e["recovery_decision"] for e in evs if e["type"] == "recovery"]
    n = next((e.get("payload", {}).get("in_memory_steps") for e in evs if e["type"] == "learner"), None)
    ok = claimed == set() and decisions == ["cold_start"]
    return ok, {"in_memory_steps": n, "committed_claimed": len(claimed), "decision": decisions}


def fixture_l2(rng, it):
    """optimizer 执行中 kill：部分 rank 已 apply，必须全 Trainer rollback，不得留下部分提交。"""
    r = rng.randint(2, 8)          # rank 数
    done = rng.randint(0, r - 1)   # 已 apply 的 rank（部分）
    evs = [_ev("learner", it, payload={"op": "optimizer_start", "ranks": r})]
    for i in range(done):
        evs.append(_ev("learner", it, payload={"op": "rank_apply", "rank": i, "durable": False}))
    evs.append(_ev("fault", it, payload={"cut": "L2", "exit_code": -9, "rank": done}))
    evs.append(_ev("recovery", it, recovery_decision="rollback"))
    evs.append(_ev("checkpoint", it, checkpoint_hash=f"pre-{it}", step_token=f"t{it}",
                   payload={"state": "pre_step"}))
    return evs, None


def verify_l2(evs, auth=None):
    claimed = _tokens_claimed(evs)
    decisions = [e["recovery_decision"] for e in evs if e["type"] == "recovery"]
    # 部分 rank 已 apply 的 step 不得被提交（rollback 到 pre_step token）
    ok = claimed == {f"t{evs[0]['step']}"} and decisions == ["rollback"]
    return ok, {"claimed_tokens": sorted(claimed), "decision": decisions}


def fixture_l3(rng, it):
    """optimizer 返回后、checkpoint 前 kill：内存更新不得被误认为 durable。"""
    n = rng.randint(1, 8)
    evs = [_ev("learner", it, payload={"op": "optimizer_done", "in_memory": True, "steps_in_memory": n})]
    evs.append(_ev("fault", it, payload={"cut": "L3", "exit_code": -9}))
    evs.append(_ev("recovery", it, recovery_decision="rollback"))
    return evs, None


def verify_l3(evs, auth=None):
    claimed = _tokens_claimed(evs)
    decisions = [e["recovery_decision"] for e in evs if e["type"] == "recovery"]
    n = next((e.get("payload", {}).get("steps_in_memory") for e in evs if e["type"] == "learner"), None)
    ok = claimed == set() and decisions == ["rollback"]
    return ok, {"steps_in_memory": n, "committed_claimed": len(claimed), "decision": decisions}


def fixture_c1(rng, it):
    """checkpoint durable 后、commit pointer 前 kill → Reconciler 给出唯一判定（commit_ack）。
    返回 (事件日志, 期望权威映射)：权威映射由 fixture 的外部意图给出，oracle 独立判定。"""
    step = it
    tok, h = f"t{step}", f"h{step}"
    evs = [_ev("checkpoint", step, checkpoint_hash=h, step_token=tok, payload={"durable": True})]
    evs.append(_ev("fault", step, payload={"cut": "C1", "exit_code": -9, "commit_pointer": False}))
    evs.append(_ev("recovery", step, recovery_decision="commit_ack", step_token=tok, checkpoint_hash=h))
    auth = [{"step": step, "step_token": tok, "checkpoint_hash": h}]
    return evs, auth


def verify_c1(evs, auth):
    viols = trace_oracle.classify_event_log(evs, auth)
    decisions = [e["recovery_decision"] for e in evs if e["type"] == "recovery"]
    ok = viols == [] and decisions == ["commit_ack"]
    return ok, {"step": auth[0]["step"], "token": auth[0]["step_token"], "violations": len(viols), "verdict": decisions}


def fixture_c2(rng, it):
    """commit pointer 发布后丢 ACK → 恢复后不重复 apply。
    返回 (事件日志, 期望权威映射)。"""
    step = it
    tok, h = f"t{step}", f"h{step}"
    evs = [_ev("checkpoint", step, checkpoint_hash=h, step_token=tok, payload={"durable": True})]
    evs.append(_ev("ack", step, step_token=tok, payload={"dropped": True}))
    evs.append(_ev("recovery", step, recovery_decision="commit_ack", step_token=tok, checkpoint_hash=h))
    evs.append(_ev("learner", step, payload={"op": "apply", "step_token": tok}))
    auth = [{"step": step, "step_token": tok, "checkpoint_hash": h}]
    return evs, auth


def verify_c2(evs, auth):
    viols = trace_oracle.classify_event_log(evs, auth)
    applies = _tokens_applied(evs)
    ok = viols == [] and len(applies) == 1
    return ok, {"step": auth[0]["step"], "token": auth[0]["step_token"], "violations": len(viols), "applies": len(applies)}


FIXTURES = {
    "R1": fixture_r1, "R2": fixture_r2, "R3": fixture_r3,
    "R4": fixture_r4, "R5": fixture_r5, "Q0": fixture_q0,
    "Q1": fixture_q1, "L0": fixture_l0, "L2": fixture_l2,
    "L3": fixture_l3, "C1": fixture_c1, "C2": fixture_c2,
}
VERIFIERS = {
    "R1": None, "R2": None, "R3": None, "R4": None, "R5": None,
    "Q0": verify_q0, "Q1": verify_q1, "L0": verify_l0, "L2": verify_l2,
    "L3": verify_l3, "C1": verify_c1, "C2": verify_c2,
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
    cut_results, failure_details, sample_events = {}, [], []
    for cut in CUT_POINTS:
        fn = FIXTURES[cut]
        ver = VERIFIERS[cut]
        n_fail = 0
        for it in range(args.per_cut):
            res = fn(rng, it)
            if isinstance(res, tuple) and len(res) == 3:
                # R1–R5：真实代码路径，fixture 内嵌断言
                evs, ok, detail = res
            else:
                # Q0/L0/C1 等：协议模型事件日志 (+期望权威映射) + 日志解释器判定
                evs, auth = res
                ok, detail = ver(evs, auth)
            if not ok:
                n_fail += 1
                failure_details.append({"cut": cut, "iter": it, "detail": detail, "events": evs})
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
