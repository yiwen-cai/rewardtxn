#!/usr/bin/env python3
"""
Phase 2A: Group Seal + Reward CAS 包装层 (slime --custom-rm-path, 兼容 batch/单样本)

相对 day2_custom_rm.py 新增 (RTX_SEAL=1 启用):
  1. Group Seal 状态机:
     - 每样本 reward 计算后登记 (group_index -> {verifiers, logical_ids, count})
     - 组完成 (count == K=8) 时检查组内 verifier 版本一致性:
       全部一致 -> SEALED (放行审计)
       版本混合 -> ABORTED (混算检测审计)
     - 审计流: seals.jsonl (group_index, status, versions, count, ts, step)
  2. Reward CAS:
     - rewards.jsonl 写入按 logical_id=(group_index,index,rollout_id) 幂等
     - 重复 logical_id -> 拒绝写入 + cas_reject 审计 (cas_rejects.jsonl)
  3. GroupManifest 雏形: seals.jsonl 即 (GroupID -> Seal 状态) 的持久化视图,
     作为 2B StepManifest / 2C Reconciler 的直接输入。

设计边界 (分阶段):
  - 2A 负责"检测 + 审计 + CAS 幂等" (把 Day2 完全静默变成显式可审计)
  - "ABORTED 组 0 混算进入训练"的完整消费侧 gate 由 2B StepToken +
    2C Selective Replay 实现 (slime 训练循环不可改, 需要包装 checkpoint/消费路径)
"""
import asyncio
import json
import os
import time

from slime.rollout.rm_hub.math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy

FAULT = os.environ.get("RTX_FAULT", "none")
START = int(os.environ.get("RTX_FAULT_START", "-1"))
END = int(os.environ.get("RTX_FAULT_END", "-1"))
RUN_DIR = os.environ.get("RTX_RUN_DIR", "/workspace/runs")
V2_MODE = os.environ.get("RTX_V2_MODE", "strict")
SEAL = os.environ.get("RTX_SEAL", "0") == "1"
K = int(os.environ.get("RTX_GROUP_SIZE", "8"))
_lock = asyncio.Lock()
_seal_state = {}  # group_index -> {verifiers:set, ids:set, count, first_ts}   # group_index -> {verifiers:set, ids:set, count, first_ts}
_seen_logical = set()


def _v1_reward(response: str, label: str) -> float:
    model_solution = response or ""
    for sep in ("</think>", "###Response"):
        if sep in model_solution:
            model_solution = model_solution.split(sep)[-1]
            break
    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return 0
    if label == "":
        return 0
    truths = [label]
    processed = []
    for truth in truths:
        truth = str(truth)
        if "\\boxed" in truth:
            pa = extract_answer(truth)
            if pa is not None:
                processed.append(pa)
        else:
            processed.append(truth)
    if not processed:
        return 0
    for gt in processed:
        if grade_answer_mathd(model_answer, gt) or grade_answer_sympy(model_answer, gt):
            return 1
    return 0


def _v2_reward(response: str, label: str) -> float:
    if "</think>" in response:
        model_solution = response.split("</think>")[-1]
    elif "###Response" in response:
        model_solution = response.split("###Response")[1]
    else:
        return 0
    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return 0
    if label == "":
        return 0
    truths = [label]
    processed = []
    for truth in truths:
        truth = str(truth)
        if "\\boxed" in truth:
            pa = extract_answer(truth)
            if pa is not None:
                processed.append(pa)
        else:
            processed.append(truth)
    if not processed:
        return 0
    for gt in processed:
        if V2_MODE == "strict":
            ok = grade_answer_mathd(model_answer, gt) and grade_answer_sympy(model_answer, gt)
        else:
            ok = grade_answer_mathd(model_answer, gt)
        if ok:
            return 1
    return 0


def _cas_write(rec: dict) -> bool:
    """Reward CAS: (group_index, index, rollout_id) 幂等写入。重复 -> False (拒绝)。"""
    lid = f"{rec['group_index']}:{rec['index']}:{rec['rollout_id']}"
    if lid in _seen_logical:
        try:
            with open(os.path.join(RUN_DIR, "cas_rejects.jsonl"), "a") as f:
                f.write(json.dumps({**rec, "logical_id": lid, "reason": "duplicate-logical-id"}) + "\n")
        except Exception:
            pass
        return False
    _seen_logical.add(lid)
    try:
        with open(os.path.join(RUN_DIR, "rewards.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    return True


def _seal_register(group_index: int, verifier: str, logical_id: str):
    """登记样本到组状态; 组完成时返回 (status, versions) 否则 None"""
    st = _seal_state.setdefault(group_index, {"verifiers": set(), "ids": set(), "count": 0, "first_ts": time.time()})
    st["verifiers"].add(verifier)
    st["ids"].add(logical_id)
    st["count"] += 1
    if st["count"] < K:
        return None
    # 组完成 -> Seal 检查
    status = "SEALED" if len(st["verifiers"]) == 1 else "ABORTED"
    versions = sorted(st["verifiers"])
    rec = {
        "group_index": group_index,
        "status": status,
        "versions": versions,
        "count": st["count"],
        "unique_samples": len(st["ids"]),
        "ts": time.time(),
    }
    try:
        with open(os.path.join(RUN_DIR, "seals.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    return status, versions


def _log(sample, reward: float, verifier: str, injected: bool):
    rec = {
        "group_index": sample.group_index,
        "index": sample.index,
        "rollout_id": sample.rollout_id,
        "reward": reward,
        "verifier": verifier,
        "injected": injected,
        "fault": FAULT,
        "seal_enabled": SEAL,
        "ts": time.time(),
    }
    lid = f"{rec['group_index']}:{rec['index']}:{rec['rollout_id']}"
    _cas_write(rec)
    if SEAL:
        _seal_register(sample.group_index if sample.group_index is not None else -1, verifier, lid)


async def _rm_one(s, is_batch_sorted_pos=None):
    in_window = START <= (s.group_index if s.group_index is not None else -1) <= END
    if FAULT == "crm_crash" and in_window:
        raise RuntimeError(f"injected RM crash (group_index={s.group_index})")
    r1 = float(_v1_reward(s.response, s.label or ""))
    r2 = float(_v2_reward(s.response, s.label or ""))
    half = K // 2
    pos = is_batch_sorted_pos if is_batch_sorted_pos is not None else ((s.index or 0) % K)
    if FAULT == "skew" and in_window:
        if pos < half:
            r, ver, inj = r1, "v1", False
        else:
            r, ver, inj = r2, "v2", True
    elif FAULT == "dup" and in_window:
        if pos < half:
            r, ver, inj = r1, "v1", False
        else:
            r, ver, inj = r2, "retry-stale", True
    else:
        r, ver, inj = r1, "v1", False
    _log(s, r, ver, inj)
    return r


async def rm_function(args, samples):
    if not isinstance(samples, list):
        return await _rm_one(samples)
    results = [0.0] * len(samples)
    groups = {}
    for i, s in enumerate(samples):
        groups.setdefault(s.group_index, []).append(i)
    async with _lock:
        for gi, idxs in groups.items():
            idxs_sorted = sorted(idxs, key=lambda i: samples[i].index)
            for pos, i in enumerate(idxs_sorted):
                results[i] = await _rm_one(samples[i], is_batch_sorted_pos=pos)
    return results
