#!/usr/bin/env python3
"""
Day 2 故障注入自定义 RM (slime --custom-rm-path, batch 模式)

签名: async def rm_function(args, samples: list[Sample]) -> list[float]

环境变量:
  RTX_FAULT         none | skew | crm_crash | dup
  RTX_FAULT_START   注入窗口起点 group_index (含, 全局递增; step = group_index // U)
  RTX_FAULT_END     注入窗口终点 group_index (含)
  RTX_RUN_DIR       instrumentation 落盘目录 (rewards.jsonl 逐样本)
  RTX_V2_MODE       strict | loose   (v2 Verifier 构造)
                    strict: 两个 grader 都通过才判对 (比 v1 严)
                    loose : 仅 mathd grader (与 v1 的 or 语义有差异)

语义 (对齐文档 R1/R2/R3):
  skew     : 窗口内每组前 K/2 条用 v1, 后 K/2 条用 v2 -> 组内跨版本混算 (Revision Skew)
  crm_crash: 窗口内样本抛 RuntimeError (模拟 RM Worker 崩溃)
  dup      : 窗口内每组一半样本使用"重试陈旧结果"(v2 结果), 模拟重复交付结果不一致
             (注: slime 同步 RM 无 at-least-once 窗口, Day 2 为弱化版, 完整语义在 Day 3)
  none     : 纯 instrumentation (v1, 无注入) —— 对照
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
_lock = asyncio.Lock()


def _v1_reward(response: str, label: str) -> float:
    """Verifier v1: 兼容版 deepscaler 判定
    修复 slime 原版 bug: 原版要求 response 含 </think> 或 ###Response 分隔符,
    Qwen2.5 输出无此标记导致恒 0。兼容版: 无分隔符时直接对完整 response 提取答案。"""
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
    """Verifier v2: 版本变体 (与 v1 均为确定性 rule-based, 但判定语义不同)"""
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
        else:  # loose: 仅 mathd
            ok = grade_answer_mathd(model_answer, gt)
        if ok:
            return 1
    return 0


def _log(sample, reward: float, verifier: str, injected: bool):
    rec = {
        "group_index": sample.group_index,
        "index": sample.index,
        "rollout_id": sample.rollout_id,
        "reward": reward,
        "verifier": verifier,
        "injected": injected,
        "fault": FAULT,
        "ts": time.time(),
    }
    if os.environ.get("RTX_LOG_RESPONSE", "0") == "1":
        resp = sample.response or ""
        rec["response_head"] = resp[:120]
        rec["response_tail"] = resp[-160:]
        rec["label"] = (sample.label or "")[:40]
    path = os.path.join(RUN_DIR, "rewards.jsonl")
    try:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


async def _rm_one(s, is_batch_sorted_pos=None):
    """单样本注入逻辑。pos: 组内位置 (batch 模式传入); 单样本模式用 index % K 推断"""
    in_window = START <= (s.group_index if s.group_index is not None else -1) <= END
    if FAULT == "crm_crash" and in_window:
        raise RuntimeError(f"injected RM crash (group_index={s.group_index})")
    r1 = float(_v1_reward(s.response, s.label or ""))
    r2 = float(_v2_reward(s.response, s.label or ""))
    half = 4  # K=8 -> 前 4 条 v1, 后 4 条 v2
    pos = is_batch_sorted_pos if is_batch_sorted_pos is not None else ((s.index or 0) % 8)
    if FAULT == "skew" and in_window:
        if pos < half:
            r, ver, inj = r1, "v1", False
        else:
            r, ver, inj = r2, "v2", True
    elif FAULT == "dup" and in_window:
        if pos < half:
            r, ver, inj = r1, "v1", False
        else:
            r, ver, inj = r2, "retry-stale", True  # 弱化: 陈旧重试结果
    else:
        r, ver, inj = r1, "v1", False
    _log(s, r, ver, inj)
    return r


async def rm_function(args, samples):
    """兼容 fully-async 单样本调用 (async_rm) 与 batch 调用 (batched_async_rm)"""
    # 单样本模式: fully-async 路径经 async_rm(args, sample) 调用
    if not isinstance(samples, list):
        return await _rm_one(samples)
    # batch 模式: 组内按 index 排序后分流
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
