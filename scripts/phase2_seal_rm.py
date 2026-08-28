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
from __future__ import annotations

import ast as _ast
import asyncio
import json
import os
import sqlite3
import time

try:
    from slime.rollout.rm_hub.math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy
except ImportError:  # 宿主机 (无 slime): 使用纯函数副本 (同 SHA 来源)
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__))))
    from phase2_verifiers import extract_answer, grade_answer_mathd, grade_answer_sympy

FAULT = os.environ.get("RTX_FAULT", "none")
START = int(os.environ.get("RTX_FAULT_START", "-1"))
END = int(os.environ.get("RTX_FAULT_END", "-1"))
RUN_DIR = os.environ.get("RTX_RUN_DIR", "/workspace/runs")
V2_MODE = os.environ.get("RTX_V2_MODE", "strict")
SEAL = os.environ.get("RTX_SEAL", "0") == "1"
AUTO_FIX = os.environ.get("RTX_SEAL_AUTO_FIX", "0") == "1"
GROUP_RM = os.environ.get("RTX_GROUP_RM", "0") == "1"
K = int(os.environ.get("RTX_GROUP_SIZE", "8"))
# 多窗口注入: RTX_FAULT_WINDOWS='{"skew":[20,39],"crm_crash":[40,43]}' (覆盖单窗口)
_WINDOWS = {}
_wenv = os.environ.get("RTX_FAULT_WINDOWS", "")
if _wenv:
    _WINDOWS = _ast.literal_eval(_wenv)
_lock = asyncio.Lock()
_seal_state = {}  # group_index -> {verifiers:set, ids:set, count, first_ts}
_seen_logical = set()
_pending_samples = {}  # group_index -> [(sample, verifier)] 单样本降级路径的 AUTO_FIX 覆写输入
# CAS index is process-local only as a fast path; the SQLite primary key is the
# cross-process source of truth.  Connections are recreated after fork and
# whenever RUN_DIR/RTX_CAS_INDEX_DIR changes (useful for offline gate tests).
_cas_conn = None
_cas_conn_path = None
_cas_conn_pid = None


def _fault_for(gidx):
    """多窗口优先, 否则单窗口语义"""
    if _WINDOWS:
        for f, (s, e) in _WINDOWS.items():
            if s <= gidx <= e:
                return f
        return "none"
    return FAULT if START <= gidx <= END else "none"


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


def _logical_id(rec: dict) -> str:
    """Return the stable CAS key, including for legacy records without the field."""
    if rec.get("logical_id") is not None:
        return str(rec["logical_id"])
    return f"{rec.get('group_index')}:{rec.get('index')}:{rec.get('rollout_id')}"


def _json_line(rec: dict) -> str:
    # Compact JSON matters for the high-frequency audit stream, especially when
    # response/label are retained for replay.
    return json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n"


def _append_json_lines(path: str, lines: list[str], lock: bool = False) -> None:
    """Append a batch with one open/lock instead of one syscall per record."""
    if not lines:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        locked = False
        try:
            if lock:
                try:
                    import fcntl
                    fcntl.flock(f, fcntl.LOCK_EX)
                    locked = True
                except ImportError:  # pragma: no cover - non-POSIX fallback
                    pass
            f.write("".join(lines))
            f.flush()
        finally:
            if locked:
                fcntl.flock(f, fcntl.LOCK_UN)


def _cas_index_path() -> str:
    root = os.environ.get("RTX_CAS_INDEX_DIR") or RUN_DIR
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "cas_index.sqlite3")


def _reset_cas_connection() -> None:
    global _cas_conn, _cas_conn_path, _cas_conn_pid
    if _cas_conn is not None:
        try:
            _cas_conn.close()
        except Exception:
            pass
    _cas_conn = None
    _cas_conn_path = None
    _cas_conn_pid = None


def _initialize_cas_index(conn: sqlite3.Connection) -> None:
    """Create/import the indexed CAS ledger once per run/index path.

    The one-time import keeps compatibility with old rewards.jsonl records that
    predate ``logical_id``.  Normal writes never scan rewards.jsonl again.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cas_claims ("
        "logical_id TEXT PRIMARY KEY, record_json TEXT NOT NULL, created_at REAL NOT NULL)"
    )
    conn.execute("CREATE TABLE IF NOT EXISTS cas_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    reward_path = os.path.join(RUN_DIR, "rewards.jsonl")
    current_size = os.path.getsize(reward_path) if os.path.exists(reward_path) else 0

    conn.execute("BEGIN IMMEDIATE")
    row = conn.execute("SELECT value FROM cas_meta WHERE key='rewards_size'").fetchone()
    previous_size = int(row[0]) if row else None
    # Rebuild if the log was truncated/replaced; rescan only on startup or
    # after an external legacy append, never once per reward.
    if previous_size is None or previous_size != current_size:
        if previous_size is not None and current_size < previous_size:
            conn.execute("DELETE FROM cas_claims")
        if os.path.exists(reward_path):
            with open(reward_path, errors="replace") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        if not isinstance(rec, dict):
                            continue
                        lid = _logical_id(rec)
                        if lid == "None:None:None":
                            continue
                        try:
                            created_at = float(rec.get("ts", time.time()))
                        except (TypeError, ValueError):
                            created_at = time.time()
                        conn.execute(
                            "INSERT OR IGNORE INTO cas_claims(logical_id, record_json, created_at) VALUES(?,?,?)",
                            (lid, _json_line(rec).rstrip("\n"), created_at),
                        )
                    except (ValueError, TypeError, sqlite3.Error):
                        continue
        conn.execute(
            "INSERT OR REPLACE INTO cas_meta(key, value) VALUES('rewards_size', ?)",
            (str(current_size),),
        )
    conn.commit()


def _get_cas_connection() -> sqlite3.Connection:
    global _cas_conn, _cas_conn_path, _cas_conn_pid
    path = _cas_index_path()
    pid = os.getpid()
    if _cas_conn is not None and _cas_conn_path == path and _cas_conn_pid == pid:
        return _cas_conn
    _reset_cas_connection()
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    # The JSONL stream remains the human-readable audit trail.  NORMAL avoids
    # an fsync for every tiny SQLite transaction while retaining transactions.
    conn.execute("PRAGMA synchronous=NORMAL")
    _initialize_cas_index(conn)
    _cas_conn, _cas_conn_path, _cas_conn_pid = conn, path, pid
    return conn


def _write_cas_rejects(items: list[tuple[dict, str]]) -> None:
    if not items:
        return
    lines = []
    for rec, lid in items:
        # A duplicate carries no new replay payload; retaining the full sample
        # here was a needless source of writes and disk growth.
        compact = {
            "logical_id": lid,
            "group_index": rec.get("group_index"),
            "index": rec.get("index"),
            "rollout_id": rec.get("rollout_id"),
            "reason": "duplicate-logical-id",
            "ts": time.time(),
        }
        lines.append(_json_line(compact))
    try:
        _append_json_lines(os.path.join(RUN_DIR, "cas_rejects.jsonl"), lines, lock=True)
    except Exception:
        pass


def _cas_write_many(records: list[dict]) -> list[bool]:
    """Indexed, cross-process CAS with one transaction per batch.

    The old implementation held a file lock while parsing the complete
    rewards.jsonl for every sample (O(N²) I/O).  SQLite's PRIMARY KEY makes the
    membership test O(log N), and group-RM callers can commit all eight samples
    in one transaction.
    """
    results = [False] * len(records)
    candidates = []
    duplicate_items = []
    batch_ids = set()
    for idx, rec in enumerate(records):
        lid = _logical_id(rec)
        prepared = {**rec, "logical_id": lid}
        if lid in _seen_logical or lid in batch_ids:
            duplicate_items.append((prepared, lid))
        else:
            batch_ids.add(lid)
            candidates.append((idx, prepared, lid))

    db_duplicates = []
    if candidates:
        conn = _get_cas_connection()
        inserted = []
        try:
            conn.execute("BEGIN IMMEDIATE")
            for idx, rec, lid in candidates:
                raw = _json_line(rec)
                cur = conn.execute(
                    "INSERT OR IGNORE INTO cas_claims(logical_id, record_json, created_at) VALUES(?,?,?)",
                    (lid, raw.rstrip("\n"), time.time()),
                )
                if cur.rowcount == 1:
                    inserted.append((idx, rec, lid, raw))
                else:
                    db_duplicates.append((rec, lid))
            if inserted:
                _append_json_lines(
                    os.path.join(RUN_DIR, "rewards.jsonl"),
                    [raw for _idx, _rec, _lid, raw in inserted],
                    lock=True,
                )
                conn.execute(
                    "INSERT OR REPLACE INTO cas_meta(key, value) VALUES('rewards_size', ?)",
                    (str(os.path.getsize(os.path.join(RUN_DIR, "rewards.jsonl"))),),
                )
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            finally:
                # If the append succeeded but commit failed, the next open
                # imports that tail once and prevents a duplicate retry.
                _reset_cas_connection()
            raise
        for idx, _rec, lid, _raw in inserted:
            results[idx] = True
            _seen_logical.add(lid)
        for rec, lid in db_duplicates:
            _seen_logical.add(lid)

    _write_cas_rejects(duplicate_items + db_duplicates)
    for rec, lid in duplicate_items:
        _seen_logical.add(lid)
    return results


def _cas_write(rec: dict) -> bool:
    """Reward CAS wrapper; membership is indexed, not a full-log scan."""
    return _cas_write_many([rec])[0]


def _seal_write(group_index: int, status: str, versions, autofix: bool = False, count: int = K):
    """写 seals.jsonl (组级模式与单样本模式共用)"""
    rec = {
        "group_index": group_index,
        "status": status,
        "versions": sorted(versions),
        "count": count,
        "autofix": autofix,
        "ts": time.time(),
    }
    try:
        _append_json_lines(os.path.join(RUN_DIR, "seals.jsonl"), [_json_line(rec)], lock=True)
    except Exception:
        pass
    return status, sorted(versions)


def _autofix_rewrite(group_index: int):
    """单样本降级路径: 组检测到 ABORTED 时, 用权威 v1 重算覆写 rewards.jsonl 记录
    (返回给训练器的值无法撤回, 但持久化数据流修正; --group-rm 组级模式为主路径)。
    追加 corrected 记录, Reconciler 读取以最后一条为准。"""
    pend = _pending_samples.get(group_index)
    if not pend:
        return 0
    fixed = 0
    lines = []
    for s, _ver in pend:
        rw = float(_v1_reward(s.response, s.label or ""))
        lid = f"{s.group_index}:{s.index}:{s.rollout_id}"
        lines.append(_json_line({
            "logical_id": lid, "corrected_reward": rw, "autofix": True,
            "group_index": s.group_index, "index": s.index,
            "rollout_id": s.rollout_id, "ts": time.time(),
        }))
        fixed += 1
    _append_json_lines(os.path.join(RUN_DIR, "rewards.jsonl"), lines, lock=True)
    return fixed


def _seal_register(group_index: int, verifier: str, logical_id: str, sample=None):
    """登记样本到组状态; 组完成时返回 (status, versions) 否则 None
    AUTO_FIX 单样本降级: 组完成且 ABORTED -> 权威重算覆写 rewards.jsonl"""
    st = _seal_state.setdefault(group_index, {"verifiers": set(), "ids": set(), "count": 0, "first_ts": time.time()})
    st["verifiers"].add(verifier)
    st["ids"].add(logical_id)
    st["count"] += 1
    if sample is not None:
        _pending_samples.setdefault(group_index, []).append((sample, verifier))
    if st["count"] < K:
        return None
    # 组完成 -> Seal 检查
    status = "SEALED" if len(st["verifiers"]) == 1 else "ABORTED"
    versions = sorted(st["verifiers"])
    autofix = False
    if status == "ABORTED" and AUTO_FIX:
        n = _autofix_rewrite(group_index)
        autofix = n > 0
    rec = {
        "group_index": group_index,
        "status": status,
        "versions": versions,
        "count": st["count"],
        "unique_samples": len(st["ids"]),
        "autofix": autofix,
        "ts": time.time(),
    }
    try:
        _append_json_lines(os.path.join(RUN_DIR, "seals.jsonl"), [_json_line(rec)], lock=True)
    except Exception:
        pass
    return status, versions


def _make_log_record(sample, reward: float, verifier: str, injected: bool, autofix: bool = False) -> dict:
    rec = {
        "group_index": sample.group_index,
        "index": sample.index,
        "rollout_id": sample.rollout_id,
        "reward": reward,
        "verifier": verifier,
        "injected": injected,
        "fault": FAULT,
        "seal_enabled": SEAL,
        "autofix": autofix,
        "ts": time.time(),
    }
    # Seal 模式下完整记录 response/label (Selective Replay 重算输入)
    if SEAL:
        rec["response"] = (sample.response or "")[:4000]
        rec["label"] = (sample.label or "")[:200]
    return rec


def _log(sample, reward: float, verifier: str, injected: bool, autofix: bool = False, register: bool = True):
    rec = _make_log_record(sample, reward, verifier, injected, autofix=autofix)
    lid = _logical_id(rec) if SEAL and register else None
    _cas_write(rec)
    if SEAL and register:
        _seal_register(sample.group_index if sample.group_index is not None else -1, verifier, lid, sample)


def _log_crash(sample, gidx):
    """崩溃前先落盘 rollout 数据 (模拟真实系统 rollout buffer 可复用)"""
    rec = {
        "group_index": sample.group_index, "index": sample.index,
        "rollout_id": sample.rollout_id, "reward": None,
        "verifier": "v1", "injected": True, "fault": "crm_crash",
        "seal_enabled": SEAL, "ts": time.time(),
    }
    if SEAL:
        rec["response"] = (sample.response or "")[:4000]
        rec["label"] = (sample.label or "")[:200]
    _cas_write(rec)


async def _rm_one(s, is_batch_sorted_pos=None):
    gidx = s.group_index if s.group_index is not None else -1
    fault = _fault_for(gidx)
    in_window = fault != "none"
    if fault == "crm_crash" and in_window:
        _log_crash(s, gidx)
        raise RuntimeError(f"injected RM crash (group_index={gidx})")
    r1 = float(_v1_reward(s.response, s.label or ""))
    r2 = float(_v2_reward(s.response, s.label or ""))
    half = K // 2
    pos = is_batch_sorted_pos if is_batch_sorted_pos is not None else ((s.index or 0) % K)
    if fault == "skew" and in_window:
        if pos < half:
            r, ver, inj = r1, "v1", False
        else:
            r, ver, inj = r2, "v2", True
    elif fault == "dup" and in_window:
        if pos < half:
            r, ver, inj = r1, "v1", False
        else:
            r, ver, inj = r2, "retry-stale", True
    else:
        r, ver, inj = r1, "v1", False
    _log(s, r, ver, inj)
    return r


def _compute_verdict(s, pos: int):
    """单样本版本判定: 返回 (reward, verifier, injected)"""
    gidx = s.group_index if s.group_index is not None else -1
    fault = _fault_for(gidx)
    in_window = fault != "none"
    r1 = float(_v1_reward(s.response, s.label or ""))
    r2 = float(_v2_reward(s.response, s.label or ""))
    half = K // 2
    if fault == "skew" and in_window:
        if pos < half:
            return r1, "v1", False
        return r2, "v2", True
    elif fault == "dup" and in_window:
        if pos < half:
            return r1, "v1", False
        return r2, "retry-stale", True
    else:
        return r1, "v1", False


async def _rm_batch_group(samples):
    """组级批调用 (--group-rm 主路径): 组完成时检测版本一致性;
    AUTO_FIX 时 ABORTED 组全部样本返回权威 v1 值 (训练器消费侧 0 混算)。"""
    results = [0.0] * len(samples)
    groups = {}
    for i, s in enumerate(samples):
        groups.setdefault(s.group_index if s.group_index is not None else -1, []).append(i)
    async with _lock:
        for gi, idxs in groups.items():
            idxs_sorted = sorted(idxs, key=lambda i: samples[i].index)
            recs = []
            for pos, i in enumerate(idxs_sorted):
                s = samples[i]
                fault = _fault_for(gi)
                if fault == "crm_crash":
                    _log_crash(s, gi)
                    raise RuntimeError(f"injected RM crash (group_index={gi})")
                r1 = float(_v1_reward(s.response, s.label or ""))
                r2 = float(_v2_reward(s.response, s.label or ""))
                p = (s.index or 0) % K
                r, ver, inj = r1, "v1", False
                if (fault == "skew" or fault == "dup") and fault != "none":
                    if p >= K // 2:
                        r, ver, inj = r2, ("v2" if fault == "skew" else "retry-stale"), True
                recs.append({"i": i, "s": s, "r": r, "r1": r1, "ver": ver, "inj": inj})
            vers = set(rec["ver"] for rec in recs)
            status = "SEALED" if len(vers) == 1 else "ABORTED"
            autofix = False
            if status == "ABORTED" and AUTO_FIX:
                for rec in recs:
                    rec["r"] = rec["r1"]  # v1 权威
                    rec["ver"] = "v1"
                    rec["autofix"] = True
                autofix = True
            # Group-RM already has all samples available; commit their audit
            # records in one indexed CAS transaction instead of opening,
            # locking, and committing once per sample.
            _cas_write_many([
                _make_log_record(
                    rec["s"], rec["r"], rec["ver"], rec["inj"],
                    autofix=rec.get("autofix", False),
                )
                for rec in recs
            ])
            if SEAL:
                _seal_write(gi, status, vers, autofix=autofix)
            for rec in recs:
                results[rec["i"]] = rec["r"]
    return results


async def rm_function(args, samples):
    if not isinstance(samples, list):
        return await _rm_one(samples)
    if GROUP_RM:
        return await _rm_batch_group(samples)
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
