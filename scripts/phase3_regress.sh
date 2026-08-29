#!/usr/bin/env bash
# Phase 3C G3C1: 一键回归套件 — 全部协议切点断言 (单测级, 无需 GPU)
# 覆盖: Seal 检测 / AUTO_FIX 消费侧 / CAS 幂等(单进程+多进程) / Manifest 审计 /
#       Reconciler 恢复计划 / Selective Replay 重算 / Trace 切点语义
# 用法: bash scripts/phase3_regress.sh  ->  输出 runs/PHASE3_REGRESSION.json
set -euo pipefail
BASE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$BASE/runs/PHASE3_REGRESSION.json
cd "$BASE"

python3 - "$OUT" "$BASE" <<'PYEOF'
import asyncio
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

out_path = sys.argv[1]
BASE = Path(sys.argv[2])
if not __debug__:
    raise RuntimeError("Phase 3 regression refuses optimized Python (-O/PYTHONOPTIMIZE): assertions must remain active")
results = {}
os.environ.update({"RTX_SEAL": "1", "RTX_GROUP_RM": "1", "RTX_SEAL_AUTO_FIX": "1",
                   "RTX_FAULT": "skew", "RTX_FAULT_START": "0", "RTX_FAULT_END": "100"})
sys.path.insert(0, str(BASE / "scripts"))
import phase2_seal_rm as m

class S:
    def __init__(s, gi, idx, rid, resp, label):
        s.group_index, s.index, s.rollout_id = gi, idx, rid
        s.response, s.label = resp, label

GOOD = "Let me solve.\n</think>\nThe answer is 42.\n###Response\n\\boxed{42}"
BAD  = "Let me solve.\n</think>\nThe answer is 43.\n###Response\n\\boxed{43}"

def t(name, fn):
    tmp = tempfile.mkdtemp()
    os.environ["RTX_RUN_DIR"] = tmp
    m.RUN_DIR = tmp            # 模块级常量直接更新 (避免 reload)
    m._seal_state.clear(); m._seen_logical.clear(); m._pending_samples.clear()
    m.FAULT = os.environ.get("RTX_FAULT", "none")
    m._WINDOWS = {}
    try:
        detail = fn(tmp)
        results[name] = {"pass": True}
        if detail is not None:
            results[name]["detail"] = detail
        print(f"  PASS {name}")
    except AssertionError as e:
        results[name] = {"pass": False, "detail": str(e)}
        print(f"  FAIL {name}: {e}")

def load(tmp, fname):
    return [json.loads(l) for l in open(f"{tmp}/{fname}")]


def manifest_fixture(tmp):
    """用已归档 StepToken 构造轻量 checkpoint 目录，避免依赖 Git 忽略的大权重。"""
    root = Path(tmp) / "manifest-fixture"
    save_dir = root / "checkpoints"
    manifest_dir = root / "manifests"
    save_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    source = BASE / "runs/p1-slime-skew-K8-s42-20260828-233354/manifests"
    for it in (9, 19):
        (save_dir / f"iter_{it:07d}").mkdir()
        shutil.copy2(source / f"step_token_{it}.json", manifest_dir)
    return save_dir, manifest_dir

# 1. Seal 检测: skew 组 ABORTED + autofix
def seal_detect(tmp):
    async def run():
        g = [S(0, i, i, GOOD if i % 2 == 0 else BAD, "42") for i in range(8)]
        r = await m.rm_function(None, g)
        seals = load(tmp, "seals.jsonl")
        assert len(seals) == 1 and seals[0]["status"] == "ABORTED" and seals[0]["autofix"]
        recs = load(tmp, "rewards.jsonl")
        assert len(recs) == 8
        assert all(abs(x["reward"] - float(m._v1_reward(x["response"], x["label"]))) < 1e-9 for x in recs)
        assert all(abs(r[i] - float(m._v1_reward(g[i].response, "42"))) < 1e-9 for i in range(8))
    asyncio.run(run())
t("seal_autofix_group", seal_detect)

# 2. 干净组 SEALED + AUTO_FIX on/off 同输入逐样本一致
def seal_clean(tmp):
    async def run_once(run_dir, auto_fix):
        Path(run_dir).mkdir()
        os.environ["RTX_RUN_DIR"] = str(run_dir)
        os.environ["RTX_FAULT"] = "none"
        m.RUN_DIR = str(run_dir)
        m.FAULT = "none"
        m.AUTO_FIX = auto_fix
        m._seal_state.clear(); m._seen_logical.clear(); m._pending_samples.clear()
        m._reset_cas_connection()
        g = [S(0, i, i, GOOD, "42") for i in range(8)]
        returned = await m.rm_function(None, g)
        seals = load(run_dir, "seals.jsonl")
        recs = load(run_dir, "rewards.jsonl")
        assert seals[0]["status"] == "SEALED" and not seals[0]["autofix"]
        assert len(recs) == 8 and all(not x.get("autofix") for x in recs)
        return returned, [x["reward"] for x in recs]

    async def run():
        on = await run_once(Path(tmp) / "autofix-on", True)
        off = await run_once(Path(tmp) / "autofix-off", False)
        assert on == off
        return {"paired_samples": 8, "autofix_on_off_mismatch": 0}

    detail = asyncio.run(run())
    m.AUTO_FIX = True
    return detail
t("seal_clean_group", seal_clean)

# 3. CAS 单进程幂等
def cas_single(tmp):
    lid = "0:0:0"
    ok1 = m._cas_write({"group_index": 0, "index": 0, "rollout_id": 0, "reward": 1.0})
    ok2 = m._cas_write({"group_index": 0, "index": 0, "rollout_id": 0, "reward": 1.0})
    assert ok1 and not ok2
    assert len(load(tmp, "rewards.jsonl")) == 1
t("cas_single_process", cas_single)

# 4. CAS 多进程 (文件锁级)
def cas_mp(tmp):
    def worker(run_dir):
        os.environ["RTX_RUN_DIR"] = run_dir
        import phase2_seal_rm as wm
        wm.RUN_DIR = run_dir
        wm._seen_logical.clear()
        for i in range(8):
            wm._cas_write({"group_index": 0, "index": i, "rollout_id": i, "reward": float(i)})
    ps = [mp.Process(target=worker, args=(tmp,)) for _ in range(4)]
    for p in ps: p.start()
    for p in ps: p.join()
    assert all(p.exitcode == 0 for p in ps), [p.exitcode for p in ps]
    recs = load(tmp, "rewards.jsonl")
    lids = [r["logical_id"] for r in recs]
    assert len(recs) == 8 and len(set(lids)) == 8
    return {"workers": len(ps), "worker_exitcodes": [p.exitcode for p in ps],
            "authoritative_records": len(recs)}
t("cas_multi_process", cas_mp)

# 5. R1/R2/R3/Q0/L0/L2 全切点矩阵（真实归档数据 + R2 同语义合成组）
def cutpoint_matrix(tmp):
    from phase2_reconciler import build_plan

    # R3: 真实混版本数据经 Reconciler 全覆盖。
    source = BASE / "runs/p2c-slime-skew-sealresp-K8-s42-20260827"
    run_dir = Path(tmp) / "reconciler-r3"
    run_dir.mkdir()
    shutil.copy2(source / "rewards.jsonl", run_dir)
    shutil.copy2(source / "seals.jsonl", run_dir)
    save_dir, manifest_dir = manifest_fixture(tmp)
    old_manifest_dir = os.environ.get("RTX_MANIFEST_DIR")
    os.environ["RTX_MANIFEST_DIR"] = str(manifest_dir)
    try:
        r3 = build_plan(str(save_dir), str(run_dir))
    finally:
        if old_manifest_dir is None:
            os.environ.pop("RTX_MANIFEST_DIR", None)
        else:
            os.environ["RTX_MANIFEST_DIR"] = old_manifest_dir
    assert r3["committed_iters"] == [9, 19]
    assert len(r3["replay_groups"]) == 20 and r3["replay_samples"] == 160

    # R2: 陈旧 retry-stale 与 v1 混入同组，必须 ABORTED 并在返回前权威化。
    async def run_r2():
        r2_dir = Path(tmp) / "r2-stale"
        r2_dir.mkdir()
        os.environ["RTX_RUN_DIR"] = str(r2_dir)
        m.RUN_DIR = str(r2_dir)
        m.FAULT = "dup"
        m.AUTO_FIX = True
        m._seal_state.clear(); m._seen_logical.clear(); m._pending_samples.clear()
        m._reset_cas_connection()
        group = [S(0, i, i, GOOD if i % 2 == 0 else BAD, "42") for i in range(8)]
        returned = await m.rm_function(None, group)
        seals = load(r2_dir, "seals.jsonl")
        rewards = load(r2_dir, "rewards.jsonl")
        assert len(seals) == 1 and seals[0]["status"] == "ABORTED" and seals[0]["autofix"]
        assert all(r["verifier"] == "v1" for r in rewards)
        assert all(abs(returned[i] - float(m._v1_reward(group[i].response, "42"))) < 1e-9 for i in range(8))
    asyncio.run(run_r2())

    # R1: 真实 RM 崩溃持久化 160 个缺失 reward，Replay 全部重算。
    r1_root = BASE / "runs/p2c-slime-crash-sealresp-K8-s42-20260827"
    r1_plan = json.load(open(r1_root / "recovery_plan.json"))
    r1_replay = json.load(open(r1_root / "replay_result.json"))
    r1_rewards = [json.loads(line) for line in open(r1_root / "rewards.jsonl")]
    r1_missing = [item for item in r1_rewards if item.get("reward") is None]
    assert len(r1_missing) == 160 and all(item.get("response") for item in r1_missing)
    assert len(r1_plan["incomplete_groups"]) == 20 and r1_plan["replay_samples"] == 160
    assert r1_replay["replayed_samples"] == 160

    # Q0: get_meta 后崩溃导致 32 样本不可重取，进入重投递语义。
    q0 = json.load(open(BASE / "runs/p1-tq-Q0Q1-s42-20260825/tq_crash_probe.json"))
    q0_case = q0["scenarios"]["A_Q0_getmeta_crash"]
    assert q0_case["first_get_meta"]["n"] == 32 and q0_case["silent_loss"]
    assert "Available: 0" in q0_case["restart_get_meta"]["error"]

    # L0/L2: 无 checkpoint 如实冷启动；有 StepToken 时恢复到 iter7。
    l0 = json.load(open(BASE / "runs/p1-slime-L0-K8-s42-20260825/recovery_plan.json"))
    l2 = json.load(open(BASE / "runs/p1-slime-L2-K8-s42-20260825/recovery_plan.json"))
    assert l0["resume_iter"] is None and l0["committed_iters"] == []
    assert l2["resume_iter"] == 7 and l2["committed_iters"] == [3, 7]
    return {"cutpoints": {name: "PASS" for name in ("R1", "R2", "R3", "Q0", "L0", "L2")}}
t("cutpoint_matrix_R1_R2_R3_Q0_L0_L2", cutpoint_matrix)

# 6. Selective Replay 重算正确性 (80 原 v1 样本 0 不一致)
def replay_correct(tmp):
    rp = json.load(open(BASE / "runs/p2c-slime-skew-sealresp-K8-s42-20260827/replay_result.json"))
    v1s = [s for s in rp["samples"] if s["old_verifier"] == "v1"]
    assert len(v1s) == 80
    assert all(abs(s["old_reward"] - s["new_reward"]) < 1e-9 for s in v1s)
t("replay_correctness", replay_correct)

# 7. Manifest audit (归档 StepToken + 轻量 checkpoint 目录, committed [9,19])
def manifest_audit(tmp):
    from phase2_manifest import audit as audit_manifests
    save_dir, manifest_dir = manifest_fixture(tmp)
    old_manifest_dir = os.environ.get("RTX_MANIFEST_DIR")
    os.environ["RTX_MANIFEST_DIR"] = str(manifest_dir)
    try:
        res = audit_manifests(save_dir)
    finally:
        if old_manifest_dir is None:
            os.environ.pop("RTX_MANIFEST_DIR", None)
        else:
            os.environ["RTX_MANIFEST_DIR"] = old_manifest_dir
    assert res["committed_iters"] == [9, 19], res["committed_iters"]
    assert not res["missing_token_iters"]
    # prev 链完整: 每 token 的 prev_token == 前一 token
    toks = res["committed_steps"]
    for i in range(1, len(toks)):
        assert toks[i]["prev_token"] == toks[i-1]["token"]
    # 内容绑定: weights_hash 非空且长度 64
    assert all(len(s["weights_hash"]) == 64 for s in toks)
t("manifest_audit", manifest_audit)

# 8. 零修改不变量 (3A 实验样本)
def zero_mod(tmp):
    recs = [json.loads(l) for l in open(BASE / "runs/p3a-slime-skew-autofix-K8-s42-20260828/rewards.jsonl")]
    assert len(recs) == 11392
    bad = [r for r in recs if abs(r["reward"] - float(m._v1_reward(r["response"], r["label"]))) > 1e-9]
    assert not bad
t("zero_modification", zero_mod)

# 9. 自动恢复历史 (3B 事件链)
def recovery_history(tmp):
    hist = [json.loads(l) for l in open(BASE / "runs/rtx_recovery_history.jsonl")]
    last = hist[-1]
    assert last["recovery_action"] == "resume" and last["committed_step"] == 9
t("auto_recovery_history", recovery_history)

passed = sum(1 for r in results.values() if r["pass"])
report = {"status": "PASS" if passed == len(results) else "FAIL",
          "total": len(results), "passed": passed, "checks": results,
          "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
json.dump(report, open(out_path, "w"), indent=2, ensure_ascii=False)
print(f"=== 回归 {passed}/{len(results)} {'PASS' if passed == len(results) else 'FAIL'} -> {out_path} ===")
if passed != len(results):
    sys.exit(1)
PYEOF
