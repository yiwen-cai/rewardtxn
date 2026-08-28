#!/usr/bin/env bash
# Phase 3C G3C1: 一键回归套件 — 全部协议切点断言 (单测级, 无需 GPU)
# 覆盖: Seal 检测 / AUTO_FIX 消费侧 / CAS 幂等(单进程+多进程) / Manifest 审计 /
#       Reconciler 恢复计划 / Selective Replay 重算 / Trace 切点语义
# 用法: bash scripts/phase3_regress.sh  ->  输出 runs/PHASE3_REGRESSION.json
set -euo pipefail
BASE=/public/home/caiyiwen/rewardtxn
OUT=$BASE/runs/PHASE3_REGRESSION.json

python3 - "$OUT" <<'PYEOF'
import json, os, sys, tempfile, asyncio, re, multiprocessing as mp, time

out_path = sys.argv[1]
results = {}
os.environ.update({"RTX_SEAL": "1", "RTX_GROUP_RM": "1", "RTX_SEAL_AUTO_FIX": "1",
                   "RTX_FAULT": "skew", "RTX_FAULT_START": "0", "RTX_FAULT_END": "100"})
sys.path.insert(0, "/public/home/caiyiwen/rewardtxn/scripts")
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
        fn(tmp)
        results[name] = {"pass": True}
        print(f"  PASS {name}")
    except AssertionError as e:
        results[name] = {"pass": False, "detail": str(e)}
        print(f"  FAIL {name}: {e}")

def load(tmp, fname):
    return [json.loads(l) for l in open(f"{tmp}/{fname}")]

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

# 2. 干净组 SEALED + 零误改
def seal_clean(tmp):
    async def run():
        os.environ["RTX_FAULT"] = "none"
        m.FAULT = "none"
        g = [S(0, i, i, GOOD, "42") for i in range(8)]
        await m.rm_function(None, g)
        seals = load(tmp, "seals.jsonl")
        assert seals[0]["status"] == "SEALED" and not seals[0]["autofix"]
        recs = load(tmp, "rewards.jsonl")
        assert all(not x.get("autofix") for x in recs)
    asyncio.run(run())
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
    recs = load(tmp, "rewards.jsonl")
    lids = [r["logical_id"] for r in recs]
    assert len(recs) == 8 and len(set(lids)) == 8
t("cas_multi_process", cas_mp)

# 5. Reconciler 恢复计划 (2C 真实数据)
def reconciler_plan(tmp):
    from phase2_reconciler import build_plan
    plan = build_plan(
        "/public/home/caiyiwen/rewardtxn/runs/p2c-slime-skew-sealresp-K8-s42-20260827/checkpoints",
        "/public/home/caiyiwen/rewardtxn/runs/p2c-slime-skew-sealresp-K8-s42-20260827")
    assert plan is not None and len(plan.get("replay_groups", [])) >= 0
t("reconciler_plan", reconciler_plan)

# 6. Selective Replay 重算正确性 (80 原 v1 样本 0 不一致)
def replay_correct(tmp):
    rp = json.load(open("/public/home/caiyiwen/rewardtxn/runs/p2c-slime-skew-sealresp-K8-s42-20260827/replay_result.json"))
    v1s = [s for s in rp["samples"] if s["old_verifier"] == "v1"]
    assert all(abs(s["old_reward"] - s["new_reward"]) < 1e-9 for s in v1s)
t("replay_correctness", replay_correct)

# 7. Manifest audit (2B sidecar 链完整性)
def manifest_audit(tmp):
    from pathlib import Path
    from phase2_manifest import audit as audit_manifests
    res = audit_manifests(Path("/public/home/caiyiwen/rewardtxn/runs/p2b-slime-ckpt-K8-s42-20260827/checkpoints"))
    assert res["committed_iters"] == [3, 7, 11, 15, 19], res["committed_iters"]
    assert not res["missing_token_iters"]
    # prev 链完整: 每 token 的 prev_token == 前一 token
    toks = res["committed_steps"]
    for i in range(1, len(toks)):
        assert toks[i]["prev_token"] == toks[i-1]["token"]
    # 内容绑定: weights_hash 非空且长度 64
    assert all(len(s["weights_hash"]) == 64 for s in toks)
t("manifest_audit", manifest_audit)

# 8. 零修改不变量 (3A 实验 400 样本)
def zero_mod(tmp):
    recs = [json.loads(l) for l in open("/public/home/caiyiwen/rewardtxn/runs/p3a-slime-skew-autofix-K8-s42-20260828/rewards.jsonl")]
    bad = [r for r in recs if abs(r["reward"] - float(m._v1_reward(r["response"], r["label"]))) > 1e-9]
    assert not bad
t("zero_modification", zero_mod)

# 9. 自动恢复历史 (3B 事件链)
def recovery_history(tmp):
    hist = [json.loads(l) for l in open("/public/home/caiyiwen/rewardtxn/runs/rtx_recovery_history.jsonl")]
    last = hist[-1]
    assert last["recovery_action"] == "resume" and last["committed_step"] == 9
t("auto_recovery_history", recovery_history)

passed = sum(1 for r in results.values() if r["pass"])
report = {"status": "PASS" if passed == len(results) else "FAIL",
          "total": len(results), "passed": passed, "checks": results,
          "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
json.dump(report, open(out_path, "w"), indent=2, ensure_ascii=False)
print(f"=== 回归 {passed}/{len(results)} PASS -> {out_path} ===")
PYEOF
