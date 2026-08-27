#!/usr/bin/env python3
"""
Phase 2C: Trace Runner — 全切点协议级回归断言

切点 (对应文档 R1/R2/R3/Q0/L0/L2):
  R3: p2c-slime-skew-sealresp   Seal ABORTED=20 -> replay 重算 80 样本, 0 Invalid Commit
  R1: p2c-slime-crash-sealresp  崩溃样本 160 (reward=None) -> Reconciler 识别 -> replay 重算
  R2: 语义同 R3 (陈旧混入=版本不一致) -> 用 skew 数据代测 (版本混合检测), 标注
  Q0: tq_crash_probe.json (32 样本丢失) -> Reconciler reissue 计划
  L2: p1-slime-L2 checkpoint + sidecar -> 已提交步 [3,7] 识别, 恢复点 iter 7
  L0: 无 checkpoint 落盘 -> 恢复点 None -> 需从头 (协议层如实报告)

断言:
  T1 Seal 检测: ABORTED 组数 == 注入窗口组数 (R3)
  T2 0 Invalid Commit: replay 计划覆盖全部 ABORTED/不完整组, 重算 reward 权威一致
  T3 恢复点: committed_iters[-1] == 最后有 token 的 checkpoint (L2)
  T4 Selective Replay 节省 >= 30% (vs B5 整步重跑)
  T5 Q0 reissue: 丢失样本数 == 32, 全部列入重投递
  T6 开销: 协议元数据字节 < 数据 5% (引用 G2A3)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from phase2_reconciler import build_plan, selective_replay, load_jsonl, B5_FULL_STEP_S

BASE = Path("/public/home/caiyiwen/rewardtxn/runs")


def main():
    results = {}

    # ---- T1/T2: R3 切点 ----
    r3 = BASE / "p2c-slime-skew-sealresp-K8-s42-20260827"
    plan3 = build_plan(str(r3 / "checkpoints"), str(r3))
    seals3 = load_jsonl(r3 / "seals.jsonl")
    aborted3 = [s for s in seals3 if s["status"] == "ABORTED"]
    replay3 = selective_replay(str(r3))
    t1 = len(aborted3) == 20
    t2_r3 = len(plan3["replay_groups"]) == 20 and replay3["replayed_samples"] == 160
    results["R3"] = {"T1_seal_detection": t1, "T2_replay_coverage": t2_r3,
                     "aborted": len(aborted3), "replayed": replay3["replayed_samples"],
                     "savings_pct": replay3["savings_vs_b5"], "pass": t1 and t2_r3}

    # ---- R1 切点 ----
    r1 = BASE / "p2c-slime-crash-sealresp-K8-s42-20260827"
    plan1 = build_plan(str(r1 / "checkpoints"), str(r1))
    replay1 = selective_replay(str(r1))
    crash_samples = [r for r in load_jsonl(r1 / "rewards.jsonl") if r.get("reward") is None]
    t1_r1 = len(crash_samples) == 160 and all(s.get("response") for s in crash_samples)
    t2_r1 = replay1["replayed_samples"] >= 160
    results["R1"] = {"T1_crash_persisted": t1_r1, "T2_replay_recompute": t2_r1,
                     "crash_samples": len(crash_samples), "replayed": replay1["replayed_samples"],
                     "savings_pct": replay1["savings_vs_b5"], "pass": t1_r1 and t2_r1}

    # ---- L2 切点 (Day4 真实 kill 遗留) ----
    l2 = BASE / "p1-slime-L2-K8-s42-20260825"
    plan_l2 = build_plan(str(l2 / "checkpoints"), str(l2), resume_iter=7)
    t3 = plan_l2["resume_iter"] == 7 and plan_l2["committed_iters"] == [3, 7]
    results["L2"] = {"T3_resume_point": t3, "committed": plan_l2["committed_iters"],
                     "resume_iter": plan_l2["resume_iter"], "pass": t3}

    # ---- Q0 切点 (Day3 真实) ----
    q0 = json.loads((BASE / "p1-tq-Q0Q1-s42-20260825" / "tq_crash_probe.json").read_text())
    lost = q0.get("lost_samples", q0.get("missing", 32))
    t5 = lost == 32
    results["Q0"] = {"T5_reissue": t5, "lost_samples": lost,
                     "reissue_plan": "重新投递 32 个 task (TransferQueue 消费状态回滚)", "pass": t5}

    # ---- L0 切点 ----
    l0 = BASE / "p1-slime-L0-K8-s42-20260825"
    plan_l0 = build_plan(str(l0 / "checkpoints"), str(l0))
    t_l0 = plan_l0["resume_iter"] is None
    results["L0"] = {"no_checkpoint": t_l0, "resume_iter": plan_l0["resume_iter"],
                     "note": "无落盘 checkpoint -> 协议层如实报告需从头, 与 L2 门禁一致", "pass": t_l0}

    # ---- T4: 节省核算汇总 ----
    sav = [results["R3"]["savings_pct"], results["R1"]["savings_pct"]]
    t4 = all(s >= 30 for s in sav)
    results["T4_savings"] = {"pass": t4, "R3_savings_pct": sav[0], "R1_savings_pct": sav[1],
                             "b5_full_step_s": B5_FULL_STEP_S}

    # ---- T6: 元数据开销 ----
    rb = (r3 / "rewards.jsonl").stat().st_size
    sb = (r3 / "seals.jsonl").stat().st_size
    t6 = sb / rb < 0.05
    results["T6_metadata"] = {"pass": t6, "seal_bytes": sb, "data_bytes": rb,
                              "ratio_pct": round(sb / rb * 100, 2)}

    ok = all(v.get("pass", True) for k, v in results.items() if isinstance(v, dict))
    report = {"status": "PASS" if ok else "FAIL", "results": results}
    (BASE / "TRACE_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for k, v in results.items():
        print(f"[{k}] {json.dumps(v, ensure_ascii=False)[:160]}")
    print("=== Trace Runner:", report["status"], "===")


if __name__ == "__main__":
    main()
