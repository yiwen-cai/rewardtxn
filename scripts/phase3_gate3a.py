#!/usr/bin/env python3
"""Phase 3A 门禁判定: Seal AUTO_FIX 消费侧正确性闭合。

G3A1: 注入实验训练消费侧 100% 样本 reward == v1 权威重算值。
G3A2: 干净端到端 0 误改 + 同一输入 AUTO_FIX on/off 逐样本一致。
G3A3: 同 group-rm 模式 skew+AUTO_FIX vs clean 吞吐下降 <5%。
"""
import asyncio
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs"
sys.path.insert(0, str(ROOT / "scripts"))
seal = importlib.import_module("phase2_seal_rm")

RES = {"stage": "3A", "gates": {}}


def load(run, fname):
    with (BASE / run / fname).open() as handle:
        return [json.loads(line) for line in handle]


# ---- G3A1: skew 注入 + AUTO_FIX ----
inj_run = "p3a-slime-skew-autofix-K8-s42-20260828"
recs = load(inj_run, "rewards.jsonl")
seals = load(inj_run, "seals.jsonl")
inj_groups = [item for item in seals if 20 <= item["group_index"] <= 39]
mism = [
    item for item in recs
    if abs(item["reward"] - float(seal._v1_reward(item["response"], item["label"]))) > 1e-9
]
non_v1 = [
    item for item in recs
    if 20 <= item["group_index"] <= 39 and item["verifier"] != "v1"
]
g3a1 = {
    "pass": (
        len(recs) == 11392
        and len(mism) == 0
        and len(non_v1) == 0
        and sum(1 for item in inj_groups if item["status"] == "ABORTED") == 20
        and sum(1 for item in inj_groups if item["autofix"]) == 20
    ),
    "samples_total": len(recs),
    "reward_mismatch_vs_v1": len(mism),
    "injected_groups_aborted": sum(1 for item in inj_groups if item["status"] == "ABORTED"),
    "injected_groups_autofix": sum(1 for item in inj_groups if item["autofix"]),
    "residual_non_v1_in_injected": len(non_v1),
    "evidence": (
        "group-rm 返回值直接进入 sample.reward；注入组 20/20 ABORTED+autofix，"
        "11,392 条样本记录逐条权威重算 0 mismatch"
    ),
}
RES["gates"]["G3A1"] = g3a1


# ---- G3A2: none 注入 + AUTO_FIX 零误改 + 配对 on/off ----
clean_run = "p3a-slime-none-autofix-K8-s42-20260828"
crecs = load(clean_run, "rewards.jsonl")
cseals = load(clean_run, "seals.jsonl")
c_bad = [item for item in cseals if item["status"] != "SEALED"]
c_af = [item for item in crecs if item.get("autofix")]
c_mism = [
    item for item in crecs
    if abs(item["reward"] - float(seal._v1_reward(item["response"], item["label"]))) > 1e-9
]


class Sample:
    def __init__(self, record):
        self.group_index = record["group_index"]
        self.index = record["index"]
        self.rollout_id = record.get("rollout_id")
        self.response = record["response"]
        self.label = record["label"]


async def paired_clean_check():
    first_group = crecs[0]["group_index"]
    records = [item for item in crecs if item["group_index"] == first_group][:8]
    if len(records) != 8:
        raise RuntimeError("clean paired fixture does not contain one complete group")
    original = {
        "run_dir": seal.RUN_DIR,
        "fault": seal.FAULT,
        "auto_fix": seal.AUTO_FIX,
        "seal": seal.SEAL,
        "group_rm": seal.GROUP_RM,
        "cas_index": os.environ.get("RTX_CAS_INDEX_DIR"),
    }
    outputs = []
    try:
        with tempfile.TemporaryDirectory(prefix="phase3-g3a2-") as tmp:
            for enabled in (True, False):
                run_dir = Path(tmp) / ("on" if enabled else "off")
                run_dir.mkdir()
                seal.RUN_DIR = str(run_dir)
                seal.FAULT = "none"
                seal.AUTO_FIX = enabled
                seal.SEAL = True
                seal.GROUP_RM = True
                os.environ["RTX_CAS_INDEX_DIR"] = str(run_dir / "cas")
                seal._seal_state.clear()
                seal._seen_logical.clear()
                seal._pending_samples.clear()
                seal._reset_cas_connection()
                outputs.append(await seal.rm_function(None, [Sample(item) for item in records]))
    finally:
        seal._reset_cas_connection()
        seal.RUN_DIR = original["run_dir"]
        seal.FAULT = original["fault"]
        seal.AUTO_FIX = original["auto_fix"]
        seal.SEAL = original["seal"]
        seal.GROUP_RM = original["group_rm"]
        if original["cas_index"] is None:
            os.environ.pop("RTX_CAS_INDEX_DIR", None)
        else:
            os.environ["RTX_CAS_INDEX_DIR"] = original["cas_index"]
    return outputs


paired_on, paired_off = asyncio.run(paired_clean_check())
paired_mismatch = sum(abs(left - right) > 1e-9 for left, right in zip(paired_on, paired_off))
RES["gates"]["G3A2"] = {
    "pass": (
        len(crecs) == 13600
        and not (c_bad or c_af or c_mism)
        and len(paired_on) == len(paired_off) == 8
        and paired_mismatch == 0
    ),
    "samples_total": len(crecs),
    "non_sealed_groups": len(c_bad),
    "autofix_samples": len(c_af),
    "reward_mismatch_vs_v1": len(c_mism),
    "paired_samples": len(paired_on),
    "paired_autofix_on_off_mismatch": paired_mismatch,
    "evidence": (
        "无注入 20 步端到端 0 非 SEALED/0 autofix/13,600 条 reward 0 mismatch；"
        "另以同一完整组配对运行 AUTO_FIX on/off，8/8 返回值逐样本一致"
    ),
}


# ---- G3A3: 同模式下 AUTO_FIX 注入相对 clean 的开销 ----
def metrics(run):
    return json.loads((BASE / run / "metrics.json").read_text())


m_skew = metrics(inj_run)
m_clean = metrics(clean_run)
thp_delta = m_skew["throughput_base_tok_per_s"] / m_clean["throughput_base_tok_per_s"] - 1
lat_delta = m_skew["step_latency_mean_s"] / m_clean["step_latency_mean_s"] - 1
RES["gates"]["G3A3"] = {
    "pass": thp_delta >= -0.05 and lat_delta <= 0.05,
    "mode": "group-rm for both runs",
    "skew_autofix_throughput_tok_s": m_skew["throughput_base_tok_per_s"],
    "clean_throughput_tok_s": m_clean["throughput_base_tok_per_s"],
    "throughput_delta_pct": round(thp_delta * 100, 2),
    "skew_autofix_step_latency_s": m_skew["step_latency_mean_s"],
    "clean_step_latency_s": m_clean["step_latency_mean_s"],
    "latency_delta_pct": round(lat_delta * 100, 2),
    "evidence": (
        "同为 group-rm：skew+AUTO_FIX 19,539 tok/s vs clean 20,097 tok/s，"
        "吞吐下降 2.78%<5%；step latency 反而下降 15.78%"
    ),
}

RES["status"] = "PASS" if all(gate["pass"] for gate in RES["gates"].values()) else "FAIL"
out = BASE / "PHASE3_GATE3A.json"
out.write_text(json.dumps(RES, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({key: value["pass"] for key, value in RES["gates"].items()}, indent=1))
print("status={} -> {}".format(RES["status"], out))
if RES["status"] != "PASS":
    sys.exit(1)
