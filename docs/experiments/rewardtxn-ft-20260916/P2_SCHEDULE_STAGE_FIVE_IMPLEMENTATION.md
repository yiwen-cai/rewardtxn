# P2 第五段：F4 CPU 事件模型

2026-09-20。economy-dev 根明确enabled，Astra medium，无递归。仅driver、两份旧测试的支持集计数/原unsupported测试改为仍pending格、新增`test_p2_schedule_f4.py`及实施文档；生产state/oracle、manifest、P1、第三方和原证据不变，审计schema不扩。

## 已有验收与本次实现分账

第四段主验收原25项/7.646秒、scope6项/3.049秒，representative-r5 12/12、stage-four-r1 80/80（10owner races）通过，去重330合同passed，78未跑、8限制、184pending；见`p2_evidence/stage-four-verification-r1.json`。它尚无新X2 schema事后审计。

本段STAGE_FIVE=F4十boundary × i00/i01/i02/i03/i04/i05/i08/i09=80格，与原330重合F4.b00.i00和F4.b06.i03；原两个分支保留。去重支持408，静态408可执行上限全部实现不等于已全运行，更不代表408不同实际状态。功能合同、实际签名数、GPU可达性继续分账。

## 有限boundary实现

| b | 真实CPU文件/API与规范事件 | O primary |
|---|---|---|
| 00 | 8唯一生成文件，4个reward_start规范事件，sample3 in-flight模型 | correct_recovered模型 |
| 01 | 8文件，4start，实际fixture-exact计算4个reward_end，无accept | correct_recovered模型 |
| 02 | 7唯一文件、4start、7个真实授权/接受，prepare拒绝K不全 | technical_invalid_unmet_window |
| 03 | 同02，额外实际写sample6重复文件及重复生成事件；仍7 unique | technical_invalid_unmet_window |
| 04 | 8文件，sample0 start；无结束/接受 | correct_recovered模型 |
| 05 | 实际计算sample0 reward_end，记录before_accept，未接受 | correct_recovered模型 |
| 06 | 7合法结果接受，第8声明exact-v2真实拒绝 | invalid_commit_if_retained |
| 07 | 8文件，声明worker_lost(samples2/3)规范事件 | correct_recovered模型 |
| 08 | 保存原response全hash，真实CPU重算8个fixture reward，前后字节不变，1次retry模型 | correct_recovered模型 |
| 09 | 8文件和8个真实授权/接受，确认completed全8且executing=None | technical_invalid_missed_window |

文件生成是固定CPU载荷，不是LLM生成。start/in-flight/lost/retry为规范marker。b07没有杀死reward进程，b08不是AReaL native retry，任何一格不声称自然F4 GPU窗口命中。b01/04/05等只覆盖其声明的局部事件/接纳合同，不把规范图correct_recovered解释成实际GPU故障恢复。

每格在任何Popen启动前写入`f4-window-contract.json`：要求8个唯一生成、存在executing sample、reward未全部完成，固定b02/3“unique不足”和b09“全完成错过”分类。执行后从实际收据复核唯一数、重复逻辑sample/字节、start/end序列、真实评分、accepted集合及准确拒绝。O从独立外部输入和预定规范构图；technical_invalid事件及合同hash一起封存，不根据方法是否获利反推无效。图时钟为模拟值，不能作为实测RTO。

b02/3的O generation事件分别7/8条但unique都7，重复仍是sample6；技术无效primary不会替代安全反例。健康K8图独立保存，另删其sample构造invalid_commit反例。b09则保存全部完成模型并technical_invalid。primary文件在modifier执行前保存。i08总是从完整健康图副本移除bootstrap optimizer_start，必须深入到`missing optimizer_start`的unverifiable，而不是让technical_invalid提前返回。i09沿用先健康control、实际checkpoint字节损坏和新owner fullhash拒绝。

其余modifier沿用已有真实实现：i01实际生成顺序；i02幂等证据；i03/4 probe旧attempt早晚；i05两个真实进程同run争锁、winner继续。每格独立目录，失败保留。没有要求SIGKILL的F4 CPU模型不补造信号；P不新增reward worker。

## 测试和交回入口

新增6项：支持集合等于静态executable；b02/3 unique及重复sample反例；b09预定分类和primary/健康/深层缺证据分离；lost/retry范围与实际保存字节；实际评分与before-admission；非legacy in-flight/verifier拒绝。checker负例修改重复sample为7、reward改错、删除retry计算、伪造executing，必须拒绝。

worker定向`F4.b03.i08`、`F4.b09.i08`通过，证据`/tmp/rtx-f4-target-r1`；lost/retry和非legacy两项测试通过（1.001秒）。未跑完整6项/80矩阵，交主任务验收。

```sh
.venv-tq/bin/python3.11 tests/ft/test_p2_schedule_driver.py -v
.venv-tq/bin/python3.11 tests/ft/test_p2_schedule_scope.py -v
.venv-tq/bin/python3.11 tests/ft/test_p2_schedule_f4.py -v
.venv-tq/bin/python3.11 tests/ft/p2_schedule_driver.py --representative \
  --output docs/experiments/rewardtxn-ft-20260916/p2_evidence/representative-r6
.venv-tq/bin/python3.11 - <<'PY'
import subprocess
args = ['.venv-tq/bin/python3.11', 'tests/ft/p2_schedule_driver.py',
        '--output', 'docs/experiments/rewardtxn-ft-20260916/p2_evidence/stage-five-r1',
        '--case-timeout', '30', '--total-timeout', '600']
for b in range(10):
    for i in (0, 1, 2, 3, 4, 5, 8, 9):
        args += ['--case', f'F4.b{b:02d}.i{i:02d}']
subprocess.run(args, check=True)
PY
```

完整验收应80/80合同通过、0harness_failure/failed、全部must_reach实证；其中24格primary为technical_invalid，是正确执行预定无效窗口合同，不计有效F4故障恢复cell。case-timeout/total-timeout沿用单调时钟，cleanup限自有Popen。此前88/96/72/80集合可按原入口回归。后续独立有界单元才扩X2/F4审计schema并审计最新全批，不伪称当前408都已签名验证；8限制/184pending保持不变。

## 主任务测试验收（全矩阵另记）

主任务已确认原driver25项/7.909秒、scope6项/3.043秒、F4新6项/3.196秒全部通过。随后启动当前版本按cell串行408格，输出`p2_evidence/full-408-r1-F1`、`F2`、`F3`、`F4`、`X1`、`X2`。本段记录测试事实，不在矩阵验收报告落盘前推断全408结果；最终实证以主任务报告为准。
