# P2首段扩展：F1全边界与X1正例

2026-09-20。仅实现 [扩展计划](P2_SCHEDULE_EXPANSION_PLAN.md) 第7节第一项；未改生产state/oracle、P1、第三方。economy-dev根status enabled，Astra medium实施，无递归。本轮没有运行完整矩阵或GPU。

## 清单修订及历史来源

v1完整compiler、静态测试、manifest已在任何修改前归档于 [schedule-v1](p2_evidence/schedule-v1)，全hash/文件大小见 [hashes.json](p2_evidence/schedule-v1/hashes.json)。v2仅为X1.b00的10格补显式optimizer start/end/成功收据→双rank11类文件及finalize→commit，期望仍committed_target；i06/i07仍pending。全局schema仍1，新增schedule_revision=2。

[变化账本](p2_evidence/schedule-v2-changes.json)列10格前后semantic hash、操作数量及当前compiler/test/manifest全hash。其余590格逐case对象不变；分类仍408可执行上限、8限制、184待补。r1的12格验收继续归属于旧源hash，不冒充新修订全部通过。

新增独立`validate_completion`按正例的实际operation序列检查prepare/commit顺序、optimizer成功收据、两rank组件及finalize；不靠canonical相等判断提交前缀完整。手写prepare-only负例保持committed_target，必须具体拒绝；额外只有commit仍缺optimizer时也拒绝。该静态检查不证明真实保存，因此driver另核对实际目标token/commit收据、optimizer/rank收据和文件全hash。

## 支持范围及有限组合

`STAGE_ONE`精确88格：F1的10个boundary×8个交错，X1.b00×8个交错；交错为00/01/02/03/04/05/08/09。`SUPPORTED`是它们与旧12格的并集，共98格。`--representative`仍只运行旧12格，不暗增分母；其他未实现格仍not_executed。

只局部增加F1和X1正例方法、已提交收据幂等函数及父进程细分检查；不解释manifest任意call，不加DSL。现有SIGKILL、owner竞争、旧cell边界路径保留。

- F1.b00…06真实生成/接收1…7项，prepare拒绝不完整K；b09生成8项仅接收7，具体拒绝。O对应safe_drop及错误保留K反例。
- F1.b07实际从方法sample7删除tokens后送摘要接收/prepare；S不声称校验原文语义，O独立同字段缺失必须unverifiable且指向tokens。b08仅在独立O原文删除logprobs，S保留完整输入。两格均先有完整健康对照，缺证据负例通过不等于恢复成功。
- X1.b00真实计算fixture-exact并记录计算，完整K授权/接收/prepare/optimizer/组件/finalize/commit。父进程检查目标token实际存在及内容、提交收据、每rank每组件和fullhash，避免只得到bootstrap token或prepared就声称committed。
- i01从冻结manifest的实际目标生成顺序构造，目标只有1项才倒序bootstrap；父进程从JSONL检查两者真实顺序。最终消费/tensor仍按sample索引排列。
- i02真实向已提交g0重投完全相同optimizer evidence两次，比较返回相同，整个generation文件映射/fullhash前后不变；不改token后证据合同。
- i03/i04父进程核对probe new→old与old→new实际顺序；有目标prepare时检查probe仍在data.pending/regenerate中。
- i05继续现有真实双进程争锁、winner同run继续。i09继续无损对照后篡改g0组件、新合法进程select具体拒绝；不把关闭owner拒绝当损坏拒绝。
- i08现在统一在独立完整健康图上删除bootstrap optimizer_start；primary报告另存。这样原始缺tokens/logprobs不会遮住修改器检查，也不将它们偷偷补成完整数据。

共享external-fixture在进程启动前冻结，S/O各自读取；新增target_arrival_order。预定方法载荷变异有before/after fullhash及字段收据，父进程仅允许F1.b07/sample7删除tokens这一精确差异，其余原文逐项相等。没有关闭全局输入比较来迁就负例。

## 针对测试与交付验收

静态测试现14项、driver测试现13项。新增实际正例运行后删除目标token、隐藏optimizer/rank收据的针对负例，父进程深层检查均不得通过；缺completion suffix的case在启动前具体拒绝。测试不依靠从state输出生成oracle expected。

本worker仅执行必要定向检查：

- 两个静态测试：手写缺suffix拒绝、v2只变10格，0.481秒通过。
- X1.b00.i02真实提交/收据幂等与缺目标token/optimizer/rank证据反例通过；最终代码修订后该测试再次通过，0.299秒。
- F1.b07.i08实际缺tokens与独立删start测试通过；与初次X1正例测试合计0.561秒。
- F1.b03.i01目标实际逆序及缺suffix启动前拒绝两个测试通过，0.251秒。
- 单格F1.b08.i09通过，原始输出在 `/tmp/rtx-p2-stage-one-missing-logprobs-r1`；没有运行88格或旧12格完整批次。

本轮未发现需将首段88格降为pending的新组合，但**完整支持声明待主任务验证**。全局完成账本不能仅凭本实现更新为88 passed。建议主任务依次运行：

```sh
.venv-tq/bin/python3.11 tests/ft/test_p2_schedules.py -v
.venv-tq/bin/python3.11 tests/ft/test_p2_schedule_driver.py -v
.venv-tq/bin/python3.11 tests/ft/p2_schedule_driver.py --representative \
  --output /absolute/new-representative-v2 --case-timeout 30 --total-timeout 420
```

首段88格用已有重复`--case`入口，直接脚本避免tests包冲突：

```sh
.venv-tq/bin/python3.11 - <<'PY'
import json, subprocess, sys
from pathlib import Path
manifest = json.loads(Path('tests/ft/fixtures/p2_schedules.json').read_text())
selected = [case['id'] for case in manifest['cases']
            if case['execution_status'] == 'executable'
            and (case['cell'] == 'F1' or case['cell'] == 'X1' and case['boundary'] == 0)]
assert len(selected) == 88
argv = [sys.executable, 'tests/ft/p2_schedule_driver.py',
        '--output', '/absolute/new-stage-one-v2', '--case-timeout', '30', '--total-timeout', '2700']
for case_id in selected:
    argv.extend(['--case', case_id])
subprocess.run(argv, check=True)
PY
```

输出目录必须新建，保存全部失败证据。旧12和首段88重合2格，全局按ID去重，不能相加成100格已验收。下一cell尚未实施，停止交回。

## 主任务首段验收 r1（已完成）

主任务完整执行通过：静态14项/10.689秒，driver13项/2.116秒；[representative-r2](p2_evidence/representative-r2) 12/12；[stage-one-r1](p2_evidence/stage-one-r1) 88/88，failed=0、harness_failure=0，11组实际owner争锁。汇总见 [stage-one-verification-r1.json](p2_evidence/stage-one-verification-r1.json)。该修订全局按case ID去重为 **98 passed、310静态可执行未跑、8限制、184待补接口**。上述“待主任务验证”段落保留为交付时记录，以本验收为准；后续源码修订仍需回归，不能跨源码hash借用通过结果。
