# P2第三段：X1评分与载荷反例

2026-09-20。仅实现扩展计划第7节第三单元；economy-dev根显式enabled，Astra medium实施，不递归。没有改manifest、compiler、静态测试、生产state/oracle、P1或第三方，没有完整矩阵/GPU运行。第二段190个唯一case的已验收记录见 [stage-two-verification-r1.json](p2_evidence/stage-two-verification-r1.json)。

## 支持集合与具体边界

`STAGE_THREE`为X1的b00/01/02/03/04/06/07/08/09乘8个当前交错，共72格；b05训练归一化仍pending，i06/i07也仍pending。支持总并集252格（此前190中已有10个X1格），不是新增72个唯一结果。`--representative`仍12格。

worker用一个有限X1负例方法执行真实文件/评分/state操作，O由另一个有限函数独立构造相同冻结输入的反例；二者不读取对方结果生成expected。原X1.b01/b09路径合并为该方法，原有业务资格和期望保持，需回归。F4混verifier路径仍独立；此前F1/F2/F3/X2逻辑未扩展。

| boundary | S实际行为 | O独立主断言 |
|---|---|---|
| b00 | 保留已有完整commit正例 | correct_recovered |
| b01 | 对真实completion/label计算exact=1及1-exact=0，声明v2；提交bad sample0被真实accept拒绝，其他7项接收后prepare拒绝 | v2未授权+独立reward mismatch，invalid_commit |
| b02 | 仅声明verifier改v2，前后reward仍1；accept与不完整prepare实际拒绝 | 明确未授权声明拒绝，数值相同不救seal资格 |
| b03 | 在实际独立authority副本把label改2，冻结原始输入仍label1；S admission/prepare不声称评分语义验证 | 对应O原文label2与冻结dataset冲突，必须命中frozen source违反；原文/tensor摘要同步，避免靠偶然tensor错误通过 |
| b04 | 先保存真实方法回答fullhash seal，再把completion改2，不更新旧seal；S只接收当前摘要 | O先封装spec及seal，再实际修改回答文件字节，保留原seal；具体hash mismatch导致unverifiable |
| b06 | 从另一份真实JSON（completion2,label1）独立计算0，替换目标sample0应为1的分数，真正送入accept/prepare | 从冻结other输入独立算0，放入目标sample及实际tensor row，必须independent reward mismatch |
| b07 | 显式授权policy0，向accept提交声明policy4的payload，实际拒绝；原生成原文policy仍0，不能伪造历史policy | O声明sample policy4、auth0、trainer1/max_staleness1，保留原raw policy0；必须同时命中授权及staleness违反 |
| b08 | 独立authority副本实际删除completion；方法原文仍完整，S不检查独立observer缺口 | O原文具体缺completion，unverifiable；不等同安全恢复 |
| b09 | 实际reward0进入accept/prepare，记录伪success/authoritative_reward0、正确值1 | 将错误0映射进真实tensor row，伪标志不能覆盖独立reward mismatch |

每个O场景先审计完整健康图；invalid主图必须命中该边界指定violations，缺证据主图必须具体指向hash或completion。unverifiable主图的错误保留K反例从独立健康图构造，不能让缺原文遮住K断言。

共享external-fixture只对X1.b06额外冻结other样本，来源completion/label取自原manifest；没有改目标输入。S的生成文件由父进程逐项比对，b04只允许已记录的sample0/completion变化且旧seal确实未更新。authority副本、跨组source文件及变异前后hash都实际落盘；O不读取这些S文件，而独立从外部fixture生成自己的原文和评分来源。

b03/b08的“authority副本”是CPU测试对observer材料的建模，不是实际部署的可信采集管线。b06另一组仅作冻结评分来源，没有伪称它参与optimizer或额外改变目标工作量。state不被赋予真实reward语义能力。

## 授权正例与交错

新增独立同组policy正例，不挪用600编号：sample0的policy=1，其他7项policy=0，各自有相符授权，trainer policy=1/max_staleness=1；真实state完成整组commit，独立O图也必须correct_recovered。该例证明组内合法policy差异不被当成混verifier；不声称state证明模型真实policy版本。

所有8交错复用现有真实逆序、g0幂等收据、probe顺序及owner竞争。b01/02/07分别核对真实坏声明payload、持久授权版本、7个实际接收索引及不完整prepare拒绝。其他负例核对8个实际接收分数与prepare深度；未消费probe必须保留pending。i08仍独立完整control删bootstrap start，i09仍无损主场景后真实g0破坏/新进程拒绝及O旧seal破坏；两者不能遮住X1原主负例。

## 新增测试与交付范围

driver测试从19增为25项，新增：版本标签单独改变但同分；错label+i08；回答字节旧seal；跨组真实分数来源；未授权policy与缺authority；同组合法policy差异真实state+独立O正例。支持集合断言更新为252，第三段72。静态14项未变。

worker必要定向检查通过：回答旧seal与跨组来源两项0.670秒；未授权policy/缺authority及合法混policy两项0.606秒。未运行25项完整suite、旧代表格或72格矩阵；没有发现需降级的组合，但完整验证留主任务。

建议主任务完整25项及旧12回归，然后串行72格。此前190条路径保持业务合同，共享payload取policy逻辑从常数0改为实际raw声明，旧fixture仍0；应保留对旧阶段的必要回归。

```sh
.venv-tq/bin/python3.11 tests/ft/test_p2_schedule_driver.py -v
.venv-tq/bin/python3.11 tests/ft/p2_schedule_driver.py --representative \
  --output /absolute/new-representative-stage-three --case-timeout 30 --total-timeout 420
```

```sh
.venv-tq/bin/python3.11 - <<'PY'
import json, subprocess, sys
from pathlib import Path
manifest = json.loads(Path('tests/ft/fixtures/p2_schedules.json').read_text())
selected = [case['id'] for case in manifest['cases']
            if case['cell'] == 'X1' and case['execution_status'] == 'executable']
assert len(selected) == 72
argv = [sys.executable, 'tests/ft/p2_schedule_driver.py', '--output', '/absolute/new-stage-three-r1',
        '--case-timeout', '30', '--total-timeout', '2220']
for case_id in selected:
    argv.extend(['--case', case_id])
subprocess.run(argv, check=True)
PY
```

新目录保留每格原文/变异/收据/全部O报告；预计72格有9组真实owner竞争，无主SIGKILL需求（i09仅组件篡改）。仅在主任务验收后才能更新全局唯一passed计数；252上限不是190+72=262，负例probe和辅助合法policy也不增加case分母。停止于本单元。
