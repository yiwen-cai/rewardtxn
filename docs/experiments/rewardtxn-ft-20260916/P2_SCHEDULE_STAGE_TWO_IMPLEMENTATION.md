# P2第二段：F2/F3存储切点组合

2026-09-20。仅实现扩展计划第7节第二个单元；economy-dev显式status enabled，Astra medium实施，不递归。没有改compiler、manifest、静态测试、生产state/oracle、P1或第三方；没有运行完整矩阵/GPU。首段98已通过的历史证据见 [stage-one-verification-r1.json](p2_evidence/stage-one-verification-r1.json)。

## 有限支持与切点事实

新增 `STAGE_TWO` 精确96格：F2.b00…09×8可执行交错，F3.b00…01×8；交错仍00/01/02/03/04/05/08/09。与之前98支持集合并为190格（其中4个storage代表格重合）。184待补接口和8限制格不变；F3.b02…08仍无ACK支持。支持数量不是已经验收通过数量。

现有storage worker分支提取为一个有限Python方法；按boundary顺序直接调用API和fixture写入，不解释任意manifest操作。以下每个切点都先发送具体收据并阻塞，由父进程核对真实进程身份、直接读取切点文件及事件，再精确SIGKILL确认-9，最后新进程acquire/select。

| 切点 | 实际optimizer start/end | 实际成功收据 | 实际组件/finalize | manifest/token | 恢复 |
|---|---|---|---|---|---|
| F2.b00 | 0/0 | 无 | 无 | 无/无；target intent也无 | 父代，rollback_intents空；不声称接收结果已可重放 |
| F2.b01 | 0/0 | 无 | 无 | 无/无；intent有 | 父代+精确target intent义务 |
| F2.b02 | 0/0；有before-start事件 | 无 | 无 | 无/无 | 同上 |
| F2.b03 | 1/0 | 无 | 无 | 无/无 | 同上 |
| F2.b04 | 1/1 | 无 | 无 | 无/无 | 同上 |
| F2.b05 | 1/1 | 有 | 无 | 无/无 | 同上 |
| F2.b06 | 0/0 | scheduler_applied=false实际提交被拒；成功收据不存在 | 无 | 无/无 | 同上 |
| F2.b07 | 1/1 | 有 | 只有actor:0的11文件及finalize | 无/无 | 同上 |
| F2.b08 | 1/1 | 有 | 两rank共22文件；**任何rank finalize都没有** | 无/无 | 同上 |
| F2.b09 | 1/1 | 有 | 两rank共22文件及两个finalize | 有/无 | 充分证据补提交target，保留执行epoch |
| F3.b00 | 1/1 | 有 | 只有actor:0完整；commit实际缺rank拒绝 | 无/无 | 父代+精确target intent义务 |
| F3.b01 | 1/1 | 有 | 两rank完整 | 有/可见token；目录fsync前 | 新进程验证可见完整token，不声称断电保证 |

`storage-cut.json` 是父进程在SIGKILL之前读取的实际文件全hash/大小、完整intent、实际持久收据和事件序号。保存切点目标identity及manifest中对应call_offset的声明调用。检查组件集合、32768字节完整内容、成功scheduler收据、各rank finalize字典、manifest/token存在与缺失；不是恢复后倒推，也不是仅写一个期望标签。F2.b06另保存实际提交的false参数和具体API拒绝收据。optimizer仍是明确CPU fixture事件，模型文件不是训练模型。

恢复后额外验证：选中generation及token实际存在；rollback_intents精确等于此target intent路径（或明确为空）；不完整候选被拒原因；intent字节未变；补提交/已可见token的execution_epoch仍为原执行epoch，retained pending与原intent相同。补提交必须report `completed durable candidate`。没有执行真实optimizer重放，不把保存的义务称为重算已经完成。

## 交错与O模型

复用首段的目标实际逆序、相同g0证据重投、probe old/new次序和真实winner同run争锁；父进程增加共用modifier核对，未降低旧路径要求。probe在target intent中的pending必须保留；b00没有intent的事实单独注明，不编造待重放快照。

i08继续先审计主场景，再从独立完整健康control删除bootstrap optimizer_start；必须命中特定缺失。i09必须先完成主SIGKILL恢复与无损O对照，再修改父代g0组件并用另一新进程select拒绝。原主恢复的结果、切点快照和之后破坏报告分开保存，不能拿损坏拒绝掩盖主恢复失败。

F2全部边界和F3.b00的O主图按原清单验证合法rollback/recompute与错误保留负例；F3.b01是实际模型load目标token、后续保存保留祖先义务。O图独立读取同一冻结外部输入，不读取S产物来生成答案；模拟事件时间不是实测RTO。特别是F2.b00…03的S没有成功物理更新，O丢失已完成更新的图是**单独的规范对照**，不是其S时间线；F2.b09 S补提交与O回滚也分别分账。

## 新增测试与必要定向验证

driver测试由13增至19项，新增六项：

- `test_storage_before_intent_preserves_no_invented_rollback`
- `test_storage_before_start_and_started_cuts_are_distinct`
- `test_scheduler_false_is_rejected_before_real_kill`
- `test_all_rank_files_without_any_finalize_then_corruption`
- `test_complete_candidate_preserves_execution_epoch_and_probe`
- `test_visible_token_cut_and_independent_missing_event_probe`

既有“unsupported case”改选仍未实现的F4.b01；支持集合检查更新为190，stage one仍88、stage two为96。没有改既有静态14项或清单。

本worker仅定向执行五项关键检查：b00及b08+i09两个测试0.886秒通过；b06+i02与b09+i03两个测试0.807秒通过；F3.b01+i08测试0.416秒通过。未跑19项完整suite、原12、首段88或第二段96批次，其余切点由主任务验证；目前未发现组合需降为pending。

## 主任务建议验收

先跑完整driver及旧12；必要时回归首段88以核对共用modifier及文件读取。然后单独串行跑第二段96，输出新目录，保留所有失败。

```sh
.venv-tq/bin/python3.11 tests/ft/test_p2_schedule_driver.py -v
.venv-tq/bin/python3.11 tests/ft/p2_schedule_driver.py --representative \
  --output /absolute/new-representative-stage-two --case-timeout 30 --total-timeout 420
```

```sh
.venv-tq/bin/python3.11 - <<'PY'
import json, subprocess, sys
from pathlib import Path
manifest = json.loads(Path('tests/ft/fixtures/p2_schedules.json').read_text())
selected = [case['id'] for case in manifest['cases']
            if case['execution_status'] == 'executable' and case['cell'] in ('F2', 'F3')]
assert len(selected) == 96
argv = [sys.executable, 'tests/ft/p2_schedule_driver.py', '--output', '/absolute/new-stage-two-r1',
        '--case-timeout', '30', '--total-timeout', '2940']
for case_id in selected:
    argv.extend(['--case', case_id])
subprocess.run(argv, check=True)
PY
```

全部96若通过，应分别观察96次真实SIGKILL/新进程恢复、12组owner争锁，96份SIGKILL前切点文件清单；不能仅凭96个status标签通过。最终按case ID去重，已验收上限将是190而非98+96=194；未执行的其余cell不能计入。没有自动继续下一实施单元。

## 主任务第二段验收 r1（已完成）

主任务完整验证：driver **19项/4.839秒**；representative-r3 **12/12**、stage-one-r2 **88/88**、stage-two-r1 **96/96**，全部无失败。第二段实际确认 **96次SIGKILL及新进程恢复、12组owner争锁**。汇总见 [stage-two-verification-r1.json](p2_evidence/stage-two-verification-r1.json)。全局按case ID去重为 **190 passed、218静态可执行未跑、8限制、184待补**。本节是该源hash修订的有效验收；后续扩展需要自身回归，不借用历史结果充通过数量。
