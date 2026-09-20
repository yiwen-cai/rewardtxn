# P2跨case实际执行签名：独立事后审计

2026-09-20。本单元只新增 `tests/ft/p2_schedule_audit.py`、`tests/ft/test_p2_schedule_audit.py` 和本文；只读既有 representative-r4、stage-one-r3、stage-two-r2、stage-three-r1，不修改driver、清单、生产state/oracle、P1或原始证据，不启动进程/GPU。economy-dev显式enabled，Astra medium实施，无递归。

## 已验与未验必须分开

第三段主任务已验证driver25项/6.616秒及原12、88、96回归、X1全72；按ID去重252格合同passed，剩余156静态可执行未跑、8限制、184待补。证据见 [stage-three-verification-r1.json](p2_evidence/stage-three-verification-r1.json)。这些证明逐格到达自身声明的合同，**不证明跨case实际行为互异**。此前只有静态semantic hash和逐格boundary/modifier检查，缺少本项整体审计。

该事后审计不导入driver/state/oracle/compiler及其验证函数，不重跑生产oracle，不读取expected来决定签名。case目录名仅在签名算完后作报告定位和同case重跑归并，case.json中的expected不参与。四批共268次观察、252个唯一编号；同case跨批观察另核对签名是否一致，不把重跑计新格。

## 封闭schema与签名层次

标准库解析已保存的trace、launch/parent收据、实际方法文件、intent/token/manifest、切点快照及独立O图。固定操作/字段表；未知操作、字段、配置、未解析引用或缺少必要材料逐条 `unverifiable_normalization`，整体 `audit_incomplete`。不为提高覆盖率把未知数据随意丢掉。

| 签名 | 保存的事实 | 不用于制造差异的内容 |
|---|---|---|
| S | 实际state API顺序与参数/返回，生成顺序、真实载荷/变异、评分来源、组件/rank映射、可读intent/token及SIGKILL前文件存在性 | PID、时间、随机owner/generation名字、操作描述性的拒绝后缀、case/expected标签 |
| P | writer/lock-loser/recovery/corruption-check按实际行为确定的角色，退出码，各进程parent收据顺序及其真实trace引用 | owner0/owner1谁赢、PID/starttime/bootid具体值、argv中的case/path、初始两个独立进程的读取顺序 |
| O | 实际保存的states/updates/groups、load/optimizer/checkpoint/event有向引用、tensor真实值、rank映射、原文及seal匹配关系；所有已保存变体作为内容多重集 | oracle目录名、fault_model.case_id、boundary标签、任意graph ID、绝对模拟时间 |
| markers | CPU optimizer/reward/before-start等规范事件及其相对API位置、边界标记和伪success声明 | case ID、自由命名的when/result/boundary描述 |
| core_joint | S+P+O | markers不参加 |
| with_markers | S+P+O+markers | 同上所有身份/标签排除项 |

真实身份按关系规范化：winner/loser不按PID排序；generation、attempt、snapshot及O节点引用按首次语义使用改名，保留相等/不等关系。O事件时间保留相对顺序、同刻分组及故障窗口的前后/超窗关系，去掉绝对值；不计算新的实测RTO。

最关键的间接标签来源不是只删顶层case_id：external-fixture含case_semantic_sha256，source_sha进入intent，owner_nonce进入train-input，随后进入manifest/token。审计解码实际JSON，将这些digest替换成规范化内容及“引用是否匹配”的关系，再计算规范化对象hash。opaque digest自身不作新调度的证据。checkpoint二进制fullhash只在能关联到实际读取的二进制内容时保留；无法解析的摘要明确unsupported。

原文缺tokens/logprobs/completion等已声明负例仍保留缺字段事实，不把缺字段补默认值。未知字段则不能猜测。O graph只把有类型引用的载荷/组件内容加入图签名，未引用的fixture残留文件仅保留读取来源hash，不伪装成额外训练行为。source配置限制为当前CPU fixture，未扩展为通用真实训练审计。

所有读取文件的原始SHA256单列在每份normalized产物里；provenance中的源码、输入树/目录和原始hash不参与语义签名。输出目录不得位于任何输入证据树内。

## 碰撞、unsupported与调查

`audit.json`输出所有观察、解析数量、各层等价组和明确调查项；完整规范化对象保存为 `normalized-NNNN.json`，可检查差异路径而不是只看两个hash。

- `core_joint`相同的不同case：`cross_case_core_equivalence_requires_investigation`。另外列marker差异；即使marker不同，也不能宣称实际API/状态不同。
- 同case重跑签名不同：`rerun_signature_disagreement`，保留两份来源和第一个结构差异。
- 未知schema/材料不足：记录原因及原case；整体audit_incomplete，不能用已解析子集代替252全覆盖。
- S/P/O单层等价组通常正常，仅作定位；不自动认为方法或case失败。联合无碰撞也仅是已解析观察范围内未发现重复，不是功能/安全证书。

程序发现调查项仍正常退出，因为调查不是程序执行失败；调用者必须检查audit.json的status/completeness及investigations，不能把exit0当审计通过。原252合同passed字段不改，`functional_passed_count`明确not_recomputed。

已完成一个局部定向调查：真实 `stage-two-r2/F2.b01.i00` 与 `F2.b02.i00` 的S/P/O及core_joint相同，markers不同（writer标记事件4与5）。前者prepare后、后者只多一个before_optimizer_start规范标记，实际state/optimizer进度相同。应由主任务据此裁决覆盖口径或计划修订；本审计不偷偷增加一个标签来“修复”重复，也不自动撤销两格逐格合同结果。

## 有界测试与验收入口

7项测试覆盖：PID/时间/generation/nonce/path及case digest完整派生链改名后签名不变；真实两格marker-only等价；实际生成顺序改变；原文字段缺失/policy变化；rank映射变化；未知事件字段显式unsupported；碰撞报告不推导功能失败。

worker只执行必要定向测试（包括完整元数据与digest级联不变、未知字段拒绝、等价调查报告）及少量真实样本解析。没有运行全部7项或252整体审计。对旧seal回答、跨组来源、补提交、可见token等已保存样本作过定向解析；新增未知schema继续如实报告，不承诺252全部可解析。

```sh
.venv-tq/bin/python3.11 tests/ft/test_p2_schedule_audit.py -v
.venv-tq/bin/python3.11 tests/ft/p2_schedule_audit.py \
  --run docs/experiments/rewardtxn-ft-20260916/p2_evidence/representative-r4 \
  --run docs/experiments/rewardtxn-ft-20260916/p2_evidence/stage-one-r3 \
  --run docs/experiments/rewardtxn-ft-20260916/p2_evidence/stage-two-r2 \
  --run docs/experiments/rewardtxn-ft-20260916/p2_evidence/stage-three-r1 \
  --output docs/experiments/rewardtxn-ft-20260916/p2_evidence/signature-audit-r1
```

输出必须新目录。父进程start/continue命令没有单独journal，不能从事后现有文件补造真实放行记录；P签名只描述已保存的进程结果、parent收据和关联。观察记录并非独立硬件见证，审计不会反哺恢复、修复输入、生成新的方法期望或声称GPU自然到达。停止于本有界单元，X2/F4扩展继续暂缓。

## 主任务完整验收（2026-09-20）

7 项测试通过（1.409 秒）。`p2_evidence/signature-audit-r1/audit.json` 完成四批 268 次观察、252 个编号的解析：unsupported=0，同编号重跑差异=0。得到 **228 个 core_joint 签名**，存在 8 个等价组；每组为同一 interleaving（i00/i01/i02/i03/i04/i05/i08/i09）的 F2.b01、b02、b03、b04 四格。四者 markers 签名均不同。

原因是这些切点的实际持久状态都停在 prepare 后、optimizer 收据写入前。b02 的 before-start、b03/b04 的 optimizer progress 仅由 CPU 规范 marker 表达，没有真实训练 backend；因此 S API/持久文件相同，P 为同一杀死与接管合同，O 独立构造相同合法回滚重算图。不能靠新增标签使它们伪装成不同实际状态。8×4=32 个编号折合 8 个 core 签名，252−24=228。

验收分账：252 个逐格 CPU 合同仍通过；228 是本批规范化后实际 core 行为数量，8 组是已解释的模型等价，整体调查结论不能替代真实 backend 覆盖。600 清单仍含 156 静态可执行未跑、8 限制、184 待补；不宣称 600 个不同实际状态。真实 optimizer 开始/执行中/结束未落盘的差异、GPU 状态丢失与真实恢复仍需接入 backend 后验证。本次审计产物保持不变；后续 X2 扩展获得单独授权，不改变本批历史结论。

## X2/F4 有界schema扩展（2026-09-20）

本次只改独立审计器/其测试和文档，未改正在运行的driver、manifest、生产state/oracle及旧证据。所有新增操作仍逐项固定字段；不支持项保持`audit_incomplete`，程序没有通用字段丢弃或动态导入。

新增S解析：真实foreign root的control登记、授权/accepted与提交收据关系；目标sample错配；实际提交conflict载荷；拒绝的重复消费intent/data；恢复policy载荷的变更前后绑定；重复生成文件、真实fixture score/retry计算。foreign-run根路径和nonce不进入签名，但不同root/owner关系、实存accepted引用一致性进入。只允许已观察的foreign单sample/head为空CPU控制文件，未知嵌套内容不猜。

b04历史payload特别处理：最终method-target-0文件已经声明policy1，不能拿它去解释原owner的policy0接受收据。只有确切recovered_policy_payload事件绑定同一文件、原生成hash与变更before hash，才用冻结原始payload重建旧值，并核对原序列化hash；新进程变更事件之后才使用policy1文件。签名保留两次声明及匹配关系，不声称实际模型权重provenance。

新增P解析：scope-resume独立进程角色、原登记身份对应已观察writer及退出码、接管epoch、新旧身份不同。存在后续corruption-check时读取最终持久control，保留最终epoch及登记到哪个真实actor；b09关闭的旧owner与合法的新checker分开，不把正常退出叫SIGKILL。PID/starttime/bootid具体值依然不参加签名。

F4 S保留7唯一+第8重复文件的实际生成操作、样本身份与真实raw一致性。O保留预先冻结窗口合同的内容/引用匹配及technical_invalid事件；缺少唯一生成和全部reward已完成两类事实不被“方法成功”覆盖。lost/retry纯规范事件在S与O都进入markers：O先验证字段，再从core事件序列和时间槽中去除这两种纯模型事件，独立保存其模型位置；因此它们不凭新增event ID、time slot或PID伪造core行为。真实CPU重评分及response字节一致性仍有独立S记录。预先window合同与check_fault_window声明在marker层保留；O的technical_invalid是独立模型分类事实，并非实际进程故障证据。

原7测试保留，新增5项覆盖foreign元数据/派生digest不变、真实退出与checker登记、foreign/sample错配、F4重复raw/technical-invalid、lost/retry模型隔离与未知字段拒绝。定向测试证实仅改变规范lost samples时core_joint保持不变，markers改变；b02与b03的真实重复写入使S不同。未运行完整12项或408整体审计，交主任务验收；没有修改历史signature-audit-r1。新增测试仅读`stage-four-r1`和已保存`full-408-r1-F4`，不执行driver。

```sh
.venv-tq/bin/python3.11 tests/ft/test_p2_schedule_audit.py -v
.venv-tq/bin/python3.11 tests/ft/p2_schedule_audit.py \
  --run docs/experiments/rewardtxn-ft-20260916/p2_evidence/full-408-r1-F1 \
  --run docs/experiments/rewardtxn-ft-20260916/p2_evidence/full-408-r1-F2 \
  --run docs/experiments/rewardtxn-ft-20260916/p2_evidence/full-408-r1-F3 \
  --run docs/experiments/rewardtxn-ft-20260916/p2_evidence/full-408-r1-F4 \
  --run docs/experiments/rewardtxn-ft-20260916/p2_evidence/full-408-r1-X1 \
  --run docs/experiments/rewardtxn-ft-20260916/p2_evidence/full-408-r1-X2 \
  --output docs/experiments/rewardtxn-ft-20260916/p2_evidence/signature-audit-408-r1
```

应等待六批矩阵全部结束后执行。调查/unsupported仍是独立验收账，不以408逐格passed覆盖；新case与旧case可合理等价，不以目录标签或digest差异冒充真实调度多样性。

## 完整408格主任务验收

扩展审计器12项测试通过（3.120秒）。六个同版本完整批次408个编号全部可解析，unsupported=0；核心联合签名384个，含规范标记408个。独立S/P/O分别322/116/110个。8组联合等价仍均为F2.b01–b04同一interleaving，原因同前述CPU optimizer规范标记限制；没有新增未解释组。原始audit保留investigation_required，不能将程序exit0改称全面功能验收。详见p2_evidence/signature-verification-408-r1.json。

408格合同通过包括对失败/拒绝的预期检查：primary模型分类为correct_recovered 168、safe_stop 64、unverifiable 32、technical_invalid 24、invalid_commit 120。尤其24项F4窗口无效不是24次成功恢复。实际进程层共有96次SIGKILL及51次owner竞争；这些均为CPU fixture，不代表GPU训练或正式RTO。600清单仍有8限制和184接口待补。
