# P2：600个CPU调度的可执行分层计划

日期：2026-09-20。仅规划，未改实现、未执行调度或GPU。继承economy-dev根 `01a0bd89-a9c3-7d00-b16d-4a0a766de676`，本次显式只读status为enabled，角色Astra medium planner，不递归委派。依据 [P2_IMPLEMENTATION_PLAN](P2_IMPLEMENTATION_PLAN.md) 第6节、[oracle_spec](oracle_spec.md) 与当前state/oracle接口。主任务已报告state 21项、oracle修复后25项及真实GSM8K CPU入口通过；这些不等于600调度完成。

## 1. 先明确能交付什么

保留六cell×10边界×10交错的全部600个稳定编号 `F1.b00.i00`…`X2.b09.i09`。当前不能诚实声称600格都具备原计划要求的执行接口：真实ACK、注入控制器journal恢复、第二故障聚合、真实训练归一化尚缺合同接入。不得以恒定返回unverifiable/N/A填满通过率。

采用以下互斥账本分类，不删格、不补seed、不改FT-v1矩阵：

| 分类 | 精确集合 | 数量 | 当前可声称范围 |
|---|---|---:|---|
| 待补原合同接口 | 全cell的i06/i07 | 120 | owner崩溃或第二轮普通state操作可以补测，但不完成注入journal/第二故障要求 |
| 待补ACK合同 | F3.b02…b08，排除i06/i07 | 56 | 不能用toy ACK、token或者直接调用返回值冒充ACK |
| 待补训练变换 | X1.b05，排除i06/i07 | 8 | 当前unsupported-transform拒绝只能是接口限制测试 |
| 接口限制验证 | F3.b09，排除i06/i07 | 8 | 证明预先N/A及ack_capability=none，不完成F3恢复验收 |
| 当前CPU合同可执行 | 上述集合的补集 | 408 | 可执行真实state API和/或独立规范图判读；仍非GPU/真实A基线 |

合计600：当前408格可以验收各自声明的CPU合同，8格可以确认限制，184格保留pending。各cell当前可执行数量为F1=80、F2=80、F3=16、F4=80、X1=72、X2=80。F3的16格只验证token前后存储切点，不验证实际ACK，因此不能汇总成“F3 supported”。上述数字是待实施计划，当前执行完成数量仍为0。

如果后续要求“600全部原合同通过”作为门禁，应先补上述接口再执行剩余格；不能将本计划的408格上限改名为原600验收完成。即使所有CPU格通过，A/F3仍预先N/A，F4真实自然窗口仍由GPU证据决定。

## 2. 三条证据线，结果不能相互替代

1. **S：真实state操作线。** 调用当前7个API，使用真实本地文件、fullhash、flock、fsync和普通CPU子进程。模型/optimizer/rank文件均明确是fixture；没有真实训练。每个run冻结verifier，attempt显式授权policy；同组可有多个合法policy，不允许混verifier。记录实际返回/异常、磁盘文件和子进程退出结果。
2. **O：独立oracle规范图线。** 使用手写语义模板构造schema 1证据，再调用 `audit_run`。不调用state来生成期望、不读取state的token判定函数。Graph构造可以参考现有test_oracle，但必须支持K=8、明确logical与physical身份、真实load执行段、实际训练rows。其时间是**模拟控制器时间**，不是CPU进程wall-clock；fixture RTO不作方法实测RTO。
3. **P：真实进程故障线。** 在S线指定系统调用/API边界确定性握手，然后由持有Popen的测试父进程只杀自己的直接子进程；另有两个真实进程争owner锁。记录SIGKILL退出码和注册身份。P不是P1 runner，也不测试P1多层worker接入、armed/fired/release协议。O图可镜像对应规范事件，但不能据此声称其GPU接缝真实。

每格输出三线各自 `pass/fail/not_executed/not_applicable_to_this_lane` 和证据路径。没有S线能力的O纯数据反例可以通过O声明，不能补一个虚假S通过。例如F4进度在测试驱动中是显式生成/评分事件模型，不是AReaL真实并发reward；X1的真实语义错误由O重算识别，S只负责拒绝版本授权错误。

A-model/R-model使用相同冻结外部payload、目标与外部事件偏序；允许各自不同恢复路径和期望。A-model是独立手写的基线行为假设，不能用state API为其添加R机制，也不能叫“官方A实测”。首版不实现完整A模拟器：O模板分别声明“安全丢弃/合法回滚/正确保留”等基线合法图与R图，结论仅针对图合同。

## 3. 10个边界：逐项落实到操作与期望

所有当前可执行S格都有共同前缀：新CPU run→登记owner/必要worker→为bootstrap完整K=8组逐样本authorize/accept→prepare→写11类组件及data→optimizer收据→全部rank finalize→commit，得到完整父代g0。目标组随后执行。这样即使目标边界是部分生成，仍有真实父token可用于回退、文件篡改和恢复测试；bootstrap不能计作目标恢复。

| cell | b00…b09（顺序固定）及主要执行线 |
|---|---|
| F1 | b00…b06分别只有1…7/8个目标样本生成完成，缺项显式保留在预期manifest，S不得prepare/commit不完整组，O分别检查安全回退与错误保留；b07目标回答文件缺失（S在完整组件/payload引用处拒绝或O缺证据，不把“token”误解为StepToken）；b08回答存在但训练logprobs缺失（O unverifiable/不完整载荷）；b09从未产生任何reward记录的第8个样本，仍由预期K发现，S不封组、O不放行 |
| F2 | b00 intent前；b01 intent持久后；b02模拟optimizer开始前；b03模拟optimizer执行中；b04模拟成功但成功收据尚未落盘；b05成功收据后、模型文件未完成；b06scheduler_applied=false/缺落实；b07两个expected rank中只写一个rank；b08全rank文件已有但缺最后finalize；b09manifest持久、token未发布。S/P验证回退或仅充分证据补提交；“optimizer执行”仅是CPU fixture事件，绝不称真实优化器崩溃 |
| F3 | b00 token发布前（缺证据必须回退）；b01 token link已可见、目录fsync未确认（P进程崩溃后若完整文件仍存在，可重新验证并确认；不是断电保证）；b02 token持久未ACK；b03 ACK发送丢失；b04执行成功收据丢失；b05重复ACK；b06旧epoch ACK；b07错generation ACK；b08已ACK重投；b09部署没有ACK接口。b02…b08当前pending，不造ACK实现；b09只验证限制 |
| F4 | b00 K完成且第4个评分执行中；b01 K完成且4个评分已完成；b02生成K缺一；b03重复sample冒充K；b04reward start后未end；b05reward已end但结果未accept；b06一条混verifier；b07一个fixture worker丢失使同组两项缺结果；b08存活回答下原生重试的**规范图**合法路径；b09所选窗口已经过去。S检查接收/封组资格，O检查完整/错误保留；没有自然可达性证明。b00/b01分别保留精确定义，不能在运行后挑有收益者 |
| X1 | b00正确fixture-exact v1；b01测试驱动真实调用另一确定函数 `1 - exact` 得到不同分数并声明v2：S拒绝未授权verifier，O发现混版本/错误reward；b02仅把版本标签改v2、数值不变：S/O仍拒绝；b03label与冻结dataset行不符；b04回答内容/摘要绑定不符；b05非identity归一化（当前pending，限制测试不充验收）；b06把另组真实reward放进目标组；b07声明policy超出授权/合法staleness；b08缺独立authority原文；b09伪造success但错误reward/实际tensor。真实不同函数只存在于fixture驱动，不扩oracle动态import白名单 |
| X2 | b00同attempt相同结果重投应幂等；b01同attempt冲突内容拒绝；b02旧attempt先到；b03旧attempt后到；b04新epoch当前合法结果；b05从未授权attempt；b06错logical group/sample；b07错run；b08重复消费映射；b09旧owner延迟提交。S用真实API拒绝/接收，O验证旧结果即使先占位也不得进入保留链；不改任何方法内部pending字典 |

F1.b07、F1.b08和X1.b04必须明确对象类型和错误位置：缺文件是缺证据，不擅自判invalid_commit；只有完整独立证据确证已保留错误输入时判invalid_commit。X1.b08单独声明缺证据期望，不把unverifiable当恢复成功。所有boundary模板至少有一条未破坏对照图用于证明具备通过深层检查的条件。

## 4. 10个交错及组合规则

| i | 固定交错 | 真实落点与避免掩盖主边界的办法 |
|---|---|---|
| 00 | 顺序 | 按边界模板原偏序运行，构成主对照 |
| 01 | 独立样本逆到达 | 将已允许到达样本按7→0排序；固定sample_idx及tensor最终顺序不变。若某边界允许样本不足2，改对bootstrap独立样本逆序，标明交错发生在prefix，不声称目标并发交互 |
| 02 | 同事件幂等重复 | 重投一个已经接受的完全相同result或持久收据；比较实际返回与文件内容不变。不能给O证据重复event ID制造schema早拒；observer规范图只保留一个业务事件，原始重复调用另存 |
| 03 | 延迟旧attempt | 给独立probe样本先后授权old/new，先接合法new，再提交旧result；明确旧result到达在目标boundary之前或之后，拒绝后继续主边界。旧result不进入oracle有效训练图；另有错误保留反例图证明oracle会识别 |
| 04 | 新授权后旧结果抢先 | probe old→new授权，旧结果先提交被拒，再接new。不得复用i03同一实际操作顺序 |
| 05 | 两真实进程争owner | 在共同prefix之前两个真实子进程同时请求新run owner锁，winner保持锁直到loser完成；必须只有一个acquire成功。winner继续该格目标边界，不能竞争完成后换一个无关run重演 |
| 06 | controller在注入确认前/后崩溃 | 原要求涉及fired/observed与不重复注入，需要P1接缝；本任务不依赖正在开发P1。全部60格pending。可记录补充S owner崩溃测试，但不将其等同控制器journal恢复 |
| 07 | 恢复后第二故障仍保留 | 当前oracle单fault_id/单target，state不持久化注入schedule。全部60格pending；禁止把第二故障删掉、改成普通消息或复制两份单故障报告后称多故障通过 |
| 08 | 删除必要证据 | 先完成主边界S/P和未破坏O对照，再从**独立observer副本**删除预定证据，例如某rank finalize或一个optimizer映射；重新封装观察副本（若是规范证据缺失）或保留旧seal（若是外部文件损坏），两种语义预先固定。必须得到指向该证据的缺口，不是任意schema错误 |
| 09 | 文件中段篡改 | 主边界完成后对本格已有父代g0或完整目标代的一个>16KB组件文件中间改1字节，保留size/首尾；S再次select必须拒绝损坏token。O独立副本保留原seal，得到fullhash mismatch。先保存主边界和未篡改对照结果，避免corruption掩盖所有主逻辑 |

probe样本使用独立逻辑身份，不能进入目标消费集合；不让额外合法probe消费改变FT工作量假设。若需要在state中accept probe而不训练，要在data pending/预期清单中明确解释其未消费状态，不能留一个看不见的游离已取出工作。

组合采用显式偏序，不简单在每个trace末尾append同一干扰。每格编译出 `prefix → interference-pre → target-boundary → resolution → interference-post → audit`；i01/02/03/04的准确插入点由boundary表规定，插入点必须存在。无法同时满足前置条件的格必须在冻结清单记录 `composition_conflict` 并转pending，不能静默退回i00。本文408是源码接口层面的可执行上限；编译发现新的真实冲突须减少“当前可执行”数量、增加pending并解释，600总数不变。

## 5. 杜绝换ID/seed凑600

拟定 `tests/ft/fixtures/p2_schedules.json` 为完全展开清单。每项至少：

```json
{
  "id": "F2.b09.i05",
  "cell": "F2", "boundary": 9, "interleaving": 5,
  "execution_status": "executable",
  "lanes": ["state", "process", "oracle-model"],
  "payload": {"k": 8, "expected_ranks": ["actor:0", "actor:1"]},
  "operations": ["...具体带参数的操作..."],
  "cut": {"operation": "commit_generation", "after": "manifest_durable", "before": "token_link"},
  "expected": {"state": "promote_complete_candidate", "oracle": {"safety": "pass"}},
  "must_reach": ["owner_contended", "intent_durable", "all_rank_finalize", "manifest_durable", "new_process_selected"],
  "semantic_sha256": "...",
  "limitations": ["CPU model files; no real optimizer"]
}
```

不得把JSON里的字符串占位留到实际冻结清单。`expected`由手写边界×交错决策表给出，不通过运行state/oracle填答案。状态结果允许指定严格集合的场景仅限事先无法区分的持久性窗口，并说明物理原因；普通进程SIGKILL后不随意允许两种结果掩盖bug。

`semantic_sha256`对规范化操作序列、真实payload、依赖边、cut、期望分类hash，去掉case_id、任意nonce、路径、PID、wall-time和seed。对逻辑ID做alpha重命名后再比较，避免只换组名骗过重复检查。同一语义trace重复即编译失败；每格还输出可读diff证明与对应i00或前一边界至少有操作、顺序、缺失位置或真实payload变化。只换seed不算独特调度。

固定seed20260920仅用于确定无语义随机性的排列/实例化；不能让随机数决定“碰巧是否命中”。一个case内的base/control/corrupt等多个probe分别计检查数，仍只算一个case；S/O两线和A-model/R-model也不把分母翻倍。

## 6. 最小驱动与进程切点

只新增测试域文件，不加生产state/oracle API、不接P1：

- `tests/ft/p2_schedule_cases.py`：10边界与10交错的小表、完全展开编译、语义去重、手写expectation；不要DSL解析器/插件框架。
- `tests/ft/p2_schedule_driver.py`：`run_case(case, output_dir)` 顺序调度S/P线，保存结果；`build_oracle_fixture(case, variant)` 独立构造O证据；`check_case(case, observations)`只比较手写期望及must_reach。现有Graph太固定于K=2，新增小构造器而不改oracle生产逻辑。
- `tests/ft/test_p2_schedules.py`：静态600清单验收、当前可执行子集、限制/pending账本、测试驱动的自检。
- `tests/ft/fixtures/p2_schedules.json`：冻结的展开结果。若清单由编译函数产生，重新生成必须字节一致并保留生成器hash。

cut实现限测试子进程：对现有 `_write` 调用做窄包装，原调用实际完成后发管道ready并阻塞；只在指定目标文件、指定第几次调用命中。token可见未目录fsync用测试子进程内包装 `os.link`：真实link后ready，在 `_sync_dir`之前等待父进程SIGKILL。不修改state保存算法。初始intent前可在公共prepare调用之前握手；组件部分写由fixture writer在明确字节数后握手；每rank普通文件写完即产生其fixture finalize收据。捕获的收据明确 `evidence_level=cpu_fixture`，不能把API边界当真实Megatron finalize。

父进程收到精确cut ID及pid身份再kill，等待returncode=-SIGKILL；随后在新子进程acquire_owner并恢复，parent在存活时不能伪称旧owner已退出。owner竞争用两根启动管道同时放行，winner继续持锁，loser必须在winner退出前报告拒绝。所有等待有固定短超时；超时为harness failure，不是方法恢复超时；清理仅Popen所持本任务子进程，无名称匹配杀进程。

最少P层必跑：F2的b01/b04/b05/b07/b08/b09×当前8个交错，F3的b00/b01×当前8个交错，以及所有当前可执行i05。交集去重按case计；其他格可以只S/O。i09不是掉电测试；fsync保证不能由SIGKILL实测外推。无需为600格各启动GPU、容器或真实AReaL。

## 7. 期望与覆盖门槛

验收输出 `inventory.json`、逐case `result.json`、`coverage.json`，字段分开：total_declared=600、executable、executed、passed、failed、pending_interface、limitation_only、harness_failure、lane-specific counts。不输出把pending/限制当passed的“600/600”。结果文件保留源码hash、Python版本、文件系统类型、case semantic hash、实际操作及cut receipts、退出码和O报告。

必须满足：

1. 六cell每个恰好100个唯一编号；语义trace去重通过；所有格都有具体执行计划或精确缺失接口，没有隐式skip。
2. 每个已执行case的must_reach全部有实际观测，不允许schema报错先于cut还算通过。S按实际API/磁盘事实记录；O证明事件完整性、root/load、optimizer绑定等预期层已达到（必要时依据报告证据引用及trace，不靠字符串包含任意error）。
3. 对i08/i09等故意缺证据/损坏格，必须先运行未破坏对照图并进入目标深度；破坏后错误必须发生在预定验证点。故意schema非法另作driver自检，不占600有效业务调度。
4. 每个当前有可执行业务格的cell，必须包含实际合法保存/恢复或合法回滚、确证错误保留链、特定缺证据负例。F3只token范围，不能以这个门槛推导ACK通过。X1要有真正不同评分值和混verifier拒绝的S/O有效覆盖，不能只跑unsupported-transform。
5. 四个核心不变量必须有正反对照：完整K、版本授权、最终保留消费唯一、真实load执行段与父state连续。包括同epoch reload旧祖先的非法父保留；同组不同合法policy为正例。
6. oracle针对同一fixture图的期望由手写规则独立给出；S的实际返回不决定O的expected。两线一致不能替代真实adapter验证。
7. 所有P case真正到达cut并确认kill或owner争用；用普通异常替代SIGKILL、sleep碰运气、kill了测试harness均失败。原始失败记录不覆盖，不为凑全绿换seed。

可用标准库 `trace` 记录fixture进程的语句覆盖作为辅助，但不作为业务覆盖替代，也不引入coverage依赖。若must_reach需要当前oracle没有暴露的深层阶段，优先从手工构造的完整对照图与具体报告证明；确有无法区分的拒绝路径时先补针对性测试/诊断字段规划，不输出虚假的覆盖百分比。

## 8. 下一项有界实施brief

先只实现 `p2_schedule_cases.py`、完整600清单与静态验收测试：冻结上述分类、编译偏序、语义去重、手写expected/must_reach；不执行批量case、不修改生产state/oracle、不补ACK/P1。交付必须能列出408上限、8限制、184待补，并对编译发现的新冲突逐格说明，不能只是打印这些常数。

第二个有界单元实现driver并执行12个代表格：F1.b00.i01、F1.b09.i04、F2.b04.i00、F2.b09.i05、F3.b00.i00、F3.b01.i09、F4.b00.i00、F4.b06.i03、X1.b01.i00、X1.b09.i08、X2.b02.i04、X2.b09.i05。这些代表前置可达、真实SIGKILL/争锁、有效版本负例、完整对照后缺证据与中段篡改；不以12格冒充600完成。

代表格验证后再执行当前实际可执行子集，最后按缺口独立规划剩余184格。拟定命令可为 `python -m tests.ft.p2_schedule_cases --check-manifest ...`、`python -m tests.ft.p2_schedule_driver --manifest ... --case ... --output ...`；这些入口尚不存在，本轮未运行。停止于本规划，不调动GPU、不碰正在实施的P1文件。
