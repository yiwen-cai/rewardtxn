# P2：12格之后的CPU调度扩展计划

2026-09-20，仅规划。economy-dev根 `01a0bd89-a9c3-7d00-b16d-4a0a766de676` 显式status为enabled，Astra medium planner，不递归。依据冻结600清单、当前state/oracle API、[原调度计划](P2_SCHEDULE_PLAN.md)与[12格实现](P2_SCHEDULE_DRIVER_IMPLEMENTATION.md)。本轮未改实现、未运行矩阵/GPU。

## 1. 起点与停止线

当前有效实证：[driver-verification-r1.json](p2_evidence/driver-verification-r1.json)，8个driver测试1.161秒、12代表格全部通过，4次真实SIGKILL、2组真实owner竞争。全局仍12已通过/396静态可执行未跑/8限制/184待补。408是编译时接口上限，不是已实现数量或必须凑出的通过数。

扩展只涉及测试域driver、针对测试与本实施文档；生产state/oracle、P1、第三方和原manifest均不自动修改。i06/i07、F3.b02…b08、X1.b05的184格不进入本轮，F3.b09的8限制格也不计CPU业务通过。发现组合冲突需保留原case与semantic hash，在运行inventory另列`contract_review_pending`及具体原因，报告有效支持上限下降；不得重写原408为已完成，也不能以固定unverifiable填满。

## 2. 最小组织方式

保留现有Child管道、身份核对、握手、精确SIGKILL、锁竞争、恢复及文件保存实现。只在新增边界需要时局部提取：

- Worker.target按六个cell分为六个普通Python方法，参数是已核实的boundary整数；有限if分支调用现有generate/authorize/accept/prepare/optimizer/checkpoint/cut。不遍历manifest的`call`字符串动态dispatch，不用eval、插件注册、AST或通用解释器。
- 将现有bootstrap/目标group的到达顺序作为显式整数列表参数。有限读取`fixture.generate_sample`与bootstrap输入是冻结fixture构造，不是解释执行清单。生成、接受、最终tensor排列是不同顺序，不能一并反转。
- 将probe、幂等收据、owner争锁与observer破坏各抽一个普通函数；`run_case`仍明确prefix→modifier pre→cell boundary→recovery→modifier post→O audit。8个当前交错通过固定分支组合，不写408个case分支。
- O保留独立完整图、safe-drop、rollback、token-load三类现有构造；按cell/boundary显式改变输入或图事实。增加有限的“错误保留”“缺证据”“窗口不满足”构造函数。手写expected对照表仅解释manifest语义，不依据state结果生成答案。
- 只增测试域的能力清单函数：输入case，输出supported或精确reason。每个组合须同时满足cell handler、modifier、恢复/变异后置条件；不能凭`execution_status=executable`即声称实现。

不拆出通用运行框架，不为格式统一重构已通过的12条路径。阶段测试必须复跑这12条，证明局部提取没有改变其切点和证据合同。源hash变化后另存新的运行目录，r1证据原样保留。

## 3. 各cell的直接实现与独立期望

以下每个b乘交错 `{00,01,02,03,04,05,08,09}`。S为真实state API，P为真实子进程，O为独立规范图；O模拟时间不报告实测RTO。每格完整健康对照必须先到K8/tensor/optimizer/保留链深层检查。

| cell/数量上限 | S/P操作 | O主合同及负例 |
|---|---|---|
| F1 / 80 | b00…06生成/接收1…7项，prepare具体拒绝；b09生成8接收7并明确第8项无reward；b07按manifest删除方法payload.tokens，继续真实授权/接收/prepare，但S仅检查摘要授权，不能声称检测字节语义；b08仅删独立observer logprobs | b00…06/b09最终只保留父代+safe_drop；错误保留不完整K必须invalid_commit。b07/b08分别命中特定缺tokens/logprobs的unverifiable；缺证据是此格预先声明的负例合同，不是恢复成功 |
| F2 / 80 | 每个b均依冻结cut握手并真实SIGKILL：b00最后accept后prepare前；b01 prepare后；b02 before_optimizer_start；b03 start后无end；b04 end后无收据；b05收据后；b06实际scheduler_applied=false收据拒绝后；b07单rank finalize；b08全文件无finalize；b09 manifest持久/token前。恢复b00父代无目标intent，b01…08父代且保留rollback_intent，b09充分证据补提交 | 延用清单要求的独立legal_rollback_recompute图：丢失physical-target→真实模型load父代→重算最终保留；另强行保留丢失更新拒绝。S b00…03并没有真实完成optimizer，O是单独的合同对照，不能声称其反映S切点；b09的O回滚也不冒充S补提交观察 |
| F3 / 16 | b00缺rank commit拒绝→kill→回父；b01真实link后目录fsync前kill→新进程验证可见完整token；沿用r1 | b00合法回滚模型及错误保留；b01实际模型load target、后续保存保留祖先。始终无ACK证明，不外推掉电持久性 |
| F4 / 80 | 明确生成及评分规范事件：b00启动4无end；b01四项end；b02仅7唯一样本；b03 sample6再生成一次但unique仍7，不能覆盖原收据后计8；b04只start0；b05 end0但before_accept；b06第8项v2实际拒绝；b07 worker_lost仅规范事件；b08回答文件保持不变的规范retry_count=1；b09评分已全结束 | b02/b03预先窗口不足、b09错过窗口为technical_invalid，事实在故障选择前冻结；不得凭“此方法失败”决定。其他model恢复图保留真实tokens/mask/logprobs映射；b06错误保留混verifier拒绝。worker_lost/retry是模型而非真实原生恢复实证 |
| X1 / 72 | b00正例见第6节已裁决修订；b01实际1-exact=0/v2，b02仅标签v2，b07未授权policy4均实际accept拒绝；b03 observer label与冻结源不一致；b04封装后改回答保留旧seal；b06从另组completion2/label1算出0替换目标分数；b08删除独立completion；b09错reward+伪success。不得为负例篡改冻结输入本身 | b01/02/03/06/07/09确证错误保留为invalid_commit，b04具体fullhash错、b08缺completion为unverifiable。b07逐sample授权及staleness，组内合法多policy另作正例，绝不恢复同组policy必须相同旧限制。b05仍pending |
| X2 / 80 | b00实际相同结果重投并比较相同receipt；b01同attempt分数冲突拒绝；b02/03新授权后old先/后到，记录顺序；b04旧登记owner真实退出、新进程同run acquire后授权policy1并accept；b05未授权attempt；b06源授权绑定target而声明other；b07真实另一run的attempt；b08真实prepare重复消费映射；b09关闭owner后真实延迟提交拒绝 | b00/b04合法当前结果；其他错误保留独立invalid_commit。b04不是拿旧owner.close当退出，b09只明确失效对象拒绝范围。b06需构造sample=other但源epoch/nonce/attempt来自target的无效attempt，不能调用现有accept不传目的身份然后期待拒绝。b07使用另一真实root/owner的nonce，不假装run字符串受生产API检查 |

F2.b08冻结清单实际是两个rank都没有finalize，不能擅自改成“只缺最后一个finalize”。新增writer需支持`finalize=False`，并记录实际没有收据。F4.b03重复生成保存第二份raw及事件，state的distinct sample资格不得随重复次数增加。

## 4. 八种交错的真实组合合同

| i | 实现及收据门槛 |
|---|---|
| 00 | 原顺序，保留target操作序列；不得复用前一case目录 |
| 01 | 根据manifest实际生成调用顺序逆序目标；目标只1项时才逆序bootstrap。父进程核对真实事件数组与冻结顺序，另核对最终sample-index排序不变。r1 F1.b00仅覆盖了bootstrap逆序，不能直接外推其他格 |
| 02 | 真实对已token的g0重复2次**完全相同optimizer evidence**；现API允许相同已有返回。比较两次返回、receipt/token/manifest全hash不变，不能换成重复新事件或仅写日志。原evidence由bootstrap实际写入时保存，用于调用，不读取它来生成O期望 |
| 03/04 | 沿用probe old/new，分别new→old和old→new实际收据顺序；合法probe仍未消费，若目标prepare，data.pending必须明确regenerate，且不算目标消费 |
| 05 | 沿用同时ready/start、winner持锁等待loser拒绝、同run继续。若组合X2.b04，先竞争再由winner完成bootstrap及目标前缀、真实退出、新进程接管；两类进程身份分别记录，不把竞争loser当退出owner |
| 08 | 主边界完成后，先审计未额外破坏的primary（它可按边界预期invalid/unverifiable/technical_invalid），另有完整健康对照。从完整健康control的独立副本删除bootstrap optimizer_start并重新seal；要求报告缺口具体指向此项。primary技术无效会提前返回，不能把其删事件后的原technical_invalid充修改器通过，详见第6节 |
| 09 | 主边界及无损对照先保存。修改g0组件offset16384，size/旧hash保留，新进程或原合法owner调用select具体拒绝；另O副本保留旧seal，具体fullhash错。已关闭/退出owner的X2b04/b09必须用新合法owner做损坏select，不把inactive owner误认损坏拒绝 |

重复与修改器收据增加在原始观察日志，O业务事件不能使用重复event ID。S/O共用事先冻结外部输入；受控负例从冻结输入复制，额外记录mutation的对象、字段、前后hash及操作次序。共享输入检查对未变异字节逐项匹配，对预定变异精确验证差异；不能全局关掉比较，也不能被正确设计的负例先行触发harness_failure。

## 5. 覆盖验收不能只看must_reach的三个通用名字

保留原must_reach逐项实证，另写测试域`boundary_checks`与`modifier_checks`：实际generation数量/唯一索引、授权/accept顺序、准确拒绝原因、prepare是否存在、各rank实际文件与finalize集合、intent/manifest/token的存在性、cut期间PID及退出、恢复选择与pending义务、O最终保留/回滚集合、特定mutation报告。这些事实必须符合手写该边界规则，而非返回一个success字段。

每格保留冻结case/full semantic hash、共享输入、实际trace、父进程收据、源hash、O spec/report及无损对照。不得以不同case ID掩盖同一实际行为：去PID、时间、路径后，将实际S阶段顺序、变异参数和O图结构生成辅助执行签名；差异应定位到manifest规定的b/i操作，不要求正常恢复结果字符串彼此不同。重复签名先报告调查，不靠添加ID修补。纯O负例也必须有对应输入或图的实质差异。

oracle缺证据及技术无效分账：某格“负例合同验证通过”与其方法status不是同一字段。fixture模型证据不升级真实cell，不输出模型RTO的性能对比。恢复目标必须是冻结目标工作，其他组保存不结束其RTO；保留原oracle对此的独立测试。

## 6. 静态深层期望审查与已决定修订

已只读检查全部60种boundary及其交错展开，不只查默认good模板。结果如下；这是源码语义审查，不是执行通过证明。

| 深层state期望 | 全部落点 | 到达依据/缺口 |
|---|---|---|
| committed_target | 仅X1.b00的10格（8静态可执行、2原pending） | 当前只到prepare，缺optimizer/checkpoint/commit；确为清单遗漏 |
| promote_complete_candidate | 仅F2.b09的10格 | prepare、成功optimizer/scheduler、两rank×11文件与finalize、commit均有；cut在manifest fsync后token link前。恢复由driver明确新进程select，不通过重演optimizer补证 |
| verify_visible_complete_token_after_process_exit | 仅F3.b01的10格 | 完整保存前缀及真实link后切点存在，driver新进程select；i09破坏发生在主恢复成功之后 |
| recover_parent_no_target_intent | F2.b00 | accept后prepare前cut；select不得编造rollback intent，bootstrap保留 |
| recover_parent_with_uncommitted_obligations | F2.b01…08、F3.b00 | 有prepare；分别缺optimizer/end/成功收据、scheduler有效收据、rank、finalize等充分证据。F2.b06是record_evidence明确拒绝后cut。driver应验证缺失事实及rollback_intent，不仅比generation |
| ack_protocol_contract_pending | F3.b02…08 | 虽有完整commit，但ACK操作不可执行，继续pending，不被提交前缀掩盖 |
| idempotent / accept_current_epoch | X2.b00 / b04 | 仅承诺接收合同，不承诺保存；前者重复accept，后者真实退出+新owner授权接收。不得添加committed结论 |

其余state期望是指定拒绝、not_claimed或result_admission_contract_only，不隐含提交成功。F4一些边界甚至没有accept调用，承诺的是规定模型事件和资格边界，不能记录虚构accept实证。O的correct_recovered_model是另线手写图，不证明S有完整提交。

**主任务已裁决X1.b00：保留committed_target正例语义，补显式completion suffix，不降成prepared。** 下一实施单元实际修改前，完整归档现有compiler、静态测试、约4.7MB manifest到 `p2_evidence/schedule-v1/`，附文件全hash。只为X1.b00所有10格补optimizer→checkpoint→commit（包括仍pending的i06/i07），生成v2；记录这10格semantic hash的前后变化，其余590格的规范语义及case内容不应无故变化。编译器源hash作为manifest全局元数据会变化，须与逐case语义变化分开解释。已验收12格均不含X1.b00，r1继续按原来源保留；新driver对新清单仍重新验证12格。

补一个**独立手工负例验收**：在副本中删掉X1.b00 completion suffix而保持committed_target，语义前置验收必须拒绝，或真实运行只prepare必须报告缺target token/commit收据、不能passed。此测试不得通过重新调用compiler得到同样expected来自证；明确手写缺失trace/落盘事实。另分别删除optimizer成功收据、一个rank finalize、commit调用，证明深层判断不是“有prepare”或“有commit字符串”就通过。不得直接修改已归档manifest。

组合限制的最小解决：i08作用于**完整独立健康对照图**而非技术无效primary。原modifier已写complete_independent_control，故无需改manifest；先保留主primary的技术无效/缺证据/语义错误报告，再独立删除健康图的bootstrap optimizer_start，要求特定missing证据。这样F4.b02/b03/b09.i08不被oracle早返回遮住，也没有把技术无效判成业务恢复通过。保留r1 X1.b09.i08历史证据，后续路径统一时另验主错reward报告仍存在。

若实现中仍出现新冲突（例如某个变异导致缺少必需切点，或生产API只能拒绝另一种错误），先留该格not_executed/contract_review_pending；给出准确IDs、原操作与接口差异，再更新支持数量。当前未因上述可解决组合直接改408静态分类，也不宣称408已全部证明可执行。

## 7. 分段实施与最小代表回归

每段交付后由主任务跑新增针对测试及原12格；实现worker只跑必要定向检查。不要在一次实现中同时补184接口。

1. **X1.b00修订+F1扩展（首个bounded单元）。** 归档v1、按裁决生成v2、补独立缺suffix负例。仅为X1.b00新增完整提交正例，以及F1六种新样本数量/两种缺证据边界；把i01/02/03/04/05/08/09有限组合落实。保留进程路径。最小新增回归：X1.b00.i00、X1.b00.i02、F1.b03.i01（目标逆序）、F1.b06.i02（提交后幂等）、F1.b07.i08（缺tokens主例+独立缺start）、F1.b08.i09（缺logprobs主例+真实组件篡改）。每个F1 b至少i00一次，随后主任务跑F1全部80格及X1.b00的8可执行格，未实现其他cell继续not_executed。
2. **F2/F3保存切点扩展。** 新增F2.b00.i00、b01.i02、b02.i00、b03.i03、b05.i00、b06.i04、b07.i05、b08.i09，加已有b04/b09和F3两切点回归。必须逐切点核对intent、receipt、rank文件及token存在性。主任务跑F2 80+F3 16全部组合，确认每一个有cut的格都实际kill并新进程恢复。
3. **X1余下评分/载荷反例。** 新增b02.i00、b03.i08、b04.i09、b06.i03、b07.i04、b08.i08，并保留b01/b09。新增同组两sample合法policy差异正例只作辅助回归，不挪用600编号。主任务跑X1全72格（含首段8格），b05仍pending。
4. **X2身份/epoch/消费。** 新增b00.i02、b01.i00、b03.i03、b04.i05、b04.i09、b05.i00、b06.i04、b07.i00、b08.i08、b09.i09；已有b02/b09必须回归。主任务跑80格。b04退出/reacquire与b09后置合法owner的实现只在本段增加，不要求P1。
5. **F4全部事件模型。** 每个未覆盖b至少i00；另b02.i08、b03.i01、b07.i05、b08.i09、b09.i08覆盖重复样本、lost/retry模型及技术无效主例与后置缺证据的区分。主任务跑80格；不把这些case通过率称为自然窗口命中率。

阶段可以按进度调整顺序，但一个阶段完成后停止交回，不因“尚有408”自动扩成通用框架。首段暂不抽象oracle所有cell，也不实现X2接管新流程。

## 8. 子集运行、时限与最终账本

沿用直接脚本及重复`--case`参数；无需新增通用filter DSL。主任务可用一个短的独立选择脚本从冻结manifest取指定cell和`execution_status=executable`的ID，构造argv列表调用driver（不shell拼接）；driver再以自身能力表拒绝未支持格。示例已有入口：

```sh
.venv-tq/bin/python3.11 tests/ft/test_p2_schedule_driver.py -v
.venv-tq/bin/python3.11 tests/ft/p2_schedule_driver.py \
  --case X1.b00.i00 --case F1.b03.i01 \
  --output /absolute/new-stage-run --case-timeout 30 --total-timeout 120
```

以上新case在实施前会not_executed，文档不声称当前已支持。每case默认30秒，阶段总上限按`30×请求数+60`秒设置（80格2460秒、96格2940秒）；最终全部408上限12300秒，推荐仍按cell分批独立目录串行运行，不让长批掩盖失败。真实耗时可由主任务测后缩短，超时一律harness_failure，不算方法RTO。磁盘预估先参考r1实际产物再预算；保留失败、不覆盖重跑，清理只有本次持有Popen。

每批验收同时检查：请求集合=能力已宣称支持集合；每格原must_reach及细分检查完整；先对照后变异；预定负例命中特定语义；实际执行签名差异有操作证据；没有schema早拒冒充主场景。4种核心合同仍要正反对照（K、授权版本、保留消费唯一、实际load连续性），不能只看整批exit0。

最终若实际408全支持且全部执行，账本才可写408 CPU合同passed/8限制/184待补；否则按精确case列passed/failed/harness_failure/可执行未跑/新增contract-review-pending。不要把各阶段重跑或control/mutation报告重复计case，也不能把最初12加到408之外。旧r1的12不是新源码408批次的免测凭证。此规划到此停止，未增加任何执行结果。
