# i06/i07：真实 CPU 控制器接入的最小计划

2026-09-20。economy-dev 根 `01a0bd89-a9c3-7d00-b16d-4a0a766de676` 已显式查得 enabled；Astra medium 规划，不递归。本次只读源码、写本文，不执行实验。已有408格合同通过；独立审计12测试通过、408全解析、384 core联合签名，8组仍为F2.b01–b04的规范marker等价。它不消除以下pending。

## 1. 可直接复用和不能冒充的能力

- `scripts/ft/run.py::Journal.add`追加JSONL并flush/fsync；`execute(..., resume=True)`在相同冻结输入且出现未闭合attempt时直接返回technical_invalid，不重启子进程。`run_attempt`顺序为armed→fired落盘→真实pidfd SIGKILL→wait确认→observed落盘。有真实两attempt现成测试，但`test_uncertain_fired_never_replayed`是人工写journal，尚未实际杀死控制器。
- `scripts/ft/faults.py::OwnedProcess`、socketpair/barrier、nonce与直接子进程身份验证可直接复用；要求Linux Python真的提供pidfd，无能力应不执行，不降级到裸PID。
- `scripts/ft/namespace_run.py`支持有序多event、可信后代登记/退出观察，`injection_assignment`支持活控制器内的already_fired；**它明确拒绝任何resume**，phase仅在内存，`signal_intent`不能冒充持久恢复成功。`atomic_json`也未提供power-loss级fsync保证。
- `container_run.py`的CPU封闭容器、全ID/nonce清理和lease可复用来包住测试。不改GPU profile，不创建新的通用进程管理器。
- P2现有`Worker`的生成、state授权/接纳、prepare、组件保存方法可以用作CPU业务fixture；原`Child`/stdin启动协议和父进程直接kill不是P1协议，不能仅替换日志标签声称接入。原oracle单fault/单target，不能把两次独立固定图的pass相加当多故障正确性。

## 2. 清单中必须先澄清的真实顺序缺口

184 pending = 全60 boundary的i06/i07共120格 + 56个ACK组合 + 8个训练变换组合。120中与ACK/变换/N/A仍重叠；按基础boundary，有51个CPU可执行、8个接口欠缺、1个N/A，因此102格仅是未来候选上限，不是本轮可完成数。

当前compiler把i06三个操作放在bootstrap之后、target生成/prepare之前；例如F2.b09真正主切点却在target manifest落盘之后。冻结清单的`primary-fault`也未定义其进程或业务证据。不能在prefix杀一个哑worker，再声称命中F2.b09；也不能静默移动操作而沿用原semantic hash。i07只追加`second-target`标签，没有第二份冻结输入、实际工作进程/恢复epoch/具体切点。

**决定：先交真实控制器协议探针，不降低184 pending计数。** 随后为少量具体格提出v3修订（完整归档v2 compiler/test/manifest及hash、逐格变化账本）；批准后才把i06放到实际主故障的fired接缝，把i07第二work/cut明确展开。未修订格一律保持pending，不批量填充102格。

## 3. 下一实施单元：两个真实i06切点探针

仅新增 `tests/ft/p2_controller_probe.py`、`tests/ft/test_p2_controller_probe.py` 和实施说明，不改生产P1/P2。探针子进程直接调用原`preflight/execute`。测试局部包裹`Journal.add`：调用原实现完成fired的fsync后通过专用管道通知guardian并阻塞；另一模式在即将调用原observed追加之前阻塞，此时原pidfd发送与wait已经真实完成。只在测试子进程安装固定两处包装，不跳过原方法、不手工制造fired/observed、不暴露生产可配置注入DSL。

固定两个场景：

1. `fired_durable_before_send`：目标确实在真实state.prepare之后的barrier等待；fired完整持久后，guardian精确SIGKILL其自有控制器Popen。不能声称目标受到了计划SIGKILL；socket关闭引起目标退出属于控制器丢失的清理后果。
2. `target_reaped_before_observed`：同一真实边界，原控制器已经通过pidfd杀死目标并wait得到-SIGKILL；observed尚未持久时杀控制器。此时实测目标死亡由已完成wait的外部握手收据和pidfd证据支持，不补写observed。

两个场景都由**新进程**以完全相同frozen/run-dir调用原execute(resume=True)。预期是“uncertain interrupted attempt; no replay”的technical_invalid，fired仅一条、无新增armed/observed/process_registered/attempt_start；不是训练恢复成功。第三次对已终结run调用应拒绝，冻结输入篡改应拒绝。

所有参与者在CPU私有namespace容器中；外层现成`container_run`使用空schedule、唯一launcher为probe guardian，内层控制器仍是自己的直接Popen。guardian只用持有的Popen/pidfd杀控制器；目标走原P1 direct-child pidfd。控制器死亡后barrier断连使worker退出，未退出的孤儿由私有namespace既有lease/容器全ID清理兜底；不在宿主按日志PID补杀、不声称guardian拥有被杀控制器的Popen子对象。每次记录容器全ID、namespace、退出观察及残留清理；外层执行完成不能覆盖内层technical_invalid。

准备阶段fsync创建的测试目录项/初始文件，只承诺进程SIGKILL下的已完成fsync，不证明机器断电/分布式一致性。journal尾部截断、坏JSON应明确拒绝且不注入，不设计本轮journal修复协议。

## 4. i07最小真实双故障：先控制协议，再业务oracle

原P1 `run.execute`已可用3个冻结attempt：attempt0在第一真实边界kill；attempt1新进程实际load后到第二边界kill；attempt2新进程实际load并完成。它是明确的CPU试验调度，不宣称AReaL native重启，也不增加A基线恢复能力。先选单一状态合同：F2.b09完整candidate第一次补提交，再独立second-target完整candidate第二次补提交；每次kill均发生在manifest fsync之后、token之前。可复用现有固定`os.link`测试拦截，但触发后走P1 barrier及pidfd注入，不再走旧P2父进程kill。

最少补四个接缝：

- 冻结primary/second-target两个独立K8输入、logical update及前后消费集合；第二intent的data.cursor/drawn/consumed必须接上真实已选父代，不能复用Worker.intent当前只假设bootstrap前缀的实现。
- CPU adapter每attempt读取同一state-run、真实acquire_owner退出证明、select_recovery，记录实际所选generation和已载入文件全hash；确认load后才进入下一目标。第一代未退出或CAS不符立即拒绝。
- 真实boundary proof由adapter在检查当前intent/optimizer/rank/finalize/manifest/token缺失后保存并fsync，绑定run/event/attempt/identity及文件hash。P1固定evidence仅携带稳定合同项，独立验收必须读proof和文件；不能认为barrier收到某字符串即已到切点。guardian/observer记录不得被adapter用于选择恢复状态。
- 两个不同event nonce、两次真实pidfd发送/退出、非目标release、最后恢复及final retained链可回查。事件2只在事件1退出、attempt1真正load之后触发；相同target别名不算第二目标。

原oracle仍不能完成`audit_multiple_targets`。第一子单元用独立只读CPU控制审计检查上述真实收据和持久链，结果字段叫`controller_protocol_pass`，训练/oracle=`not_evaluated`，**i07正式case仍pending**。下一单独规划才定义多fault全局祖先链与每目标消费/RTO；不得按fault切成两张相互看不见的图、不得从state产物生成expected，更不触碰用户待答的正式评分合同。真正训练tensor/optimizer/GPU连续性仍需真实backend。

## 5. 门槛、交付顺序与计数

第一个bounded brief只实施第3节及两个无故障control；可直接复用P1，不需先改其API。建议未来入口：`python -m unittest discover -s tests/ft -p 'test_p2_controller_probe.py' -v`，具备pidfd的CPU容器内运行；每probe 20秒、整组180秒，超时保留证据并按所有权清理。不跑600矩阵。

必须通过：两个切点真实控制器-SIGKILL；一个场景目标尚未被注入、一个目标已被真实注入；真实resume无新注入；重复ready无重复signal；错误nonce/未知身份/坏journal拒绝；不中断control能到observed；退出前fired尾字节与fsync握手绑定；所有信号/清理目标皆有可验证所有权。不得只检查fired数量而忽略是否发生真实process death。

第二bounded brief再实施第4节单组两个目标与三attempt，先不批量扩边界；应另有事件2提前到达/错target/第一代仍活/第二代文件不完整等反例。之后才提交v3具体case修订和多目标oracle范围，由独立审计扩schema记录真实API/进程及marker分层。

本计划立即可交付的新增600完成数为0；这是两个真实协议缺口的最小验证路径，不是用标签增加模拟数量。F1实际生成并发、F2真实optimizer中间态、F4自然reward窗口与lost/native retry必须GPU/训练后端；F3 ACK和X1训练归一化不属于这两个modifier，本轮绝不合成ACK、也不以identity变换代替正式训练变换。

外层生存/清理是强制门槛：guardian与CPU容器宿主supervisor必须在内层控制器SIGKILL后继续存活；guardian在创建控制器后立即保留pidfd并核对身份，用该fd发SIGKILL，再Popen.wait确认。不能依赖被杀控制器的finally。目标断连退出只是快路径，所有孤儿最终由私有PID namespace终结与已登记容器全ID清理收敛，需验收退出/容器不再运行后才结束probe。无隔离或无法证明所有权时整项not_executed，禁止在宿主直接试杀控制器。恢复审计读取**原journal**且保存重启前后字节/记录集合，明确没有新attempt、新worker PID或第二次注入；保留不确定状态，不声称训练恢复。
