# Native r4 timeout：local shell/tee 与孤儿进程诊断

2026-09-20。仅只读源码/既有证据诊断；未修改代码、未运行CPU/GPU、未清理其他任务。native r4 为有效单trainer注入后，在本工程运行期限内没有进入原生retry；不是因A未恢复而重新归类technical_invalid。formal_samples=0。

## 1. 已证实结果与计时口径

证据根：`p1_evidence/native-gpu-r4/`。

- `injection-verification.json`：一次SIGKILL，完整身份与pending trainer绑定；step0完整save发生在第二次successful optimizer之前，注入位于post_optimizer_pre_save。
- `launcher.log:213–244`：torchrun PID649报告trainer PID667退出码-9，并抛ChildFailedError；证明训练进程确实失败，不是ready但没杀到。原生launcher只出现run_id=0,is_recover_run=False。
- `events.jsonl`：signal_sent=549314613129771、process_exit_observed=549316274555426、method timeout=549925021360442（同controller_monotonic_ns时钟）。**signal→timeout=610.408230671秒**；pidfd退出确认→timeout=608.746805016秒。900秒是controller整run预算，含预检/初始化/故障前训练，不能写成故障后完整900秒恢复窗。
- `independent-verification.json`：恢复验收不通过；没有第二trainer incarnation、没有run1/恢复后update/final checkpoint；method_observation为timeout。该结果不证明step0文件不可加载，只证明本次没有进入新进程load。
- `supervisor_result.json` / `cost.json`：host supervisor墙钟911.4975174730644秒，四卡分配1.012775019414516 GPU·小时；cleanup_confirmed=true、cleanup_errors=[]。实际controller执行完成/host failure=null不等于方法恢复成功。
- 完整CID `0d99eb8107330da144991f10902d6861d9935168b606d59b76aec95abc8c0aab`、network ID `67dbd0f6fbc959e1892ee790ff979be1caa5ac3004e1bc53f3e6d425f2538462` 的清理及独立删除确认见inspect/network/independent-cleanup证据。

真实恢复RTO为空/未观测；不能填900秒、610秒为“实际RTO”，也不能把此次工程限时直接套用正式单故障900秒失败惩罚。此前r1/r3预检/启动失败与本次有效注入后的method timeout须分栏，不能混为同类。

## 2. 为什么torchrun退出了，native local却没有retry

源码与现场共同给出完整阻塞链：

1. `third_party/areal/areal/infra/launcher/local.py:123–164` 的 `LocalLauncher.submit_array` 先 `build_target_cmd(..., use_stdbuf=True)`，再拼接 `2>&1 | tee -a <log>`，以 `subprocess.Popen(...,shell=True)` 启动。`self._jobs`保存的是**整条pipeline的shell Popen**，不是actor trainer、也不是torchrun实际PID。
2. `LocalLauncher.wait:198–260` 对该shell PID做 `psutil.Process(pid).status()`。睡眠/等待态被映射RUNNING；循环每2秒检查。它不检查torchrun退出状态、trainer PID或管道写端。local_main调用wait没有内部timeout参数。
3. SIGKILL667之后，torchrun649已经报告ChildFailedError并退出；但 `postkill-stdio.json` 显示 shell648仍为PID7的子进程，其tee650仍存活，stdin为 `pipe:[2796824288]`。
4. 同一快照显示10个PPID=1、PGID=667的孤儿python进程，stdout/stderr仍指向该pipe写端：1132、1134、1137、1141、1148、1152、1156、1162、1631、1632。只要还有任何写端打开，tee不能收到EOF；即使这些进程从此不输出任何字节，也会保持tee读等待。
5. 前8个PID可由本run `areal/pilot_events/events-<pid>-*.jsonl` 中真实reward_start/reward_done事件直接确认是reward worker。1631/1632目前只确认为继承写端的孤儿python，不能无证据把10个都称reward worker。PPID1是trainer死亡后的namespace收养，不是controller创建了这些评分进程。
6. shell等待pipeline末端tee，tee等待孤儿持有的写端；shell因此继续RUNNING。`local_main:432–450` 只有wait抛JobException等之后才会stop_all、run_post_exit_hook、检查recover.retries并递归进入run1。此次这条异常/重试分支没有被触发。

这解释了“retries=1但没有run1”：不是次数用尽、不是checkpoint检测失败、不是new trainer load错误，而是**上层作业退出检测被日志pipeline生命周期拖住**。`set -o pipefail`不能解决这个活性问题：它只影响pipeline完成后的状态，仍须等待tee终止。

`api/reward_api.py:AsyncRewardWrapper` 使用共享ProcessPoolExecutor并注册atexit shutdown；SIGKILL不会执行trainer的atexit/finalizer，所以不能指望该清理路径关闭子进程stdout。源码与现场支持该机理。当前证据不足以逐一解释所有孤儿最终阻塞在哪个Python队列/线程，也不需要该推断才能证明pipe EOF阻塞；不宣称已采集它们的栈或全部FD flags。

当前P1 PID1在run结束时整体namespace退出，能够收尾；它没有在单trainer被杀时主动关闭孤儿FD或杀整个trainer PGID。若现在加这类动作，就从“精确杀单trainer”变为额外进程干预，会给方法提供未冻结的生命周期恢复能力。此前真实torchrun CPU fixture没有创建实际reward子孙，所以那次通过并不覆盖本次pipeline被孤儿写端保持的情形。

## 3. 不修改A时，可以继续什么

**推荐下一步保持r4失败，继续与自动重启无关的CPU/无故障接缝工作。** 不再原样重复占四卡等待同一现象；没有新证据/修订不应把重跑当修复。

可推进：

- P3a draw WAL/loader/blob的已通过CPU合同与后续独立授权的R数据/attempt/tensor接口合同；它们不依赖A自动恢复成功。
- 独立CPU最小复现此退出检测问题：真实local submit_array／实际torchrun或普通command，子孙保持stdout，父进程退出，检查shell/tee仍RUNNING与retry分支不可达。仅诊断，不修改A；若做反事实关闭FD必须在单独CPUfixture中明确标记干预，不能算原生恢复通过。
- R无故障真实持久化/实际update/checkpoint闭环和**明确标记为新进程手动启动的load round-trip**可以作为组件验收规划；它们不能报告native automatic retry、RTO或FT恢复收益。是否启动另需正常资源/实施门禁，本诊断不自动发起GPU。
- 继续完善独立oracle证据真实性/最终保留链反例，baseline无token、未重启、合法回滚分别按规范判断。

不应设置“A必须恢复成功才准记录有效实验结果”的门禁：有效命中后A timeout是真实method outcome，应保留，不要求把A修到成功后才承认分母。P1“安全命中与收尾”此次通过；P1“所选local部署可以自动trainer恢复”的**能力门禁没有通过**；这两个结论不冲突。

但不能把门禁直接删除后宣称A+R故障闭环具备执行条件：R目前只补数据/提交，不拥有另一个进程恢复器。同一local pipeline若同样卡住，R也不会获得新trainer进程。允许两臂均timeout作为真实结果，与“已完成R故障→replay→新update闭环”是不同命题。后者仍需实际可触发的原生重启路径、或被明确批准/披露的公共launcher修订，不能由controller替R拉起下一代。

继续当前未经修改的A时，应如实登记本部署该工程切点的失败；不能事后N/A、技术无效、偷偷删掉reward pool、重定向其FD、改SIGTERM、杀整个trainer树、缩小pool或给torchrun新增内部retry来换结果。这些都会改变方法/部署/故障语义，需要独立设计决定。

## 4. 官方替代入口的只读核查

| 入口 | 本checkout事实 | 是否是同部署的已验证替换 |
|---|---|---|
| `infra/launcher/ray.py` | 仍有SPMD `_AllocationMode`、原生check_if_recover和recover.retries；submit使用ray.remote(run_func)，wait通过ray.get/RayTaskError判FAILED，异常后stop_all并进入ray_main(run_id+1)。不是local的shell/tee等待模型。main调用ray.init；有runtime_env和placement groups。 | **不是**。需要独立Ray服务/进程/资源拓扑与权限、env传递、实际失败类型和cleanup验证；Ray worker突然死亡是否由当前catch捕获不能只凭RayTaskError分支保证。没有现场Ray部署能力核验，不推荐直接迁移并称等价。 |
| `infra/launcher/slurm.py` | 同样解析SPMD allocation，构造sbatch/srun，查询scheduler作业状态；JobState.FAILED/NOT_FOUND可进入原生retry。 | **不是**。需要真实Slurm allocation/partition/凭据，不能把现独立Docker本地四卡替换为一个命令就称同部署；未核验现场调度环境。 |
| 单controller `scheduler.type=local/ray` | 是另一条trainer调度路径，不等同SPMD local.py。RecoverHandler._ensure_recover_supported对GatewayTrainController（v2）明确NotImplemented。 | **不是**。v1局部支持也不能推导同拓扑/状态/故障恢复成立；需重新审计实际engine/controller类，不能顺从deprecated提示切到v2后沿用当前恢复结论。 |
| 直接torchrun增加max-restarts | torchrun的CPU一次重启已验证，但绕开/叠加官方local_main恢复层，故障后server/name-resolve/data语义不同。 | 不作为本次“未经修改官方local恢复”的替代证据，也不能自动给A加这一层。 |

本诊断只读现有源码，没有查询/启动Ray或Slurm资源，没有网络/依赖安装。替代官方路径可能绕开当前特定等待模式，但尚不能证明其清理、状态恢复、pending或本deployment可行。

## 5. 若决定公共修补：具体待审设计，不自动实施

这是可选的**新共同基线版本**，不是把r4改判成功。最小待审补丁可只针对日志pipeline与作业句柄的耦合：

- 目标 `LocalLauncher.submit_array`：保留现有build_target_cmd/env/stdbuf/用户command字符串，去掉 `| tee -a ...`；以追加打开的角色log文件作为Popen stdout、stderr=STDOUT，仍保留现有shell执行命令和 `_jobs` 生命周期。父进程启动后关闭其文件句柄，子孙继承普通文件FD不再使shell等待日志读者EOF。不是仅在wait外层增加超时并强杀。
- 第一版不偷偷改正常COMPLETED也retry的原有逻辑，不增加controller重启器，不改故障目标/信号。日志不再自动tee到launcher stdout，须在文档明确：角色log仍完整，公共observer/离线collector读相同role log；如需console镜像，可用不参与job completion判定的独立有界只读collector，不能让collector又成为wait依赖。
- 上述改动只解决检测活性；**不保证孤儿reward进程或异步writer清理**。原stop_all依赖父子树，reparent后的孤儿可能不在原shell后代中。不能把“触发run1”当作完整生命周期恢复通过。若还需每job进程组/cgroup所有权和停止策略，必须另列更大补丁、真实归属认证及两臂披露，不能隐含在日志修补里。
- 必需CPU反例：普通成功/失败command、实际shell参数/env/stdbuf一致、子孙继承stdout且父失败、日志持续输出/异常/有限flush、无关进程哨兵、COMPLETED原retry语义；用精确job身份验证，不按名称杀。之后才做两臂共同受限GPU工程验证、实际full-state load与资源无泄漏。
- 保存原版与补丁版源码完整diff/hash、镜像/配置、失败r4及新run分别归档。A与A+R必须相同公共补丁、相同资源/故障/日志观察开销；报告标为“官方AReaL＋公共launcher日志生命周期修补”，不再称未经修改的官方基线。旧结果不得与新版本成对混用。

`third_party/areal/AGENTS.md`明确 **“Ask first … Changing launcher or scheduler logic.”** 这项补丁属于该边界，即便通过项目monkeypatch实现也不能绕过。若选择此路线，先准备上述范围的可审查补丁/CPU证据，再按用户当前授权与该规则处理最终确认；本文不实现、不申请启动、不自动扩大授权。对于当前只读诊断，没有需要用户确认后才能完成的工作。

## 6. 主报告应引用的精确事实

1. 预检通过、真实step0保存与第二optimizer后单trainer SIGKILL命中；注入成功≠恢复成功。
2. torchrun649已报告trainer667的-9/ChildFailedError；local只run0，没有新trainer/load/update。
3. shell648/tee650及pipe2796824288；10个孤儿写端，其中8个有reward事件证明、2个角色未证实。保留“源码＋一次postkill快照支持的退出检测阻塞诊断”，不杜撰栈。
4. 900秒总run期限；signal后实际观察610.408230671秒。恢复RTO未观测，formal_samples0，不套正式900秒惩罚。
5. supervisor911.4975174730644秒/1.012775019414516 GPUh；CID/network独立删除；controller/cleanup通过而method timeout。
6. 原生自动trainer恢复能力门禁失败，不抹掉A有效失败结果；CPU R draw合同仍是独立限定通过，不能因此声称R故障闭环或泛化“原生A没有恢复”。
