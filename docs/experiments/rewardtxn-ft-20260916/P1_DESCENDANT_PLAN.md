# P1 可信后代登记与原生恢复接入最小方案

日期：2026-09-20。状态：**仅规划，未实现、未启动测试/GPU，不升级能力表**。economy-dev 根会话 `01a0bd89-a9c3-7d00-b16d-4a0a766de676`，Astra medium planner。已有直接子进程合同11项通过；真实 RLVR 三步及新进程 resume 的限定证据见 [PILOT_REPORT.md](PILOT_REPORT.md)，不能代替故障恢复验证。

## 决策与现场依据

采用**每 run 独立 Docker PID namespace/cgroup，容器内 P1 controller 为 PID 1，官方 SPMD launcher 只启动一次**。控制器只负责认证、观察、一次精确注入及整个实验的最终收尾。launcher 决定是否和如何恢复；不把现有 CPU `attempts` 启动序列用作基线恢复。

| 已核实文件/符号 | 对实施的约束 |
|---|---|
| `scripts/ft/run.py:run_attempt` | 目前只有直接 Popen 角色；第一次 pidfd_open 失败仍可能留下未登记子进程。后代不能套用 Popen.wait。 |
| `scripts/ft/faults.py:OwnedProcess` | pidfd 信号可复用，直接父子和 PGID=PID 的条件不适用于 torchrun/reward pool；PID、start-time、boot-id、cgroup 仍保留。 |
| `scripts/ft/areal_pilot.py:main` | 已在 trainer 的真实入口安装 hooks，适合在 PPOTrainer 构造前登记 trainer，不需要修改第三方入口。 |
| `scripts/ft/areal_pilot_hooks.py:ObservedRLVRWorkflow._compute_rewards / observed_gsm8k_reward` | 分别位于真实生成完成与官方 ProcessPool 内实际评分 callable；可登记实际评分进程和关联 invocation。当前异步写盘日志没有控制收据，不够作瞬时故障判据。 |
| `third_party/areal/areal/infra/launcher/local.py:LocalLauncher.submit_array`（约149行） | 官方链包含 shell/tee，不是直接 torchrun 子进程；不能根据一个 PGID 或命令名决定所有权。 |
| 同文件 `local_main`（约267、410、424–451行） | 原生 `run_id` 递增、`recover.retries` 上限及10秒等待；COMPLETED 也进入 recover_states，正常结束可能触发额外原生重试。保持原行为并记录。 |
| `third_party/areal/areal/api/reward_api.py:AsyncRewardWrapper.__call__ / _recreate_executor` | BrokenProcessPool 会原生重建并重试，reward 故障未必引起整个 SPMD job 恢复；不得关闭该能力或强行提升为整 job 重启。 |

根用户 AGENTS 要求中文、局部改动、明确假设与验证。本轮没有发现覆盖 `scripts/ft`/本 docs 目录的附加 AGENTS；已读取 `third_party/areal/AGENTS.md`。其 “Changing launcher or scheduler logic” 属 Ask first 范围。本方案不修改或 monkeypatch launcher/scheduler 决策，不新增依赖；若后续需要改这些逻辑，必须另行准备具体补丁并处理该约束。本轮不递归委派。

## 1. 隔离、启动与失联收尾

1. host supervisor 使用 argv 数组执行 `docker create`，保存返回的**完整 container ID**和独占 run_nonce label，检查其镜像 ID、PID 模式、挂载、GPU UUID、资源限制；先取得 CID 再 start，不以名称搜索并杀容器。不用 `--rm`，完成后先采集 inspect/退出证据，再按已保存 CID 清理本容器。官方 launcher 内部已有 shell 实现原样保留，外层不再拼 shell 命令。
2. 容器使用默认独立 PID namespace，显式拒绝 `--pid=host`、共享他人 PID namespace、特权模式、Docker socket 挂载及共享可写 cgroup。维持已有内部网络以满足 gethostip；不用曾失败的 network=none。源码只读、每 run 输出独占。使用已验证非 root UID，并去除不需要的 capabilities，防止伪造发送者身份/迁移 namespace。
3. **直接 exec Python controller 为 PID 1**，不用前置 timeout/shell 或 `--init` 把其他进程放成 PID 1。启动自检 `getpid()==1`、pid namespace inode、cgroup、pidfd 能力，任何不符在启动 launcher 前失败。controller 安装 SIGTERM/SIGINT 有界退出处理；不忽略 SIGCHLD、不设置 SA_NOCLDWAIT，正常跟踪并 reap 直接 launcher，另清理被收养的孤儿僵尸。
4. 首次 Popen 后 pidfd_open 无论因 FD 耗尽还是其他错误失败，都进入 **fatal run abort → PID 1 退出**；不继续运行，不重试启动，不退回 PID kill。独立 namespace 的 PID 1 退出会由内核终止其余进程，闭合当前 provisional 之前的漏洞；若只是一般 Python 函数返回而 PID 1 仍活着，则不算闭合。
5. controller 崩溃/被 host SIGKILL 同样终止本 namespace；controller 卡死由 host supervisor 的独立 heartbeat/绝对期限兜底，按已保存 CID 执行有界 stop，超时再 kill 本容器。host 与 controller 之间的 heartbeat 断开也作为技术失败；容器内另有绝对 run deadline，避免 host 掉线后无限运行。所有清理命令本身设置超时；Docker daemon 不可达时报告清理未证实，不能报无残留。
6. 成功注入后保持容器和 launcher 活着，让原生恢复运行；只有 controller 技术失效、总窗结束或方法自然结束才做容器收尾。记录 stop 前容器状态、退出码/OOMKilled、资源成本，等待同 CID not-running 并验证 cgroup 无本 run 存活任务。保留输出，不触碰外部容器、共享 Ray、其他 GPU 任务。

PID 1 退出终止 namespace 是内核语义，仍须在所用 Docker 配置做验收；controller 的 daemon heartbeat 线程不能取代 host 的卡死兜底。[Linux PID namespaces 手册](https://man7.org/linux/man-pages/man7/pid_namespaces.7.html)

## 2. 可信登记：内核身份 + 启动祖先链，nonce 不当所有权

最小新增一个容器内 Unix `SOCK_SEQPACKET` listener，路径位于本 run 私有目录；**每个进程自己新建连接**，不把 parent socket 跨 fork 直接复用。trainer 在入口登记，评分进程在第一次实际 reward callable 登记。registration/ready/receipt 全部用 recvmsg 并要求 `SO_PASSCRED` 的 SCM_CREDENTIALS；`SO_PEERCRED` 可作连接初检，不能单独证明继承 FD 后每次实际发送者。

登记步骤：

1. controller 先从实际 Popen 得到官方 launcher root 身份与 pidfd。新连接的 kernel PID/UID 必须与声明一致，UID 为冻结 UID；JSON 中的 PID、rank、nonce 只提供关联，不能授予信号权。每条消息要求同一登记进程凭据，拒绝其他子进程借用连接。
2. 读取 kernel PID 对应 `/proc` 的 start-time、boot-id、PID namespace inode、NSpid、cgroup、PPID/PGID；必须处于本次已认证 namespace/cgroup。容器内 cgroup 路径可能只是 `/`，因此不能把路径字符串当容器身份；要与 host CID→init PID/namespace/cgroup 的 inspect 证据共同绑定。
3. 沿 PPID 实际读取到持有 pidfd 的 launcher root，逐级保存 `(pid,start-time,namespace)`，允许 shell/tee/torchrun/trainer 等多层、不同 PGID；反向复查整链，链变化/祖先消失就拒绝该次登记。孤儿到 PID 1 不能仅因“仍在容器”补授予目标身份。已经合法登记后发生 reparent 要记录失去原链，默认不接受新的注入，交原生退出或技术 abort。
4. sender 自己 `pidfd_open(os.getpid())`，通过 SCM_RIGHTS 发送一个 pidfd；controller 将 fdinfo 中本 namespace PID 与 kernel 凭据及 `/proc` 快照交叉验证，再重复检查身份/祖先链。这样避免“登记消息已排队，controller 后按旧数字 PID 打开新进程”的窗口。FD 缺失、多个 FD、非 pidfd、截断控制消息、已退出/UID不符均拒绝并关闭收到的 FD。若部署不支持自 pidfd 传递/读取校验，能力不足 fail closed；不改成只读 worker 提交 PID。
5. role/rank 在通过所有权后，结合实际入口、trainer ancestry 和冻结 topology 登记。reward role 必须是已登记 trainer 的后代，且携带真实 group/sample/invocation；不能把 SGLang/observer 伪装评分目标。进程 incarnation 与 native launcher epoch 由 controller 分配，worker 不能自行递增 epoch。

此认证防误目标、PID重用及普通错误通知，假设冻结方法代码与同容器受控进程不是主动恶意攻击者；它不声称抵御同 UID 进程任意 ptrace 篡改。nonce 负责隔离与幂等，不是密钥认证。Kernel 凭据和 pidfd 传递依据见 [Unix sockets 手册](https://man7.org/linux/man-pages/man7/unix.7.html)。

## 3. 信号和 observed：不得等待非子进程的 Popen

保留现有 journal 的 ready→armed→fired→observed→release/abort，但字段分清 `signal_intent`（写前日志，可能未发送）、`signal_sent`（pidfd_send_signal 成功）、`process_exit_observed`。现有 fired 意义为 uncertain intent，迁移时不能拿它冒充实际命中时间；中断处仍拒绝自动再杀。

- armed 要求 frozen event_id/group/sample、native epoch、incarnation、nonce 与真实边界证据匹配。紧邻发送前重查 pidfd 仍存活、身份与所属关系；仅向该 pidfd 发送 SIGKILL，绝不 PGID 广播或按名称选择。
- 对非 controller 子进程，使用 poll/selector 等待 **pidfd 可读**来证明已退出，不调用 `Popen.wait()`/强行 `waitpid()` 抢官方父进程收尸。`waitid(P_PIDFD)` 不保证可用于非子进程。pidfd 可读不等于“退出码=-9”：日志明确存为“SIGKILL 成功发送＋同 pidfd 随后退出”，不伪造 wait status；若父进程已有可读取的官方 exitcode 证据再追加。
- 必须在实际边界仍成立时发送；若退出在发送前发生、条件已过、信号失败或 observed 超时，保留 technical_invalid。自然退出与并发退出的歧义不能靠 pidfd 可读单独消除。
- observed 后释放所有本事件仍存活 waiters，收齐幂等收据；目标不等 release。控制失联/收据超时属于技术失败，触发容器有界收尾。不能把屏障缺陷计入方法 RTO。

pidfd 支持 poll 退出通知，但读取/等待子进程状态存在单独限制，见 [pidfd_open 手册](https://man7.org/linux/man-pages/man2/pidfd_open.2.html)。

## 4. native attempt 与 schedule：只观察，不接管恢复

官方 local launcher 的完整 argv/config 在容器内原样运行一次。保持 `recover.mode`、`recover.retries`（正式上限3）及 reward 原生 retry 配置，冻结并分别记录。runner 不修改 RecoverInfo、补 pending、重投任务、重启评分池或重新执行 launcher。

通过每轮 trainer 实际祖先链中的 torchrun incarnation 区分 native launcher epoch；多 rank 归属同一 torchrun。与官方 `local_main ... run_id=...` 日志交叉核对，只有一致才激活对应 schedule，不能把 worker 自报 attempt 当真。旧 epoch 的 ready/receipt 不得命中新 epoch。评分池 replacement PID 只增加 role incarnation，不增加 native launcher epoch；invocation 次数和可读取的官方 retry 日志分栏，无法精确归属时标证据不足，不猜层级。

在同一持久 controller 内，方法 native restart 不改变 run_nonce、不清空 schedule；完成事件只记一次，后续事件继续等待实际 epoch/逻辑更新条件。controller 自身重启不重建未知进程或重放 uncertain fired；旧容器先关闭，原 run 标技术无效。CPU `--resume` 不能冒充真实训练恢复。

**COMPLETED 也重试是当前官方行为。** 短 probe 设置有限 retries 并保留额外启动/原始退出码；达到目标工作量后仅按预先冻结终止规则收尾，不能事后改 exit 1 为0。若这一行为妨碍正式停止，另提 launcher 修复而不在 observer wrapper 中吞异常改变基线。

## 5. F4：先认证，再验证自然切点可命中

pilot 只证明自然候选窗口存在，约0.65–8.66ms；并未证明控制器能命中。FT-v1 的“实际执行到K/2”仍需冻结为具体开始/完成边界，不能把两个定义混用；建议先分别报告第4次评分开始和第4次评分完成的可观测性，正式采用哪个须在注入前写入 schedule 语义，不能看到收益再选。

预先登记可能评分进程、建立连接与 pidfd，预先 armed 指定逻辑组/attempt；不要在毫秒窗口里首次建连接、扫 `/proc` 树或全量 hash。真实 generation/reward 通知通过非阻塞控制通道同步关联序号和 controller 收据时间；现有异步 JSONL 仅作旁证。发生目标边界时只允许暂停**该目标 callable 的公共故障屏障**，其余 reward/generation 保持原调度，rollout asyncio 路径不可同步阻塞。

即使暂停单目标，其他评分也可能推进；必须记录边界进入到 signal_sent 的时间区间及并发事件，验证实际故障时冻结谓词仍成立。若只能证明过去某刻有4个完成、发送时数量未知，判未证实命中；不能靠迟到日志顺序假称同时成立。不暂停全组奖励、不设置全K生成门闩、不sleep所有reward、不串行化评分、不延长自然窗口、不造远程RM。若严格同时性需要这些改变，停止该映射并按原计划修订设计；保留原F4主检验未执行状态。

先做预先限定次数的工程命中探针；每次未命中和控制成本都保留，不转挑获利组。正式 target 是冻结更新附近对应组，不能选择事后窗口最长组。评分池被杀后的额外受影响组/任务必须记录，两臂杀同类角色、同样范围，不能把一个进程共享的损失算成仅一个sample。

## 6. 实施顺序与验收门槛

只对本项目 `run.py`/`faults.py`、必要的一个 AReaL 公共控制 hook 文件及对应 `tests/ft` 做局部变更；host Docker 生命周期入口只承担 create/start/inspect/wait/stop，避免通用插件框架。pilot 原观测代码保留为来源，可在确定入口引用最小控制模块，不改第三方源码。

| 顺序 | CPU验收（全部真实进程，证据独立保存） |
|---|---|
| 1 容器生命周期 | 同时启动无关哨兵容器；本 run 构造三层子进程。注入第一次 pidfd_open失败、register失败、controller异常/被kill、host通道断开、controller卡死；在冻结期限内本CID退出且本run任务清空，哨兵存活。无PID kill回退。 |
| 2 可信后代 | shell→Python launcher替身→trainer→ProcessPool；不同PGID，合法后代登记成功。外部同UID PID、同容器非launcher后代、旧start-time、伪role、继承连接代发、错误/缺失pidfd、跨namespace信息均拒绝。真实进程快速退出再新建不能使旧登记信号击中新目标。 |
| 3 退出与屏障 | 杀真实孙进程只使它退出；pidfd poll确认，真实父进程仍能取得退出状态。重复通知仅一次信号；release丢失/abort丢失/日志损坏都技术失败并有界退出。 |
| 4 多epoch | 两次启动及不同PID替身只验证controller协议；后续schedule保留，旧epoch拒绝，fired无sent/有sent无observed均不自动重放。明确fixture不证明官方launcher。 |
| 5 实际reward库 | CPU调用未经修改的AsyncRewardWrapper与实际ProcessPool，杀真实评分worker，观察其自身BrokenProcessPool/重建/重试路径；控制器不得调用_recreate_executor。返回/失败均保存，不能当GPU恢复通过。 |

按三个可独立审查的增量实施：①仅容器生命周期与CPU可信归属合同；②接入trainer/reward真实hook并验证实际AsyncRewardWrapper受故障路径；③进行下述短GPUprobe。主代理已验证真实评分器CPU入口可运行，这不代替第②项实际worker故障验证。

CPU先通过才进行两类短GPU工程probe，沿用已经验证的四卡actor1＋SGLang3、同步Megatron checkpoint、真实RLVR入口与小模型，重新冻结实际设备UUID/配置/代码/镜像；不把四卡布局叫多actor rank验证：

1. **native launcher probe**：先保存一个真实完整checkpoint，再在预定后续训练边界精确杀一个trainer。仅官方launcher处理stop/retries；记录run_id变化、原生10秒等待、checkpoint实际加载、数据游标与随后真实更新。没有自动重启则是观察结果，不由P1补启动。范围限原生恢复路径可运行，不声称pending正确恢复或受影响工作已提交。
2. **F4 probe**：另一个独立run，在预注册评分进程后尝试冻结自然切点；逐项核验全K生成、实际评分进度、kernel PID/祖先链、signal_sent、pidfd退出、所有存活waiter收据、池原生重建或job原生重试。未命中标技术无效，命中后方法不能恢复保留方法结果；独立oracle尚缺则affected-work为unverifiable。不得合并两个probe形成不存在的一次成功。

每probe设启动/握手/注入/恢复/总run/清理各层上限，短工程总窗建议沿用此前900秒容器上限；这不是把正式30步45分钟、多故障120分钟或900秒方法窗改短。主代理记录实际时长、GPU分配成本、残留检查和全部失败。完成后才决定P1放行范围；正式FT1/FT2、R闭环及F4主比较仍按独立门槛推进。

## 仍需验证的风险与停止条件

- 所用内核/容器是否允许所需pidfd、SCM_RIGHTS和凭据、所需/proc可见性；本轮只查规范，不声称已做传递验收。
- `/proc`父链不是原子快照；采用双读、pidfd和暂停登记进程降低竞态，祖先变化就拒绝，不能默认成功。
- 主代理另核实MathVerifyWorker.verify的ThreadPoolExecutor上下文可能在future.result(timeout=5)之后继续等待线程；5秒参数不是强制进程截止。保持独立controller/container总watchdog，不在本轮修改上游评分器。
- host Docker daemon故障使清理不可证明；明确报告而不修改别的容器或主机服务。
- 官方shell/tee可能掩盖退出码，COMPLETED恢复行为可能耗尽重试；不得靠外层重写状态掩盖。
- F4毫秒窗口与严格同时性可能无法在不改变算法下可靠命中；这是映射可行性问题，不能以协议fixture补证。
- 原生RecoverInfo游标先于实际训练、未存dispatcher pending的缺口仍存在，进程重启成功不等于受影响工作恢复。

本轮只新增本文。后续实现先闭合隔离与身份，再验证官方恢复，最后验证F4自然精确命中；任何一项失败都保留证据，不升级为正式结果。
