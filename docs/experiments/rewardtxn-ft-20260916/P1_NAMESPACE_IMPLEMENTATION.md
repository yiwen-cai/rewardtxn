# P1 增量①：CPU 容器生命周期与可信后代合同

2026-09-20。**增量①限定CPU验收通过；不是 AReaL 接入、原生恢复或 GPU cell 验证，P1未全部完成。** 范围来自 [P1_DESCENDANT_PLAN.md](P1_DESCENDANT_PLAN.md) 的增量①。原 `run.py`、`faults.py` 及其11项合同测试未修改，真实 pilot/oracle/state 文件未修改。

## 文件与边界

| 新文件 | 职责 |
|---|---|
| `scripts/ft/descendants.py` | kernel SCM_CREDENTIALS 逐消息认证、self-pidfd SCM_RIGHTS传递/校验、祖先链复查、pidfd精确信号及readiness、CPU Client。 |
| `scripts/ft/namespace_run.py` | 必须作为独立PID namespace内PID1，单次Popen launcher，登记后代、按顺序注入CPU屏障、收release收据、host lease与绝对期限，最终直接退出PID1。 |
| `scripts/ft/container_run.py` | CPU专用host入口，create→保存完整CID→inspect→start→lease/heartbeat→按CID有界stop/kill→保存final inspect/logs→rm。不接受任意Docker选项，不启动GPU。 |
| `tests/ft/namespace_fixture.py` | 真实 launcher→middle→target/waiter CPU树、父进程wait收据、睡眠子树与仅测试使用的controller故障注入。 |
| `tests/ft/test_descendants.py` | 8项内核/配置合同；不需要Docker socket，pidfd相关项在缺能力时明确skip。 |
| `tests/ft/test_namespace_container.py` | 4个显式启用Docker的CPU测试方法，失联测试含3个子场景；另起独立哨兵容器检查不受影响。 |

这里role只是冻结协议槽，不证明AReaL实际角色。允许受控同UID方法代码注册其真实后代，防误目标、FD/PID身份错配和继承连接代发；不声称抵御同容器恶意代码的任意ptrace或篡改。

## CLI与完整配置

host（非root用户、有Docker权限）执行；镜像可用既有AReaL镜像，但CPU控制器不导入AReaL/Torch，不申请GPU：

```sh
python -m scripts.ft.container_run \
  --source /public/home/caiyiwen/rewardtxn \
  --config /absolute/input.json \
  --run-dir /absolute/new-output \
  --image sha256:ACTUAL_IMAGE_ID \
  --python python
```

`--source`必须包含本次控制器代码，以只读方式挂到`/workspace`；`--run-dir`必须不存在，以可写方式挂到`/output`。路径不接受逗号/换行。`--python`指定镜像中的Python入口；推荐使用实际核实的绝对解释器路径。config恰好包含以下四个字段：

```json
{
  "argv": ["/usr/bin/env", "python", "/workspace/tests/ft/namespace_fixture.py", "launcher", "--count", "2"],
  "env": {
    "PYTHONPATH": "/workspace",
    "PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"
  },
  "timeouts": {"run": 20, "handshake": 2, "lease": 4},
  "schedule": [
    {"event_id": "event-0", "target": "target", "waiters": ["target", "waiter"], "evidence": {"boundary": "cpu_fixture"}},
    {"event_id": "event-1", "target": "target", "waiters": ["target", "waiter"], "evidence": {"boundary": "cpu_fixture"}}
  ]
}
```

- `argv`完整字符串数组，首元素为绝对可执行文件；不会shell拼接，不会自动补恢复参数。例中env解析的Python必须由冻结PATH解析为预期版本。
- `env`复用旧runner白名单，只包含显式提供的值，不复制父进程环境；controller另注入本run的socket/nonce/握手期限。不得把密钥放argv。
- `timeouts`单位秒，各值有限且在[0.1,7200]；强制`lease >= handshake + 1`，正常pidfd等待不超过handshake。host还提供独立绝对期限。写日志或其他操作真实卡住时heartbeat停止并触发技术收尾，不用后台心跳掩盖主控制循环卡死。
- `event_id`全run唯一；target必须包含在waiters；roles为合法identifier；evidence为非空对象，必须与真实ready逐项完全一致。这里fixture evidence不证明业务语义。
- 多事件按schedule顺序发信号；新role incarnation只能替换已经退出的同role进程。controller只启动一次launcher，两个事件由CPU launcher自己创建后续进程，不是P1重启方法。

内部容器入口为`python -m scripts.ft.namespace_run --config /output/config.json --output /output --nonce VALUE --host-pidns INODE`，应由host入口生成，不能单独把任意inode当隔离证据。host同时核验完整CID、nonce label、Docker配置和init namespace。`check_namespace`要求实际PID1且其namespace不同于host；无前置shell、timeout或`--init`。

namespace入口不支持resume：目录中已有`events.jsonl`或`frozen.json`即拒绝，uncertain signal_intent/signal_sent不会自动再杀。原direct-child runner的只读preflight和CPU resume语义保留，不用它恢复此namespace运行。

## 认证与精确退出

每个进程自己新建Unix SOCK_SEQPACKET连接，注册时发送自己的pidfd。服务端开启SO_PASSCRED，使用recvmsg+MSG_CMSG_CLOEXEC，每消息核对kernel PID/UID与已登记PID；fork继承连接后代发会被拒绝。注册JSON仅允许kind/pid/nonce/role。

收到FD必须恰好一个，`/proc/self/fd`类型为`anon_inode:[pidfd]`，fdinfo的`Pid`等于kernel sender，且pidfd仍存活、signal0可用。不要求fdinfo存在NSpid。错误、多个、缺失、非pidfd、跨sender FD及控制消息截断均拒绝；所有已收到FD在错误路径关闭。MSG_CMSG_CLOEXEC返回flag自身不是截断。

注册时沿真实PPID到实际Popen launcher，记录并反向复查每层PID/start-time/boot-id/namespace/cgroup/PGID/NSpid，不要求共享PGID。信号前再查目标身份及同一祖先链。已登记后被reparent不继续授予新注入；外部同UID进程不能因持有nonce变成owned target。

journal分别记录`armed`、`signal_intent`、`signal_sent`、`process_exit_observed`。后代仅通过同pidfd的poll/readiness确认退出，不用waitpid抢父进程收尸，不编造退出码=-9；fixture的真实middle独立wait到-9作为另一份证据。目标退出后release所有仍活着的本事件waiters，并收幂等receipt。重复ready不重复发信号，旧incarnation不能复用旧事件。

## 生命周期、证据与当前限制

host先`docker create --cidfile`，即便create CLI超时也尝试读取已落盘CID；仅对完整CID且nonce label匹配的对象执行后续动作。若create超时且CID文件也未落盘，无法安全确认目标，报告未确认而不按名称或全机搜索清理。

容器非root、cap-drop ALL、no-new-privileges、只读根fs、独立PID namespace、CPU2/内存2GiB/pids128、无GPU、network=none。network=none仅用于本CPU增量，不可直接用于AReaL真实launcher（pilot已证明其gethostip需求）。本轮不挂载Docker socket。

Popen后首次pidfd获取失败不走PID kill回退；记录技术失败后`os._exit`结束PID1，由内核结束其namespace内尚未登记的后代。正常收尾、controller崩溃也同样闭合。host失联由controller检测lease过期；controller挂起由host检测heartbeat过期，再有界stop/kill持有CID；独立绝对run期限继续约束两侧。Docker命令一般单次10秒超时，stop/kill另有有限超时；这些外层操作耗时单独记入总墙钟，不声称清理必定只用handshake秒。

输出保留`launch.json`、原始`config.json`、`frozen.json`（原始config SHA256与nonce）、`events.jsonl`、launcher/container日志、完整CID、created/started/before_cleanup/final inspect、host init identity及supervisor_result。每轮源码hash由主代理验收记录；host入口没有自动制作不可变源码快照，正式freeze仍未实现。host CLI退出码也反映controller非零退出；`execution_complete`仅是控制协议完成，oracle未评估。

`cleanup_confirmed`依据该CID不再running和原init身份消失，再结合独立PID namespace终止语义；未逐项枚举host cgroup内所有PID，也未处理Docker daemon故障导致的清理不可证。原始inspect/cgroup路径保留，不把容器内`/`字符串作为唯一隔离证明。

本增量不做长run孤儿僵尸回收，以namespace退出统一收尾；没有waitpid(-1)。不实现native epoch归属、AReaL角色证明、真实reward hook、原生重试观察、F4语义映射、GPU恢复或独立oracle。abort目前是尽力通知后退出namespace，未实现abort receipt收集；有界退出由containment提供，不可直接宣称完整正式屏障验收。

## 验证命令与证据边界

不访问Docker socket的内核合同，在支持pidfd的现有隔离容器内执行：

```sh
python -m unittest discover -s tests/ft -p 'test_descendants.py' -v
```

Docker生命周期套件必须在host显式启用，建议每次给新的持久证据根：

```sh
FT_NAMESPACE_DOCKER_IMAGE=sha256:ACTUAL_IMAGE_ID \
FT_NAMESPACE_EVIDENCE_DIR=/absolute/new-evidence-root \
python -m unittest discover -s tests/ft -p 'test_namespace_container.py' -v
```

可另设`FT_NAMESPACE_DOCKER_PYTHON`为镜像中实际解释器路径。指定EVIDENCE_DIR时，每case和哨兵创建独立新目录，成功/失败均保留，不覆盖；未指定时才用TemporaryDirectory。哨兵保存完整CID、启动/收尾inspect及清理记录。

测试覆盖两事件/role重建仍只启动一个launcher、真实父进程wait收据、首次pidfd失败留下未登记睡眠子树、注册失败、controller KILL/STOP、host lease停止及独立哨兵存活；内核合同另覆盖FD类型/数量/截断关闭、外部目标、继承连接代发、身份错配、旧pidfd和uncertain日志不重放。没有实际强制造成数值PID复用、跨不同Docker namespace发送登记或Docker daemon不可用测试；拒绝逻辑/内核契约不能冒充这些场景已实测。

实现worker只做py_compile。主代理最终实测：8项内核合同通过（0.182秒），4项Docker测试通过（16.869秒，包含controller失效3种子场景），退出码均0。6个run均`cleanup_confirmed=true`且`cleanup_errors=[]`，无关哨兵全程存活；按7个保存完整CID再次独立inspect，确认6个run容器与哨兵均已移除。

最终源码全量SHA256与限定结论见 [namespace-verification-r1.json](p1_evidence/namespace-verification-r1.json)；原始日志见 [descendant-tests-r2.log](p1_evidence/descendant-tests-r2.log)、[namespace-tests-r1.log](p1_evidence/namespace-tests-r1.log)，全部run及哨兵inspect/journal/收尾记录位于 `p1_evidence/namespace-r1/`。早期7项通过只保留开发历史，不作为最终源码凭证。

这些结果不覆盖真实AsyncRewardWrapper子进程恢复、AReaL多worker/native epoch合同、自然F4命中或GPU恢复。特别是现协议拒绝新incarnation再次ready同一个已触发event；真实库重试接入仍需只读注入状态分配以避免重杀，不得由worker自行持久化故障sentinel来补充基线恢复能力。
