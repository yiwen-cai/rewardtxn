# P1 真实 AsyncRewardWrapper CPU 故障探针

2026-09-20。限定CPU合同及真实库探针验收通过，P1整体尚未完成。**仅验证单评分进程池的原生BrokenProcessPool路径，不是F4、rollout、SPMD launcher恢复或GPU训练恢复。** 现有P1 namespace的独立PID1容器、只读来源、无网络、CPU2/内存2GiB和完整CID收尾保持不变。

## 接口依据与实施范围

已核对未修改的 `third_party/areal/areal/api/reward_api.py`：`AsyncRewardWrapper.__call__` 在 `BrokenProcessPool` 分支调用其自己的 `_recreate_executor`，并在 `range(max_retries + 1)` 内重试。项目探针没有调用/替换 `_recreate_executor`，没有另建pool、手工重投、吞掉库异常或mock评分/ProcessPool。官方 `areal.reward.gsm8k.gsm8k_reward_fn` 调用原 `MathVerifyWorker.verify`，评分代码未修改。

- 新增 `scripts/ft/reward_pool_probe.py`：pickleable项目级包装器、真实AsyncRewardWrapper调用、每次实际pool callable的进程与控制边界记录、官方分数、源码hash、内存测量。
- 局部扩展 `scripts/ft/descendants.py` / `namespace_run.py`：注册时可请求一个具体event的一次性注入分配；允许只登记的observer角色；namespace专用环境白名单增加USER/LOGNAME。旧 `run.py` / `faults.py` 和第三方代码未改。
- 新增 `tests/ft/test_reward_pool_probe.py`：4项协议定向测试，加1项显式启用的真实库Docker测试（无故障、有故障两个子场景）。没有扩展训练调度或启动GPU。

故障发生在**实际ProcessPool worker进入项目包装器之后、官方gsm8k函数调用之前**。被杀进程已实际运行官方pool的任务，但尚未计算该次官方分数；必须称reward callable entry故障，不能称评分中途或“评分完成K/2”。替换worker在官方库自己的重建/重试路径执行同一包装器并实际调用官方评分函数。

## 最小协议扩展

原 `Client(role)` 仍兼容。新 `Client(role, event_id=...)` 在注册包中附加所请求event_id；每消息内核凭据、self-pidfd、真实祖先链与incarnation校验保留。

注册回复可含：

```json
{"injection":{"event_id":"reward-entry-0","status":"pending","event_nonce":"controller-derived-nonce"}}
```

第一次分配为pending，绑定已认证incarnation；原进程在未触发前丢失时拒绝把pending资格转给新进程。事件已observed、进入release阶段后，替换进程取得同一event_nonce、status为`already_fired`，不再发送ready，直接执行原评分函数。状态只控制注入幂等，不携带prompt、回答、reward、checkpoint、replay或方法恢复建议；方法从不读取observer产物决定分数/重试，没有worker自写故障sentinel。

controller仍拒绝新incarnation重复ready旧event，不能靠忽略重复请求隐含再杀。unknown event、observer角色请求不属于它的schedule目标、uncertain状态都拒绝。`observer_roles`只添加登记权限，不添加signal权限：故障权仍来自冻结schedule的target/waiters。

原先未请求event_id的两事件CPU fixture保持原行为，须做定向回归。这里没有实现native epoch编号，也不支持多worker评分池、持续服务的任意同PID重新登记；范围是一个max_workers=1 pool的一次真实调用及BrokenProcessPool替换。

## 配置与CLI

沿用host入口：

```sh
python -m scripts.ft.container_run \
  --source /public/home/caiyiwen/rewardtxn \
  --config /absolute/reward-pool-fault.json \
  --run-dir /absolute/new-reward-pool-run \
  --image sha256:ACTUAL_IMAGE_ID --python /opt/.venv/bin/python
```

故障配置示例（当前镜像已现场核实解释器为`/opt/.venv/bin/python`）：

```json
{
  "argv": ["/opt/.venv/bin/python", "-m", "scripts.ft.reward_pool_probe", "--max-retries", "1", "--fault"],
  "env": {
    "PYTHONPATH": "/workspace:/workspace/third_party/areal",
    "PATH": "/opt/.venv/bin:/usr/local/bin:/usr/bin:/bin",
    "USER": "ft_cpu_probe", "LOGNAME": "ft_cpu_probe", "OMP_NUM_THREADS": "1"
  },
  "observer_roles": ["reward"],
  "timeouts": {"run": 120, "handshake": 5, "lease": 10},
  "schedule": [{"event_id":"reward-entry-0", "target":"reward", "waiters":["reward"], "evidence":{"boundary":"official_gsm8k_before_call"}}]
}
```

无故障配置仅删除argv末尾`--fault`并置`schedule=[]`，保留observer_roles、官方包装器及全部pool/评分参数。两个场景都使用固定输入“2+2”、完成文本`The answer is \\boxed{4}.`的单反斜杠LaTeX形式、answer=`4`、空token ID列表；官方评分并不使用这些token列表。期望1.0来自固定算术样例，不是训练独立oracle。

pool固定`max_workers=1`、`timeout_seconds=15`；CLI `--max-retries`只接受0–3，当前预设对照均为1，即最多两次尝试。本单元不执行重试耗尽扩测。官方MathVerifyWorker的5秒thread future超时不是强制进程截止；namespace/host总watchdog独立保留。

USER/LOGNAME只在namespace配置中显式允许，解决无passwd记录容器的getpass导入问题；不读取父进程其他环境，不扩大旧direct-child入口白名单。PYTHONPATH优先本地未修改AReaL checkout，不安装新依赖。

## 产物与验收

- `probe_started.json` / `probe_result.json`：trainer替身的真实身份、输入、pool参数、最终score/异常、Python/依赖版本与multiprocessing实际start method。
- `reward-PID-invocation.jsonl`：每次真实pool callable进入、内核身份、注入分配、边界、官方评分开始/返回；逐次flush/fsync，不回读到方法。
- controller原journal：真实注册与父链、pending/already_fired分配、唯一signal_sent、同pidfd退出证明。不得从pidfd readiness编造wait退出码。
- `launcher.log`：官方库自己的broken/recreated日志；测试要求明确出现`ProcessPoolExecutor broken (attempt 1/2)`与`Recreated ProcessPoolExecutor with 1 workers`。
- provenance全内容SHA256包括实际导入的reward_api.py、官方gsm8k.py、reward/__init__.py及本项目probe/控制模块；测试与只读checkout逐项比对。容器生命周期原inspect/CID/log/cleanup产物继续保留。

内存报告区分Linux进程峰值RSS（KiB）和容器cgroup峰值（bytes），不把父子RSS相加冒充总内存。仅在host inspect证明private cgroup namespace、进程位于该私有根且实际挂载匹配时读取v2 `memory.peak`或v1 `memory.max_usage_in_bytes`；无法证明或读取时峰值为null并附原因，不读host总量冒充本run。保持2GiB上限，若实测OOM/不足才另行调整并记录失败，不预先扩资源。

精确验证命令：

```sh
# 4项注入分配合同；不启动Docker或导入官方库。
python -m unittest discover -s tests/ft -p 'test_reward_pool_probe.py' -k InjectionAssignmentTests -v

# 显式启用真实官方CPU pool两场景；保存原始失败。
FT_REWARD_POOL_DOCKER_IMAGE=sha256:ACTUAL_IMAGE_ID \
FT_REWARD_POOL_EVIDENCE_DIR=/absolute/new-reward-pool-evidence \
python -m unittest discover -s tests/ft -p 'test_reward_pool_probe.py' -v

# 旧两事件/不同incarnation仍只启动一个launcher的定向兼容回归。
FT_NAMESPACE_DOCKER_IMAGE=sha256:ACTUAL_IMAGE_ID \
FT_NAMESPACE_EVIDENCE_DIR=/absolute/new-two-event-regression \
python -m unittest discover -s tests/ft -p 'test_namespace_container.py' -k two_events -v
```

镜像Python路径不同可设置`FT_NAMESPACE_DOCKER_PYTHON`。显式证据根下每run独立新目录，失败不覆盖/删除。真实测试要求无故障1个worker、有故障2个不同PID且同真实trainer父进程；仅一次signal/observed；故障首worker没有official_score_start，replacement确实官方score=1.0；两场景输入/源码hash/最终分数一致；launcher均自然返回0且容器清理已确认。

实现worker运行4项注入分配合同：0.023秒全部通过，py_compile通过。主代理最终r2实测5项测试通过（24.016秒），包含真实无故障与故障两场景：两者官方score均为1.0；故障场景实际仅一次SIGKILL，日志证实官方BrokenProcessPool attempt 1/2→原生重建pool→新PID执行官方评分成功。另有descendant回归8项通过（0.179秒）、旧两事件定向回归1项通过（2.369秒）。源码hash、精确命令和日志索引见 [reward-pool-verification-r2.json](p1_evidence/reward-pool-verification-r2.json)。

本次保持2GiB内存上限完成两场景，无需扩容。父进程peak RSS分别为923468和922884 KiB；reaped children指标分开保留，不相加冒充总量。因host inspect未证实private CgroupnsMode，cgroup_peak为null，不声称测得容器总峰值。两run约13.3/10.7秒只用于工程墙钟/资源记录，不能比较得出故障收益或方法RTO。

r1两个失败run、r2两个成功run以及namespace-r2的run/哨兵共6个保存完整CID，主代理再次独立inspect均确认已移除，见 [reward-pool-cleanup-verification.json](p1_evidence/reward-pool-cleanup-verification.json)。全部原始输出保留在对应p1_evidence子目录。

未覆盖多worker、native SPMD epoch、真实生成/评分窗口、训练状态/受影响工作恢复、GPU与正式FT矩阵；任何本探针PASS不升级这些能力。


首次实际运行失败记录保留在`p1_evidence/reward-pool-r1/`：两臂均在导入AReaL时缺colorlog，尚未创建pool或注入。主代理现场诊断确认原测试PATH误写`/opt/venv/bin`，导致选中系统`/usr/bin/python`；实际依赖位于`/opt/.venv/lib/python3.12/site-packages`，正确解释器是`/opt/.venv/bin/python`。见`p1_evidence/reward-pool-python-diagnosis.log`。测试已冻结绝对解释器及正确PATH，未安装依赖；仍可显式设置FT_NAMESPACE_DOCKER_PYTHON。旧纯stdlib fixture通过不构成真实依赖环境验证。探针已补充导入/provenance失败也写probe_result并原样抛错，测试先验证方法真实退出再读取结果。
