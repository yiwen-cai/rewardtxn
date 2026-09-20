# P1 通用运行控制实现

后续增量：多层进程身份与独立PID namespace收尾见 [P1_NAMESPACE_IMPLEMENTATION](P1_NAMESPACE_IMPLEMENTATION.md)；官方评分池真实CPU故障验证见 [P1_REWARD_POOL_PROBE](P1_REWARD_POOL_PROBE.md)。下文描述原直接子进程runner；其首次pidfd失败缺口没有在旧路径中改写，新隔离路径另行验证，不能混为同一实现。

日期：2026-09-20。实现范围是 CPU 可执行的控制协议和精确故障注入；P1 尚未全部验收完成；**不是 AReaL adapter、原生恢复验收或 GPU cell 通过**。P0 中 F4 自然切点、完整 RecoverHandler、多层 launcher 的真实 worker 身份登记仍未解决。

## 文件职责

- `scripts/ft/run.py`：只读 JSON preflight、原始配置/schedule 全内容 SHA256、独立 run 目录/nonce、有效 argv 和白名单 env、进程生命周期、追加 fsync 日志、有限等待与恢复后 schedule 保留。
- `scripts/ft/faults.py`：Linux PID/start-time/boot-id/PGID 身份、pidfd 信号、同步 ready/release/abort 客户端 watchdog。调用方必须在真实语义边界提供证据；模块不证明业务证据真实性。
- `tests/ft/test_runner.py`、`test_faults.py`：真实普通 Python 子进程合同测试。包含两次 attempt 故障、重复 ready/receipt、错误 PID、外部目标拒绝、controller 断开/超时、release 收据缺失、失败保留、未完成 attempt 不重放、后续 schedule 保留。全部为 CPU fixture，不代替 GPU 或原生库验证。

## 最小接口

入口从仓库根执行：

```sh
python -m scripts.ft.run --config /absolute/config.json --schedule /absolute/schedule.json --dry-run
python -m scripts.ft.run --config /absolute/config.json --schedule /absolute/schedule.json --run-dir /absolute/new-run
python -m scripts.ft.run --config /absolute/config.json --schedule /absolute/schedule.json --run-dir /absolute/existing-run --resume
python -m unittest discover -s tests/ft -p 'test_*.py' -v
```

config 只接受 `attempts`（每项为 role→完整 argv 数组）、`env`、`handshake_timeout`、`run_timeout`、`cleanup_timeout`。例：

```json
{"attempts":[{"target":["/usr/bin/python3","/absolute/worker.py"]}],"env":{"PYTHONPATH":"/absolute/rewardtxn"},"handshake_timeout":5,"run_timeout":30,"cleanup_timeout":2}
```

schedule 例：

```json
[{"event_id":"cpu-cut-1","attempt":0,"target":"target","waiters":["target"],"evidence":{"boundary":"cpu_fixture"}}]
```

CPU worker 在实际执行点调用 `from scripts.ft.faults import barrier; barrier('cpu-cut-1', {'boundary':'cpu_fixture'})`。正式 adapter 必须用真实业务证据替换 fixture。所有同事件 waiters ready 后注入，只有已确认 SIGKILL 退出才 observed；所有存活 waiter 接收 release 并确认 receipt。身份不符、未就绪、观察失败、release 丢失属于 technical_invalid。控制通道失联触发 worker 有界异常退出；abort 收据和 watchdog 记录保留。

`attempts` 是显式冻结的 CPU 启动序列，至多初次加三次；不是方法原生重试实现，不自动推断 checkpoint 参数，不为 A/C 提供新恢复能力。每次使用所配置的完整 argv。没有安排的 attempt 不创建；不清空后续故障 schedule。同一个角色进程只适合一次目标 kill；重启后的故障应放在后续 attempt。方法内 retry 尚需 adapter 独立记录。

只继承显式白名单环境变量（PATH、LANG、LC_ALL、PYTHONPATH、CUDA_VISIBLE_DEVICES、OMP_NUM_THREADS、MKL_NUM_THREADS、TOKENIZERS_PARALLELISM）；不读取/倾倒父进程环境。argv 是已授权运行参数，调用方不得把密钥放入 argv；认证等其他环境需求应在真实 adapter 设计时明确扩充，当前不推测。

## 身份和恢复边界

每个 role 由控制器实际 Popen、独立 session 启动，经直接父子关系及 session leader PGID 验证后持有 pidfd；每个 role 独立继承 socket。拒绝 worker 指定任意 PID；信号前再次对照完整身份，信号通过 pidfd，缺少 pidfd API 立即拒绝执行。宿主现有 Python 构建缺少该 API，主代理已核实 AReaL 容器 Python3.12 有此能力。没有 pkill、共享 Ray stop、PGID 群杀或普通 kill 回退。

**当前仅接入直接子进程角色。** 不接受未经 launcher 所有权证明的多层 worker 注册，不假设 AReaL 子进程共享 PGID；不声称清理其尚未登记的后代。未来 adapter 必须在实际启动接缝建立可信所有权链再接入，不能把当前 CPU 启动序列部署为 AReaL 全进程监督。

独占文件锁防止两个 controller 使用同一 run。resume 要求配置及 schedule 冻结一致，已写 result 的 run 不可继续。存在 attempt_start 但无 attempt_end 时标 technical_invalid 并拒绝重放，即使有 fired/observed 也不自动再杀；已有完整 attempt_end 时才允许启动后续冻结 attempt。旧失败记录不覆盖，不自动换 seed/补跑。controller 被强杀时等待方会因 FD 断开或独立 watchdog 退出，但非屏障中的方法进程尚需真实 launcher/cgroup 退出策略；该缺口不得计作方法 RTO。

结果使用控制层 `execution_complete` / `technical_invalid` 与 `oracle_status=not_evaluated`；方法退出码、有效命中后的运行超时独立记 `method_observation`。runner 不产出 correct_recovered、safety pass 或 oracle-valid，不判断受影响组是否恢复。注入屏障失败不记方法性 timeout；正式结果需后续独立 oracle 合并。

主代理在隔离 AReaL 容器中运行目录发现测试：11 项通过，1.375 秒，退出码 0。实际命令、源码全量 SHA256 和日志见 `p1_evidence/tests-r1.json`、`p1_evidence/tests-r1.log`。首次按模块名执行因现有 tests 包名冲突未收集到测试，改用上述 discover 命令解决；先前开发中版本曾通过 8 项，不作为最终版本凭证。没有修改第三方源码、启动训练/GPU 或提交。

仍开放的异常路径：Popen 成功后第一次 `os.pidfd_open` 本身失败时，尚无可用 pidfd 进入 provisional 清理集合；当前通过 pidfd API 存在性检查不能证明该调用一定成功。正式接入前须与独立 job/container 生命周期策略一起闭合并测试，不能将当前测试通过解释为所有启动失败都无遗留进程。
