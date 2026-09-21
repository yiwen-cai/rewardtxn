# 公共 launcher 作业生命周期修订提案（2026-09-21）

状态：用户已明确批准“A 和 A+R 共用公共补丁，继续验证”。补丁已应用；6项CPU验收与GPU单故障工程验证均通过，见 [自动恢复报告](P3_AUTOMATIC_RECOVERY_REPORT.md)。旧 native r4 的真实故障后 timeout 结果保留，不能改判成功。

## 为什么需要修订

已核实原生 local launcher 将 torchrun 和 tee 组成 shell pipeline。trainer 被 SIGKILL 后，孤儿评分进程继承 pipe 写端，tee 不结束，launcher 不触发原生 retry。详见 [原失败诊断](P1_NATIVE_TIMEOUT_DIAGNOSIS.md)。直接重复旧 GPU 探针不能验证新的恢复能力。

## 可审查改动

- [local-job-lifecycle.patch](local-job-lifecycle.patch)：仅修改 LocalLauncher.submit_array 的进程创建和日志方式；GPU分配、环境、stdbuf、原始command、job注册、原生local_main/retries逻辑保持原样。
- `scripts/ft/job_lifecycle.py`：每个原生job由独立subreaper管理，等待原root真实退出后，只对自己收养的后代使用pidfd回收，循环直到ECHILD；记录root退出码、清理信号/身份及时间。TERM后1秒升级KILL，最多10秒清理。
- 清理失败时写quarantine凭据并保持job未完成，等待外层实验总超时清理容器。不能简单返回非零，因为原生local_main对任何完成码都可能retry；不能让下一代与未退出的旧writer重叠。
- 日志改为追加普通角色文件，不再tee到launcher stdout。角色日志完整保留；独立观察器须读取角色日志。
- supervisor不执行重启、不读RewardTxn状态；重启仍属于原生local_main。公共修订必须同时用于A和A+R，清理/日志开销计入方法成本。

## 已完成 CPU 验证

`p3_evidence/job-lifecycle-cpu-r2`：固定已有CPU镜像、无GPU、无网络，5项通过，14.566秒，无跳过；容器退出0且独立删除核验通过。见 [摘要](job-lifecycle-cpu-verification.json)。

覆盖真实root退出0/7/SIGKILL、双层孤儿与setsid/忽略TERM、无关哨兵不受影响、supervisor收到TERM后有界收尾、人工注入清理失败时不结束job，以及在临时源码副本上执行真实LocalLauncher.submit/wait/stop和env/stdbuf/角色日志。这5项本身尚未验证完整local_main的实际retry或GPU恢复；随后r4新增实际retry验收，见下。

r1保留：4项进程生命周期通过，原生接口测试错误复用了已stop_all的launcher对象，触发上游job_counter断言；按原生retry每次创建新launcher的实际方式修正测试后r2通过。宿主受限执行不适用于subreaper孤儿收养验收，其双fork案例超时，不作为通过证据；精确临时脚本已检查无残留。以私有PID namespace容器结果为准。

新增 `job-lifecycle-cpu-r4`：6项通过，32.954秒，无跳过。实际torchrun进程SIGKILL、孤儿持有stdout并忽略TERM后，由公共层清理，原生local_main等待原有10秒间隔并自动启动第二进程；第二root退出0。CPU分配接口模拟，torchrun/Popen/wait/stop/retry真实。r3为CPU平台设备环境变量名为空导致命令无效的测试问题，完整保留；r4正确模拟该硬件接口并保留临时源码/角色日志/退出凭据。

## 已批准的工程验收

先验证实际原生retry的CPU路径，再在最新空闲4卡上运行A+R真实RL训练。固定一次故障：首步完整提交后，第二次真实optimizer成功但尚未保存时，pidfd精确SIGKILL trainer；生命周期层回收其后代，原生launcher自动重启。核验唯一故障、新trainer身份、最后保留checkpoint精确加载、未提交候选不作为消费权威、后续更新/提交，以及容器与网络无残留。保留完整失败，不将手动新进程启动作为通过。

这仍是单rank工程验收；不擅自启动正式矩阵，不改变评分口径，不将写入中未完成writer的安全接管、多rank或正式RTO比较宣布完成。

## 确认来源

`third_party/areal/AGENTS.md` 明确要求：**“Ask first … Changing launcher or scheduler logic.”** 已在准备可审查补丁及CPU证据后询问，用户回复“批准公共补丁，继续验证”，随后应用并启动GPU。本次没有自动审批拒绝。
