# 原生 trainer 故障工程探针

2026-09-20。r4 已结束：有效 trainer 注入后，原生方法在 900 秒运行期限内未重试，恢复验收失败。控制器及资源收尾成功；正式样本 0。

## CPU 前置验证

真实类接口/保存合同 7 项通过（最终 r2 23.979 秒）；身份与 RecoverInfo 证据负例 4 项通过（0.011 秒）；GPU profile/UUID 合同 4 项通过（0.007 秒）；后代 pidfd 8 项、容器生命周期 4 项、一次性事件协议 4 项通过。真实 torchrun CPU r1 在 network=none 下 hostname rendezvous 超时，r2 使用显式 loopback 后通过（13.214 秒）。失败证据及两容器独立删除确认均保留。

## GPU 工程运行账本

- native-gpu-r1：四卡 CUDA/JIT 执行后，PyTorch UUID 不带 GPU- 前缀导致比较误拒；37.840803 秒，0.0420453 GPU 小时。训练未启动，CID/network 删除已独立确认。原源码副本 hash 与当次记录一致。
- native-gpu-r2：启动前 GPU 0 已被外部任务占用，空闲检查拒绝；无容器/网络/实际分配，GPU 成本 0。
- native-gpu-r3：人工改选空闲同型 H100 1–4；CUDA/JIT/UUID/缓存/network 预检通过，官方 launcher 因缺 name_resolve 目录拒绝；43.994615 秒，0.0488829 GPU 小时。CID/network 删除已独立确认。
- native-gpu-r4：补齐 GPU profile 输出目录，前置合同通过后启动。源码完整副本在 native-gpu-r4-source，运行时 hash 另存。固定 900 秒期限结束，method_observation=timeout；含创建/收尾共 911.497517 秒，1.012775019 GPU 小时。仅 run_id0、无第二 trainer、无恢复 load；容器及网络已独立确认删除。

## r4 最终证据

真实第一次 successful optimizer、step0 完整同步 DCP/RecoverInfo 保存均有见证。第二次 successful optimizer 返回后，唯一 ready/armed/SIGKILL/同 pidfd 退出链存在；injection-verification.json 已独立核对 trainer PID667 的完整 identity/incarnation、唯一信号/退出及 step0→第二次更新→ready 顺序。

故障后直到期限结束均未见 run_id1。只读 postkill-stdio.json 表明：trainer tee PID650 的 stdin 是 pipe:[2796824288]，多个 PPID1 的孤儿 python3 仍持有同管道 stdout/stderr；外层 shell PID648 仍在。官方 LocalLauncher.submit_array 使用 shell 管道到 tee，wait 观察 shell 状态。这支持日志管道未关闭使 launcher 未触发 retry 的诊断；不把该观察说成已测到内部 wait 系统调用。

运行期间未额外杀孤儿进程、未改官方 launcher、未用自定义重启替代原生行为；最终由隔离容器收尾。independent-verification.json明确passed=false，supervisor_result.json的failure=null/退出0仅说明控制流程完成。不把注入成功称为训练恢复成功，不把 method timeout 自动改成接口 N/A。同步单 actor 工程探针不覆盖异步、多rank、pending/R replay 或自然 F4。

四轮本探针合计 1.103703262 GPU 小时（r2 未分配为0），独立于旧pilot成本；含容器创建/收尾的保守分配时长。此有效命中后的方法失败不删除、不转为N/A、不补样冒充成功。P1可核实注入/超时收尾能力，但原生自动恢复正例门禁未通过；后续脚本验收须区分可靠分类与方法恢复能力。

计时口径：900秒是本工程探针的整run预算；signal_sent至timeout实际610.408230671秒。未测满正式设计的故障后900秒观察窗，不将本次结果套用正式900秒惩罚/RTO统计。孤儿写端中8个可由reward事件确认为reward worker，其余2个只确认是python进程。
