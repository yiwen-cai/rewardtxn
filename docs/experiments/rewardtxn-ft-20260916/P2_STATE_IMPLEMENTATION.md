# P2 state CPU 合同实现

2026-09-20。`scripts/ft/state.py` 与 `tests/ft/test_state.py` 已实现，最新r3的21项测试通过（1.285秒，见state-verification-r3.json）。完整日志、命令、源码SHA256、Python和文件系统范围见 `p2_evidence/`。这不是P2全部验收、GPU恢复或真实replay通过。

采用本地主机POSIX文件锁、文件/目录fsync、不可变文件发布与全文件SHA256；测试目录及workspace现场均为ext4。checkpoint由后端直接写独立generation目录，不复制第二份快照。仅接受 `cpu_contract` scope，GPU/未知scope拒绝；登记进程的PID/start-time/boot-id和存活检查不证明未登记后代已退出。

七个API见模块。run固定verifier版本；`authorize_attempt(..., expected_policy_version=...)` 显式持久绑定每个attempt的policy版本，允许跨更新演进及组内不同的已授权policy。accept、prepare、commit复核授权；组内混verifier、旧attempt先到、未经授权结果拒绝。policy的实际来源和逐样本staleness资格由adapter及独立oracle核查，不要求全组policy相同。该层不证明声明版本对应真实verifier，更不替代离线重评分。

intent在更新前持久化，消费/已取出/pending集合必须相符；缺rank或组件不能提交。token发布后证据只允许相同已有收据幂等读取。部分候选仅在完整收据、内容和父链成立时补提交；损坏token安全拒绝。合法未提交工作保留rollback_intents供未来replay，不声称已恢复。

测试包含真实子进程owner争用、已登记存活worker拒绝接管、新进程选代、token前后SIGKILL、文件中段篡改、缺rank/组件、cursor/pending错配、旧attempt/版本混用、提交后证据不可变。小文件表示组件，收据为合同fixture；尚未验证真实backend finalize、实际训练tensor语义、完整数据预取恢复、GPU fencing、独立oracle或六cell各100调度。当前pending实现只支持显式regenerate解释，回答复用及重评分待replay接入。
