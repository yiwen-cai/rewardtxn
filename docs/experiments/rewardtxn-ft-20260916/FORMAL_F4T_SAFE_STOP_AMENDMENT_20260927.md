# F4′ 修订 A2：原生恢复无法继续时记为安全停止（2026-09-27，用户批准）

## 触发
p02（s4602，R→A）：R 为 `correct_recovered`，保留 37/39。A 臂的 SIGKILL 正好落在 AReaL `recover_handler` 写恢复检查点期间（`main_thread_last_event_before_kill=recover_handler_dump_start`）。当时 `step_info.json` 等描述文件已写完，DCP 分片尚未写完。重启后 `recover_handler.load` 报 `EOFError`，重试后仍失败，故障后没有任何成功更新，也没有最终状态。验收因此停在 `blocked_missing_final_native_state`。这是合法的故障命中（valid_hit，无外部 GPU 进程），不是技术无效。

## 规则
在 F4T 正式矩阵中，若某臂同时满足以下条件，记为 `safe_stop`，算作安全、有效的结果：
- 故障有效命中且只发了一个信号；
- 所有 job 已清理；
- 训练未返回，故障后成功更新 0 次；
- 没有最终原生状态。

`safe_stop` 臂没有最终保留链，同次执行保留数按定义为 0。分母沿用 `verify_inflight` 的定义：信号前返回、由父进程接受、且未进入信号前已生效更新的评分执行。实现为 `check_ft_minimal_source.verify_safe_stop`。该口径在 3 个已验收的臂（redo2 A 40、redo2 R 69、p02 R 39）上复算，结果与原值一致。`safe_stop` 臂保留全量证据。两臂适用同一规则。

## 代码
- `tests/ft/check_ft_minimal_source.py`：新增 `verify_safe_stop`。
- `tests/ft/run_ft_formal.py`：F4T 正式配对遇到上述情况时登记 `safe_stop` 并继续，不再中止整个矩阵。
- `tests/ft/analyze_ft_f4t.py`：把 `safe_stop` 纳入安全类别，只对非 `safe_stop` 臂检查保留链和重载。
- 新冻结 `FORMAL_F4T_FREEZE_20260927-A2.json`，用于 p03–p10。seed、顺序、目标、主判据和统计都不变。

## p02 结果
A 为 `safe_stop`，保留 0/43；R 为 `correct_recovered`，保留 37/39；差值 +0.949。

## 待办
`analyze_ft_f4t.py` 目前只认原冻结和逐对 GPU 分配清单，还不认补做名（p01-redo2）和 A1/A2 冻结。汇总前要补上。
