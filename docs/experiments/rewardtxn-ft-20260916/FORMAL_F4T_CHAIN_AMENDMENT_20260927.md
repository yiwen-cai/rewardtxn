# F4′ 验收脚本修订 A1（2026-09-27，用户批准）

## 问题
p01 redo2 的 R 臂验收停在 `blocked_at_chain`。`check_ft1_chain.py` 和 `check_ft1_input_audit.py` 都断言 batch_taken、train_batch、optimizer_end 三类事件数量相等。F4′ 的 SIGKILL 落在第 4 批 train_batch 之后、optimizer_end 之前（update `09eb2f16…`），计数为 12/12/11。这一批从未生效，重启后重新训练；重启后成功更新 8 次，总计 11 次。

## 修订
两个脚本都只放行一种情况，其余断言不变：
- 场景为 F4T；
- 恰好一个 train_batch 没有对应的 optimizer_end；
- 它是所在进程的最后一个 pilot 事件，前面有对应的 batch_taken；
- 控制器的 `signal_sent` 在它之后，此后所有事件都在信号之后，且属于新进程。

放行后，把这一对 batch_taken/train_batch 从配对序列中剔除，并记为 `killed_in_flight_update_id`。CPU 单测见 `tests/ft/test_chain_killed_update.py`（6 项，含 5 个拒绝用例）。

## 影响与证据
- redo2 的 R 臂证据未做任何改动，按修订后的脚本完整重跑验收，全部通过，结果为 `correct_recovered`；同次执行保留 66/69。原始验收状态保留为 `acceptance-status-blocked_at_chain-20260927.json`。
- 已删除大分片的旧证据，在修订前后报相同的错误，不属于回归。
- p02–p10 改用新冻结 `FORMAL_F4T_FREEZE_20260927-A1.json`：只有这两个脚本的哈希变化；seed、顺序、目标、判据、统计、镜像和配置都不变。
