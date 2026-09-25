# R 臂逐步变慢修复：实现与验证记录

日期：2026-09-23。依据：`R_SLOWDOWN_FIX_PLAN.md`（v2）、`R_SLOWDOWN_FIX_PLAN_AUDIT.md`。用户已批准 v2 §8 的 4 项决定：实施 C1、批准 30 步 F2 门控（非正式 pilot）、FT-v1 §9 正式修订、另修陈旧 head 活性问题。

**版本批次**：本修改结束旧版本批次，此前所有 FT1 试跑都属于修复前版本，不与之后的结果合并。备份：被修改文件旁的 `*.bak-20260923-100804`。

## 1. 代码改动

| 文件 | 改动 | 对应方案 |
|---|---|---|
| `scripts/ft/state.py` | `_head(..., verify_head_content=False)`：历史代只校验 token/manifest/intent 哈希链，外加文件集合与大小（stat）；`select_recovery` 只对恢复要加载的 head 做内容哈希，并与原 `_validate_token(head)` 合并 | L1 |
| 同上 | `commit_generation` 的第二次全量哈希改为重新枚举整棵树并比较 `(dev, ino, size, mtime_ns, ctime_ns, mode, nlink)`；新增 `prehash()`，`_inventory(cache=)` 只在 stat 身份不变时复用摘要；候选代提升复用 `_complete` 的摘要 | L2、C1b、M5 |
| 同上 | `pin_generation` / `pinned` / `prune_generations`：按 token 链定位；marker 内容确定、先于删除、续删时不重写；abandoned、无 token、候选代和 pin 都不剪；`_check_files` 按 marker 豁免已剪文件，无 marker 缺文件即拒绝 | L3、M1、M2 |
| 同上 | `CONTENT_HASHED`：只记录不干预，记下被内容哈希的目录，供门控统计恢复时的哈希次数 | §6.2 |
| `scripts/ft/training_adapter.py` | `native_snapshot_r`：R 专用，逐张量哈希并行（8 线程），输出与原函数逐字节一致；两臂共用的 observer 仍用原串行 `native_snapshot` | C1a |
| 同上 | 保存时让 `prehash(checkpoint)` 与 `native_snapshot_r` 并发，commit 复用摘要 | C1b |
| 同上 | 恢复时 pin `recovery_loaded`，恢复后第一次提交 pin `first_commit_after_recovery`；每次提交后 `prune_generations(prunable=native/*.distcp, keep=2)`；新增事件 `recovery_selected`（含耗时与被内容哈希的代） | L3、§6.2 |
| 同上 | `prepare` 的 parent 改用 token 链派生的 head，不再用可能陈旧的 `control['head']` | 活性修复（§8-4） |
| `tests/ft/p2_schedule_driver.py` | `*.i09` 改为篡改 token 链上的 head（原先篡改祖先 bootstrap） | v2 §6.1-1 |
| `tests/ft/offline_generation.py`（新） | 离线逐代文件检查，支持 marker 和 pin；与方法状态代码独立 | S4 |
| `tests/ft/check_ft1_chain.py`、`check_training_fault.py`、`check_training_integration.py` | 改用 `offline_generation.check_files`；`check_ft1_chain` 按 `ft1-case.json` 的 `steps` 参数化 | S4 |
| `tests/ft/check_ft1_fault.py`、`check_ft1_load.py`、`check_ft1_input_audit.py` | F2 故障点序号和步数从 case 读取，默认值仍为 10 步、第 2 次更新 | §6.2 |
| `scripts/ft/ft1_fault_hooks.py`、`scripts/ft/areal_ft1.py`、`tests/ft/run_training_fault.py` | `FT1_F2_ORDINAL` 只允许 2 或 12，`FT1_STEPS` 只允许 10 或 30；30 步 run 上限 2700 秒 | §6.2 |
| `tests/ft/run_ft1_gate.py`、`tests/ft/check_ft1_gate.py`（新） | 30 步 F2 门控试跑与判定，判据在运行前写入 `ft1-case.json` | §6.2 |
| `RewardTxn 已发表方法容错优势补充实验方案.md` | 追加"修订 R1"（§9 证据保留），两臂同样适用 | §8-3 |

另：`tests/ft/run_ft1_acceptance.py` 在本日较早时已改为可重入（每次重跑用新输出目录），并加了 900 秒超时 kill。

## 2. CPU 验证

| 项 | 结果 |
|---|---|
| 宿主 `test_state`（21）＋ `test_state_fork_lock`（1）＋新增 `test_state_efficiency`（17） | 39/39 通过 |
| 容器（固定镜像，无网络，只读）全部 `tests/ft/test_*.py` | 除下述环境问题外全部通过：P2 driver 25、oracle 25、p2_schedules 14、p2_schedule_audit 12、rlvr_replay 12、strict_scoring 11、replay 9、descendants 8、runner 8、native_trainer_probe 7、training_replay 5 等 |
| 容器对照（HEAD 基线 vs 修改后，同一环境，补 `USER/LOGNAME`） | 9 个模块结果逐项相同；新增 `test_native_snapshot_r` 2/2 通过（并行与串行逐字节一致） |
| P2 全部 60 个 `*.i09` 用例（driver） | 基线与修改后都是 51 通过、9 个 driver 不支持而未执行，逐项相同；篡改对象现为 token 链 head |
| 旧证据只读回归（新版 `check_ft1_chain`） | `ft1-f1-s421-r-r2`、`ft1-f1-s421-a-r1`：与原判定逐字段一致（R 多出 `pruned_files=0`、`pins={}` 两个新字段）；`ft1-smoke-s401-r-r1`、`ft1-f4-s417-r-r1`（checkpoint 已被人工删除）判为失败，即不可复验，符合预期 |

环境问题（与本修改无关，基线相同）：
- 容器内 uid 1028 没有 passwd 条目，不设 `USER` 时 `getpass` 失败；
- `tests.ft` 包名被 `third_party/areal` 的同名 `tests` 包遮蔽，`test_state` 的子进程用例只能在宿主跑；
- `test_writer_recovery` 在宿主因缺运行环境失败，容器内通过；
- `test_areal_pilot_contracts` 在两组都是 0 项。

`check_training_fault.py`、`check_training_integration.py` 对应的旧证据目录（`training-*-gpu-r*`）本机已不存在，只做了编译检查，未做旧证据回归。

## 3. 已披露的边界
1. L1：不被加载的祖先 checkpoint 如发生同大小内容篡改，在线不发现，由离线审计全量哈希覆盖（`test_ancestor_same_size_corruption_is_online_boundary`）。
2. L2/C1b：首次哈希之后、同一时间戳刻度内的同大小覆写（包括 unlink 后同尺寸重写并复用 inode）不会改变 stat 身份；前提是 writer 已 join，且有 finalize 收据和 pending gate。
3. L3：R 在运行中只保留链上最近 2 代及 pin 的权重分片；已剪代的权重内容无法再复验。

## 4. GPU 门控
见 `R_SLOWDOWN_GATE_REPORT.md`（运行后填写）。
