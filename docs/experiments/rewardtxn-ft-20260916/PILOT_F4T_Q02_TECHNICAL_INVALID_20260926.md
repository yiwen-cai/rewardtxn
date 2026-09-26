# 试点第 2 对（F4'，s7987）技术无效与重做

- **原尝试** `pilot-f4t-q02-s7987-20260926`（顺序 R→A）。
  - R：容器已创建但从未启动（`inspect-final.json` 中 State=created，墙钟 5.8 s），没有产生 `events.jsonl`，验收在 `check_ft1_fault` 处因缺文件失败。
  - A：GPU 预检报"selected GPU has a compute process"，拒绝启动。
  - 现场：分配的 GPU-40b9…、GPU-1b3b… 被外部 `VLLM::EngineCore`（pid 1778331、1778259）占用，每卡约 72 GB。
  - 故障信号未发出，按冻结规则属技术无效，原件保留。
- **重做**：按规则做一次完整配对重做，同种子、同顺序，新目录 `pilot-f4t-q02-s7987-20260926-redo1`。冻结见 `PILOT_F4F1_FREEZE_20260926_REDO1.json`，只改第 2 对的名称并加 `redo_amendment`。
- **其余**：第 3、4 对照常执行；启动器等待任意 4 张同时空闲的 H100。

## 重做结果（2026-09-26）：再次技术无效，按规则停止
- `pilot-f4t-q02-s7987-20260926-redo1-r`：
  - 启动时四卡空闲；运行约 3 分钟后完成第 1 次更新。
  - 权重同步 `/update_weights_from_distributed` 返回 400 / 服务端断连。`llm_server.log` 记录 CUDA OOM：GPU 1 上外部进程 1949262 占用 72.26 GiB。
  - 此后训练停滞，控制器在 2400 s 截止时仍未注入故障，判 `technical_invalid`（`run deadline before injection/release completion`）。与正式 F2 p02 原尝试的失败机制相同。
- 本对已用掉唯一一次重做，按冻结规则停止，不再自动补做。F1 两对（第 3、4 对）因启动器随之退出而**未启动**。
- 两次失败的原因都是外部作业在运行中占用了共享 GPU，与 R/A 的实现无关。原件全部保留。
