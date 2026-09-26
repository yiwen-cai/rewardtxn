# 试点第 2 对（F4'，s7987）技术无效与重做

- **原尝试** `pilot-f4t-q02-s7987-20260926`（顺序 R→A）。
  - R：容器已创建但从未启动（`inspect-final.json` 中 State=created，墙钟 5.8 s），没有产生 `events.jsonl`，验收在 `check_ft1_fault` 处因缺文件失败。
  - A：GPU 预检报"selected GPU has a compute process"，拒绝启动。
  - 现场：分配的 GPU-40b9…、GPU-1b3b… 被外部 `VLLM::EngineCore`（pid 1778331、1778259）占用，每卡约 72 GB。
  - 故障信号未发出，按冻结规则属技术无效，原件保留。
- **重做**：按规则做一次完整配对重做，同种子、同顺序，新目录 `pilot-f4t-q02-s7987-20260926-redo1`。冻结见 `PILOT_F4F1_FREEZE_20260926_REDO1.json`，只改第 2 对的名称并加 `redo_amendment`。
- **其余**：第 3、4 对照常执行；启动器等待任意 4 张同时空闲的 H100。
