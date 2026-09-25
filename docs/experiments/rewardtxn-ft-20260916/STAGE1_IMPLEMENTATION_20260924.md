# 最小 FT 实验阶段 1 实施记录

日期：2026-09-24。范围：来源证据、固定配对 pilot 入口、验收后存储清理、定向 CPU 验证。未运行 GPU pilot；正式样本数为 0。

## 实施

- `scripts/ft/areal_ft1.py`：只在真实 `engine.agenerate` 返回后记录请求和执行 ID；在评分子进程返回时记录执行 ID 和分数。旧 `generation_complete` 标明仅是评分入口观察。
- `scripts/ft/areal_pilot_hooks.py`：A 的每个样本执行 ID 随张量经过 group、`batch_taken`、`train_batch`，进入来源验收。身份列在进入原生模型和 loss 前剥离。
- `tests/ft/check_ft_minimal_source.py`：A 以执行 ID 核对进入训练的生成、评分、批次与最终持久更新，并单列未入训练的预取；R 核对 response/reward/tensor 原件、真实请求、唯一物理评分返回和 adoption 的原始 receipt→新 attempt 链，再比较被杀更新与最终链中的 32 条来源。相同文本的新请求不算复用；来源含糊时停跑。
- `tests/ft/run_ft_minimal.py`：固定 seed、显式 A/R 顺序、四个固定 GPU UUID，串行执行 F2 或无故障配对 pilot，并逐 run 记录磁盘峰值。首次及每次新 run 前暂要求 64 GiB 可用。
- `tests/ft/run_ft1_acceptance.py`：无故障 run 走 smoke、链、独立输入和完整状态加载验收。
- `tests/ft/minimal_storage.py`：仅在验收与来源均通过后，列出并哈希该 run 的 DCP 分片，再删除；失败时保留原件并停止配对。

## 验证

离线镜像：`sha256:c0573bb8412a8db753d44b02c9e04504d2392c54f019707f58acf26dd68d2469`。容器使用 `--network none --read-only`、仓库只读挂载，`/tmp` 和 `/output` 为临时文件系统，未分配 GPU。运行 `pytest -q -p no:cacheprovider tests/ft/test_areal_pilot_contracts.py tests/ft/test_ft1_preflight.py tests/ft/test_ft_minimal_stage1.py`。另外执行 `python3 -m compileall -q` 检查改动的 Python 文件与 `git diff --check`。

结果：受影响测试运行 **13 passed**；新增无故障验收用例后，阶段 1 测试文件单独运行 **5 passed**，合计覆盖 14 项。Astra 评审指出的在途预取与逻辑评分 nonce 两项缺口已补定向回归；R 的新 attempt 重封装、缺失 adoption、重复物理评分也有定向断言，修订后阶段 1 测试文件 **5 passed**。`compileall` 与 `git diff --check` 均通过。CPU 测试覆盖原生 PPO 分批身份传递、物理生成事件、评分子进程返回、固定配置、A/R 同文本新请求判定、无故障完整验收路径，以及独立加载验收前禁止清理。合成测试不能替代 F2 实际故障切点和完整 GPU 加载验证。

## 下阶段门槛

按方案各跑一对 F2 与无故障 pilot。R F2 必须 32/32 同执行来源复用；A 的安全丢弃须与来源证据一致。四个 pilot 的磁盘峰值决定正式 run 的 `P + 20 GiB` 门槛。若任一 run 验收或来源不明，保留完整原件并停跑。
