# 正式实验 GPU 分配补充记录（2026-09-24）

原始冻结清单 `FORMAL_FREEZE_20260924.json` 的 SHA-256 为
`c0e386e88dda06fcb7231506636506af246cb3ffcfe03718de591e69b280343b`。
首对 A 已按原冻结清单使用 GPU 1/5/6/7，训练、故障/输入/来源/链验收及独立 native 加载完成；协调进程在分片清理和配对登记前退出。随后在不重训、不改写原始验收文件的条件下，对该 A 的分片做 SHA-256 记录并删除，回收 6,917,500,928 B，补登记在首对配对记录中。

用户随后允许任意四张空闲 GPU。自第二对起，每对开始前选择四张同时空闲且型号相同的 H100；该对 A/R 固定使用这四张，运行前均重新检查 GPU 空闲、冻结来源及空间门槛。每对实际 UUID 与原冻结清单 SHA-256 写入该对专用的 `FORMAL_FREEZE_<pair>.json`，其 SHA-256 再写入配对记录。原 seed、A/R 顺序、故障窗口、训练、验收、统计、失败停止及最小存储规则保持原冻结值。修改仅涉及宿主调度和每对设备分配。

第二对先于首对 R 启动，因为首对指定的 GPU 1 一度被其他任务占用，而其他四张 H100 已空闲。首对 R 已登记恢复等待任务：其余各对完成后，在原来的 GPU 1/5/6/7 均空闲、来源及磁盘检查通过时执行，保留首对原冻结清单和同组设备。若任何配对失败，正式调度停止并保留证据待审查。

旧故障验收器 `check_ft1_fault.py` 在 `functional-verification.json` 中把 `formal_sample` 固定写为 `false`，并沿用 pilot 范围文案；它不控制本次运行的参数或正式配对判定。本次正式样本身份由执行前写入的 `ft1-case.json`（`formal_sample=true`）、对应配对记录（`formal_sample=true`）和该对冻结清单判定。保留原验收器与其输出，公开记录这一元数据不一致，不回填或改写原始结果。

调度入口：`tests/ft/launch_ft_formal_dynamic.py`；首对 R 恢复入口：`tests/ft/resume_ft_formal_first_pair.py`。运行日志分别为 `FORMAL_DYNAMIC_LAUNCH_20260924.log` 和 `FORMAL_FIRST_PAIR_RESUME_20260924.log`。
