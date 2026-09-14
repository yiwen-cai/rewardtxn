# RewardTxn 当前交接

更新：2026-09-14

目标：排查 R（RewardTxn）相对 O（Oracle）的准确率回退。当前仅完成最小实验方案，新实验未实现、未启动。

已有结论：
- SQLite 单 seed191 的 R/DBM/R重复各完成500步，validation100正确数69/68/74；DBM在线RM成本下降15.3%，固定回答CPU成本下降21.9%，未证实准确率收益。R重复跨重启，不能当作稳定波动范围。
- 日志、分组/优势/mask/损失处理及奖励/格式审计未发现足以解释 R 回退的特有异常；历史完整训练logprob和梯度缺失，不能排除整个学习链。
- pilot 的 R−O 为−4/−5/−1pp，初筛为−5/−4/+1pp，复用seed11/23/37，不能合成6个独立seed。“稳定回退”及根因均未确认。

下一方案：O/R × 原异步/批同步，训练seed193，顺序OA→RA→OS→RS，各500步；数据seed42、评估seed29，GPU1为actor、2/3/4为rollout。完整预算2000步，另拟四组各10步工程预检。比较同步前后R−O差距，区分R改善与O下降；单seed只筛查机制。

下一步是实现同一driver内的同步分支：同时关闭提前生成和后台跨批预取，验证零在途及全部引擎权重确认，保存选定步实际训练logprob。候选复审、5%公共开销及同版本smoke缺口仍在；SQLite例外不迁移，启动前重新核验资源与冻结。失败即停，不自动重试。

旧LITE−R确认最后三轮继续暂缓；已有21份成功凭证，未读取该批终点质量，不作为新方案前置。

[实验方案](/public/home/caiyiwen/rewardtxn/docs/experiments/rewardtxn-async-sync-plan-20260914/PLAN.md) · [已有证据](/public/home/caiyiwen/rewardtxn/docs/experiments/rewardtxn-reproducibility-20260914/REPORT.md)
