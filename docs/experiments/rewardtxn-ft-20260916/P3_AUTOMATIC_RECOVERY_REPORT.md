# P3 真实故障自动恢复工程验收（2026-09-21）

状态：**验收通过**。公共生命周期补丁下，单 rank、一次 post-optimizer/pre-save SIGKILL 后，原生自动重启、完整状态精确加载、样本重放和后续提交闭环均已实测。

[可回查独立验收摘要](training-fault-verification-r1.json)；[原始验收结果](p3_evidence/training-fault-gpu-r1/verification.json)；[补充核验](p3_evidence/training-fault-gpu-r1/supplementary-verification.json)。

## 验证范围

用户明确批准 A 和 A+R 共用 [公共 launcher 生命周期补丁](JOB_LIFECYCLE_PROPOSAL.md)。本次检验 A+R 的单 rank、一次 post-optimizer/pre-save SIGKILL 自动恢复闭环。原生local_main负责重启，namespace controller只启动一次launcher；不会替方法拉起第二代。旧未修订部署native r4的timeout结果保留。

先完成step0完整异步保存与提交，然后在第二次真实optimizer成功、scheduler/save尚未发生时，controller以凭据认证的pidfd精确杀死trainer。公共生命周期层回收该job后代并退出，原生retry启动新trainer。验收要求加载先前保留的完整状态、未提交候选没有消费权威、其32样本重放后进入提交链，并继续完成step1/2。

## 预检与代码

- CPU公共生命周期6项通过，32.954秒，无跳过；含真实torchrun SIGKILL与原生local_main自动重试，见 [CPU摘要](job-lifecycle-retry-cpu-verification.json)。
- `scripts/ft/areal_training_fault.py` 只添加该故障切点与身份/状态见证；训练实现使用已验收的training_adapter。
- `tests/ft/run_training_fault.py` 使用固定已有镜像，现场挑选4张空闲H100，私有PID namespace、只读workspace、32CPU/128GiB/4096PID/16GiBshm、内部网络、host lease和完整ID清理；总controller运行窗口1200秒（不写作故障后1200秒RTO）。
- `tests/ft/check_training_fault.py` 独立核对controller身份/唯一SIGKILL、native run0/run1、生命周期顺序、4次物理optimizer/3次提交、完整checkpoint哈希及消费/重放链。
- 原生 RecoverInfo 探测仍看原默认路径；R使用自己的retained generation。因此原生日志is_recover_run可能为False，但是否恢复必须由实际R checkpoint加载和完整状态精确比较证明，不能只看该标签。

## GPU 运行

`p3_evidence/training-fault-gpu-r1`，现场空闲卡0/1/2/3。源码、配置、镜像与UUID在启动前记录，运行中未热改源码；独立重算源码哈希一致。

| 验收项 | 实际结果 |
|---|---|
| 控制器启动次数 | 仅1次launcher_created，无controller重启 |
| 真实注入 | 唯一SIGKILL，trainer PID688，kernel credentials和同一pidfd退出确认 |
| 原生重试 | local_main run_id 0→1，新trainer PID2193；pending→already_fired，无二次注入 |
| 生命周期 | 两轮trainer与两轮inference job共4份receipt齐全，全部cleanup.empty=true；前轮清理结束早于新轮启动 |
| 原生完整加载 | 保留step0 generation的模型、optimizer、scheduler及各RNG内容精确相等 |
| 物理更新/有效提交 | 4次optimizer success，其中故障前第2次未保存；最终3次有效提交 |
| 消费链 | 96个唯一样本，64个跨进程复用后提交；故障前未提交的32样本均重放并最终消费 |
| 废弃候选 | g-cb018c2ce49a481fa8ed560127c45aa8 无token，不属于最终保留链 |
| checkpoint哈希 | 20,754,258,566字节文件重新哈希，3代token/manifest/intent/文件清单链一致 |
| 收尾 | 恢复trainer正常返回，最后trainer job root退出0；原生launcher保留COMPLETED→JobException/退出1语义，controller容器退出0 |
| 资源 | 容器和网络完整ID独立确认删除；GPU0/1/2/3均回到0%、各4MiB |

保留generation顺序：

1. step0 `g-6c04282533c9421ea47d21c451c9eeac`，首trainer提交。
2. step1 `g-5f79d52532404bebad5e05fcb9060cf2`，恢复trainer提交。
3. step2 `g-179278dd38384542874608c7df6c67e8`，恢复trainer提交。

从同一单调时钟的signal_sent开始，到恢复后第一次optimizer success为 **173.198439346秒**；到恢复后第一次commit为 **233.613593145秒**。这是该单次工程样本，包含清理、原生重试等待、服务/模型启动、加载与校验，不能称正式对照平均RTO或收益。整个run墙钟679.567秒、分配0.755075 GPU·小时；1200秒是预设整个controller运行窗口，并非本次实际恢复时长。

## 复算

```bash
.venv-tq/bin/python3.11 tests/ft/check_training_fault.py docs/experiments/rewardtxn-ft-20260916/p3_evidence/training-fault-gpu-r1
.venv-tq/bin/python3.11 tests/ft/check_training_fault_supplement.py docs/experiments/rewardtxn-ft-20260916/p3_evidence/training-fault-gpu-r1
```

主验收器启动前冻结；补充验收器在运行结束后只读补查每轮全部job凭据、实际网络ID绑定及源码哈希，不改变运行结果。Docker创建前NetworkSettings.NetworkID为空，因此网络绑定使用创建时HostConfig.NetworkMode及退出时实际NetworkID共同核对。

原始证据包括 `events.jsonl`（controller）、`rewardtxn/events.jsonl`、`areal/logs/.../trainer.log`、`job-lifecycle/*.json`、完整retained state、配置/源码hash、容器inspect、清理和成本。角色日志使用普通文件，不能只看launcher.log推断训练是否恢复。

## 边界

不覆盖保存写入中SIGKILL、多rank、换PID namespace或多次故障。pending writer gate仍会拒绝未关闭写入的接管。本次不是正式配对矩阵或性能比较，正式GPU样本0、P2的184项待补不变；评分继续现有official_call_returned工程语义，不提前冻结正式评分合同。
