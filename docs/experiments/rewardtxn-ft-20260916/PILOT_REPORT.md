# 早期 RLVR 接入与新进程恢复探针

2026-09-20。真实无故障3步及新进程加载后下一步均通过限定核验。**这不是故障注入、完整pending恢复、A+R闭环、FT1或正式实验通过。**

## 已核实结果

- nofault-r4：Qwen2.5-0.5B-Instruct，4张H100，actor1＋SGLang3，隔离6373行训练集，K8/U4，每批32回答；3次真实optimizer调用成功，3次同步DCP与RecoverInfo保存。
- 每次输入有4个完整组、32个唯一组内样本；观察键经过真实PPO合同对照，未改变优势计算值。首步有效LR为0（保留官方warmup），后续为1e-6，不能称三个非零LR权重更新。
- 最终checkpoint及metadata约6.92GB，全量文件SHA256独立重算一致；metadata含dataloader_info.pkl与step_info.json。原生路径覆写，因此只对最终文件做了独立重算，早期保存保留当时observer清单。
- resume-r1：将上述原始产物复制到独立目录，新容器经原生RecoverHandler实际加载model/optimizer/scheduler/RNG，读取清单与原保存完全一致；policy version3、LR1e-6继续一次32回答更新，保存global_step3并独立校验最终完整文件。
- 两轮训练正常返回；官方SPMD launcher在retries=0时将trainer的JobState.COMPLETED作为异常抛出，外层退出码均1。原始退出码完整保留，未改成0；判定依据是实际训练/保存事件和产物。

## 待处理工作与F4观察

nofault-r4产生158条generation/reward记录、19个完整组及一个部分组；实际训练12组。保存的sampler.samples_yielded=28、7个dataloader batch；新进程下一步后为48、12个batch。RecoverInfo不保存dispatcher pending，游标推进明显先于训练，完整受影响工作恢复仍不成立。这里未注入故障，不给目标工作恢复结果或RTO。

20个有生成记录的组中，8组自然出现“全部8条回答已生成，已有4条评分完成且仍有评分执行”的窗口；记录的连续子区间约0.65–8.66ms。也有8组在第4次评分开始前已完成全部8条生成。这只是带当前公共观测开销的自然候选窗口；没有控制器收据时钟、精确PID故障、release或独立oracle，不能据此宣称正式F4已命中或改变主终点定义。

## 失败记录与成本

nofault-r1：官方launcher预检要求name-resolve根目录存在；未开始训练。r2：network=none无法提供官方gethostip需要的非回环IP；停止本任务容器后保留退出143。r3：三个推理服务成功启动，但launcher直接读原始YAML缺省字段失败；显式补入原默认use_deterministic_algorithms=false后重跑。没有修改第三方、驱动或依赖。

5次工程运行（含上述失败、nofault-r4及resume-r1）累计分配GPU约0.935小时，按4卡×实际容器运行墙钟计算。原始命令、设备UUID、镜像ID、配置/代码hash、时间与退出码见各run的launch/execution.json。这些均不进入正式样本或收益统计。容器已退出，所用卡已释放，本任务内部网络已移除。

## 证据索引与边界

- `pilot_evidence/nofault-r4/verification.json`：无故障7项限定核验。
- `pilot_evidence/resume-r1/verification.json`：新进程恢复与下一步7项限定核验。
- `pilot_evidence/nofault-r4/natural_f4_observation.json`：逐组自然窗口。
- `pilot_evidence/dataloader_audit.json`：真实保存游标。
- `pilot_evidence/completed_engineering_runs.json`：失败分母和工程GPU成本。

真实RL新进程探针没有独立比较加载后的内存model/optimizer/scheduler/RNG逐项值或下一批连续性；P0单actor固定下一批测试仍是另一层证据。异步保存完成、多rank状态、自然窗口精确注入、原生job自动重启、P1后代所有权、R完整持久化/replay和独立oracle继续待办。正式矩阵尚未启动。
