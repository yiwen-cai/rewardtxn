# FT1 F1 公共注入 hook（已批准并安装）

用于两臂相同的 F1：目标组已有一个真实生成返回，且同组仍有请求在实际 SGLang GPU worker 的 run_batch 执行时，向该 worker 注入一次 SIGKILL。正常 batch 选择算法、推理参数与原生重启规则不变。

- 安装点：`third_party/areal/areal/v2/inference_service/sglang/scheduler.py`，PPSchedulerBridge.bind 之后；仅 FT1_SCENARIO=F1 生效。
- 完整补丁：`p3_evidence/ft1-f1-contracts-r3/scheduler-observer.patch`。
- helper 源码：同目录 `sources/ft1_scheduler_observer.py`、`sources/ft1_f1_trainer.py`；拟安装到 `scripts/ft/`。
- 独立审查后增加：trainer 实时 pidfd 存活与完整身份校验；active 标记必须晚于本 scheduler attach，拒绝一致但旧的标记；两臂共同短文件锁只保护 marker/claim，不等待生成或评分。claim 后握手失败不换目标。
- 真实数据 6,373 条完整 tokenized prompt 唯一性核验通过，target source_row_id=5518；只建立组级映射，不声称 sample_idx 映射。证据 `p3_evidence/ft1-f1-target-r1/f1-target.json`。
- CPU 12 项 scheduler 正反例通过，覆盖命中、仅前缀、finished、trainer 不一致、未返回、已 claim、歧义、错误 source/nonce、旧启动代、trainer 已退出、两份一致错误身份。真实 namespace/pidfd/信号；scheduler/CUDA/请求为 fixture，不是 GPU 命中证据。
- trainer marker CPU 测试通过：真实 RLVR 方法包装接口，内部执行为 stub；验证转发不变、只标记目标组、token 漂移拒绝、第二任务歧义拒绝。

此 hook 改变启用时 scheduler 执行路径：在有效切点与控制器握手并等待预定杀进程。因此适用 `third_party/areal/AGENTS.md` 的 “Ask first ... Changing launcher or scheduler logic.”。用户已明确批准该公共 hook；已在 seed409 配对结束后安装，安装后 18 项隔离合同＋6 项入口测试通过。GPU F1 尚未执行。
