# 两臂公共评分状态补丁（2026-09-21）

状态：**补丁已应用，最终回归通过**。用户已批准评分修订，并接受首批正式拓扑为单训练 rank＋3个推理实例。该拓扑选择尚不等于正式配置冻结。

[验收摘要](scoring-status-verification-r3.json)、[公共AReaL补丁](common-scoring-status.patch)、[容器独立清理核验](scoring-status-independent-cleanup.json)。补丁不包含此前已批准的launcher生命周期修改；相关R适配器源码由本次source hash单独绑定。公共patch在实际工作树上通过反向apply检查，未提交或推送代码。

## 行为

| 情况 | 处理 |
|---|---|
| 正常判对/判错 | 有效1/0，允许R缓存复用 |
| 模型回答无可提取表达式 | 保持有效零分，不把格式错误自动当作系统故障 |
| 标准答案无可解析表达式 | `invalid_gold`，不重试、不转零、不静默替换数据 |
| 评分执行异常 | `error`，对同一输入有限重试，耗时计入调用 |
| 评分等待超时 | `timeout`，终止并回收该次独立评分进程，再决定重试 |
| 进程启动/序列化/通信/清理异常 | 专用终态基础设施失败；不能进入普通丢弃/补样本路径 |
| 重试仍失败 | `RewardEvaluationError`；停止本次批次准备，失败不产生有效reward/tensor缓存或训练提交 |

预检默认每wrapper并发1、每次执行等待15秒、最多1次重试（共2次尝试）；无效gold只尝试一次。等待评分并发名额的排队时间另计入调用墙钟，不冒称15秒为整个排队调用的硬上限。回收阶段允许短暂TERM等待，必要时KILL并join；不能回收则终态失败，不继续重试。这是预检设置，正式参数待FT1配对预检后冻结。

GSM8K的公共`AsyncRewardWrapper`根据显式`strict_scoring`声明进入独立进程路径。A的观察评分包装与A+R的ReturningReward均传递该声明，使用相同默认策略。旧非严格callable仍保留历史执行语义，不能将其标作严格评分已验证。每次严格评分使用本次调用自有Linux fork进程，不终止共享旧进程池，不修改同组“生成后立即评分”的顺序，也未另建RM服务。

## 修改位置

- `MathVerifyWorker.verify_strict`和GSM8K：使用math-verify已提供的`raise_on_error=True`暴露公共parse/verify边界原本被转为[]/False的异常；原提取配置、精度和比较规则保留。评分失败不再由GSM8K外层catch转零。数学库内部有意使用的解析/比较fallback不被统一重新解释为系统故障。
- 去掉严格路径内层ThreadPool超时；原实现的上下文退出会等待线程，不能形成实际执行期限。现在外层独立评分进程负责超时和回收；直接同步调用`verify_strict`不自带超时，运行与数据预检都应经过公共wrapper。
- `GroupedRolloutWorkflow`在终态评分错误/取消时收齐其余sample任务；`WorkflowExecutor`将专用失败写入已有dispatcher失败槽并唤醒主线程，避免普通`Exception→None→补新prompt`行为。
- R严格评分使用schema2、status=`scored`。旧call-return envelope不能冒充严格有效评分；失败不写有效reward/tensor缓存。训练端verifier指纹覆盖GSM8K、worker、wrapper、状态/进程实现及math-verify parser/grader源码，防止跨语义复用旧状态。
- `scripts/ft/validate_scoring_data.py`提供只读数据检查，记录数据完整hash与verifier指纹，按原始行号报告问题，不改标签、不删样本。64行一组通过同一有界执行器检查标准答案提取和自比较。

## 数据检查

`p3_evidence/strict-scoring-data-r1/data-verification.json`：当前`runs/diagnosis-20260910/train.jsonl`共6,373行，全部检查通过。

数据SHA256：`e348030a2faa24c7b732cf588d16dd409b83cdd18545144490e4e87e2a2b13c5`。这是格式/可解析性/自比较检查，不证明标签事实正确。

## 测试证据

固定已有镜像`sha256:c0573bb8412a8db753d44b02c9e04504d2392c54f019707f58acf26dd68d2469`；无GPU、无网络、只读workspace、每容器4CPU/8GiB/256PID。每次运行前记录源码hash，运行中不修改源码。

- `strict-scoring-cpu-r3`：28项通过，95.213秒，无跳过。
- `strict-scoring-legacy-r3`：88项通过，42.13秒；唯一pytest警告是只读workspace不能写cache。
- 合计116项测试通过。最终日志无`Task was destroyed`或`Exception ignored in:`；真正Grouped失败时八个sample任务均完成/取消收尾后，主线程才收到终态失败，accepted=0。
- 顽固评分进程测试使用0.15秒测试期限、两次真实执行，均经SIGKILL回收，总墙钟断言小于3秒。普通有效评分以及取消后的下一调用正常完成；六次连续调用验证Process fd未累积。
- 两个最终测试run的冻结源码哈希一致。数据预检使用的全部评分/runtime源码也核对一致；其后只补了Grouped收尾和相关测试，数据评分指纹未变，无需重复扫描相同数据。
- 本轮7个CPU容器均已按完整ID独立确认删除，无网络/GPU资源需要回收。

覆盖正常0/1及格式零分、底层异常、错误gold、真实重试、顽固worker超时KILL、调用取消、进程启动/序列化失败、Process句柄释放、A/R相同默认策略、R有效零分复用/失败不缓存/旧envelope拒绝、跨进程重放，以及真实Grouped→WorkflowExecutor→AsyncTaskRunner的终态传播和同组清理。

保留调试历史：r1新套件25项有1失败/2错误，其中旧writer错误消息断言已过时、旧超时测试仍期待fallback零分/迟到结果、新Grouped fixture遗漏额外kwargs导致未进入预期打点。r2的28项断言通过，但退出日志暴露未等待的同组协程；增加实际任务收齐断言并修复后重跑，r2不作为最终完整通过凭据。旧官方相关套件r1/r2各88项通过，最终r3再次通过。

## 复算与边界

```bash
# 固定镜像、PYTHONPATH包含/workspace、/workspace/third_party/areal、/workspace/tests/ft
python -m unittest test_strict_scoring test_rlvr_replay test_training_replay -v
python -m pytest third_party/areal/tests/test_math_verify_reward.py third_party/areal/tests/test_async_reward_wrapper.py third_party/areal/tests/test_grouped_rollout_workflow.py -q
python -m scripts.ft.validate_scoring_data /workspace/runs/diagnosis-20260910/train.jsonl /output/data-verification.json
```

CPU测试必须启用`FT_RLVR_REPLAY_CPU=1`并提供本地`FT_RLVR_TOKENIZER`；完整实际argv/env见各证据目录launch.json。生成和训练张量来自明确标识的CPU fixture，不将本轮算作GPU训练或正式配对样本。公共评分执行边界已变化，既有GPU恢复结果保留为旧版本工程证据；新版本仍需FT1配对预检，尤其F4实际评分进程和故障时刻的重新映射。未启动正式矩阵，未重跑准确率、正常吞吐或5%开销实验。
