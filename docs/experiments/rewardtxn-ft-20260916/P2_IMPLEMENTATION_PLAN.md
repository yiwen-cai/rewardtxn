# P2 状态与独立 oracle 最小实施合同

日期：2026-09-20。本文仅规划，不代表实现或验收完成。依据 [SCRIPT_REPAIR_PLAN](SCRIPT_REPAIR_PLAN.md) 与 [oracle_spec](oracle_spec.md)。本轮只读源码和文档、写本文；不接触运行中的 GPU pilot，不修改第三方。economy-dev 根会话 `01a0bd89-a9c3-7d00-b16d-4a0a766de676` 已显式只读查询为 enabled；角色 Astra medium，不递归委派。

## 1. 范围与先决条件

先实现 `scripts/ft/state.py`、`scripts/ft/oracle.py` 及针对性 CPU 合同测试。state 只服务 R，不能用于给 A 补 pending、去重或重投。oracle 离线读取独立证据，不导入 state 的验证、选代或提交判定函数。两者可以共享书面 JSON 规范，首版不另造抽象后端、插件或数据库层。

当前 A 原生 RecoverInfo 没有 dispatcher pending/data_generator 的完整持久状态；该缺口只观察分类。A/F3 预先 N/A；F4 自然窗口待真实证据。600 个合同调度不改变这些能力状态。P2 不实现 AReaL adapter、真实 ACK、GPU 调度或 reward replay 闭环，也不能因此宣布 P3 完成。

旧脚本只作反例素材：phase2_manifest 的头尾4KB hash 与 sidecar 存在不能证明提交；phase2_seal_rm 的 logical_id INSERT OR IGNORE 不能拒绝旧 attempt 先到，也不能把 SQLite 与 JSONL 双写视为同一事务；phase2_reconciler 只扫描已有 reward 行会漏掉从未写出的样本，CPU 重评分没有进入真实 optimizer，历史每步/保存秒数禁止进入新成本。

## 2. 单节点存储与故障模型

限定一个 Linux 主机、一块支持本地 POSIX rename/link、flock、文件及目录 fsync 的文件系统。运行前记录 mount 类型；不假定 NFS、对象存储、跨节点共享盘成立。不声称分布式共识、拜占庭防护或实测断电安全。CPU SIGKILL 测试验证进程崩溃；掉电持久性依赖已声明的文件系统保证，须与测试结果分开。

每个 run 一个独占目录：

```text
run/
  owner.lock                  # controller 生命周期持有，禁止 worker 继承锁FD
  mutation.lock               # 每次权威元数据变更持有
  control.json                # epoch、owner_nonce、active attempts、head hint
  generations/<gid>/
    intent.json               # 更新前持久化，禁止修改
    checkpoint/               # 真实后端直接写独立新目录
    receipts/                 # optimizer与各rank finalize持久收据
    data.json                 # cursor/shuffle/pending及payload文件引用
    manifest.json             # 最终完整内容清单与语义映射
    token.json                # 最后发布，不覆盖
  acknowledgements/           # 仅部署存在真实ACK时，追加独立收据
```

所有 JSON 使用 UTF-8、排序键、固定 separators、禁止 NaN/重复键；hash 为完整字节 SHA256。路径只能是 run 内规范相对路径，拒绝绝对路径、`..`、symlink及越界引用。大 tensor/checkpoint 保持后端原格式，JSON只记录完整文件摘要和语义引用，不二次发明模型序列化。

普通控制文件采用同目录临时文件→flush/fsync→os.replace→父目录fsync。不可变文件采用临时文件fsync→同文件系统 hard-link 到最终名字（目标已存在即拒绝覆盖）→目录fsync→清理临时名；幂等重试仅允许已有字节完全相同。发布 token 的 link 是可见性点，目录fsync成功后才能返回持久提交成功；崩溃后若发现已存在完整token，重新验证后可完成持久确认。latest/head仅缓存，不能覆盖token事实。fsync失败不返回成功。

checkpoint必须先全rank停止写该generation、完成后端finalize，再逐文件fsync/hash并fsync目录；固定路径边写边拷贝/轮询不能充当快照。R独立generation目录及等待成本归R。不可变是受约束程序协议，不靠chmod声称对同UID恶意写入免疫；加载和oracle均再次检查内容。

## 3. 具体 JSON 合同

以下是字段示意；省略的数组必须在实际产物中完整，不接受以字符串“complete”代替内容。

```json
{
  "schema": 1,
  "run_nonce": "...",
  "generation": "g000002-<nonce>",
  "parent": {"generation": "g000001-...", "token_sha256": "..."},
  "owner": {"epoch": 2, "nonce": "..."},
  "config_sha256": "...",
  "expected_ranks": ["actor:0"],
  "updates": [{
    "logical_update_id": "u2",
    "physical_update_id": "epoch2-u2-attempt1",
    "optimizer_index": 0,
    "groups": [{"logical_group_id": "dataset-hash:epoch:row:occurrence", "samples": [
      {"sample_index": 0, "attempt": "a2", "recovery_epoch": 2,
       "response_sha256": "...", "reward_sha256": "...", "policy_version": 1,
       "verifier_version": "v1", "tensor_input_sha256": "..."}
    ]}],
    "train_input_sha256": "..."
  }],
  "data_snapshot_id": "cut2",
  "ack_capability": "none"
}
```

这是更新前 `intent.json`：完整列举每组预期K个样本，逻辑更新ID不能直接等于可重用global_step；保存顺序、同一step内多个optimizer更新均明确。每条实际训练tensor映射到组/样本/attempt；原始reward及归一化/advantage等变换配置、顺序均可回查，不能拿原始标量直接比较变换后tensor。

`data.json` 包含 data/source全内容hash、epoch、shuffle状态或确定的恢复状态引用、cursor、已取出序列、已消费集合、pending清单。pending逐项写明逻辑身份、当前合法attempt、生成/评分阶段、可用tokens/logprobs/masks/versions等payload完整hash，或者明确 `regenerate` 及同prompt/K重建所需输入。从未返回reward的样本仍必须在预期清单中。用“cursor往前、pending为空”掩盖预取工作必须拒绝。pending不是必须复用全部回答，但每个已取出未消费工作必须有可执行恢复解释。

`manifest.json` 引用 intent/data完整hash，列出所有payload/checkpoint文件的path/size/sha256，并包含：model；optimizer master/moments/step；scheduler state/next_lr；每个expected rank的Python/NumPy/Torch CPU/device/tracker RNG；policy version；真实optimizer成功及scheduler推进收据；全rank同一snapshot_id的finalize收据。空optimizer、缺rank或缺组件不能用“后端支持”豁免。声明不适用的组件必须有冻结后端依据，例如无CUDA的CPU合同fixture；不得沿用到GPU。

`token.json` 只含 schema/run/gid/parent_token_sha256/intent_sha256/manifest_sha256/commit_epoch/owner_nonce；token标识为其规范字节完整hash，不做自引用hash。首个root也有冻结基座与初始完整状态凭据。每次保存的状态包括父代之后所有实际更新，不能遗漏其中一次成功optimizer操作。

事件至少包含event_id/type、run、source角色/rank、PID/start-time/boot-id/cgroup、attempt/epoch、逻辑身份、source单调时间、controller接收时间、payload引用。token前持久收据属于R自身数据；observer副本另存且不向R开放。state检查收据结构和一致性，不能仅凭布尔值证明真实GPU行为；真实接缝真实性是P3及独立observer验收项。

## 4. 最小 API 与提交顺序

state.py公开接口建议限制为以下七项，使用小dataclass或dict，不建通用仓储框架：

1. `acquire_owner(run_dir, expected_epoch, prior_owner_exit_proof) -> Owner`：非阻塞获得生命周期owner锁；mutation锁内检查CAS并持久递增epoch、换nonce。首次epoch=0不需要前owner退出凭据。
2. `authorize_attempt(owner, logical_sample, expected_attempt, new_attempt) -> Attempt`：先检查当前epoch/nonce，再CAS合法attempt。必须先授权新attempt再接收结果。
3. `accept_result(owner, attempt, payload_manifest) -> Receipt`：在权威写入之前检查epoch、nonce、当前attempt及版本；同attempt相同内容幂等，冲突内容拒绝。不以logical_id第一次写入获胜。
4. `prepare_generation(owner, intent, data_snapshot) -> Generation`：确认完整合法组与父token；持久写intent后，才允许adapter启动该物理optimizer操作。
5. `record_evidence(owner, generation, evidence) -> Receipt`：保存optimizer成功/scheduler推进/各rank finalize等不可变收据。拒绝非预期rank、不同snapshot、重复冲突；不主动执行模型保存。
6. `commit_generation(owner, generation) -> Token`：验证完整证据→fullhash→发布manifest→再次检查fence与父head→发布token。每run首版最多一个未提交更新generation，不支持流水提交。所有权限复查和最终token发布在mutation锁内。
7. `select_recovery(owner) -> RecoveryDecision`：完整验证token链和候选；返回可加载路径、需重新生成/评分的自有pending引用、明确拒绝原因。它不调用训练/ACK、不读取observer、不宣称恢复完成。

大文件保存/hash不长持mutation锁，但finalize后不得再写；重入提交锁时必须重新验证owner和父代，失去owner的候选只能隔离。下游真实optimizer开始前与结果回收后都必须校验fence。文件锁不能中断已在GPU执行的旧owner：恢复前必须证明旧actor/worker已退出或被可靠隔离，再加载新actor。仅owner.lock释放而后代仍活着，返回blocked/safe_stop，不能自称GPU fencing成功。持有锁期间不按超时抢锁，不实现猜测性lease租约。

严格顺序为 intent持久化→实际optimizer成功→scheduler状态落实→同切点所有rank保存finalize→内容与数据/pending验证→token持久发布→真实ACK（若存在）。首版A接入 `ack_capability=none`，token仅是R提交凭据；不产生伪ACK事件。ACK适用时，丢ACK只允许调用原接口幂等确认，不重复训练。

无token候选仅当预先持久intent、完整optimizer/scheduler收据、全rank同切点finalize、完整文件/data/pending及父链均可验证，且旧执行者已退出，才允许新owner在当前epoch补发布token；token保留原执行epoch并记录补提交owner。缺任何一项回退已验证父代并重算。token存在但内容损坏/缺文件时拒绝该token并报告corruption；默认安全停止，不静默当作未提交候选倒退。若未来允许回退必须作为显式冻结策略并保留损坏证据。分叉多个已提交孩子同样拒绝自动挑“最大步”。

## 5. oracle 独立实现

`audit_run(evidence_dir, frozen_spec) -> report` 为唯一核心入口；CLI可为 `python -m scripts.ft.oracle --evidence ... --spec ... --output ...`。只允许只读证据目录和写报告目录。禁止导入state、replay、旧reconciler或被测方法的success判定；源码测试可检查导入依赖边界。verifier由冻结白名单选择，不能根据不可信日志动态import任意路径。

按以下顺序判读：

1. 验证冻结输入、事件完整性、时间线及故障实际observed；输入丢失/乱序未能解释则记录具体missing_evidence，不默认成功。只有源码预先判N/A可得not_applicable；有效命中后方法失败不改N/A。
2. 以实际checkpoint_loaded事件及其全内容证据定位恢复点；构建执行分段和最终持久状态祖先链，保留恢复点之前的祖先消费集合，不能仅从恢复后的step开始去重。若没有重启，使用冻结初始状态作为起点。A不要求R token，依据真实保存/load/state证据建链；证据不足为unverifiable。
3. 将物理optimizer操作投影到最终保留链。父代之外被回滚的操作不算duplicate；同一冻结logical_update或同一消费义务在保留链重复才判重复。合法多epoch再次读取相同行不是重复，必须用dataset epoch/occurrence区分。hash相同不等于相同更新，数值非确定性也不直接等于invalid_commit。
4. 对每个保留更新独立读取原始prompt/label/回答/tokens/masks/logprobs，调用冻结真实verifier重新评分，再验证训练变换与实际optimizer输入映射、K完整性、policy staleness及attempt/epoch资格。旧attempt先占位即使reward值恰好一样，也不能以“数值相同”放行。只有observer原始副本中的authoritative事实可作为依据，方法自报reward/success不算。
5. 给出三个独立维度：safety(pass/invalid_commit/unverifiable)、training_continuation(continued/safe_stop/timeout/unverifiable)、affected_work_recovery(recovered/safely_dropped/unresolved/unverifiable)。确证违规优先报告invalid_commit；纯缺证据为unverifiable，不能把缺日志推断成方法作弊。
6. RTO从controller observed到角色可服务且目标组首次oracle-valid持久提交两条件同时成立，且该提交在最终保留链中。首次后来被回滚的“提交”不充终点。若角色再次故障，角色可服务状态按事件区间判断，不能盲取两个历史首时刻的max。最终安全性违规取消correct_recovered。未恢复RTO=null、censored=true；900秒仅另列penalized_score。继续训练其他prompt不等于目标恢复；30步完成不提前截断目标观察窗。

报告还含实际加载/最终状态ID、证据引用、缺口清单、回滚/重复生成/丢弃/verifier/I/O/GPU分配时间实测成本。无法观测成本字段为null并说明，不用历史常数补齐。observer数据不挂给方法读取，控制消息仅release/abort；权限/挂载与协议测试拒绝回传payload。

## 6. 预先冻结的600个有效调度

新增 `tests/ft/fixtures/p2_schedules.json`，六cell各100项，固定生成器版本和seed（建议独立测试seed20260920，不改FT-v1正式seed）。每cell采用10个不同边界变体×10个交错变体；每项记录实际操作序列、payload、cut位置、expected结果、覆盖标签和完整hash。A-model/R-model使用相同外部payload与故障顺序，允许不同期望结果，不要求A实现R协议。

| cell | 10个边界变体应覆盖的范围（每项在fixture中展开成确定步骤） |
|---|---|
| F1 | 完成1至7个样本的生成边界共7项；只有token payload缺失；已写回答但缺logprob；整个未返回样本从预期清单发现 |
| F2 | intent前/后、optimizer开始前/执行中/成功未记收据、成功收据后、scheduler未落实、单rank部分写、全rank finalize前、manifest完成token前 |
| F3 | token前、token可见未确认durable、token持久未ACK、ACK发送丢失、ACK执行收据丢失、ACK重复、ACK迟到旧epoch、ACK错generation、已ACK重投、无真实ACK部署（预期N/A） |
| F4 | K完整时第4评分执行中/4评分完成的分别标注模型、K缺一、重复sample冒充K、reward启动未完成、完成未落盘、版本混合、损失worker同时影响同组多个样本、存活原生池重试、目标窗口已过去 |
| X1 | 正确v1、真实v2不同分数、仅版本标签伪造、错误label、回答hash错、训练归一化映射错、混组reward、过时policy、缺authority原文、方法伪造success |
| X2 | 相同结果重复、冲突重复、旧attempt先到/后到、新epoch合法结果、未授权attempt、错误group、错误run、重复消费映射、旧owner延迟提交 |

10种交错固定为：顺序；独立样本逆到达；重复同事件；延迟旧attempt；合法授权新attempt后旧结果抢先；两真实controller进程争owner锁；controller在已确认/未确认注入间崩溃；恢复后保留第二故障；删除一份必要证据；篡改一个完整文件的中段字节。每种在对应cell明确落到具体操作，不允许相同trace换seed凑100项；相同trace/payload哈希重复导致测试集验收失败。组合前先生成满足该cell前置条件的trace，再在合法位置应用交错；若语义冲突，登记为预期拒绝的负例，不能把它算有效故障命中。

覆盖报告同时列正常合法轨迹、有效命中后的方法结果、被拒绝技术无效、规范N/A、缺证据五类数量；每cell至少含合法恢复/合法回滚重算、不完整证据、确证错误保留链。测试为每项声明应触发的分支/不变量并检查达到；不能600项都早早卡在schema验证。额外反例补充缺rank、缺optimizer映射、token缺文件、cursor跳pending、observer反哺、PID复用、fired状态不明重复注入拒绝。这些可成为上述组合的具体payload，覆盖不足再加针对性测试，不改“六cell各100”主集合。

独立expected由小型手写规范状态表/fixture明确给出，不能调用state生成oracle期望。state产物交给独立oracle还不够：需手工合法链与定点损坏链验证oracle本身。进程竞争与中断用真实multiprocessing/普通子进程＋确定性握手，在每个持久写与token发布前后SIGKILL；不以sleep概率制造切点。模型checkpoint小文件测试写入中断，不冒称Megatron保存中断。部分写入/异常注入和真实进程SIGKILL分别标注。

## 7. 下一项有界实施 brief

先只实现 `scripts/ft/state.py` 与 `tests/ft/test_state.py`，不接GPU、P1 runner或AReaL，不实现replay和600调度生成器。交付上述7个API、单节点持久原语、完整字段/路径验证、epoch/attempt CAS、token发布/校验及补提交条件；CPU小文件表示完整组件，明确fixture级别。

针对性验收至少：完整提交/新进程读取；中段篡改拒绝；缺rank/optimizer/pending映射拒绝；旧attempt先到拒绝；两个进程只有一个owner；旧owner失效后延迟提交拒绝；token前足证据补提交/缺证据回父；token后head未写仍能找到正确链；token损坏安全停止；owner退出但旧worker存活不得接管；已提交重试不产生第二token。建议命令 `python -m unittest tests.ft.test_state -v`（若tests非包则用discover），只运行本项测试，不扩大到GPU套件。

后续再单独派发oracle＋手写反例，最后固定600调度并集成。任何CPU合同通过均只更新P2合同状态，真实RecoverHandler pending、F4可达性、后代pidfd所有权及真实保存/ACK适配继续由pilot/P3凭据决定。
