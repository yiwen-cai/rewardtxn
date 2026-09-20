# P2 独立 oracle：规范化证据合同实现

日期：2026-09-20。实现范围是 `scripts/ft/oracle.py`、`tests/ft/test_oracle.py`。本轮不接 GPU、不改 pilot、不运行600调度。独立手写规范图不使用 `state.py` 生成证据或期望；oracle不导入state、replay或旧reconciler。本文不表示真实GPU oracle验收通过。

## 调用与输出

```sh
python -m scripts.ft.oracle --evidence /absolute/observer-evidence \
  --spec /absolute/observer-spec.json --output /absolute/new-report.json
python -m unittest discover -s tests/ft -p test_oracle.py -v
```

核心接口 `audit_run(evidence_dir, frozen_spec)` 返回dict；第二参数为spec路径或已解析dict。CLI输出必须位于evidence之外，不能覆盖spec或已有文件；不修改任何证据。缺证据、不支持的语义和来源不匹配返回unverifiable，不靠程序异常冒称方法失败。输出包含status、safety、training_continuation、affected_work_recovery、真实RTO或null、censored、独立惩罚分数、保留states/updates、回滚physical updates以及具体缺证据/违规清单。

`correct_recovered` 仅表示本规范图的判读结果；必须结合 `evidence_level=fixture`，不能将其用于真实capability状态。当前 `observer-normalized` 输入也会因未实现真实训练reward变换而得到unverifiable。没有真实GPU oracle-valid结果。

## 可信边界与文件封装

spec分两部分：`freeze` 为预运行冻结合同；`observer_seal` 为事后由独立observer提交的 `{相对文件名: 全内容SHA256}`。事后文件hash不伪称运行前已知。`evidence.json` 自身也必须位于seal中，并包含预freeze规范JSON的 `freeze_sha256`。规范JSON为UTF-8、sort_keys、separators=(',',':')、ensure_ascii=False，禁止NaN和重复键。输入文件还必须匹配freeze.inputs中的运行前hash。

文件只允许evidence根内的普通相对路径；拒绝绝对路径、空路径段、`.`、`..`及symlink。加载时重算每个seal文件的完整hash。seal与freeze必须由调用者从可信observer通道提供；hash不是身份认证。被测方法同时伪造seal和证据的攻击不由本模块解决。实际部署仍需observer独立权限/挂载、只出不进控制协议与原始日志留存证明；本模块不声称仅从JSON就能证明这种物理隔离。方法的success、authoritative_reward不参与结论。

## schema 1

freeze必要字段：

```json
{
  "schema": 1,
  "run_nonce": "...",
  "evidence_level": "fixture",
  "timeline": "controller-clock-id",
  "run_start": 0,
  "inputs": {"dataset.json": "<full sha256>"},
  "initial_state": "s0",
  "expected_ranks": ["actor:0"],
  "fault_id": "f1",
  "target_group": "dataset:epoch:row:occurrence",
  "target_role": "reward",
  "k": 8,
  "max_policy_staleness": 2,
  "reward_transform": "identity",
  "recovery_window_seconds": 900,
  "verifier": {"name": "fixture-exact", "version": "v1"},
  "applicability": {"status": "applicable"}
}
```

N/A必须在freeze中预先写status=not_applicable、source、reason、decided_at，且决定时间不晚于run_start。无有效observed或observer/controller/external明确技术异常为technical_invalid；方法性超时不根据收益倒推N/A。

evidence.json包含 `run_nonce, freeze_sha256, events, states, updates, groups, final_state, end_time`。events按同一已校验controller时间线非降序排列，每项都有 `id/type/time/timeline/run_nonce/role/rank/pid/start_time/boot_id/cgroup/event_nonce/source_monotonic`。不相减不同进程的source_monotonic。各事件ID唯一，重复传输应由未来observer adapter保留原始日志并规范化为一个业务事件，而不能在这里重复算更新。

最小事件载荷：

| type | 额外字段 |
|---|---|
| initial_state | state |
| checkpoint_loaded | state、epoch；epoch=0也明确初始真实load |
| authorize | group逻辑ID、sample组内序号、attempt、epoch、policy_version、verifier_version |
| optimizer_start | update物理ID、epoch、tensor_sha256 |
| optimizer_end | update物理ID、epoch、successful、scheduler_applied |
| checkpoint_schedule / checkpoint_persisted | state、snapshot_id |
| checkpoint_finalize | state、snapshot_id、actor_rank |
| fault_observed | fault_id |
| role_ready / role_down | 公共role标明被冻结目标角色 |
| work_dropped | group逻辑ID |
| safe_stop | 公共时间字段 |
| technical_invalid | origin只能是observer/controller/external；方法失败不用此事件 |

state字段为 `id,parent,epoch,updates,data_file,components,snapshot_id,save_event,finalize_events,persist_event`。updates列父代后全部真实physical update，不能只挑结果正确的更新。components必须精确覆盖model、optimizer_master/moments/step、scheduler、rng_python/numpy/torch_cpu/device/tracker、data、policy；每组件映射所有expected_ranks到非空已sealed文件列表。CPU fixture文件可以是标明性质的小JSON，绝不冒充真实后端tensor。root没有updates，其persist_event指initial_state；非root要求全rank同snapshot完成且先schedule后finalize后persist。data_file包含drawn、consumed、pending逻辑组ID数组及cursor：三集合/游标须吻合，子代consumed必须等于父代加本代实际消费。

update字段为 `id`（physical）、`logical_id`（预冻结消费义务，不是global_step）、`epoch,policy_version,loaded_state,start_event,end_event,groups,tensor_file`。groups引用下述group记录ID。实际load必须是该更新所在最终链父代的祖先，且在持久保存后、optimizer前发生；这个祖先条件本身不充分。实现按实际checkpoint_loaded事件ID划分执行段，即使epoch数字未变，reload也会开启新段。子保存只有两种合法父关系：直接load父state，或与父保存处于同一无reload执行段；从旧祖先reload后不能继续挂接已丢失的中间父state。每个optimizer与其保存必须在同一执行段，且在父持久化之后开始；optimizer结束与保存之间reload也拒绝。最后load还须与final_state可追溯。每个保存窗口中observer记录的真实成功optimizer列表必须与state.updates一致，不能通过漏记映射藏掉已知更新。

group字段为 `id`（此次完整组记录）、`logical_id`（包含dataset epoch/occurrence）、`k,prompt_sha256,source_file,source_index,samples`。source_file必须为冻结dataset JSON数组，索引行含prompt和label。每sample含 `index,attempt,policy_version,verifier_version,reward,payload_file`。payload JSON含 `prompt,label,prompt_sha256,completion,policy_version,tokens,loss_mask,logprobs,prompt_ids,completion_ids`。前者用于冻结输入/回答身份与训练载荷检查，后两者为真实GSM8K verifier参数。每组恰好0..K-1、verifier版本一致；不同样本policy可不同，逐样本核查授权与staleness范围；optimizer入口前最新授权必须匹配attempt/epoch/version。版本声明一致不是版本真实性证明；真实采集由adapter承担。

tensor_file为 `{"transform":"identity","rows":[...]}`；rows保持实际训练入口顺序，每行含 `group,sample,attempt,payload_sha256,reward,tokens,loss_mask,logprobs`。optimizer_start绑定该文件fullhash，oracle逐行对照实际tokens/mask/logprobs值、样本身份与独立重评分；仅payload_sha相同不能绕过值比较。此表示是CPU规范模型的tensor见证，不是对任意PyTorch batch通用序列化；padding/normalization/advantage→optimizer的转换尚需真实adapter和变换验证。现有pilot的reward scaling、bias、group normalization不能由此版本默认放行。

## verifier白名单

- `fixture-exact`：对completion与label分别strip后精确相等得1，否则0。只用于手写fixture；不是正式任务评分器。
- `areal-gsm8k`：仅导入固定 `areal.reward.gsm8k.gsm8k_reward_fn`。freeze.verifier还必须提供 `source_sha256`（键gsm8k.py和__init__.py，对应实际导入的AReaL reward两文件）、`math_verify_version` 和 `parameters={try_extract_without_anchor:true,precision:6,timeout:5.0}`。运行时核对真实文件fullhash、已安装math-verify版本、当前worker参数及extraction配置；不匹配/缺依赖为unverifiable。日志不能提供任意Python模块路径。当前可在fixture图中验证真实评分函数调用，但仍不证明GPU完整训练输入转换。

本模块没有复制被测reward结果当authority。合法版本标签但伪造reward会由原始回答/label独立重算识别；改变payload内label还会与预冻结dataset行冲突。依赖版本校验不是依赖全部传递源码认证，正式freeze仍需镜像/源码provenance；此入口不替代运行环境冻结。

## 保留链与结果

从final_state沿parent回到冻结root，覆盖恢复点以前保留的消费义务。同logical更新或组只在最终保留链重复才判invalid_commit；被回滚的physical update单列，不因重算同logical ID误判。全state fullhash、各rank、实际load、optimizer映射缺失为unverifiable；确证不完整K、旧attempt、混verifier、错误reward、tensor与组不符、保留链重复为invalid_commit。模型成功标记无效。

目标提交仅取最终保留链中的目标逻辑组；其他组继续训练单独列training_continuation。fault_observed隐含目标角色down；role_ready/down按时间区间判断，目标提交时角色不在线则等下一次ready，不能使用较早已失效的ready。后来回滚的目标提交不终止RTO。只有最终安全性pass且900秒窗口内两条件同时成立才correct_recovered；其余RTO为空/右删失，900只作独立penalized_score。真实成本计量、本规范外合法替代prompt、多故障多目标聚合、ACK与R特定token协议专项审计未在本单元实现，不通过虚构字段补齐。

## 测试及尚未验证项

有界修复补充：主任务指出“保存first→load root→保存second却挂parent=first”的伪保留路径。现按实际load执行段拒绝，包括同epoch reload。新增4项手写测试覆盖非法旧祖先保留、合法同段连续保存、合法父恢复（同/跨epoch）、optimizer后保存前reload；连同原合法回滚重算共5项定向测试通过（0.033秒）。本次未改state/P1/pilot；此修复后的全oracle回归交主任务运行，下文21项结果为修复前版本。

手写图覆盖：无Rtoken合法baseline；合法回滚重算；恢复前父代重复消费；保留重复更新；回滚目标不能充RTO；角色二次下线；安全丢弃与其他组继续；缺optimizer/rank；旧attempt先到；K/版本错误；伪reward/success；冻结label冲突；真实tensor映射错；不支持变换；全hash篡改；事前N/A与未命中；任意import拒绝；独立模块依赖边界。

实现时只运行两个必要定向测试：无token合法baseline、合法回滚重算，均通过。主任务随后运行完整模块21项测试，0.102秒全部通过，日志及源码hash见 p2_evidence/oracle-tests-r1.log 与 oracle-verification-r1.json；真实areal-gsm8k入口也已在既有镜像的无网络/无GPU容器内验证：正确/错误回答分别为1/0，源码、依赖、参数不匹配均拒绝。首次容器USER/LOGNAME缺失导致导入失败，补齐后通过；两次日志与检查脚本均保留于p2_evidence/verifier-real-library-*。此检查不覆盖训练reward变换或GPU审计。未运行600调度、未运行GPU、未声称observer隔离或真实训练接缝已验收。

主任务修复后完整回归：25项，0.130秒，全部通过。凭据为p2_evidence/oracle-tests-r2.log及oracle-verification-r2.json；r1保留为修复前历史。
