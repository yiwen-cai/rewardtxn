# i06真实控制器崩溃：CPU隔离探针实现

2026-09-20，economy-dev enabled，Astra medium。仅新增`tests/ft/p2_controller_probe.py`、`tests/ft/test_p2_controller_probe.py`及本文。生产P1/P2、manifest、历史证据不改。184 pending不减少，训练与业务oracle始终not_evaluated。

## 固定场景和真实边界

复用`test_state.py`的K1 CPU合同fixture，目标进程实际调用state.acquire_owner/authorize/accept/prepare；落盘intent后保存`target-boundary.json`（身份、generation、文件全hash），再调用原P1 barrier。fixture是小文件合同，不是真实reward模型或训练optimizer。

内层控制器使用原`run.preflight/execute/run_attempt`。测试进程内局部包裹Journal.add，不改源码：fired调用原add完成flush/fsync后可精确阻塞；observed调用前可精确阻塞，此时原run_attempt已经完成真实pidfd发送和Popen.wait。注册与发送包装均委托原实现，不手写journal伪装注入。

| 场景 | 精确切点 | guardian观察 | 后续 |
|---|---|---|---|
| before-send-crash | fired已fsync，send尚未调用 | 已验证目标pidfd仍不可读/活着 | guardian以控制器pidfd发SIGKILL并wait=-9 |
| after-wait-crash | 目标已被原P1 kill、wait=-9，observed尚未写入 | 同一目标pidfd已可读；原send完成通知与原wait参数同时保留 | 同上 |
| before-send-control | 同第一切点 | 同样绑定真实prepare和活目标 | guardian放行，原P1继续signal/wait/observed |
| after-wait-control | 同第二切点 | 同样独立确认目标已退出 | guardian放行observed落盘 |

目标pidfd由原OwnedProcess.register后经仅guardian/controller持有的socketpair传递，guardian在目标仍活时核对fdinfo、/proc身份、父PID、namespace。target通过原P1单独的barrier socket，不继承测试见证端点。切点journal字节hash、大小与实际记录由guardian重新读取核对。后切点不是只相信控制器自报：guardian独立持有目标的内核pidfd并确认退出；目标退出码仍准确标作原控制器的wait结果，不声称guardian能wait不是自己Popen的目标。

每个crash场景用新进程读取原配置/原run目录调用**未包装的**execute(resume=True)。要求原journal只追加一条technical_invalid：`uncertain interrupted attempt; no replay`，无新attempt、worker登记、fired或observed。对照已经execution_complete，resume应拒绝且原journal不变。再次resume、冻结配置变更均拒绝；坏JSON尾仅对复制的run做负例，原证据不变。另有两个真实controller/target负例，在ready报错nonce或PID，必须在fired前拒绝。

## 生存、隔离和清理

只允许现有container_run CPU profile：namespace PID1为原namespace_run，空schedule只监护唯一guardian launcher。guardian创建内层控制器，保留其pidfd；**不在宿主直接试杀控制器**。所有执行入口核对/output/launch和frozen绑定、PID1命令、私有namespace与CPU参数，无pidfd能力拒绝，无降级。

被SIGKILL控制器的finally不会执行。guardian和外层supervisor保持存活，guardian等目标pidfd退出；barrier断连/超时负责快路径，失败则整个探针失败，私有namespace PID1最终退出收敛所有孤儿。现成supervisor用已登记完整容器ID/nonce保存final inspect、确认不运行并移除容器。测试要求supervisor无failure、cleanup_confirmed且无cleanup_errors；内层pass不能覆盖外层清理失败。guardian不对日志PID补杀目标；仅清理自有Popen。

## 资源与精确入口

镜像 `areal-project/areal-runtime:v2.0.0-sglang`，容器Python `/opt/.venv/bin/python`。CPU profile固定2CPU、2GiB、128PID、network none、readonly rootfs、无GPU、cap-drop ALL、非root。每主probe guardian绝对alarm=25秒，两个ready负例各15秒；外层run=160秒、handshake=5秒、lease=10秒。宿主startup默认15秒/cleanup默认3秒，外层已有绝对超时与全ID清理。没有矩阵。

```sh
.venv-tq/bin/python3.11 tests/ft/p2_controller_probe.py \
  --make-config /tmp/rtx-controller-probe-config-r1.json
.venv-tq/bin/python3.11 -m scripts.ft.container_run \
  --config /tmp/rtx-controller-probe-config-r1.json \
  --source /public/home/caiyiwen/rewardtxn \
  --run-dir docs/experiments/rewardtxn-ft-20260916/p2_evidence/controller-probe-r1 \
  --image areal-project/areal-runtime:v2.0.0-sglang \
  --python /opt/.venv/bin/python
.venv-tq/bin/python3.11 tests/ft/test_p2_controller_probe.py \
  --evidence docs/experiments/rewardtxn-ft-20260916/p2_evidence/controller-probe-r1 -v
```

配置生成和输出均拒绝覆盖。worker已生成上述/tmp配置，主任务可直接使用；重跑另取新r编号。测试共7项（2 guard、5隔离结果验收）；不提供evidence时5项明确skip，不能宣称完整通过。

## 证据和未验项

`probes/source-sha256.json`记录探针/P1/state/fixture来源。每场景保存containment、目标prepare proof、cut-witness、原controller日志、process-exits、journal-before/after-resume（hash和完整记录）、新进程resume/repeat/changed结果、坏尾复制证据和CPU合同result。顶层原supervisor保留容器ID、host/init身份、created/started/final inspect与cleanup结果。原journal仍位于每场景run/events.jsonl，不补observed。

worker只运行AST语法和两个无进程guard测试（0.002秒）以及生成配置；未启动容器、未验收真实SIGKILL。完整隔离验收交主任务。这里证明的是本机进程故障下fsync后不重复注入，**不证明断电一致性、训练恢复或i06旧清单主boundary**；i07/多fault oracle和正式评分合同不在本单元。

## 主任务隔离验收

controller-probe-r1在固定镜像sha256:c0573bb8412a8db753d44b02c9e04504d2392c54f019707f58acf26dd68d2469中完成（supervisor 6.995秒），无GPU/网络。保存证据的7项验收通过（0.006秒，无skip）：两个实际控制器SIGKILL窗口、新进程原journal拒绝重放、两个正常对照、错误nonce/PID与坏journal拒绝。after-wait及两个control共有3次计划target SIGKILL；before-send target未被计划注入，随后因控制连接断开退出。full CID独立inspect确认已移除，见p2_evidence/controller-probe-r1/verification.json。184 pending不变，不宣称训练/oracle恢复完成。
