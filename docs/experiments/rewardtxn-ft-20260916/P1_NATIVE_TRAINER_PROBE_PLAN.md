# P1 原生 trainer 故障最小工程探针计划

2026-09-20。**仅规划，未实现、未启动GPU。** economy-dev根会话`01a0bd89-a9c3-7d00-b16d-4a0a766de676`，Astra medium planner。本计划沿用已验收PID1容器/后代pidfd/一次性注入分配；不做R、replay、F4、正式freeze或FT样本，不声明dispatcher pending或受影响组恢复。

## 决策：一次故障、一次官方重试、同run完成

使用已运行的Qwen2.5-0.5B、`sglang:d3p1t1+megatron:d1p1t1`、K8/U4、同步DCP、同一真实RLVR入口。**恰4张GPU，其中单actor rank；只杀一次trainer进程。** 新run从空输出开始，不复制既有checkpoint；首次同步recover checkpoint及RecoverInfo完成后，在第二次成功optimizer返回、该次更新尚未保存时SIGKILL该actor trainer。官方SPMD launcher由P1只Popen一次，`recover.mode=auto`、**`recover.retries=1`**；其原生路径负责清理、等待10秒及重启。

`total_train_steps=3`即可：初次执行global_step0保存，global_step1的optimizer成功后被杀；原生加载step0并继续到最终step2。实际是否重算同组不作为本探针成功条件，必须分别报告。保持首步warmup有效LR为0的事实，不把optimizer成功调用自动写成非零LR权重变化。

选择retries=1是本次工程probe的预设最小预算，不修改FT-v1最多3次规则或既有pilot.yaml。官方COMPLETED也触发恢复；本次故障已消耗唯一一次重试，因此正常完成后的run_id1不再启动第三轮，**预计官方launcher仍抛JobException、原始exit1保留**。不捕获COMPLETED改成0，不替它增加重启，不提前终止来掩盖原生行为。

## 1. 已核对的实际接缝

| 现场源码/历史证据 | 实施含义 |
|---|---|
| `areal/infra/launcher/local.py:local_main`，约267、300、410、424–451行 | 同一launcher内递归run_id+1；10秒RECOVER_TIME_INTERVAL；trainer经shell/tee→torchrun；run_id原生日志可观察。 |
| `areal/utils/recover.py:check_if_recover`，约511行 | `_run_id`明确未使用，auto-recover实际由RecoverInfo及checkpoint存在/可读决定；不能靠给run_id造值触发恢复。 |
| `LocalLauncher.submit_array`约147–153行；`infra/utils/proc.py:build_target_cmd`约26–61行 | Popen未传env，继承launcher环境；构造的KEY=value只覆盖显式字段，**没有env -i**。FT_CONTROL_SOCKET/FT_RUN_NONCE/FT_HANDSHAKE_TIMEOUT可通过实际子进程链继承。 |
| `infra/utils/launcher.py:BASE_ENVIRONS`约18–50行 | 官方设置cache路径、TOKENIZERS_PARALLELISM、CUDA_DEVICE_MAX_CONNECTIONS；默认AREAL_CACHE_DIR依赖`/tmp/areal-{getpass.getuser()}`。不改这些库逻辑。 |
| `scripts/ft/areal_pilot.py:main`；`areal_pilot_hooks.py:install_hooks` | pilot先设置观察目录、安装幂等hooks，再构造PPOTrainer；已有optimizer/save/load/metadata观察可复用，但异步JSONL不能单独授权精确注入。 |
| `megatron_engine.py:optimizer_step`约1091–1106行 | 原返回dict包含`update_successful`、grad_norm、lr；仅成功标志为1时计数，异常/未成功不算第二次。 |
| `rl_trainer.py:_save_recover_checkpoint`约1374–1400行 | 先RecoverHandler.dump，再SPMD barrier与device synchronize；外层完整返回比目录出现、单个metadata事件更强。 |
| `recover.py:RecoverHandler.dump/_save_checkpoint`约275–318、416–441行 | 原生先engine.save(with_optim)，再写RecoverInfo；metadata多文件close，无显式fsync。此探针是进程SIGKILL，不是掉电持久性测试，不额外fsync方法文件补强基线。 |
| `PILOT_REPORT.md`及nofault-r4/resume-r1 execution | 既往3步与新进程下一步通过限定核验；两次总墙钟约318.5/296.7秒，均launcher exit1。此为900秒预算依据，不是新故障结果或RTO。 |

以上路径的`areal/...`相对`third_party/areal`。实现前再次记录实际checkout/hash，不能将旧探针镜像/代码PASS直接复制给新profile。

## 2. 最小文件与hook顺序

建议新增一个项目文件`areal_native_trainer_probe.py`，包含短entrypoint和本次两处外层控制hook；复用原pilot.main、install_hooks、file_manifest与Client，不复制训练循环。新增一份独立`native_trainer_probe.yaml`（由pilot.yaml复制，仅改experiment/trial、retries=1及新输出位置；其余关键配置保持）。对应CPU合同放`tests/ft/test_native_trainer_probe.py`；GPU验收脚本只做证据核对，不控制恢复。

安装顺序必须固定并在CPU验证：

1. torchrun启动新项目entrypoint；保留原sys.argv供`pilot.main`的`load_expr_config`使用，顶层launcher实际argv只换trainer脚本路径和新YAML路径。
2. 调用原`areal_pilot_hooks.install_hooks()`，取得已包裹的optimizer与观察行为；再安装本次外层hook。随后调用原`areal_pilot.main(args)`，其中install_hooks因原幂等标志不会覆写新外层hook。不在新hook里再次执行原optimizer/save。
3. trainer入口创建`Client('trainer', event_id='trainer-post-update-0')`，每进程只登记一次；检查torchrun实际RANK=0、WORLD_SIZE=1和冻结topology，其他角色不得请求此event。首incarnation获得pending，新trainer获得already_fired后不再触发屏障。
4. 外层包裹`PPOTrainer._save_recover_checkpoint`：调用完整原方法后，确认实际RecoverHandler.last_step_info=global_step0、config async_save=False/no_save_optim=False/no_load_optim=False；关联已有真实engine save返回与RecoverInfo dump证据。使用官方`Saver.get_recover_checkpoint_path(..., name='default')`及`RecoverHandler.recover_info_path`取得路径，全内容hash checkpoint与metadata。缺文件/必要状态证据或切点不一致时拒绝建立保存前提，不能仅看目录或返回None。
5. 外层包裹当前已观察的`MegatronPPOActor.optimizer_step`：先调用原包装器一次，保留原返回对象；若update_successful=1才增加本进程成功ordinal。初始incarnation ordinal=2且已有上述step0完整保存前提时，记录当前`_pilot_update`、实际lr/统计和前代manifest标识。此时还在optimizer返回接缝，调用栈未进入该次后续scheduler/weight update/save；只在此发送ready并等候注入。若第二次成功时保存前提不成立，明确技术无效/未命中，不等待第3次来偷换预定切点。
6. ready之前排空本进程pilot观察队列，并在独立小型控制见证JSONL中同步flush/fsync所需切点证据。原pilot EventWriter.flush只是排空/写文件，不能把它描述成方法checkpoint的fsync。全部fullhash、长日志排空在ready之前完成，避免controller握手窗内阻塞。

控制配置中的evidence固定为边界名称、成功ordinal2、前代global_step0等谓词；动态update_id/文件hash写单独观察见证，并在离线验收交叉核对，不把observer文件或hash返回给恢复方法。若缺证据，本run不算有效命中。对保存返回的“完整”认定限已验证同步、单actor rank后端，不扩展到异步/多rank。

本次只一个trainer waiter/target，因此被杀者不需要release；其他后台rollout/reward不因本hook暂停。不能为简化清理让controller杀SGLang；后续其他角色退出/重建是官方launcher.stop_all的原行为，记录真实故障传播范围。

## 3. 原生run_id、torchrun incarnation与加载证据

controller持有一次Popen的launcher root，trainer登记必须经过真实祖先链到该root，不要求共享PGID。保存各层PID/start-time/namespace/cgroup和必要cmdline，辨认本run已拥有的torchrun节点；命令名只作角色关联，不授予信号权。新trainer应具有新的trainer及torchrun incarnation，而root launcher不变。

保留官方`LocalLauncher ... run_id=0/1, is_recover_run=...`原始日志。通过已认证torchrun/trainer的创建与注册顺序、日志时间/文件offset进行离线对应；单actor、两轮应形成无歧义映射。没有唯一对应就记epoch证据不足，不修改launcher注入run_id字段；`TORCHELASTIC_RESTART_COUNT`不能代称SPMD run_id。当前one-shot注入只依赖本run持久controller状态，不需要在线猜native epoch来恢复。

加载证明至少包括：

- 初次完整step0 checkpoint和RecoverInfo全文件hash；故障时该次成功update的ID不在已完成保存中。
- signal_sent与同pidfd可读退出，各自controller时刻；只一次SIGKILL，非Popen子进程不被controller waitpid收尸。
- 官方原生检测结束、清理、run_id1启动与实际`RecoverHandler.load`调用返回；`MegatronPPOActor.load`的实际path/with_optim及全内容文件manifest，和故障前step0保存manifest精确对照。RecoverInfo（含cursor）读取路径/内容同时关联。
- 新trainer实际产生成功optimizer、完整后续保存；最终step2 checkpoint/metadata由离线验收重新全内容hash。旧固定路径会覆写，早代只使用当时完整读出的保存/加载清单，不伪称事后还能重算已经覆盖的旧文件。
- 记录恢复后policy version、lr、实际训练组/样本、sampler游标；**这些不证明pending未丢、不证明同故障组恢复或随机轨迹连续**。已知cursor ahead of training的限制继续披露，不给基线补缓存/ACK/重投。

记录的“加载后首次真实更新时间”只是工程里程碑，不冒称受影响工作oracle-valid RTO。文件hash相等证明加载来源文件一致；未逐项比较加载后内存model/optimizer/RNG或下一批连续性时必须明说。

## 4. 明确受限的4GPU profile，CPU入口不放宽为任意Docker参数

建议在现有host supervisor加一个固定枚举profile `native-trainer-4gpu`，默认仍`cpu`；显式提供四个`--gpu-uuid`才允许该profile。只把已有有限生命周期代码复用于两个封闭配置，不做插件/任意extra-args。旧CPU profile继续拒绝所有GPU请求、network变更和额外权限，原测试必须保持。

GPU profile拟固定：

| 项目 | 预设与核验 |
|---|---|
| GPU | 启动前现场nvidia-smi查询完整UUID/型号/总显存/used/utilization及compute-apps，选恰4个不同、同型且空闲UUID，临启动再查。空闲要求无compute进程且指标符合本机已记录idle基线；可疑卡不碰，不以旧pilot快照作当前证据。inspect DeviceRequests精确等于这4 UUID，不允许`all`。Docker CLI多UUID的`--gpus`值复用已实际成功的argv格式，见下文。 |
| 布局 | 保留actor1＋SGLang3、单节点4卡；容器内CUDA_VISIBLE_DEVICES=0,1,2,3对应本次UUID排序。记录官方分配的实际role→local device→UUID，不臆定actor一定第0卡。 |
| 网络 | supervisor建立本run独占Docker `--internal`网络，保存完整network ID/nonce label/inspect，零端口映射；容器仅接此网，获得gethostip需要的非回环IP。结束后按持有ID清理，不接外网或外部服务。 |
| 内存/CPU/shm | 参照已运行pilot：32核、128GiB、shm16GiB。pids上限拟显式4096，属于工程配置不是已量测需求；不沿用CPU的128。启动前核查主机预算与其他任务，记录实际资源上限和可能的触顶。 |
| 文件系统/tmp | repo/model/data只读，输出独占；只读根fs保留。用本run新建私有`output/tmp`绑定到`/tmp`承接CUDA/Triton缓存，HOME=/tmp；不能沿用CPU的64MiB tmpfs。记录tmp/I/O占用，保证输出磁盘足以容纳约6.92GB recover文件及临时保存/日志峰值，预算不足先停止，不清理历史结果。 |
| 身份/权限 | 非root当前UID:GID、cap-drop ALL、no-new-privileges、独立PID namespace、controller直接PID1、无Docker socket/host PID/privileged。先实测CUDA设备可访问、pidfd和/proc祖先身份可读；不因故障自动去掉隔离或增加CAP_SYS_ADMIN。 |
| Python | controller和顶层launcher固定`/opt/.venv/bin/python`；PATH以`/opt/.venv/bin`开头，现场核验python3及torchrun也解析到该环境。之前误选/usr/bin/python的失败不能重现。 |
| 环境 | 显式保留USER/LOGNAME、HOME=/tmp、PYTHONPATH=/workspace:/workspace/third_party/areal、CUDA_VISIBLE_DEVICES、HF_HUB_OFFLINE=1、WANDB_MODE=disabled、OMP_NUM_THREADS=4、PYTHONDONTWRITEBYTECODE=1，以及下文已核实的LD_LIBRARY_PATH/CUDA_HOME和必要CUDA PATH；只在GPU profile对应配置允许这些所需键，不倾倒宿主环境。不要覆盖官方BASE_ENVIRONS及NCCL_CUMEM_ENABLE/NCCL_NVLS_ENABLE等实际设置。 |

**设备参数按已成功实参复用。** `pilot_evidence/nofault-r4/launch.json`中，`--gpus`后一个argv元素包含字面双引号；替换为本轮现场UUID后保留这一Docker CLI格式，例如JSON表示：

```json
["--gpus", "\"device=GPU-A,GPU-B,GPU-C,GPU-D\""]
```

这仍是一个argv元素，不经过新shell展开；不要凭记忆改成另一种逗号解析写法。最终以Docker inspect DeviceRequests和容器实际可见UUID集合核验解析结果。

**child env不能继续只沿用CPU白名单。** 当前namespace controller用`env=dict(config["env"], FT_...)`启动方法，会清空镜像其余环境；CPU评分池通过不证明CUDA环境完整。已只读核对同一镜像`sha256:c0573bb8412a8db753d44b02c9e04504d2392c54f019707f58acf26dd68d2469`在`p1_evidence/reward-pool-r2/reward-pool-khpy5fw3/run/inspect_created.json`中的指定字段：

- `LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64`。
- `CUDA_HOME=/usr/local/cuda`；PATH包含`/opt/.venv/bin`、`/usr/local/cuda/bin`、`/usr/local/cuda/nvvm/bin`、`/usr/local/nvidia/bin`及系统bin目录。GPU profile建议显式保留这些必要项，删除镜像中无关的root个人工具目录；该精简后的最终PATH仍须预检。
- 镜像NVIDIA_DRIVER_CAPABILITIES为`compute,utility`；镜像NVIDIA_VISIBLE_DEVICES为`all`，**不能将此默认值当成本run设备选择或复制为选择列表**，run权限以显式4 UUID设备请求及实际可见设备校验为准。
- HOME/HF_HUB_OFFLINE/WANDB_MODE取此前真实pilot显式值`/tmp`/`1`/`disabled`；可显式设AREAL_CACHE_DIR为本run `/tmp`下唯一目录，保持官方派生cache变量逻辑。若新增这个cache根配置，记录差异，不读取/继承宿主全部环境。

这些值来自已保存镜像inspect，不是本轮GPU可用性证明。CPU阶段核对最终构造env经build_target_cmd到trainer/server的传播；GPU preflight必须在最终受限profile、同一最终child env及官方BASE_ENVIRONS覆盖后检查torch CUDA context、极小张量/计算和最小JIT缓存写入，记录实际LD路径、CUDA_HOME和缓存位置。预检失败不改全量环境继承、不安装依赖或扩大权限；先保留日志再定位缺失项。

官方launcher会启动`python3`和`torchrun`且拼接其参数；因此新项目脚本及YAML使用固定简单容器路径（无空格/元字符）。外层仍argv数组，不修改官方shell/tee行为。源码/模型/数据只读不等于已正式冻结：记录本轮实际镜像ID、源码/配置/数据及既有模型内容清单，前后检查本轮所用源码hash未改变。

GPU profile必须另做inspect字段校验（设备集合、internal network ID、固定资源、挂载和权限），不能删除当前`check_containment`对CPU DeviceRequests的拒绝来“顺便兼容”。新增GPU profile文件或局部分支是实现选择，验收要求保持上述封闭边界。

## 5. 期限、CPU先验与一次短GPU验收

控制器总run上限建议**900秒**，握手10秒、lease20秒；host controller启动心跳仍短期限，CUDA初始化期间controller事件循环继续心跳，不把首次trainer登记拖到几分钟误判为controller挂死。握手从实际连接/ready开始，不能拿它约束两次CUDA冷启动。host绝对窗覆盖900秒加已有有界启动/清理开销；所有额外开销计工程GPU分配成本。900秒由既往318.5+296.7秒及原生10秒等待、hash开销留余量估计，不保证一定足够；超时保留结果，不当场延长。

CPU验收在支持真实库依赖的镜像内完成，**通过后才允许上述GPU profile真正启动**：

1. hook调用计数/顺序：原pilot安装幂等，新外层不会被pilot.main再次安装覆盖；原optimizer/save各调用一次，返回值与异常不变。成功/失败optimizer标志、缺metadata、缺checkpoint、错误global_step、异常save均不授权ready；第二次成功没有正确前代则明确失败，不移到后续ordinal。
2. 用小型CPU合同对象验证以上状态机和临时真实文件全hash，明确这不是Megatron保存PASS；另用已安装真实类检查包装签名/安装顺序，不假装CPU对象是GPU engine。
3. 真实torchrun CPU小程序检验环境继承（通过已查官方build_target_cmd构造的shell链或等价受控链，注明范围），确保FT三个控制变量、USER/LOGNAME、PYTHONPATH、python3/torchrun解析正确；不以此声称已运行GPU LocalLauncher。
4. 真实进程树fixture：保存前提→成功optimizer2→仅一次信号→launcher替身新建trainer→already_fired不重杀，root只启动一次。错误rank/world-size、陌生PID、旧incarnation拒绝。保留原8项后代合同和旧两事件定向回归。
5. host GPU profile的纯配置/inspect负例先验证：不是恰4 UUID、重复卡、host PID、外部network、任意Docker args、错误镜像Python或不足资源都拒绝。CPU合同可验证profile生成/拒绝逻辑，但不能称CUDA权限通过。

随后才进行一次受限GPU工程run，起始先在同profile内做短实际权限/设备preflight（计入成本）：CUDA见恰4个指定UUID、微小张量分配可用、internal网IP/bind可用、只读模型/训练数据及必要output/name_resolve目录可用、pidfd/SCM_CREDENTIALS/祖先/proc权限保持；失败立即按CID结束，不创建训练任务。正式probe的P1只Popen一次官方SPMD launcher，不能由preflight wrapper重试方法。资源检查不等于模型预热，不另外运行完整训练。

成功的限定验收是：**有效F2式切点一次命中＋同launcher原生run_id1恢复＋实际加载相同完整前代文件＋新trainer真实成功更新及最终保存＋完整收尾**。最终launcher exit1作为独立原始事实保留；controller自身exit0只表示控制完成，不能替代方法退出码。命中后原生不能恢复/加载失败/方法超时是方法观察结果；身份/观察/屏障/profile缺陷是technical_invalid；缺所需证据则unverifiable。以上均不是R正确性或affected-work恢复结论。

## 6. 停止条件、Ask first与交付

- 不足恰4张空闲卡、其他任务抢占、CPU/内存/磁盘不足、权限不兼容、源码/镜像/配置不符：不启动或技术中止，不kill外部任务。
- 单actor/world_size不为1、async_save开启、optimizer未成功、第一代保存/metadata不完整、预定切点错过、pidfd或祖先身份不符：不注入，不换切点。
- controller丢失、uncertain signal、release/观察超时：按本CID与network ID有界收尾，保留原记录，不自动重放。
- 已有效命中后官方恢复失败：记录失败，不能修改launcher或补自写恢复直到“看起来成功”。第二次故障、R/F4、多actor rank、扩大重试数/总窗均不在此最小run内。

已读取`third_party/areal/AGENTS.md`：其Ask first明确包括“Modifying config structures in areal/api/cli_args.py”“Adding new dependencies”“Changing launcher or scheduler logic”。本方案只用既有配置字段，观察/暂停hook在项目文件，官方launcher/scheduler及其COMPLETED行为不改，因此本轮没有该类待批准补丁。若后续必须改官方run_id传播、重试判断、退出状态或新增依赖，须先准备具体补丁并处理该Ask first约束，不能以本计划视作授权。

计划实现交付应包括：最小项目entrypoint/hook、新YAML与control schedule、封闭GPU profile及CPU负例、GPU事件/配置/源码hash/加载清单验收、全部CID/network/UUID及清理证据。4×本次容器墙钟记录GPU·秒（含冷启动、原生等待、hash和收尾），另记preflight成本；不加入正式统计、不用先前CPU两run的墙钟推断收益。

本轮只新增本文，未修改任何脚本、第三方源码或配置，未运行CPU/GPU probe。下一步先实现并验收上述CPU合同，再决定是否放行一次4GPU工程run。
