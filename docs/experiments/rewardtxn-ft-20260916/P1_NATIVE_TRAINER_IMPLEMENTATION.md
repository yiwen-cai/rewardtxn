# Native trainer 单故障工程探针（实现交接，未跑 GPU）

2026-09-20。本单元仅实现；既有 nofault/resume 和 CPU reward pool 证据不替代本探针验收。未进行 formal freeze，不是 R/replay/F4，不证明 pending 恢复。旧 pilot、第三方源码均未修改。

## 固定行为与文件

- `scripts/ft/areal_native_trainer_probe.py`：原 pilot hooks 先安装，再包真实 `PPOTrainer._save_recover_checkpoint(self,epoch,epoch_step,global_step)` / `MegatronPPOActor.optimizer_step(self)`。原调用各一次，返回值/异常保留。只支持单 actor rank0/world1、3steps、retries1、同步保存、freq1。step0 保存返回且 RecoverHandler.last_step_info、全部六个 RecoverInfo 文件、DCP `.metadata` 和非空 `.distcp` 实际存在才记录完整 SHA256 manifest。第二个 successful optimizer 返回后、尚未下一次 save 时发送固定 ready；无 step0 完整前提直接失败，不顺延。新 trainer 获 already_fired，仅观察真实 update/save/load，不能再 ready。
- `scripts/ft/native_gpu.py`：封闭四卡检查、GPU 空闲原始快照、明确 env 合同、源码 hash；方法 root 启动有界预检子进程（120s，失败则不启动方法），wait 完成后 `execv` 官方 `areal.infra.launcher.local`，PID/所有权不变。没有自定义重启。预检检查四个实际 CUDA UUID、每卡微小 tensor/Jiterator 编译执行、缓存写入、internal IP/bind、cap/no-new-privileges、pidfd、只读 workspace/model/data；不是性能测量或完整预热。
- `scripts/ft/container_run.py` / `namespace_run.py`：仅增加显式 `cpu`（默认）和 `native-trainer-4gpu`。CPU 仍拒绝 DeviceRequests，原配置兼容。GPU 仅允许固定 bootstrap argv/一次事件/900,10,20 秒时限、显式受限环境；只能指定恰四个 UUID，无 arbitrary Docker args。32CPU/128GiB/shm16GiB/pids4096、非root、capdropALL/nnp、readonly repo/root、本run output/tmp。独立 internal network，完整 network ID/nonce/inspect 后清理；create 失败可按唯一 nonce 标签取得完整 ID 再校验归属，不按名字删资源。
- `scripts/ft/verify_native_trainer_probe.py`：离线工程验收，输出 JSON/退出码，不改原方法结果、不称 oracle-valid。
- `native-trainer.yaml`：旧 pilot 配置独立副本，仅 trial_name/retries 改变，seed211/K8/U4/3steps/4卡 allocation 不变。host 拷入 output，官方使用 `/output/native-trainer.yaml`。
- `native-trainer-control.json`：argv `/opt/.venv/bin/python -m scripts.ft.native_gpu`；唯一事件 `trainer-post-update-0`，target/waiter 都为 trainer；evidence 为 successful_ordinal=2、saved_global_step=0、phase=post_optimizer_pre_save。

源码 SHA 清单在启动时保存，bootstrap 再核对；image ID 从保存的 Docker inspect 读取。controller journal 保存实际 PID/starttime/boot/cgroup/祖先链、sent/同pidfd退出；trainer witness 额外记录 torchrun 祖先 argv/身份和 incarnation。动态 manifest 只写 observer 文件，不传给原生恢复方法。旧 pilot checkpoint_load_returned 的实际完整 hash 与被杀前 manifest 离线比较；最终 step2 再独立 rehash。

## 验证命令（主任务执行）

CPU 新合同，不调用 nvidia-smi、不启动 CUDA：

```sh
FT_NATIVE_REAL_IMPORT=1 /opt/.venv/bin/python -m unittest discover -s tests/ft -p 'test_native_trainer_probe.py' -v
/opt/.venv/bin/python -m unittest discover -s tests/ft -p 'test_native_gpu_profile.py' -v
```

在既有 AReaL 镜像的 CPU 容器执行，显式 `PYTHONPATH=/workspace:/workspace/third_party/areal`、`USER/LOGNAME`、`HOME=/tmp`、`PATH=/opt/.venv/bin:...`，不传 `--gpus`；前者开启真实类导入/签名核对。host 仅定向执行该文件：6 pass、真实导入 1 skip（0.095s）；这是 CPU 状态合同，不是 CUDA/native retry 证据。新增 GPU profile 合同采用 fake Docker CLI 验证完整ID清理和固定 argv，不能替代真实容器验收。

回归：`test_descendants.py`、`test_namespace_container.py`、`test_reward_pool_probe.py`（至少一次性分配协议与旧两事件合同）。原 Docker 测试沿用 `FT_NAMESPACE_EVIDENCE_DIR` 保存失败证据；不要删除已有失败。

CPU 验收通过、现场资源确认后，才由主任务执行下述 GPU 工程命令；本轮没有执行：

```sh
.venv-tq/bin/python -m scripts.ft.container_run \
  --config docs/experiments/rewardtxn-ft-20260916/native-trainer-control.json \
  --source /public/home/caiyiwen/rewardtxn \
  --run-dir /ABS/NEW/RUN \
  --image areal-project/areal-runtime:v2.0.0-sglang \
  --python /opt/.venv/bin/python --profile native-trainer-4gpu \
  --gpu-uuid GPU-UUID1 --gpu-uuid GPU-UUID2 --gpu-uuid GPU-UUID3 --gpu-uuid GPU-UUID4
.venv-tq/bin/python -m scripts.ft.verify_native_trainer_probe /ABS/NEW/RUN
```

UUID 必须现场选择四张唯一同型号卡，启动前保存完整 nvidia-smi GPU/compute-apps 原始输出，无 compute PID、memory.used≤100MiB、utilization=0%，create 后 start 前再查。不自动选卡/抢占。Docker 多 UUID 参数保留既有成功格式：一个带字面双引号的 `"device=GPU-A,..."` argv。预检子进程退出后 CUDA 资源随进程释放，正式角色才开始初始化；900s 包含预检/两次原生 CUDA 冷启动/10s retry 等，controller/host heartbeat独立运行。仍存在检查后资源被外部任务占用的竞态；无通用 GPU reservation 服务，现场冲突即停。

## 验收与停止边界

必须同时得到：一次确切 SIGKILL＋同pidfd退出、两个可信 trainer/torchrun incarnation、官方 run_id0→1/RecoverInfo step0 实际 load fullhash一致、新进程两次真实 successful update、最终step2完整save及事后fullhash、CID/network全部有归属清理证据。原生 launcher 正常 COMPLETED 最终仍 JobException/exit1，必须原样保存；controller exit0只表示控制过程完成，单独不能判方法成功。验收器检查实际 `method_observation.launcher_exit_code=1`，不改成0。

缺任何预检/完整保存/正确 ordinal/恢复 load/实际 update/清理证据均不通过；不自动换卡、加权限、扩大memory/deadline、补重试或补方法恢复。预检尚未实际执行，受限 CUDA/JIT/权限兼容及900s预算尚待实测。多actor、多epoch、重复故障、异步save、R/replay、自然F4和pending恢复仍未覆盖。全量文件hash与同步observer flush计入工程扰动，不比较性能收益。没有本探针GPU结果前不能宣布P1全完成。


## GPU 前 CPU 增量与证据关联（2026-09-20）

主任务已验证初版新 native 合同 7 项/25.413s（包含真实 AReaL 类导入），GPU profile 命令合同 2 项/0.008s、descendants 8 项/0.189s、namespace 生命周期 4 项/16.622s、一次性分配协议 4 项/0.023s；记录 `p1_evidence/native-cpu-verification-r1.json`，namespace 7 个完整 CID 独立 inspect 均 absent。这些是 CPU/生命周期验收。

新增 `tests/ft/torchrun_cpu_fixture.py` + `test_torchrun_cpu.py`：真实官方 `build_target_cmd`/`BASE_ENVIRONS` 与 shell/stdbuf/tee 传播到真实 torchrun，max_restarts=1；controller只启动一次fixture root，精确杀首个CPUworker；torchrun自行生成replacement，后者获 already_fired，并验证再次 ready 在客户端发送前被拒。核对内核身份/祖先链、两轮完整白名单环境与BASE_ENV、RANK/WORLD_SIZE、唯一signal/exit/ready，无 GPU DeviceRequests。这里只验证 torchrun 协议；不将其 restart_count 当作官方 SPMD run_id 或原生恢复证明。

首次 r1 用 `--standalone` 自动 c10d，networknone 下广告的容器 hostname 不能解析；120s deadline 收尾，测试121.072s，cleanup_confirmed，失败证据 `p1_evidence/torchrun-cpu-r1` 保留。r2 仅fixture改显式 static rendezvous、127.0.0.1:29517、node_rank0，仍真实torchrun，生产CPU网络不变。主任务实测 1 项/13.214s 通过，证据 `p1_evidence/torchrun-cpu-r2`。

```sh
FT_TORCHRUN_DOCKER_IMAGE=areal-project/areal-runtime:v2.0.0-sglang \
FT_TORCHRUN_EVIDENCE_DIR="$PWD/docs/experiments/rewardtxn-ft-20260916/p1_evidence/torchrun-cpu-NEW" \
.venv-tq/bin/python -m unittest discover -s tests/ft -p 'test_torchrun_cpu.py' -v
```

验收代码检查发现并修补：旧检查未将信号/pending注册/witness严格关联，且旧pilot `recover_handler_load_returned` 没有返回RecoverInfo内容。新增项目外层 `RecoverHandler.load` 只调用原方法一次，原异常/None/对象原样保留；返回非None后记录实际 `last_step_info`，与当时step_info.json核对，记录全部RecoverInfo metadata文件完整hash。每条新witness附完整内核身份和控制incarnation，registered另带旧pilot writer incarnation；旧pilot/第三方源码不变。

验收现在把controller registration/assignment/ready/signal/exit和所有witness按完整identity+控制incarnation绑定；DCP实际load按PID/starttime/boot/cgroup+pilot writer incarnation绑定。必须有replacement实际返回step0 RecoverInfo、metadata fullhash等于前代完整save，且DCP load→RecoverInfo返回→两次真实成功update→最终step2 save顺序成立。动态manifest仍不提供给恢复方法。

新增 `test_native_evidence.py` 4 项纯CPU合成证据合同：正例仅用于检查验收逻辑；负例覆盖错误signal/ready/witness/pilot身份、缺RecoverInfo、错误step或metadata以及加载晚于update；wrapper验证原调用一次、原对象/None/异常保留。worker定向 4 项/0.010s 通过；主任务请复跑并记录原始输出（GPU仍未执行）：

```sh
.venv-tq/bin/python -m unittest discover -s tests/ft -p 'test_native_evidence.py' -v
# 既有CPU镜像内，保持真实依赖环境
FT_NATIVE_REAL_IMPORT=1 /opt/.venv/bin/python -m unittest discover -s tests/ft -p 'test_native_trainer_probe.py' -v
```
