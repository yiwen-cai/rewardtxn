# P3a-1 draw WAL / loader overlay / blob 实施交接

2026-09-20。仅新增 `scripts/ft/replay.py`、CPU fixture/tests 和本文；未修改 state.py、P1/GPU入口、原pilot、第三方源码或训练workflow。没有 consumed authority、generation commit、GPU fencing、评分重放或训练能力；不把draw重投叫作GPU selective replay。前序native恢复门禁仍须独立通过。

## 实现范围

`BlobStore.put(bytes)/get(ref)` 以SHA256和size寻址，文件fsync后不可覆盖发布并同步目录；重复相同bytes幂等，冲突、损坏、非法路径/symlink拒绝。读取完整复算hash。该类仅字节存储，不解释tensor或外部pickle。

`DrawLoader(base, root, run_nonce, source_sha256, loader_config_sha256, *, k)` 限定普通list-of-dict batch，非空 `messages` list，项目保留键 `_r_draw` 不得预先出现；单rank、num_workers=0、epoch0、固定batch_size/drop_last/shuffle/seed、单迭代线程。实际base必须提供StatefulDataLoader使用的state_dict/load_state_dict/iterator、num_workers/batch_size/drop_last/len、sampler.num_replicas/rank/shuffle/seed/set_epoch。RDataset/远端prefetch、多worker、多epoch、动态batch和GPU未支持。

首次使用先固定epoch0、创建底层iterator并持久initial_state/genesis。每次fresh next在同一RLock内取得完整batch、序列化after_state与batch、持久两个blob、发布顺序draw记录；draw目录fsync后才交付。任何next后持久化失败使当前代理poisoned，不允许靠已推进的RAM cursor继续。snapshot线程共用锁，只见完整prefix。

`state_dict()` 返回schema/run/source/config合同、wal_sequence、record hash、after_state引用和occurrence count的独立descriptor。`load_state_dict()` 只可在新代理首次迭代/交付前调用；它验证snapshot N准确锚定WAL前缀，但始终恢复到完整WAL M的after_state，并优先重投全部1..M记录。构造器打开已有root也执行同一恢复流程，因此无外部snapshot时仍不丢durable尾部。每个batch全部slot预先登记，交付给调用者不等于提交。

WAL由不可变 `draws/000000000001.json` 等记录构成，不使用可截断append-JSONL。sequence连续，前hash链接genesis/前record；occurrence按已有durable item数量分配；group ID由run/source/epoch/occurrence确定，相同prompt不同occurrence仍为不同组。重投保留全部ID，返回 `_r_draw`；此字段还未接真实workflow。

最终命名record的截断、序号缺口或引用blob错，即使仅损坏最后一条，也拒绝恢复。未发布 `.tmp-*` 和未引用blob不进入prefix；首版不做GC。blob/record发布失败后禁止在同一iterator猜测继续。root持有排他flock；fork子进程关闭继承owner，不能通过代理继续写。不是GPU/远程写者证明。

## pickle与接口限制

只反序列化本方法受信root中，已校验genesis/完整record链和blob hash的底层loader state_dict。**hash不使未知pickle安全**；禁止把外部上传checkpoint、observer目录或不可信下载内容作为该root。source/config hash由调用者提供，必须对应实际数据和冻结loader配置/torchdata版本；本模块不会重新扫描dataset证明调用者声明。真实torchdata迁移版本不在支持范围。

sampler代理将cycle_dataloader的set_epoch(0)视为幂等：底层epoch已在restore前设置，不能在load后重置进度；set_epoch(1)明确拒绝。`__iter__`为标准单一iterator对象，允许同线程重复iter()返回自身，拒绝另一线程取得next权限；不是两个独立并行iterator。

不支持生产consumed参数：所有durable draw都pending。重启后允许再次交付上次已交付的batch，不能推导训练exactly-once。len保留原epoch长度，不包含重投扩展的物理交付量，因此这仍是独立CPU接缝合同，不能未经P3下一单元的消费/训练步语义适配直接接trainer。

## 验证与命令

新增：

- `tests/ft/replay_fixture.py`：普通Python loader、确定性fault gate和真实SIGKILL子进程；不包含训练。
- `tests/ft/test_replay.py`：9项CPU机制测试，其中process-crash一项覆盖10个切点，另覆盖线程锁、owner/fork、snapshot尾部、损坏/冲突等。
- `tests/ft/test_replay_real_loader.py`：显式opt-in真实torchdata+官方AReaL create_dataloader+cycle_dataloader；shuffle=True、batch4、num_workers0、32项。snapshot停在N=1，WAL到M=3，新代理先重放12组再得到正确后续20组；检查真实class/config/API与schema行为。无reward、GPU或完整WorkflowExecutor实例。

worker必要定向检查：3项/0.699s通过（durable tail、真实10窗口SIGKILL、snapshot互斥）。py_compile通过。主任务随后完成全量限定验收：机制测试9项/0.773s通过；既有AReaL CPU镜像内真实 create_dataloader / torchdata / cycle_dataloader 测试1项/9.939s通过。下述命令供复核。

```sh
.venv-tq/bin/python -m unittest discover -s tests/ft -p 'test_replay.py' -v
```

真实loader使用既有AReaL镜像（无需新增依赖），固定 `/opt/.venv/bin/python`，显式 `PYTHONPATH=/workspace:/workspace/third_party/areal`、USER/LOGNAME/HOME；只CPU，不传GPU设备。容器内：

```sh
FT_REPLAY_REAL_LOADER=1 FT_REPLAY_EVIDENCE_DIR=/output/p3a-loader \
/opt/.venv/bin/python -m unittest discover -s tests/ft -p 'test_replay_real_loader.py' -v
```

FT_REPLAY_EVIDENCE_DIR指定时每次建立独立 `loader-*` 子目录、不删除失败；写provenance（torch/torchdata/datasets实际版本、StatefulDataLoader和官方loader/cycle源码完整hash），成功后才写result.json。没有此变量默认临时目录；正式工程验收必须指定持久证据根并保留原始测试stdout/stderr。测试运行Python至少3.10（os.waitstatus_to_exitcode等）；真实镜像为Python3.12。未开启opt-in时真实loader测试明确skip，不是PASS。

## 崩溃切点与未覆盖项

私有 `_cut` 仅供测试patch，生产没有env启用的故障开关。真实子进程覆盖：genesis后、next后、blob临时写截断、blob fsync后、blob发布后、draw临时JSON截断、record发布/目录fsync前、record durable后、yield前、交付batch但只读取首item后。发布前恢复上一prefix；完整record已存在则恢复重投该record，禁止直接跳过。进程SIGKILL不等于断电/存储设备丢写测试。

尚未覆盖真实完整WorkflowExecutor构造/缓存实例、真实workflow reward/tensor replay、consumed/commit authority、state schema升级、GPU writer fencing、异步checkpoint或多epoch/多rank。真实loader测试若发现iter/load顺序冲突，保留失败并按实际API修订，不能mock绕过。本文不为任何P3 GPU运行授权或验收。


## 主任务验收事实

- `p3_evidence/replay-tests-r1.log` / `.json`：机制9项通过，0.773s。
- `p3_evidence/loader-real-r1.log` / `.json`：真实CPU loader 1项通过，9.939s；没有通过mock替代真实torchdata接口。
- `p3_evidence/loader-real-r1/loader-2y2l8frq/provenance.json` / `result.json`：实际依赖版本、源码hash及snapshot N=1/WAL M=3的完整重投与后续fresh draw结果；该目录保留实际draw records/blobs。

本次验收范围只包含CPU持久draw与真实loader接口合同。未扩展consumed authority、训练workflow、完整executor或GPU；前序GPU/native重启结果不影响该限定CPU结论，也不能由此CPU通过推断GPU恢复通过。
