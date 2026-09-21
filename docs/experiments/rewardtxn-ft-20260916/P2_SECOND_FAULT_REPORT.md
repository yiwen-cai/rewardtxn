# 双故障 CPU 隔离探针验收

日期：2026-09-21。结果：6项检查全部通过，无跳过。未修改探针或生产源码。

## 执行与证据

- 配置由 `tests/ft/p2_second_fault_probe.py --make-config /tmp/rtx-second-fault-20260921.json` 生成，运行副本保存在证据目录 `config.json`。
- 使用 `scripts.ft.container_run` CPU profile，固定镜像 `sha256:c0573bb8412a8db753d44b02c9e04504d2392c54f019707f58acf26dd68d2469`，容器 Python `/opt/.venv/bin/python`；2 CPU、2 GiB、128 PID、无网络、无 GPU，总 run 上限150秒。
- 完成证据：`p2_evidence/second-fault-20260921-r2/`。supervisor耗时3.261798489秒，failure=null、cleanup_confirmed=true、cleanup_errors=[]。
- 首次 r1 在 sandbox 内访问 Docker socket 被拒绝，未创建容器；原记录保留。获得工具提权后使用新目录 r2 执行。
- 独立验证命令：`.venv-tq/bin/python3.11 tests/ft/test_p2_second_fault_probe.py --evidence docs/experiments/rewardtxn-ft-20260916/p2_evidence/second-fault-20260921-r2 -v`。退出0，6 tests / OK，耗时0.022秒。
- 独立执行 `docker inspect 764551d9a35567e6dc1b46116c1784a9b5537fe036e1712b5957aaadaa3e4a6e` 返回退出1、`No such object`，确认本次容器已删除。

## 验收结果

1. 冻结输入包含两个不同的K=8故障目标。
2. 两次真实SIGKILL退出码均为-9，身份与nonce不同，三个attempt完成。
3. 首次恢复实际加载状态后才开始第二目标；第二次恢复加载最终状态。
4. 独立校验三代保留链、每代两rank共22个组件文件、payload哈希、consumed/pending与cursor。
5. 提前第二次故障与错误目标均被拒绝，没有故障实际注入。
6. 不完整第二候选未被提升，恢复保留第一目标及第二目标pending；旧owner存活时拒绝接管且epoch不变。

## 结论边界

这是CPU fixture状态链与控制协议验收，不包含真实optimizer、GPU训练或多目标训练oracle。`training`和`oracle`仍为`not_evaluated`，P2的184项待补不变，正式GPU样本仍为0。i06/i07业务映射、P3真实训练适配及正式冻结仍待完成。
