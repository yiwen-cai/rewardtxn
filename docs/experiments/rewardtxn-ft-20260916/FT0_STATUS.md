# FT0 最新状态：2026-09-20

P0接口审计交付完成；62项官方CPU测试、真实RecoverInfo文件检查，以及单H100同步Megatron checkpoint新进程连续性验证通过。交付见 [P0_REPORT](P0_REPORT.md)、[接口审计](P0_INTERFACE_AUDIT.md)、[能力表](capability_matrix.csv)、[oracle规范](oracle_spec.md)。

完整RecoverHandler/RL恢复、数据游标、异步保存完成语义及F4运行时切点仍未验证；完整R适配和正式freeze未实施，不放行FT1/FT2正式矩阵。A选定路径F3预先N/A；C适配条件阻塞。可推进P1隔离runner及CPU合同。旧资源快照不用于启跑。

## 历史记录（以下为9月16日当时状态）

# FT0 状态

更新：2026-09-16 09:52 +0800

## 本轮已做
- AReaL 快照 `b83d1f40196e5bd7d9f83092563443561870d550`（b83d1f4 docs: clarify terminology in the M2PO guides (#1635)）
- 恢复入口 `third_party/areal/areal/utils/recover.py`
- 能力表 `docs/experiments/rewardtxn-ft-20260916/capability_matrix.csv`
- ByteCheckpoint：已 checkout `6f00167`（LICENSE 在；运行时未验）
- RobustRL：artifact 未确认

## recover 符号（节选）
- `class
class`
- `class InValidRecoverInfo`
- `class RecoverHandler`
- `def __init__`
- `def _ensure_recover_supported`
- `def _is_gateway_train_controller`
- `def _load_checkpoint`
- `def _normalize_recover_engines`
- `def _require_colocate_rollout_protocol`
- `def _save_checkpoint`
- `def _should_run_awex_colocate_transfer`
- `def check_if_auto_recover`
- `def check_if_recover`
- `def dump`
- `def load`
- `def recover_info_path`

## 相关路径
- `third_party/areal/areal/utils/recover.py`
- `third_party/areal/tests/test_recover.py`

## 未完成
1. 官方 recover 测试 + 短恢复示例
2. ByteCheckpoint checkout/锁定
3. A+R / C+R 状态路径与 oracle
4. 新 FT freeze
5. 按 cell 填完 capability_matrix

## 资源门禁
GPU0-6 空；GPU7 外部占用。FT2 正式矩阵未启。

## 门禁更新 2026-09-16 15:52
用户放宽：任意 ≥4 张空闲 GPU 即可启跑（不再锁定 device=1,2,3,4）。现场当时 GPU3–7 空闲；0/1/2 为 zhouhannan。
工程未完成项（官方 recover 测试、A+R/C+R、新 freeze）仍阻塞 FT1/FT2 正式矩阵；资源窗开时优先推进 FT0 短恢复示例，不重跑 LAST_THREE。
