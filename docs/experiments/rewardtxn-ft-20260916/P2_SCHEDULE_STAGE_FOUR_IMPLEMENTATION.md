# P2 第四段：X2 作用域合同

2026-09-20，economy-dev 根状态 enabled，Astra medium 实施；无递归。本段仅扩展 `tests/ft/p2_schedule_driver.py`，更新原 driver 测试支持集计数，新增 `tests/ft/test_p2_schedule_scope.py`。manifest、生产 state/oracle、P1、第三方及既有证据不变。

## 范围与证据

STAGE_FOUR 为 X2 的 10 boundaries × i00/i01/i02/i03/i04/i05/i08/i09，共80格。与已有252格重合 X2.b02.i04、X2.b09.i05 两格，旧分支保留；支持集合去重330。这是实现可执行集合，不是330格已验收。原252功能合同和228 core signatures的历史分账见 `P2_SCHEDULE_SIGNATURE_AUDIT.md`。

| boundary | S/P实际操作 | O独立建模 |
|---|---|---|
| b00 | 同一授权/载荷重复提交，核对原收据相同 | 合法当前结果；另构造非法保留K反例 |
| b01 | 实际提交reward=0而原已接受reward=1，要求conflicting result | 独立重算发现保留的错误reward |
| b02/b03 | CAS至新attempt，旧先到/新先到两种真实调用顺序；旧拒绝，新接受 | 新授权后仍保留旧attempt非法 |
| b04 | 原登记owner正常退出并wait；新进程持久身份退出证明接管epoch1，CAS授权recovered/policy1并接受 | reload父状态、新epoch合法重算；sample0 policy1和其他policy0均合法 |
| b05 | never-authorized attempt实际提交被拒绝 | 未授权attempt保留非法 |
| b06 | target:0授权改为other:0 sample引用，真实拒绝 | 保留非目标授权attempt非法 |
| b07 | 建立真实foreign-run root，独立owner与授权、实际接受，向目标root提交被拒绝 | 保留非本组授权attempt非法 |
| b08 | 真实intent重复sample消费条目，再prepare拒绝 | 保留重复sample的K映射非法 |
| b09 | 关闭本owner后提交bootstrap generation commit，inactive拒绝 | epoch已替换而保留旧授权非法 |

b04恢复payload的policy字段改为1并保存变更前后hash；独立O从冻结外部fixture施加同一声明变更，训练row digest相应重算。它验证授权绑定，**没有实际模型policy provenance**，也没有训练优化器backend。S只接受目标sample0，不声称恢复后的旧epoch其余七个结果可一起prepare。

O schema没有run-root字段。因此b05/b06/b07在O层映射到同一“非本目标授权”合同，不能凭不同attempt名称宣称不同O语义；它们的实际S差异由提交sample引用、独立foreign root与其control/accepted文件证明。每格O的完整健康control先审计，非法保留反例另存。b04正例使用合法回滚图，只证明独立规范模型，不从S生成expected。所有O时钟为模拟值，不报告实测RTO。

i01从实际生成收据核对倒序，i02重放相同bootstrap证据，i03/i04使用既有probe早晚旧attempt，i05真实两个Popen争同run锁且winner继续；i08完整独立健康control后移除指定optimizer_start，i09主边界结束后再损坏bootstrap checkpoint fullhash。特别是b09i09：inactive旧owner拒绝后，corruption-check是原进程已退出后合法接管的**另一个owner**。b04i09的epoch依次为0→1→2。无SIGKILL要求的X2不补造kill；b04为真实正常退出。

各case保持独立目录/原始错误；boundary检查核对初始8个实际接受、重复收据、旧/新实际次序、精确拒绝收据、foreign root登记和接受内容。b04额外记录`exited_owner_new_epoch_acceptance`，核对原身份在持久登记、新身份不同、epoch1与policy1。新scope tests覆盖上述合同，并删除拒绝/篡改sample/反转序列作为独立检查器负例。

## 验收入口与当前状态

worker仅定向运行b04i09、b07i00（`/tmp/rtx-x2-target-r1`）及冲突/旧到达、重复mapping/inactive两项测试（1.191秒）通过。最后新增b04载荷policy变更收据和b01提交参数留证后，完整回归交主任务；未运行80格矩阵。

```sh
.venv-tq/bin/python3.11 tests/ft/test_p2_schedule_driver.py -v
.venv-tq/bin/python3.11 tests/ft/test_p2_schedule_scope.py -v
.venv-tq/bin/python3.11 tests/ft/p2_schedule_driver.py --representative \
  --output docs/experiments/rewardtxn-ft-20260916/p2_evidence/representative-r5
.venv-tq/bin/python3.11 - <<'PY'
import subprocess
args = ['.venv-tq/bin/python3.11', 'tests/ft/p2_schedule_driver.py',
        '--output', 'docs/experiments/rewardtxn-ft-20260916/p2_evidence/stage-four-r1',
        '--case-timeout', '30', '--total-timeout', '600']
for b in range(10):
    for i in (0, 1, 2, 3, 4, 5, 8, 9):
        args += ['--case', f'X2.b{b:02d}.i{i:02d}']
subprocess.run(args, check=True)
PY
```

主任务另运行既有88/96/72集合回归。验收要求80格各自全部must_reach、0失败/harness_failure；单格/总deadline明确，清理只针对driver自己的Popen。跨case审计尚未扩展新scope事件schema；现有审计器遇到新事件应显式unsupported，不为兼容静默丢字段。本段不改审计器，不把80格逐case通过等同80个不同core状态。F4其余格与184待补接口不在本段。

## 主任务验收实证

原 driver 25项/7.646秒、新scope 6项/3.049秒全部通过；representative-r5为12/12，stage-four-r1为80/80（10次真实owner争锁），无失败。`p2_evidence/stage-four-verification-r1.json`记录结果。全局去重330合同passed、78静态可执行未跑、8限制、184待补。X2新事件尚未扩入事后审计封闭schema，不能把330合同视为330不同core签名；历史252的审计结论保持原样。
