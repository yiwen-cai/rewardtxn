#!/usr/bin/env bash
# 实验配置预设 (profile)。宿主入口在 RTX_PROFILE 非空时 source 本文件。
#
#   RTX_PROFILE=smoke        短冒烟: 4 步, 高频 checkpoint, 低并发 (验证链路用)
#   RTX_PROFILE=phase3b      可恢复实验: 保留 optimizer state + 2 份 checkpoint
#   RTX_PROFILE=long         不可恢复长程: --no-save-optim + 只留 1 份 checkpoint
#
# 所有变量都可被显式设置的环境变量覆盖 (${VAR:-default} 语义)。
set -euo pipefail

case "${RTX_PROFILE:-}" in
  "" )
    : # 无 profile, 使用各脚本默认
    ;;
  smoke )
    export RTX_NUM_ROLLOUT=${RTX_NUM_ROLLOUT:-4}
    export RTX_SAVE_INTERVAL=${RTX_SAVE_INTERVAL:-2}
    export RTX_SGLANG_CONCURRENCY=${RTX_SGLANG_CONCURRENCY:-16}
    export RTX_NO_SAVE_OPTIM=${RTX_NO_SAVE_OPTIM:-0}
    export RTX_CKPT_KEEP=${RTX_CKPT_KEEP:-2}
    ;;
  phase3b | recoverable )
    export RTX_NO_SAVE_OPTIM=${RTX_NO_SAVE_OPTIM:-0}
    export RTX_CKPT_KEEP=${RTX_CKPT_KEEP:-2}
    export RTX_SAVE_INTERVAL=${RTX_SAVE_INTERVAL:-10}
    ;;
  long | nonresumable )
    export RTX_NO_SAVE_OPTIM=1
    export RTX_CKPT_KEEP=${RTX_CKPT_KEEP:-1}
    export RTX_SAVE_INTERVAL=${RTX_SAVE_INTERVAL:-100}
    ;;
  * )
    echo "unknown RTX_PROFILE='${RTX_PROFILE}' (smoke|phase3b|long)" >&2
    exit 2
    ;;
esac
