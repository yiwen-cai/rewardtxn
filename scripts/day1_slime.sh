#!/usr/bin/env bash
# Day 1: Slime 4 卡最小全异步 GRPO (Qwen2.5-0.5B) —— 文档 7 节 Day 1
# 修正点 (对比文档原文命令):
#   1. 挂载锁定源码 third_party/slime -> /root/slime (文档锁定 commit a6272da0)
#   2. 挂载模型 -> /root/models (脚本默认 MODEL_DIR=/root/models/Qwen2.5-0.5B-Instruct)
#   3. 挂载数据集 -> /root/datasets (脚本默认 DATA_PATH=/root/datasets/dapo-math-17k/dapo-math-17k.jsonl)
#   4. 容器内 pip install -e . --no-deps 使挂载源码生效
set -euo pipefail

EXP_DIR=/public/home/caiyiwen/rewardtxn
NAME=rtx-day1-slime
# GPU 2-5 被外部 vLLM 任务占用时回退 3 卡冒烟 (0,1,6)；空闲时可改回 4 卡 0,1,2,5
GPUS='"device=0,1,6"'
NUM_GPUS=3

# 清理旧容器
docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" --gpus "$GPUS" -e NUM_GPUS=$NUM_GPUS \
  --ipc=host --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$EXP_DIR":/workspace \
  -v "$EXP_DIR/third_party/slime":/root/slime \
  -v "$EXP_DIR/models":/root/models \
  -v "$EXP_DIR/models/datasets":/root/datasets \
  slimerl/slime:v0.3.1 \
  bash -c "
    set -e
    git config --global --add safe.directory /root/slime
    cd /root/slime
    pip install -e . --no-deps -q
    NUM_GPUS=${NUM_GPUS:-4} bash examples/fully_async/run-qwen2.5-0.5B-fully_async.sh
  "

echo "container $NAME started, logs: docker logs -f $NAME"
