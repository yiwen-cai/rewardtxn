#!/usr/bin/env bash
# RewardTxn H100 动态环境预检脚本 (文档 3.2 节)
set -euo pipefail

echo "=== 1. 检查 CUDA 与 驱动 ==="
export PATH=/usr/local/cuda/bin:/public/home/caiyiwen/.local/bin:$PATH
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader

echo "=== 2. 检查 Docker NVIDIA Runtime ==="
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu20.04 nvidia-smi -L >/dev/null && echo "Docker GPU Runtime: OK"

echo "=== 3. 检查 /public 与 /dev/shm 容量 ==="
AVAIL_PUB=$(df -BG /public | awk 'NR==2 {print $4}' | tr -d 'G')
if [ "$AVAIL_PUB" -lt 200 ]; then
    echo "ERROR: /public 可用空间仅剩 ${AVAIL_PUB}GB，低于安全门禁 200GB！请先清理磁盘。"
    exit 1
fi
echo "/public 可用空间: ${AVAIL_PUB}GB (合格)"

echo "=== 4. 检查当前 GPU 空闲情况 ==="
BUSY_GPUS=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | wc -l)
echo "当前运行中的 GPU 计算进程数: ${BUSY_GPUS}"
