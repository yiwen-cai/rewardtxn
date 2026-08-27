# RewardTxn 环境搭建记录 (2026-08-24/25)

## 1. 目标机与预检结果 (check_env.sh 实测通过)

- 本机 = 文档目标机 `h100` (10.160.4.102, hostname: master)
- 8× H100 PCIe 81559MiB, Driver 580.126.09, Docker 24.0.7, /public 1128GB 可用, /dev/shm 252G
- Docker GPU Runtime: OK (nvidia/cuda:12.6.2-base-ubuntu20.04 容器内 nvidia-smi 正常)
- 动态占用: GPU 2-5/7 被外部 vLLM 任务占用 (34.8GB×4 + 76.6GB); 空闲 0/1/6
  → Day 1 冒烟按拓扑回退为 3 卡 (0,1,6), 未干扰外部任务

## 2. 三栈环境状态

| 栈 | 版本锁定 | 容器/环境 | 验证结果 |
|---|---|---|---|
| THUDM/slime | v0.3.1 = a6272da0d4f3d0a08520c99a2f3b4f6c887960dc (GitHub 实测 tag 一致) | `slimerl/slime:v0.3.1` (44.7GB, digest sha256:2feaad36b157... 与文档锁定前缀一致) | ✅ 3 卡 fully-async 冒烟: ray job succeeded, sglang 775×generate, actor step 0-2 正常 |
| areal-project/AReaL | b83d1f40196e5bd7d9f83092563443561870d550 (GitHub 实测 = HEAD) | `ghcr.io/areal-project/areal-runtime:v2.0.0-sglang` (53.4GB, 本地导入) | ✅ 容器验证: torch 2.9.1+cu129 CUDA OK, sglang 0.5.10.post1, ray 2.55.1, areal import OK; 挂载锁定源码 + uv pip install -e 验证通过 |
| Ascend/TransferQueue | 8497a52a5c4347c4d67c97f7ce500e544426a08a (= release/v0.1.10) | `.venv-tq` (Python 3.11: torch 2.13.0+cu130 CUDA OK, ray 2.58.0) | ✅ import + tutorial 01/03 (含 get_meta/metadata) 通过 |

## 3. 模型与数据 (国内源)

- `models/Qwen2.5-0.5B-Instruct` (1.9GB): ModelScope 下载 (HF 不可达)
- `models/datasets/dapo-math-17k/dapo-math-17k.jsonl` (10.5MB, 17398 条): hf-mirror.com 下载 (slime 官方数据源 zhuzilin/dapo-math-17k)
- `models/Qwen2.5-0.5B-Instruct_torch_dist` (943MB): 容器内 GPU0 转换完成 (tools/convert_hf_to_torch_dist.py, CONVERT_EXIT=0, 494M 参数)

## 4. 网络问题与国内镜像实测

| 目标 | 状态 |
|---|---|
| registry-1.docker.io (Docker Hub) | ❌ 不可达 (直连超时) |
| docker.m.daocloud.io | ✅ 可用 (slime 44.7GB 经此拉取成功) |
| docker.1ms.run / docker.xuanyuan.me / hub.rat.dev | ✅ 可达 |
| ghcr.io 直连 | ✅ API 通, blob 3.66MB/s (curl 直连) |
| ghcr.nju.edu.cn | ⚠️ API/小层通, 大层代理下卡死 |
| ghcr.m.daocloud.io | ❌ 白名单限制 (areal-runtime 不在) |
| huggingface.co | ❌ 不可达 |
| hf-mirror.com | ✅ 可用 (数据集经此) |
| modelscope.cn | ✅ 可用 (模型经此) |
| pypi.org / 清华 / 阿里镜像 | ✅ 全部可达 |

**关键根因**: dockerd 经 systemd 配置全局 HTTP 代理 10.129.41.197:7890,
该代理间歇性不可达 (proxyconnect i/o timeout), 导致 docker pull 大层卡死。
无 sudo 无法修改 daemon 配置 → 绕过方案 (scripts/pull_ghcr_manual.sh + oci2docker.py):
crane/curl 直连下载全部 blob (HTTP/1.1, 断点续传) + 构造 docker-save tar + docker load。
注: docker load 对 OCI archive 支持有 bug (blobs/json 错误), 需转 docker-save 格式。

## 5. 文档审查发现的问题 (已修正/已记录)

1. 🔴 镜像拉取路径: 文档未给国内镜像方案 → 已用 daocloud 拉 slime, crane 拉 areal
2. 🔴 Day 1 命令缺模型/数据集挂载 → scripts/day1_slime.sh 已修正 (挂载 /root/models, /root/datasets, 锁定源码 /root/slime)
3. 🔴 GPU 动态占用冲突 → 冒烟回退 3 卡 (0,1,6)
4. 🟡 Phase 0 门禁要求 K∈{4,8,16}+2 类故障, 脚本仅 K=4 单类 → 未改 (实验阶段处理)
5. 🟡 torch.std 默认 ddof=1 与 GRPO 总体 std 惯例不符 → 未改 (实验阶段处理)
6. 🟡 L2 定义 (Optimizer 中崩溃) 与 Day4 注入 (Backward 中杀) 不一致 → 未改 (实验阶段处理)
7. 🟡 状态机图 ABORTED→PREPARED 箭头与 I9 矛盾 → 未改 (文档修订)

## 6. 使用方式

```bash
# Day 1 slime 冒烟 (4 卡需 GPU 2/5 空闲; 当前脚本默认 3 卡 0,1,6)
bash scripts/day1_slime.sh && docker logs -f rtx-day1-slime

# 环境预检
bash scripts/check_env.sh

# TransferQueue 环境
source .venv-tq/bin/activate

# Phase 0 差分 oracle (需 .venv-tq, CPU 即可)
.venv-tq/bin/python scripts/phase0_diff_oracle.py

# AReaL (镜像就绪后)
docker run -it --rm --gpus all --network host --shm-size 64g \
  -v /public/home/caiyiwen/rewardtxn:/workspace \
  ghcr.io/areal-project/areal-runtime:v2.0.0-sglang bash
```
