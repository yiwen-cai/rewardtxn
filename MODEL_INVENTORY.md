# 模型资产清单 (Model Inventory)

记录日期: 2026-08-29
记录人: caiyiwen (自动化巡检)
用途: 训练 (Megatron torch_dist) / 推理 (SGLang/HF)

---

## 一、~/models (home 目录, 推理/候选训练模型)

### 1. Qwen3-30B-A3B (MoE) ✅ 完整
- 路径: `/public/home/caiyiwen/models/Qwen3-30B-A3B`
- 大小: 57G | 16/16 分片 + config/tokenizer OK
- 架构: qwen3_moe, Qwen3MoeForCausalLM
- 配置: hidden=2048, layers=48, heads=32, kv_heads=4, intermediate=6144
  vocab=151936, max_pos=40960, dtype=bf16, tie_word_embeddings=False
  MoE: experts=128, experts_per_tok=8, moe_intermediate=768
- 参数: 约 30.5B 总参 / 3.3B 激活
- 状态: **未转换 torch_dist** (训练前需 convert_hf_to_torch_dist.py)

### 2. Qwen3-30B-A3B-FP8 (MoE, FP8 量化) ✅ 完整
- 路径: `/public/home/caiyiwen/models/Qwen3-30B-A3B-FP8`
- 大小: 31G | 7/7 分片 + config/tokenizer OK
- 架构: 同上, 含 quantization_config: fp8 (blockwise)
- 用途: 推理/rollout 权重 (训练仍用 BF16 原始权重)

### 3. Qwen3-8B (dense) ✅ 完整
- 路径: `/public/home/caiyiwen/models/Qwen3-8B`
- 大小: 16G | 5/5 分片 + config/tokenizer OK
- 架构: qwen3, Qwen3ForCausalLM
- 配置: hidden=4096, layers=36, heads=32, kv_heads=8, intermediate=12288
  vocab=151936, max_pos=40960, dtype=bf16, tie_word_embeddings=False
- 状态: **未转换 torch_dist**

### 4. modelscope_speed_test_Qwen3-30B-A3B ❌ 不可用 (测试残留)
- 路径: `/public/home/caiyiwen/models/modelscope_speed_test_Qwen3-30B-A3B`
- 大小: 1.1G | 仅 1 个分片, 无 tokenizer
- 结论: modelscope 下载测速残留, 不建议使用

### ⚠️ 异常文件
- `/public/home/caiyiwen/models/Qwen3-30B-A3B/model-00001-of-00016.safetensors_0_167772159`
  (60MB, data 类型, mtime 1970-01-01) — modelscope 断点续传临时残留, 非模型分片;
  验证正常分片 model-00001 (3.99G) 可加载后可删除

---

## 二、rewardtxn/models (训练用, 已转换 torch_dist)

| 模型 | HF 大小 | torch_dist 大小 | 状态 |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 1.9G | 943M | ✅ 训练可用 (当前实验) |
| Qwen2.5-1.5B-Instruct | 5.8G | 5.8G | ✅ 训练可用 |
| Qwen2.5-3B-Instruct | 5.8G | 5.8G | ✅ 训练可用 |
| Qwen2.5-7B-Instruct | 15G | 15G | ✅ 训练可用 |
| Qwen3-4B | 7.6G | **无** | ⚠️ 未转换, 仅推理权重 |

### 详细配置

- Qwen2.5-0.5B: hidden=896, layers=24, heads=14, kv_heads=2, intermediate=4864,
  vocab=151936, max_pos=32768, tie_emb=True
- Qwen2.5-1.5B: hidden=1536, layers=28, heads=12, kv_heads=2, intermediate=8960,
  vocab=151936, max_pos=32768, tie_emb=True
- Qwen2.5-3B: hidden=2048, layers=36, heads=16, kv_heads=2, intermediate=11008,
  vocab=151936, max_pos=32768, tie_emb=True
- Qwen2.5-7B: hidden=3584, layers=28, heads=28, kv_heads=4, intermediate=18944,
  vocab=152064, max_pos=32768, tie_emb=False
- Qwen3-4B: hidden=2560, layers=36, heads=32, kv_heads=8, intermediate=9728,
  vocab=151936, max_pos=40960, tie_emb=True, dtype=bf16

### 转换日志 (models/ 下)
- convert_3b.log / convert_7b.log / convert_qwen3.log (均 ~42KB, 2026-08-27)

---

## 三、数据集

- dapo-math-17k: `rewardtxn/models/datasets/dapo-math-17k/dapo-math-17k.jsonl` (10.5MB)
- 其他数据集: 见 `rewardtxn/models/datasets/`

---

## 四、磁盘占用

- ~/models 合计: ~105G (57G + 31G + 16G + 1.1G)
- rewardtxn/models 合计: ~62G
- /public 可用: 1.4T (2026-08-29 巡检)

## 五、待办/备注

1. [ ] 删除 modelscope 临时残留文件 (60MB, 验证后)
2. [ ] 如需 Qwen3-4B 训练: 运行 convert_hf_to_torch_dist.py 转换
3. [ ] 如需 Qwen3-8B / Qwen3-30B-A3B 训练: 转换 torch_dist + 确认 slime
     模型配置脚本 (scripts/models/qwen3-4B.sh / qwen3-30B-A3B.sh 等)
4. 训练配置入口: RTX_MODEL_DIR + RTX_MODEL_CONFIG 环境变量
