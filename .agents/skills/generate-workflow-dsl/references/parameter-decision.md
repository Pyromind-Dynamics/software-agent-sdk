# 工作流自动配参决策链

决策链：**数据集(N, L) → 模型类型(LLM/VL) → 模型规模 → LoRA/Full → GPU 资源**。
N = 已清洗数据集条数，L = P95 序列长度（token）。用户上传 Storage 数据时采用
`preview_dataset` 的结果；数据集标识采用用户提供或文档中已确认的信息。无法确认时采用
保守档位并说明假设，不调用数据清洗流程来补齐。

参数优先级：用户明确要求 > 修改任务中未要求改变的现有有效值 > 当前数据与资源按本决策表
算出的值 > 示例模板值。模板主要提供拓扑，不能覆盖更高优先级信息。

## Step 1：数据档位

| 档位 | 条数 N | 对后续决策的影响 |
|------|--------|------------------|
| S | < 500 | 优先 LoRA，epoch 多，lr 低 |
| M | 500 ~ 5K | LoRA 默认，rank 16~32 |
| L | 5K ~ 50K | rank 可升，epoch=1 |
| XL | ≥ 50K | 资源够时可考虑 Full |

| 长度 L | 档位 | 显存压力 |
|--------|------|----------|
| ≤ 2K | 短 | 低 |
| 2K ~ 8K | 中 | 中 |
| 8K ~ 32K | 长 | 高 |
| > 32K | 超长 | 极高，需特殊并行 |

## Step 2：LLM vs VL

数据含 image/video 字段，或选了 VL 基模 → 按 VL 处理：

- `batch_size` = 同规模 LLM 的 1/2
- `learning_rate` = 同规模 LLM 的 1/2（更保守）
- 显存需求约为同规模 LLM 的 1.5~2 倍（GPU 档位升一档）
- `max_seq_length` 默认取 min(P95, 4096)，确有需要才升到 32K

## Step 3：模型规模分档

| 档位 | 参数量 | 平台可选例 | Full 可行性 |
|------|--------|-----------|-------------|
| Small | ≤ 3B | Qwen3-0.6B/1.7B、Qwen3-VL-2B | 资源充足可 Full |
| Medium | 4B ~ 14B | Qwen3-4B、Qwen3-VL-4B | 需 ≥ 2×80G |
| Large | 32B ~ 72B | （Download Model 引入） | 需 ≥ 4×80G + FSDP |
| XL | > 72B | （Download Model 引入） | 仅 LoRA/QLoRA |

## Step 4：LoRA vs Full

默认 LoRA。仅当**全部**满足才选 Full：N ≥ 5000、L ≤ 8192、有明确"深度对齐/能力迁移"
需求、GPU 满足 Full 最低要求（见 Step 5）。

LoRA rank 决策：

| N × L 组合 | lora_rank | lora_alpha |
|------------|-----------|------------|
| S + 短 | 8 ~ 16 | rank × 2 |
| M + 中 | 16 ~ 32 | rank × 2 |
| L + 中 | 32 ~ 64 | rank × 2 |
| XL 任意 | 64 | rank × 2 |

按任务类型：SFT 冷启动 / DPO / GRPO 都默认 LoRA；小数据(N<500)格式强约束用 rank=16
（Full 易过拟合）；领域深度注入(N>10K)可 rank=64 或 Full（看资源）。

## Step 5：GPU 资源

LoRA 最低显存（bf16，含 optimizer + 激活）：

| 模型规模 | LLM, L≤2K | LLM, L=8K | VL, L≤2K | VL, L=8K |
|----------|-----------|-----------|----------|----------|
| ≤ 3B | 1×24G | 1×40G | 1×40G | 1×80G |
| 4B ~ 8B | 1×40G | 1×80G | 1×80G | 2×80G |
| 14B | 1×80G | 2×80G | 2×80G | 4×80G |
| 32B | 2×80G | 4×80G | 4×80G | 8×80G |

Full 最低显存：≤3B 需 1×40G（L≤2K）/ 1×80G（L=8K）；7B~8B 需 2×80G / 4×80G（ZeRO-2/3）；
14B 需 4×80G / 8×80G（FSDP）。

并行策略：单卡用 `distributed_type: NO`（**单卡 LoRA 不要开 DeepSpeed**，对应
`AccelerateConfigBuilderNode` 的 `zero_stage=0`）；2~4 卡 LoRA 用 DDP、Full 7B+ 用
FSDP；4 卡以上 FSDP。

## 完整决策矩阵（LLM + LoRA 默认路径）

| N | L | 模型 | GPU | batch | accum | lr(SFT) | epoch | rank |
|---|---|------|-----|-------|-------|---------|-------|------|
| <500 | ≤2K | ≤8B | 1×40G | 8 | 4 | 1e-5 | 3~5 | 16 |
| <500 | 8K | ≤8B | 1×80G | 2 | 8 | 5e-6 | 2 | 16 |
| 500~5K | ≤2K | ≤8B | 1×40G | 8 | 4 | 2e-5 | 2 | 16 |
| 500~5K | 8K | ≤8B | 1×80G | 4 | 4 | 1e-5 | 1 | 32 |
| 5K~50K | ≤2K | ≤8B | 1×80G | 16 | 2 | 1e-5 | 1 | 32 |
| 5K~50K | 8K | 8B | 2×80G | 4 | 4 | 5e-6 | 1 | 32 |
| >50K | ≤2K | 8B | 2×80G | 16 | 4 | 5e-6 | 0.5~1 | 64 |

VL + LoRA：在上表基础上 batch 减半、lr 减半、GPU 升一档。

DPO 在 SFT 基础上：

```text
learning_rate = SFT_lr × 0.05 ~ 0.1
num_epochs    = min(SFT_epoch, 1)
lr_scheduler  = constant
beta          = 0.1（N<500 用 0.2）
max_grad_norm = 0.5
```

GRPO 在 SFT 基础上：

```text
learning_rate      = SFT_lr × 0.1 ~ 0.2
num_generations    = 4（N<2K）或 8（N≥2K）
max_completion_len = min(512, L/4)
beta               = 0.04
```

## 资源不够时的降级顺序

按优先级依次尝试，**不要先降 lr**：

1. Full → LoRA
2. lora_rank 64 → 32 → 16
3. batch_size 减半，grad_accum 翻倍（保持有效 batch）
4. max_seq_length 8192 → 4096 → 2048（需确认数据 P95）
5. gradient_checkpointing = true
6. 换更小基模（8B → 4B → 1.7B）
7. 拒绝任务，提示扩容
