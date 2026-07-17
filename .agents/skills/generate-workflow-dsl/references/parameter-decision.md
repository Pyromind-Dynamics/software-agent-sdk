# 训练参数整组决策

一次完成：**数据 N/L → LLM/VLM → 模型规模 → LoRA/Full → GPU → 任务修正**。不要逐个独立猜参数。

## 1. 数据与长度档位

| 档位 | N | 默认策略 |
|---|---:|---|
| S | < 500 | LoRA、较多 epoch、低 lr |
| M | 500～5K | LoRA、rank 16～32 |
| L | 5K～50K | LoRA、epoch 1、rank 32 |
| XL | ≥ 50K | 资源足够时才考虑 Full |

| L（P95 token） | 压力 |
|---:|---|
| ≤ 2K | 低 |
| 2K～8K | 中 |
| 8K～32K | 高 |
| > 32K | 超出常规模板，要求用户确认资源与截断策略 |

`max_seq_length` 默认取 `min(P95, 4096)`；P95 未知时用 4096 并声明假设。不得把
`previewed_rows` 当作完整 N。

VLM 在同档 LLM 方案上：batch 减半、learning rate 减半、GPU 升一档；至少保持 batch=1。

## 2. 模型与训练方式

| 模型规模 | 参数量 | LoRA 最低资源（L≤2K） | Full 最低资源（L≤2K） |
|---|---:|---:|---:|
| Small | ≤ 3B | 1×40G | 1×40G |
| Medium | 4B～14B | 1×80G | 2×80G（4～8B）/4×80G（14B） |
| Large | 32B～72B | 2～4×80G | 不走常规模板 |
| XL | > 72B | 仅 LoRA/QLoRA | 不支持 Full |

默认 LoRA。只有 N≥5K、L≤8K、用户明确需要深度能力迁移且资源达到最低要求时才选 Full；
Full 不创建或连接 `LoraConfigBuilderNode`。

LoRA rank：S=16，M=16（中长序列取 32），L=32（明确领域深度注入可 64），XL=64。
`lora_dropout=0.05`，target modules 默认
`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。

## 3. LLM + LoRA 基准矩阵

| N | L | GPU | batch | accum | SFT lr | epoch | rank |
|---|---|---:|---:|---:|---:|---:|---:|
| <500 | ≤2K | 1×40G | 8 | 4 | 1e-5 | 3～5 | 16 |
| <500 | 2K～8K | 1×80G | 2 | 8 | 5e-6 | 2 | 16 |
| 500～5K | ≤2K | 1×40G | 8 | 4 | 2e-5 | 2 | 16 |
| 500～5K | 2K～8K | 1×80G | 4 | 4 | 1e-5 | 1 | 32 |
| 5K～50K | ≤2K | 1×80G | 16 | 2 | 1e-5 | 1 | 32 |
| 5K～50K | 2K～8K | 2×80G | 4 | 4 | 5e-6 | 1 | 32 |
| ≥50K | ≤2K | 2×80G | 16 | 4 | 5e-6 | 1 | 64 |

有效 batch = `batch_size × grad_accum × gpu_count`。调整显存时尽量保持它不变。

## 4. 任务修正

在 SFT 基准方案上整体修改：

| 参数 | DPO | GRPO |
|---|---|---|
| learning_rate | SFT × 0.05～0.1 | SFT × 0.1～0.2 |
| num_epochs | `min(SFT epoch, 1)` | 1；主要由 max_steps 控制 |
| lr_scheduler_type | `constant` | `constant_with_warmup` |
| max_grad_norm | 0.5 | 1.0 |
| num_generations | — | N<2K 为 4，否则 8 |
| max_completion_length | — | `min(512, max(128, L/4))` |
| max_steps | — | 用户指定优先，否则 200 |

GRPO 的 `max_prompt_length + max_completion_length` 不应超过 `max_seq_length`。

## 5. GPU 与降级

- 单卡 LoRA：`zero_stage=0`；不要开 DeepSpeed。
- 多卡 LoRA：按平台支持使用 DDP/Zero；没有对应节点参数时不要编造 DSL 字段。
- 资源不足按顺序处理：Full→LoRA；rank 64→32→16；batch 减半且 accum 翻倍；
  max sequence length 8192→4096→2048（先确认截断）；最后换更小基模或提示扩容。
- 不要先降低 learning rate；lr 是优化目标参数，不是主要显存旋钮。

## 6. 决策输出

写 DSL 前内部确认一组结果：模型、训练方式、GPU、max sequence length、batch、accum、有效
batch、lr、scheduler、epoch、rank，以及 GRPO 的 steps/generations/长度。若某值来自用户，标记
为锁定值；修改现有工作流时，未被需求影响的锁定值保持不变。
