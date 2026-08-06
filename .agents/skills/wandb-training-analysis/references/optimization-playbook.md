# 优化手册:异常 → 机制 → 探针实验

按 `analysis-methodology.md` 的四阶段,把常见异常映射到候选机制与可操作的单变量
探针实验。每行探针实验可直接交给 `generate-workflow-dsl` 的
`references/parameter-decision.md` 落参数。

## 异常 → 候选机制

| 异常信号(wandb 数据) | 候选机制 | 优先探针(单变量) |
|---|---|---|
| loss 出现 NaN | 学习率过大 / 梯度爆炸 / 数据含异常样本 | `learning_rate` 减半;若仍 NaN 查数据 |
| loss 高频尖峰(相邻步跳变 >10x) | lr 过高或 batch 过小,梯度方差大 | `learning_rate` 减半 |
| loss 发散(最终值 >> 最小值) | lr 过大 | `learning_rate` 减半 |
| loss 平台期过高、不再下降 | 容量不足 / 欠训练 / 数据问题 | 1) `num_epochs` +1;2) 无效则 `lora_rank` 翻倍 |
| loss 下降过慢 | lr 过小 / 有效 batch 过小 | 1) `learning_rate` ×2;2) `batch_size` ×2(保持有效 batch) |
| 训练/验证 gap 大 | 过拟合 | `num_epochs` -1 或增大数据 |
| GPU 利用率低 | batch 过小 / 数据加载瓶颈 | `batch_size` ×2 |
| GRPO reward 震荡/崩溃 | reward 尺度 / KL 惩罚不匹配 | 调整 KL 系数或 reward 归一化 |
| 同配置两 run 差异大 | 随机种子 / 数据顺序 | 固定 seed 后重跑对照 |

## 探针实验表(可直接落参数)

| # | 探针 | 参数变更 | 验证方式 |
|---|---|---|---|
| 1 | 降 lr | `learning_rate` → ×0.5 | `compare-runs` 新旧 run: 尖峰数/NaN 应下降 |
| 2 | 升 lr | `learning_rate` → ×2 | 对比: loss 下降速度应提升且不引入不稳定 |
| 3 | 加 epoch | `num_epochs` → +1 | 对比: 平台期 loss 应下降 |
| 4 | 升 rank | `lora_rank` → ×2 | 对比: 平台期 loss 应下降(容量假说) |
| 5 | 加 batch | `batch_size` → ×2(同步调整 `grad_accumulation_steps` 保持有效 batch) | 对比: 曲线应更平滑 |
| 6 | 固定 seed | `seed` 设为固定值 | 两 run 曲线应接近 |

## 使用规则

- 每轮只执行一行探针;新 run 完成后用 `compare-runs` 验证,再决定收敛或下一个。
- 修改 `batch_size` 时同步调整 `grad_accumulation_steps`,保持有效 batch
  (= batch × accum × gpu_count)不变,避免混淆变量。
- 若某机制的两个探针都无效,放弃该机制,回到异常表选下一个候选。
- 涉及模型/数据规模类变更(如换模型、扩数据)超出 wandb 数据分析范围,
  转交 `generate-workflow-dsl` 完整流程处理。
