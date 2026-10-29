# AVI 单轮需求清单

## 必填项

| key | 需要确认的内容 | 有效示例 |
|---|---|---|
| `dataset_uri` | 用户可访问的数据集位置 | `datasets/avi/train.jsonl` |
| `dataset_format` | 文件格式和一条样本的结构 | `jsonl + images/system_prompt/user_prompt/gt` |
| `image_roles` | 每类图片的业务含义 | `defect, diff, gt` |
| `label_schema` | 标签枚举和别名 | `pass, fail, defect` |
| `split_policy` | train/val/test 的来源和隔离规则 | `按批次隔离，test 不参与选模` |
| `inference_model` | 被测模型或模型入口 | `Qwen/Qwen3-VL-4B-Instruct` |
| `train_mode` | `sft` 或 `dpo` | `sft` |
| `evaluation_rule` | 评判预测对错的确定性规则 | `result 精确匹配；主指标 fail_recall` |
| `artifact_goal` | 本轮必须产出的模型或报告 | `LoRA checkpoint + ATIF 评测报告` |

## 按条件补充

- DPO 必须补充 chosen/rejected 的来源，以及两者如何保证质量差异。
- 有框或区域判断时，补充坐标口径和 IoU 阈值。
- 有 reason 字段时，明确 reason 是必须精确匹配还是只做展示。
- 数据跨多个批次时，补充重复样本和 split 泄漏检查口径。
- 用户要求上线门禁时，补充候选与 baseline 的比较阈值。

## 提问方式

先总结已经确认的内容，再只问缺失项。一次最多提出一组相关问题，避免逐字段盘问。
用户给出模糊答案时说明会如何影响 Graph，不要自行补默认值。

示例：

```text
已确认：数据在 datasets/avi/train.jsonl，使用 defect/diff/gt 三图和
pass/fail/defect 标签，目标模型是 Qwen3-VL-4B，选择 SFT。
还需要确认：
1. train/val/test 是否按批次隔离？
2. fail 样本的验收规则和主指标是什么？
3. 本轮需要 LoRA checkpoint，还是还要合并后的完整模型？
```
