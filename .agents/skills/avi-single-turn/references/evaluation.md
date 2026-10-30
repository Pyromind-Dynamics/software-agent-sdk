# AVI 评测口径

## 输出

Evaluator 的输入来自 Inference predictions 和数据集 GT。每条样本至少输出：

```json
{
  "sample_id": "report_001",
  "gt_result": "fail",
  "pred_result": "fail",
  "gt_reason": "存在划痕",
  "pred_reason": "存在划痕",
  "box_iou": 0.82,
  "reward": 1.0
}
```

## 指标

首期沿用 AVI demo 的业务口径：

- 主指标：`fail_recall`，即 GT 为 fail 的样本中预测也为 fail 的比例。
- `fail_recall` 完全持平时，使用严格 `accuracy` 作为兜底比较。
- 同时保留 `result_accuracy`、各分类 Precision/Recall/F1 和 confusion matrix。
- 有框时增加 box Precision/Recall/F1 和 IoU 统计。

默认不能把 `pass_rate` 或训练 loss 当作上线门禁。用户明确改变主指标时，把选择写入
`evaluation_rule`。

## 确定性要求

- 同一 predictions 重复评分必须得到相同结果。
- 不在 Evaluator 内调用被测模型。
- 不根据模型自述 reason 判定正确，除非用户明确把 reason 纳入规则。
- 结果、指标和逐样本问题清单都要能追溯到数据集版本和模型产物。

## ATIF 交付

评测完成后生成 ATIF 轨迹和汇总指标。ATIF 至少支持：

- 单条样本图片回看；
- GT 与 prediction 对照；
- pass/fail/defect 混淆；
- fail 漏检和误报筛选；
- 训练前后同口径比较。

当前阶段只定义格式和 Graph Evaluator 配置，不启动真实评测。
