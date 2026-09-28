# Agent 编写 Rubric

评价项来自测试集字段、GT 和任务输出要求，运行时不会创建新评价项。
每项评价一个目标，权重为正；权重自动归一化。required 项失败会否决整体通过。
避免对同一个错误重复计分，不添加无关风格偏好。

```json
{
  "mode": "agent_rubric",
  "rubrics": [
    {"name":"label_accuracy","criterion":"预测类别与 GT 相同","weight":4,"required":true,
     "evaluator":{"type":"field_equals","prediction_path":"label","reference_path":"label"}},
    {"name":"confidence_valid","criterion":"confidence 在 0 到 1 之间","weight":1,
     "evaluator":{"type":"number_range","prediction_path":"confidence","min":0,"max":1}}
  ]
}
```

算子：exact_match（文本一致）、json_valid（可提取 JSON 对象）、required_fields（prediction_paths
所列字段存在）、field_equals（字段与 GT 相等）、non_empty（非空）、number_range（数值范围）、
list_count_match（列表长度一致）、bbox_iou（xyxy 框按 IoU 阈值匹配后的 F1，惩罚漏检和多检）。
字段路径支持点分隔对象字段和列表索引，空路径表示整个值。bbox_iou 不评价类别，类别需单独指标。
开放式生成仅评价可确定性验证的部分，交付时说明覆盖范围；不使用语义 Judge。
