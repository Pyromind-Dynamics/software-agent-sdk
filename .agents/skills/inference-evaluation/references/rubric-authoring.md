# Agent Rubric 制定范式

Rubric 是当前 Agent 在预览测试集后生成的静态评测契约。它可以随数据集变化，但在工作流运行期间
保持不变；被测模型既不生成检查项，也不修改权重。

## 一、从测试集提取评测契约

按以下顺序分析代表性样本：

1. 找出 input、媒体、ground truth 和样本 ID 的真实字段。
2. 从 system/user prompt 中提取明确输出约束，例如 JSON schema、类别集合、坐标格式和取值范围。
3. 判断 GT 是纯文本、JSON 对象、JSON 字符串，还是包含说明文本与末尾 JSON 的混合格式。
4. 列出“任务正确性”和“输出合法性”，合并会重复惩罚同一错误的项目。
5. 为每一项选择下面已有的确定性 evaluator；没有可复现判断方法的项目不要伪装成代码指标。

## 二、Rubric JSON 结构

```json
{
  "mode": "agent_rubric",
  "pass_threshold": 0.8,
  "rubric_pass_threshold": 0.7,
  "rubrics": [
    {
      "name": "result_correctness",
      "criterion": "prediction.result 与 ground truth 的 result 完全一致",
      "weight": 4,
      "required": true,
      "evaluator": {
        "type": "field_equals",
        "prediction_path": "result",
        "reference_path": "result"
      }
    }
  ]
}
```

每个 Rubric 必须包含 `name`、`criterion`、`weight` 和 `evaluator`。`name` 使用稳定的 snake_case；
`criterion` 必须写清比较对象和通过条件；`weight` 必须大于 0。核心结论、关键安全条件或关键定位项
增加 `required: true`，该项失败时样本直接失败，不被其他低价值得分抵消。

## 三、支持的确定性 evaluator

| type | 必要参数 | 评分语义 |
|---|---|---|
| `exact_match` | 可选 `case_sensitive` | prediction 全文与 GT 全文一致 |
| `json_valid` | 无 | prediction 能提取出 JSON 对象 |
| `required_fields` | `prediction_paths` | 指定路径全部存在 |
| `field_equals` | `prediction_path`、`reference_path` | 两侧字段值相等 |
| `non_empty` | `prediction_path` | 字段不是空字符串、空列表或 null |
| `number_range` | `prediction_path`，可选 `min`/`max` | 数值落在闭区间内 |
| `list_count_match` | `prediction_path`、`reference_path` | 两侧列表长度相等 |
| `bbox_iou` | `prediction_path`、`reference_path`、`iou_threshold` | 阈值匹配后的 bbox precision/recall F1，漏框和多余框都会扣分 |

路径使用点号访问嵌套对象，例如 `result.label`；空路径表示整个值。脚本会尝试从 prediction 和字符串
GT 中提取最后一个 JSON 对象，因此可兼容“解释文本 + JSON”的常见输出。

## 四、权重建议

- 核心答案、分类或任务结论：3～5。
- 关键结构化字段、数量、定位：2～4。
- 格式合法性、范围约束：1～2。
- 默认 2～5 项，总分按权重归一化；结构化检测任务的 `pass_threshold` 建议从 0.8 起。

若格式失败会使所有字段检查自然失败，降低 `json_valid` 权重，避免格式错误被重复过度惩罚。

## 五、示例

分类输出：

```json
{
  "mode": "agent_rubric",
  "pass_threshold": 0.8,
  "rubric_pass_threshold": 0.7,
  "rubrics": [
    {
      "name": "label_accuracy",
      "criterion": "预测类别与 ground truth 类别一致",
      "weight": 4,
      "required": true,
      "evaluator": {
        "type": "field_equals",
        "prediction_path": "label",
        "reference_path": "label"
      }
    },
    {
      "name": "confidence_valid",
      "criterion": "confidence 是 0 到 1 的数值",
      "weight": 1,
      "evaluator": {
        "type": "number_range",
        "prediction_path": "confidence",
        "min": 0,
        "max": 1
      }
    }
  ]
}
```

目标检测可增加 `bbox_iou` 并按业务要求设为必需项；纯文本短答案可使用 `exact_match`。`json_valid`
只判断能否解析 JSON，不验证字段完整性；有 schema 要求时应使用 `required_fields`。开放式生成若没有
独立 Judge，则只评估可确定性验证的部分，并在交付说明覆盖范围，不得让被测模型自行生成或裁决
Rubric。
