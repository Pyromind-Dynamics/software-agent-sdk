# Agent 编写 Rubric

评价项来自测试集字段、GT 和任务输出要求，运行时不会创建新评价项。
每项评价一个目标，权重为正；权重自动归一化。required 项失败会否决整体通过。
避免对同一个错误重复计分，不添加无关风格偏好。

## GT 与字段路径

`reference_field` 先从原始样本取出 GT；`reference_path` 再从这个 GT 中取值，
不是相对于整条样本。`prediction_path` 相对于模型回复解析后的值。
对象和列表直接使用；字符串中的 JSON 对象会被提取，允许对象前后有思考文本或
Markdown 代码围栏；无法提取对象时保留去除首尾空白的字符串。
`exact_match` 例外，直接比较原始 GT 与回复文本，不做 JSON 字段提取。

路径支持点分隔对象字段和列表索引，如 `answer.result`、`items.0.label`；
省略或 `""` 表示整个值。字段名和枚举值来自真实数据与输出契约，不要求叫 `label`。

| 样本中的 GT | `reference_field` | `reference_path` |
|---|---|---|
| `"label":"defect"` | `"label"` | `""`，比较整个字符串 |
| `"gt":{"result":"defect","boxes":[]}` | `"gt"` | `"result"` 或 `"boxes"` |
| `"gt":"<think>分析</think>\n{\"result\":\"defect\",\"boxes\":[]}"` | `"gt"` | 同上，先提取 JSON 对象 |

例如样本的 `gt` 为 `{"result":"defect","boxes":[[10,20,30,40]]}`，
模型也输出含 `result`、`boxes` 的 JSON。数据映射配置为
`{"user_prompt_field":"prompt","reference_field":"gt"}`，评测配置为：

```json
{
  "mode": "agent_rubric",
  "rubrics": [
    {"name":"label_accuracy","criterion":"预测类别与 GT 相同","weight":4,"required":true,
     "evaluator":{"type":"field_equals","prediction_path":"result","reference_path":"result"}},
    {"name":"box_f1","criterion":"预测框与 GT 框按 IoU>=0.5 匹配","weight":1,
     "evaluator":{"type":"bbox_iou","prediction_path":"boxes","reference_path":"boxes","iou_threshold":0.5}}
  ]
}
```

这里两个指标共用 `gt`；类别比较 `gt.result`，框比较 `gt.boxes`。
若改为 `reference_field="label"`，GT 就是字符串，不能再用 `reference_path="boxes"` 取框。

## 评分算子

每个 evaluator 必须有 `type`，各算子支持的其他字段如下；不支持的字段不能添加。

| type | 支持字段 | 含义 |
|---|---|---|
| `exact_match` | `case_sensitive` | 整段文本与 GT 一致 |
| `json_valid` | 无 | 可提取 JSON 对象，不检查字段、类型或枚举值 |
| `required_fields` | `prediction_paths` | 非空路径列表；得分为存在路径的比例，值为 null 或空列表仍算存在 |
| `field_equals` | `prediction_path`, `reference_path` | 字段与 GT 相等 |
| `non_empty` | `prediction_path` | 字段非空 |
| `number_range` | `prediction_path`, `min`, `max` | 数值范围，未指定的边界不限制 |
| `list_count_match` | `prediction_path`, `reference_path` | 列表长度一致 |
| `bbox_iou` | `prediction_path`, `reference_path`, `iou_threshold` | 框按 IoU 阈值匹配后的 F1，惩罚漏检和多检 |

`case_sensitive` 默认 false；`iou_threshold` 默认 0.5，范围为 [0, 1]。

`bbox_iou` 接收框列表，单框可为 `[x1,y1,x2,y2]`、包含 `bbox` / `box` 的对象，
或包含 `x1,y1,x2,y2` 的对象。预测与 GT 必须对应同一图像并使用相同坐标系；
算子直接计算坐标，不缩放或归一化，没有 `normalized`、`image_size` 参数。
它不评价类别，类别需单独指标。
汇总沿用这些 rubric，无需新增配置：字符串 `field_equals` 生成分类混淆矩阵和各类
Precision/Recall/F1、Macro-F1（按 GT 中出现的类别平均），并给出各真值类别的预测流向比例；矩阵未出现的组合计 0。
缺失或非法输出、请求失败仍计入已知 GT 的 support，记为 unclassified，不会抬高召回率；
无法提取 GT 的样本单列 excluded_references。`json_valid.valid_rate` 以全部样本为分母。
`bbox_iou` 复用逐例匹配规则汇总框级 Precision/Recall/F1，不把逐例 F1 均值当总体指标；
空分母记 0，非法预测单列 invalid_predictions。指标位于 metrics.json 的 rubric_metrics，
HTML 的“评测指标（全部样本）”可展开查看；请求延迟另见 mean_latency_seconds。
开放式生成仅评价可确定性验证的部分，交付时说明覆盖范围；不使用语义 Judge。
