# AVI 数据契约

## 图片角色

参考 AVI demo 和 trajectory_viewer 的 defect 场景，首期约定三张输入图：

| 角色 | 含义 |
|---|---|
| `defect` | 当前待检缺陷图 |
| `diff` | 与参考图或上一版图像差分后的结果 |
| `gt` | 人工确认或用于对照的真值图 |

角色顺序不能只依赖数组位置。JSONL 中应保留显式角色字段，或在
`image_roles` 中按相同顺序声明。缺少某个角色时在需求阶段确认是否可以省略。

## 标签

首期支持：

```text
pass
fail
defect
```

允许用户使用同义标签，但必须在 `label_schema` 中给出到上述三类的映射。不要把
`NG`、`bad`、`异常` 等别名直接写入 Graph 配置而不说明映射。

## 推荐训练样本

```json
{
  "sample_id": "report_001",
  "images": {
    "defect": "assets/report_001/defect.jpg",
    "diff": "assets/report_001/diff.jpg",
    "gt": "assets/report_001/gt.jpg"
  },
  "system_prompt": "你是 PCB AVI 图片质检模型。",
  "user_prompt": "判断缺陷图中的结果为 pass、fail 或 defect，并说明原因。",
  "gt": "{\"result\":\"fail\",\"reason\":\"存在划痕\"}",
  "rejected_answer": null
}
```

`rejected_answer` 只在 DPO 数据中需要。SFT 数据可以为空或省略。

## ATIF

评测轨迹可以参考 trajectory_viewer 的 ATIF-v1.7 结构。每条样本至少保留：

- `trajectory_id`
- 用户步骤中的图片及 `image_roles`
- 被测模型输出
- `ground_truth`
- `gt_result` 与 `pred_result`
- reason、box IoU、reward（存在时）

ATIF 是评测和可视化产物，不替代训练 JSONL。两者通过 `sample_id` 和数据集版本关联。

## 首期边界

- 不在 Skill 内执行数据清洗。
- 不要求立刻生成 `train.jsonl`、manifest 或 split 文件。
- 不在 Project 创建请求中写入本机绝对路径或签名媒体 URL。
- 在线 feeder、span 封批和持续采集留到后续阶段。
