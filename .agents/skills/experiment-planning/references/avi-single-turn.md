# AVI 单轮实验 reference

用于 PCB AVI/AOI 图片质检、单轮 SFT/DPO、缺陷图与 GT 对照类实验。本文是场景
reference，不覆盖其他业务场景，也不修改通用 `data-processing`、`sandbox` 或训练能力。

## 需求清单

| 内容 | 需要确认的信息 | 示例 |
|---|---|---|
| 数据位置 | 用户可访问的 Storage 路径和样本入口 | `/target-workspace/datasets/avi/train.jsonl` |
| 数据格式 | 文件格式及单条样本结构 | JSONL + images/prompts/gt |
| 图片角色 | 每类图片的业务含义和顺序 | defect、diff、gt |
| 标签 | 标签枚举和别名映射 | pass、fail、defect |
| 划分 | train/val/test 来源和批次隔离 | test 不参与选模 |
| 模型 | 被测基模或已有模型入口 | Qwen3-VL 系列 |
| 训练方式 | SFT 或 DPO | sft |
| 评测 | 确定性评分规则和主指标 | result 匹配；主指标 fail_recall |
| 交付物 | 本轮必须产出的模型或报告 | LoRA checkpoint + ATIF 报告 |

按条件补充：

- DPO 必须说明 chosen/rejected 的来源和质量差异；
- 有框或区域判断时确认坐标口径和 IoU 阈值；
- reason 参与评分时明确匹配方式，否则只做展示；
- 多批次数据确认重复样本和 split 泄漏检查；
- 上线门禁需要候选和 baseline 的比较阈值。

## 数据契约

首期图片角色：

| 角色 | 含义 |
|---|---|
| `defect` | 当前待检缺陷图 |
| `diff` | 与参考图或上一版图像差分后的结果 |
| `gt` | 人工确认或用于对照的真值图 |

角色不能只依赖数组位置。JSONL 应保留显式角色字段，或在 `image_roles` 中按相同
顺序声明；缺少角色时先确认是否可以省略。

首期标签为 `pass`、`fail`、`defect`。用户可以使用同义标签，但必须在
`label_schema` 中给出映射，不要把 `NG`、`bad`、`异常` 等别名直接写入 Graph。

推荐样本：

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

`rejected_answer` 只在 DPO 数据中需要。开发态通过 `data-processing` 检查真实文件、
清洗、转换和验证数据，不在本 reference 中执行数据处理。

## Graph 蓝图

计划中的 `skill` 仍为 `experiment-planning`。本场景使用四个阶段：

```text
dataset -> inference -> evaluator -> train
```

- `dataset`：检查、清洗、转换并验证真实 Storage 数据，绑定 `data-processing`；
- `inference`：用选定模型对固定评测集推理，记录预测和模型版本；
- `evaluator`：按确定性规则评分并生成标准评估产物；
- `train`：基于已验证数据和评测门禁执行 SFT 或 DPO。

节点配置从计划中的 `resources` 和 `tasks` 生成。首期尚未接入真实业务的执行器使用
`contract_probe`，只证明图、Run、状态和 Artifact 契约可执行，不产生业务成功结论。

## 评测口径

每条样本至少保留：

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

首期默认：

- 主指标为 `fail_recall`；
- `fail_recall` 持平时用严格 `accuracy` 兜底；
- 同时保留 `result_accuracy`、分类 Precision/Recall/F1 和 confusion matrix；
- 有框时记录 box Precision/Recall/F1 和 IoU；
- 不把 `pass_rate` 或训练 loss 当作默认上线门禁。

同一 predictions 重复评分必须得到相同结果。Evaluator 不调用被测模型，也不根据模型
自述 reason 判定正确，除非用户明确把 reason 纳入规则。

评测结果和轨迹使用标准 ATIF/ATIF-like 结构，至少支持图片回看、GT/prediction 对照、
分类混淆、badcase 筛选和训练前后同口径比较。ATIF 是评测和可视化产物，不替代训练
JSONL；两者通过 `sample_id` 和数据集版本关联。

## 首期边界

- 在线 feeder、Kafka、checkpoint 轮询、热切换和持续采集暂不进入本参考 Graph；
- 不在 Project 创建请求中写入本机绝对路径、凭据或签名媒体 URL；
- 不把 AVI 专属字段和规则加入通用数据处理实现或后端 Graph 路由。
