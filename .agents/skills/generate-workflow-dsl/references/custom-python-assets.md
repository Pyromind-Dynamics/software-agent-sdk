# 自定义 Metrics 与 Reward

只在平台内置项不能表达业务目标时生成脚本。训练生成场景不生成数据清洗脚本，也不调用
`run_dataset_cleaning`。

## 统一资产链路

1. 根据真实样本和业务目标确定可复现的评分规则与函数名。
2. 用 `file_editor` 在 `public_data/<name>.py` 创建单个脚本；不得写会话根目录、知识库或 Skill 目录。
3. 检查函数签名、返回类型、空输入和异常样本；能本地导入时做一次最小调用。
4. 调用 `upload_file_to_pyromind`，等待成功 observation。
5. 只使用 observation 返回的 `storage_path`，拼成 `<storage_path>:<function_name>`。
6. 上传失败立即停止；不得猜测 `/workspace/...` 路径或先写 DSL 后补传。

## Metrics

调用本文件前应已按阶段契约排除内置 Metric；能用系统 `MetricsConfigBuilderNode` 时不要生成脚本。

自定义指标函数契约：

```python
from typing import Any


def business_metric(
    gt_text: str,
    pred_text: str,
    sample: dict[str, Any],
    *,
    metrics_name: str | None = None,
) -> dict[str, float] | None:
    key = metrics_name or "business_metric"
    score = 1.0 if pred_text.strip() == gt_text.strip() else 0.0
    return {key: score}
```

- 分数通常归一化到 0～1；无法评分的样本可返回 `None`。
- 返回 dict 必须含 `MetricsConfigBuilderCustomNode.name` 对应的键。
- `entry` 填上传结果，例如 `/returned/path/metric.py:business_metric`。

## Reward

调用本文件前应已按阶段契约排除内置 Reward；业务自定义项使用
`RewardItemBuilderCustomNode`。

自定义 reward 接收批量模型输出并返回等长分数列表：

```python
from typing import Any


def business_reward(
    completions: list[Any],
    ground_truth: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    references = ground_truth or [""] * len(completions)
    return [
        1.0 if str(completion).strip() == str(reference).strip() else 0.0
        for completion, reference in zip(completions, references)
    ]
```

- 相同输入必须得到相同分数；不要调用外部不可控服务或读取明文凭证。
- 返回长度必须与 completions 一致；处理嵌套 completion 前先规范化结构。
- `entry` 使用上传返回路径；`kwargs` 非空时代表 factory 参数 YAML。
- 多个 reward 通过 `RewardConfigBuilderNode.reward_item_1...reward_item_5` 组合。

## DSL 回填

```python
metrics = MetricsConfigBuilderCustomNode(
    id="40",
    entry="/returned/path/metric.py:business_metric",
    name="business_metric",
)
reward = RewardItemBuilderCustomNode(
    id="41",
    entry="/returned/path/reward.py:business_reward",
    name="business_reward",
    weight=1.0,
)
```

路径必须与工具 observation 完全一致；不要把 Agent 工作区本地文件路径填入节点。
