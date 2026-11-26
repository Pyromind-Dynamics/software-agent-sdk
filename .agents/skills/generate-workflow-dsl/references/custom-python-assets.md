# 自定义 Metrics 与 Reward

只在平台内置项不能表达业务目标时生成脚本。训练生成场景不生成数据清洗脚本，也不提交
`cleaning/run` 计划或旧清洗 Tool。

## 统一资产链路

1. 根据真实样本和业务目标确定可复现的评分规则与函数名。
2. 用 `write` / `edit` 在 `public_data/<name>.py` 创建单个脚本；不得写会话根目录、知识库或 Skill 目录。
3. 检查函数签名、返回类型、空输入和异常样本；能本地导入时做一次最小调用。
4. 远端 Storage 已挂载为 `/workspace`，先确认真实目标目录，再将脚本写入或复制到该目录。
   用文件大小和 SHA-256 核对目标文件与已验证脚本一致，不再经过上传 Tool。
5. 只使用实际落位并核对过的 `/workspace/.../<name>.py`，拼成 `<storage_path>:<function_name>`。
6. 文件落位或核对失败立即停止；不得猜测路径或先写 DSL 后补传。

本机执行环境没有 Storage 挂载时，只有实际清单仍列出 `upload_file_to_pyromind` 才使用
该入口，并核对其成功结果中的真实 `storage_path`。本机文件不因路径相似就已发布到 Storage。

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
- `entry` 填已核对的 Storage 文件路径，例如 `/workspace/<实际目录>/metric.py:business_metric`。

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
- `entry` 使用已落位并核对的 Storage 路径；`kwargs` 非空时代表 factory 参数 YAML。
- 多个 reward 通过 `RewardConfigBuilderNode.reward_item_1...reward_item_5` 组合。
- 任务已提供冻结评分入口时，直接用原 `entry`，将其以 `weight=1`、空 `kwargs` 接入
  `RewardConfigBuilderNode`，保持 `normalize=False` 并连接到 GRPO 的 `reward_config`。

## DSL 回填

`entry` 填 `<storage_path>:<function_name>`，节点签名查 `workflow-contracts.md` 的 Reward 与
Metric 节。路径必须与已核对的 Storage 文件一致；不要把 `public_data/...` 等 Agent 工作区
本地文件路径填入节点。
