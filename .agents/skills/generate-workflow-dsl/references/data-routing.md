# 数据路由与格式门禁

## 数据源

| 用户输入 | 工具动作 | DSL 入口 |
|---|---|---|
| Storage 相对文件/目录 | 对原路径调用一次 `preview_dataset` | `PathJoinNode → LoadDataset` |
| 平台预置数据集 | 不 preview | `CloneAndCacheDataset` |
| Hugging Face/ModelScope 标识 | 不 preview | `DownloadAndCacheDataset` |
| 未提供数据 | 索要 Storage 路径 | 仅明确要演示时用测试集 |

不要仅凭字符串里含 `/` 判断来源；以用户是否说明“已上传到 Storage”为准。

## Storage 数据画像

从 preview 结果内部记录：

- `preview_file_path`：目录预览实际选中的 Storage 文件；DSL 使用它。
- `sample_file_path`：Agent 工作区内的本地样本副本；只能分析，严禁写进 DSL。
- `num_rows`：完整读取时的 N；为空时把 `previewed_rows` 当样本量下界，不伪装成总条数。
- `p95_sequence_length`：配参用 L；为空时按样本保守估算并说明不确定性。
- `columns`、`sample_rows`、`has_vision`：字段映射、模态和训练类型依据。
- `preview_error`/`error_code`：有结构错误时先处理格式门禁，不猜字段。

Storage 标准链：

```python
storage_path = PathJoinNode(
    id="1",
    base_path="/workspace/",
    subpath="datasets/my_data/train.jsonl",
)
dataset = LoadDataset(id="2", source_dir=storage_path.joined_path)
dataset_config = DatasetConfigBuilderNode(
    id="5",
    train_data_path=dataset.dataset_path,
    dataset_kind_config=dataset_kind.dataset_kind_config,
)
```

如果 preview 的输入是目录，将 `subpath` 换成返回的具体 `preview_file_path`。若返回值仍包含
`/workspace/` 前缀，先去掉该前缀再作为 Storage 相对 `subpath`，避免重复拼接。

Clone/Download 已输出本地 `dataset_path`，可直接传给 `DatasetConfigBuilderNode`；只有需要目录内
具体文件时才追加 `PathJoinNode(base_path=<dataset_path>, subpath=<file>)`。

## 训练格式门禁

满足任一形态才继续：

| 形态 | 必要结构 | Builder |
|---|---|---|
| 文本监督 | 可识别的 prompt/user 与 response/assistant 字段 | `DatasetConfigBuilderTextNode` |
| 对话监督 | `messages` 数组；每项有 `role`、`content` | `DatasetConfigBuilderMessageNode` |
| 多模态 | prompt/messages 中存在 image/video 内容，或独立媒体字段 | `DatasetConfigBuilderVisionNode` |
| DPO | 同一输入有 chosen 与 rejected 回答 | 对应 Builder 的 `rejected_field` |
| GRPO | prompt 加 ground truth/可程序化验证信息，且能定义 reward | 对应 Builder + Reward 配置 |

`messages[].content` 可为字符串，也可为显式内容块数组；内容块必须有 `type`，文本使用 `text`，
图片/视频使用可访问的 `url` 或 `path`。允许 `assistant.tool_calls` 与 `tool` 角色结果。

字段名不必固定。根据真实样本把 `prompt`、`response`、`messages`、`chosen`、`rejected`、
`image`、`ground_truth` 等实际列名填入 Builder，禁止仅按常见名字猜测。

## 训练类型推断

1. 有 chosen/rejected：DPO。
2. 只有 prompt 且存在客观可复现的答案或 reward：GRPO。
3. 其他可监督样本：SFT。
4. 用户的行业背景和业务目标可影响模型、Metric、Reward，但不能覆盖真实数据结构。

## 不合规处理

停止写工作流，列出具体缺口，并给最小目标 JSONL；不得调用清洗工具或换占位数据。

```json
{"messages":[{"role":"user","content":"问题"},{"role":"assistant","content":"答案"}]}
```

```json
{"prompt":"问题","chosen":"更好回答","rejected":"较差回答"}
```

多模态示例：

```json
{"messages":[{"role":"user","content":[{"type":"image","path":"images/a.png"},{"type":"text","text":"描述图片"}]},{"role":"assistant","content":"答案"}]}
```
