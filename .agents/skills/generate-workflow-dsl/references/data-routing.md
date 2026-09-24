# 数据路由与格式门禁

## 数据源

| 用户输入 | 工具动作 | DSL 入口 |
|---|---|---|
| Storage 相对文件/目录 | 对原路径读一次 `storage/...` 取数据画像 | `PathJoinNode → LoadDataset` |
| 平台预置数据集 | 不读 Storage | `CloneAndCacheDataset` |
| Hugging Face/ModelScope 标识 | 不读 Storage | `DownloadAndCacheDataset` |
| 未提供数据 | 索要 Storage 路径 | 仅明确要演示时用测试集 |

不要仅凭字符串里含 `/` 判断来源；以用户是否说明“已上传到 Storage”为准。

## Storage 数据画像

同一路径已有画像时复用。只有路径变化、上次读取失败、用户明确要求刷新或结果可能
过期时才重新读取。

画像用 `read`/`terminal` 直接对 `storage/<path>` 取，记录：

- `file_path`：目录输入实际选定的具体 Storage 文件；DSL 使用它。目录先用 `ls`
  看清结构，只读选中的那个文件。
- `num_rows`：完整读取时的 N（如 `wc -l`）；拿不到准确值时说明它是样本量下界，
  不伪装成总条数。
- `p95_sequence_length`：配参用 L；拿不到时按样本保守估算并说明不确定性。
- 字段、样例行、模态：字段映射、模态和训练类型依据。样例用 `head` 或小脚本取，
  不要把整个文件读进上下文。
- 结构错误（非法 JSONL、字段缺失）：先处理格式门禁，不猜字段。

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

如果输入是目录，将 `subpath` 换成目录内实际选定的那个文件。写入 DSL 前将其规范为
Storage 相对路径：去掉可选的 `/workspace/` 前缀和开头 `/`，避免 PathJoin 被绝对路径
覆盖或重复拼接。

Clone/Download 已输出本地 `dataset_path`，可直接传给 `DatasetConfigBuilderNode`；只有需要目录内
具体文件时才追加 `PathJoinNode(base_path=<dataset_path>, subpath=<file>)`。

## 训练格式门禁

满足任一形态才继续：

| 形态 | 必要结构 | Builder |
|---|---|---|
| 文本监督 | 可识别的 prompt/user 与 response/assistant 字段 | `DatasetConfigBuilderTextNode` |
| 对话监督 | `messages` 数组；每项有 `role`、`content` | `DatasetConfigBuilderMessageNode` |
| 多模态 | prompt/messages 中存在 image/video 内容，或独立媒体字段 | `DatasetConfigBuilderVisionNode` |
| 偏好监督 | 同一输入有 chosen 与 rejected 回答 | 对应 Builder 的 `rejected_field` |
| 可验证信号 | prompt 加 ground truth/可程序化验证信息，且能定义 reward | 对应 Builder + Reward 配置 |

`messages[].content` 可为字符串，也可为显式内容块数组；内容块必须有 `type`，文本使用 `text`，
图片/视频使用可访问的 `url` 或 `path`。允许 `assistant.tool_calls` 与 `tool` 角色结果。

字段名不必固定。根据真实样本把 `prompt`、`response`、`messages`、`chosen`、`rejected`、
`image`、`ground_truth` 等实际列名填入 Builder，禁止仅按常见名字猜测。

数据清洗产物若为顶层严格的 `prompt`、`chosen`、`rejected`，直接配置 DPO：
`user_prompt_field=prompt`、`assistant_response_field=chosen`、
`rejected_field=rejected`。不得把该产物改判为 SFT 或 GRPO。本文件只识别数据形态；
训练阶段由主 Skill 决定。

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
