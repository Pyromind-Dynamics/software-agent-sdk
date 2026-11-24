# 数据来源与路径

## 数据源

| 用户输入 | DSL 入口 |
|---|---|
| Storage 具体文件 | `PathJoinNode → DatasetConfigBuilderNode` |
| Storage 目录 | 确认实际训练/验证文件；需要目录检查时用 `PathJoinNode → LoadDataset` |
| 平台预置数据集 | `CloneAndCacheDataset` |
| Hugging Face/ModelScope 标识 | `DownloadAndCacheDataset` |
| 未提供数据 | 确认数据来源；明确要演示时可使用公开测试集 |

来源以用户说明和已有资产为依据，不仅凭标识包含 `/` 判断。平台或外部数据集不需要先在 Storage 查找副本。

已有信息不足以确定文件、字段映射或媒体路径时，定向查看目录或样本；数据量和序列长度统计按当前决策需要获取，不要求每次建立完整画像。字段与 Builder 契约见 `workflow-contracts.md`。

## 路径语义

用户 Storage 在 Agent、终端和平台节点中统一使用 `/workspace`：

| 使用位置 | 路径示例 |
|---|---|
| Agent 的 `read` / `terminal` / `write` | `/workspace/datasets/my_data/train.jsonl` |
| 训练节点访问挂载数据 | `/workspace/datasets/my_data/train.jsonl` |
| `submit_workflow.workflow_path` | `public_data/workflow_canvas/workflow.py`, or an existing `/workspace/.../workflow.py` |

`public_data/...` 是会话文件路径，`submit_workflow` 直接接受该路径并在执行边界解析；
不要先上传或复制 `workflow.py`。已有 Storage 工作流直接使用 `/workspace/...`，
正式的 Storage 输入和输出统一使用 `/workspace/...`。

默认平铺 Vision collator 直接打开 `image_field` 中的路径，`PathJoinNode` 只定位清单文件。检查实际绑定的 train/val 清单：每个图片引用都应是训练可读的 `/workspace/...` 路径。相对引用按原始图片根转成训练副本并绑定副本；保留多图顺序及监督字段。自定义 collator 按其实际解析基准处理。

## 文件与目录绑定

已有可访问的具体文件时，直接拼接训练路径：

```python
storage_path = PathJoinNode(
    id="1",
    base_path="/workspace/",
    subpath="datasets/my_data/train.jsonl",
)
dataset_config = DatasetConfigBuilderNode(
    id="2",
    train_data_path=storage_path.joined_path,
    dataset_kind_config=dataset_kind.dataset_kind_config,
)
```

`LoadDataset.source_dir` 只接目录，具体文件从其 `dataset_path` 拼接。PathJoin 的 `subpath`
使用 Storage 相对目录或文件名，去掉已知挂载根，避免绝对路径覆盖基础目录或重复拼接。
train、val 共用数据根时可复用该根。

Clone/Download 输出 `dataset_path`，可直接传给 `DatasetConfigBuilderNode`；需要目录内具体文件时
追加 `PathJoinNode(base_path=<dataset_path>, subpath=<file>)`。多文件合并或格式转换取决于实际输入契约与训练范围，不是默认准备阶段。
