# Dataset Config Builder 节点

![alt text](/imgs/DatasetConfigBuilderNode/DatasetConfigBuilderNode.png)

## 1.1 功能概述

构建 SFT/GRPO/DPO 训练的基础 `dataset_config` YAML 字符串。需配合 **Dataset Kind Config Builder**（Vision / Text / Messages）和 **Dataset Extra Config Builder** 节点，分别提供 `dataset_kind_config` 与 `dataset_extra_config` 输入。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| train_data_path | STRING | 是 | 训练数据路径 |
| val_data_path | STRING | 否 | 验证数据路径。默认值：空字符串 |
| dataset_kind_config | STRING\|DATASET_KIND_CONFIG | 否 | 来自 Kind Config Builder 节点的 dataset kind 配置 YAML。默认值：空字符串 |
| dataset_extra_config | STRING\|DATASET_EXTRA_CONFIG | 否 | 来自 Dataset Extra Config Builder 节点的额外配置 YAML。默认值：空字符串 |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| dataset_config | STRING\|DATASET_CONFIG | 已组装的数据集配置 YAML 字符串 |
## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/DatasetConfigBuilderNode/DatasetConfigBuilderNode.json`](../../workflow/DatasetConfigBuilderNode/DatasetConfigBuilderNode.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/DatasetConfigBuilderNode/DatasetConfigBuilderNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `datasetConfigBuilderNode` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
