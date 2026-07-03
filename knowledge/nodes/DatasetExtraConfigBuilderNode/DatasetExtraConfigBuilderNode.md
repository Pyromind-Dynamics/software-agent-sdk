# Dataset Extra Config Builder 节点

![alt text](/imgs/DatasetExtraConfigBuilderNode/DatasetExtraConfigBuilderNode.png)

## 1.1 功能概述

构建 `dataset_extra_config` YAML 片段（collator 入口、最大样本数、最大序列长度等）。输出连接至 `DatasetConfigBuilderNode.dataset_extra_config`。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| train_max_samples | INT | 否 | 最大训练样本数（0 表示全部）。默认值：`0` |
| val_max_samples | INT | 否 | 最大验证样本数（0 表示全部）。默认值：`0` |
| sft_collator_entry | STRING | 否 | SFT collator 入口（`module:function`）。默认值：`train.sft_collator:make_collate_fn` |
| dpo_collator_entry | STRING | 否 | DPO collator 入口（`module:function`）。默认值：`train.dpo_collator:make_collate_fn` |
| grpo_collator_entry | STRING | 否 | GRPO collator 入口（`module:function`）。默认值：`train.data.default_vision_grpo_collate:create_grpo_collate_fn` |
| max_seq_length | INT | 否 | 最大序列长度。默认值：`4096` |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| dataset_extra_config | STRING | 来自 Dataset Extra Config Builder 节点的额外配置 YAML |

## 1.4 Workflow JSON 定义

完整 workflow 定义见 [`workflow/DatasetExtraConfigBuilderNode/DatasetExtraConfigBuilderNode.json`](../../workflow/DatasetExtraConfigBuilderNode/DatasetExtraConfigBuilderNode.json)。

## 1.5 运行 Workflow

```bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/DatasetExtraConfigBuilderNode/DatasetExtraConfigBuilderNode.json [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--name` | `datasetExtraConfigBuilderNode` | 任务名称 |
| `--output` | - | 输出结果文件 |
| `--poll-interval` | 5 | 轮询间隔（秒） |
| `--timeout` | 600 | 最大等待时间（秒） |
| `--pretty` | false | 美化 JSON 输出 |
| `--max-retries` | 0 | API 请求最大重试次数 |
| `-h, --help` | - | 显示帮助信息 |
