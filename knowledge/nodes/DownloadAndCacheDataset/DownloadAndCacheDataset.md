# Download Dataset 节点

![alt text](/imgs/DownloadAndCacheDataset/DownloadAndCacheDataset.png)  

## 1.1 功能概述

从远程仓库下载指定数据集，缓存到本地目录，并输出数据集路径供下游节点使用。

- **数据源选择**：支持 `huggingface`、`modelscope` 两种下载源
- **本地缓存**：将数据集下载到 `cache_dir` 指定的目录；若目录不存在会自动创建
- **路径输出**：返回缓存后的 `dataset_path`，供 LoadDataset、数据预处理或训练节点使用

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| dataset_name | STRING | 是 | 数据集名称或仓库 ID。默认值：`openai/gsm8k`。示例：`openai/gsm8k` |
| cache_dir | PATH | 是 | 本地缓存目录。默认值：`/workspace/datasets/openai/gsm8k` |
| download_source | STRING | 是 | 下载源。默认值：`huggingface`。可选：`huggingface`、`modelscope` |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| dataset_path | STRING | 下载完成后数据集的本地路径。例如 `cache_dir=/workspace/datasets/openai/gsm8k` 时，输出 `/workspace/datasets/openai/gsm8k` |
