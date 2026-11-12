# Download Dataset 节点

![alt text](/imgs/DownloadAndCacheDataset/DownloadAndCacheDataset.png)  

## 1.1 功能概述

从远程仓库下载指定数据集，缓存到本地目录，并输出数据集路径供下游节点使用。

- **数据源选择**：支持 `huggingface`、`modelscope` 两种下载源
- **本地缓存**：将数据集下载到 `cache_dir` 指定的目录；若目录不存在会自动创建
- **路径输出**：返回缓存后的 `dataset_path`，供 LoadDataset、数据预处理或训练节点使用

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| dataset_name | STRING | 是 | `"openai/gsm8k"` | — | 数据集名称或仓库 ID |
| cache_dir | PATH | 是 | `"/workspace/datasets/openai/gsm8k"` | — | 本地缓存目录 |
| download_source | STRING | 是 | `"huggingface"` | 可选值：`"huggingface"`、`"modelscope"` | 下载源 |
| filename | STRING | 否 | `""` | — | 仅下载仓库内指定文件；空字符串表示下载整个仓库 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| dataset_path | STRING | 下载完成后数据集的本地路径。例如 `cache_dir=/workspace/datasets/openai/gsm8k` 时，输出 `/workspace/datasets/openai/gsm8k` |
