# Download Model 节点

![alt text](/imgs/DownloadAndCacheModel/DownloadAndCacheModel.png)  

## 1.1 功能概述

下载大模型资源

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| modelname | STRING | 是 | `"Qwen/Qwen2.5-1.5B-Instruct"` | — | 模型资源路径 |
| cache_dir | STRING | 是 | `"/workspace/models/Qwen/Qwen2.5-1.5B-Instruct"` | — | 目标文件夹 |
| download_source | STRING | 是 | `"huggingface"` | 可选值：`"huggingface"`、`"modelscope"` | 下载源 |
| filename | STRING | 否 | `""` | — | 仅下载仓库内指定文件；空字符串表示下载整个仓库 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| model_path | STRING | 模型引用 |
