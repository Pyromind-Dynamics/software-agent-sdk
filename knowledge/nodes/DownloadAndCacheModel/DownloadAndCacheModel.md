# Download Model 节点

![alt text](/imgs/DownloadAndCacheModel/DownloadAndCacheModel.png)  

## 1.1 功能概述

下载大模型资源

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述                                                    |
|------|---------|------|-------------------------------------------------------|
| modelname | STRING | 是 | 模型资源路径  默认值 : `Qwen/Qwen2.5-1.5B-Instruct` |
| cache_dir | STRING | 是 | 目标文件夹。  默认值 : `/workspace/models/Qwen/Qwen2.5-1.5B-Instruct` |
| download_source | STRING | 是 | 下载源。默认值：`huggingface` Options: huggingface, modelscope |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| model_path | STRING | 模型引用 |
