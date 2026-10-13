# Clone Model 节点

![alt text](/imgs/CloneAndCacheModel/CloneAndCacheModel.png)  

## 1.1 功能概述

从 pyromind 环境中的预置模板快速克隆模型到工作区缓存。模型已存储在环境中，克隆速度极快。克隆后的模型可被下游训练或推理节点使用。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| model | STRING | 是 | 模型标识。从 pyromind 环境中可用的模型模板动态列表中选取（例如 `Qwen/Qwen3-0.6B`、`Qwen/Qwen3-8B`）。列表在运行时从模板目录加载。 |
| target_path | STRING | 是 | 工作区中克隆模型的目标目录。默认值：`/workspace/models/` Options: /workspace/models/ |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| model_path | STRING | 克隆到工作区缓存的模型引用，可传递给下游接受模型输入的训练或推理节点。 |
