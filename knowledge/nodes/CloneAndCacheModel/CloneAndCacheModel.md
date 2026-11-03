# Clone Model 节点

![alt text](/imgs/CloneAndCacheModel/CloneAndCacheModel.png)  

## 1.1 功能概述

从 pyromind 环境中的预置模板快速克隆模型到工作区缓存。模型已存储在环境中，克隆速度极快。克隆后的模型可被下游训练或推理节点使用。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| model | STRING | 是 | 模型标识，必须从当前环境的可用模板列表中选择。 |
| target_path | STRING | 是 | 工作区中克隆模型的目标目录。默认值：`/workspace/models/` Options: /workspace/models/ |

截至 2026-10-27 核对，`us-west-1` 环境的 Clone Model 下拉列表包含：

- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-14B`
- `Qwen/Qwen3-235B-A22B-FP8`
- `Qwen/Qwen3-32B`
- `Qwen/Qwen3-32B-FP8`
- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-VL-235B-A22B-Instruct`
- `Qwen/Qwen3-VL-32B-Instruct`
- `Qwen/Qwen3-VL-4B-Instruct`
- `openai/clip-vit-base-patch32`
- `openpi/big_vision`
- `openpi/pi05_base`
- `pretrained/gui-ck900-lora`
- `stabilityai/stable-video-diffusion-img2vid`

模板列表在运行时从部署环境加载，其他环境可能不同，应以当前环境的下拉列表为准。
不在列表中的模型不能使用本节点，应改用 `DownloadAndCacheModel`。

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| model_path | STRING | 克隆到工作区缓存的模型引用，可传递给下游接受模型输入的训练或推理节点。 |
