# Save Model 节点

![alt text](/imgs/SaveModel/SaveModel.png)  

## 1.1 功能概述

保存训练好的RL模型

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| model | STRING | 是 | 字符串。默认值：- |
| save_path | STRING | 是 | 字符串。默认值：`path/to/save/model` |
| model_name | STRING | 是 | 字符串。默认值：`rl_model` |
| save_format | STRING | 是 | 字符串。默认值：`pkl` Options: pkl, onnx, torchscript, safetensors |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| save_path | STRING | 字符串 |
