# Save Model 节点

![alt text](/imgs/SaveModel/SaveModel.png)  

## 1.1 功能概述

保存训练好的RL模型

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| model | STRING | 是 | 未声明 | — | 字符串 |
| save_path | STRING | 是 | `"path/to/save/model"` | — | 字符串 |
| model_name | STRING | 是 | `"rl_model"` | — | 字符串 |
| save_format | STRING | 是 | `"pkl"` | 可选值：`"pkl"`、`"onnx"`、`"torchscript"`、`"safetensors"` | 字符串 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| save_path | STRING | 字符串 |
