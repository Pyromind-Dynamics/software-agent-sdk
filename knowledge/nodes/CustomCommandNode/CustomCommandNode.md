# Custom Command 节点

![alt text](/imgs/CustomCommandNode/CustomCommandNode.png)

## 1.1 功能概述

使用可配置的 CPU、内存和 GPU 资源执行自定义 shell 命令，并将命令输出作为字符串返回。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| command | STRING | 是 | `"echo \"hello from custom command\"\necho \"param=[$param]\""` | — | 要执行的 shell 命令 |
| cpu | INT | 是 | `4` | 最小 1；最大 64 | CPU 核数限制（1–64） |
| memory | INT | 是 | `32` | 最小 1；最大 256 | 内存限制，单位 GiB（1–256） |
| gpu_count | INT | 是 | `1` | 最小 1；最大 8 | 要申请的 GPU 数量（1–8） |
| gpu_product | STRING | 是 | `"NVIDIA-H100-NVL"` | 可选值：`"NVIDIA-H100-NVL"`、`"NVIDIA-L40S"` | gpu_count 大于零时要申请的 GPU 产品型号 |
| param | STRING | 否 | `"param-value"` | — | 传递给 shell 命令的可选参数，命令中通过 `$param` 引用 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| result | STRING | 命令捕获到的标准输出和标准错误。 |
