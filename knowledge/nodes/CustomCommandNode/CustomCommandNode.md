# Custom Command 节点

![alt text](/imgs/CustomCommandNode/CustomCommandNode.png)

## 1.1 功能概述

使用可配置的 CPU、内存和 GPU 资源执行自定义 shell 命令，并将命令输出作为字符串返回。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| command | STRING | 是 | 要执行的 shell 命令。默认值：echo hello from custom command |
| cpu | INT | 是 | CPU 核数限制（1–64）。默认值：4 |
| memory | INT | 是 | 内存限制，单位 GiB（1–256）。默认值：32 |
| gpu_count | INT | 是 | 要申请的 GPU 数量（0–8）。默认值：0 |
| gpu_product | STRING | 是 | gpu_count 大于零时要申请的 GPU 产品型号，必须与集群匹配：`us-west-1` 支持 `NVIDIA-H100-NVL`、`NVIDIA-L40S`；`us-west-2` 支持 `NVIDIA-H200`、`NVIDIA-H100-80GB-HBM3`。 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| result | STRING | 命令捕获到的标准输出和标准错误。 |

## 1.4 Workflow JSON 定义

已脱敏的单节点 workflow 示例见 [workflow/CustomCommandNode/CustomCommandNode.json](../../workflow/CustomCommandNode/CustomCommandNode.json)。

## 1.5 运行 Workflow

~~~bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/CustomCommandNode/CustomCommandNode.json --pretty
~~~
