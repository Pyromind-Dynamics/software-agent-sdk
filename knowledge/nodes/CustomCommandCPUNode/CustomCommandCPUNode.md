# Custom Command CPU 节点

![alt text](/imgs/CustomCommandCPUNode/CustomCommandCPUNode.png)

## 1.1 功能概述

使用可配置的 CPU 和内存资源执行自定义 shell 命令（不申请 GPU），并将命令输出作为字符串返回。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| command | STRING | 是 | `"echo \"hello from custom command\"\necho \"param=[$param]\""` | — | 要执行的 shell 命令 |
| cpu | INT | 是 | `2` | 最小 1；最大 64 | CPU 核数限制（1–64） |
| memory | INT | 是 | `8` | 最小 1；最大 256 | 内存限制，单位 GiB（1–256） |
| param | STRING | 否 | `"param-value"` | — | 传递给 shell 命令的可选参数，命令中通过 `$param` 引用 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| result | STRING | 命令捕获到的标准输出和标准错误。 |
