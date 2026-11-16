# Python File Processor 节点

![alt text](/imgs/PythonFileProcessorNode/PythonFileProcessorNode.png)

## 1.1 功能概述

执行定义了 process(input_file_path, output_file) 的 Python 脚本。脚本向 output_file 写入文本或字节数据，节点返回生成文件的路径。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| file_path | STRING | 是 | `""` | — | 输入文件路径 |
| output_file_path | STRING | 是 | `"/workspace/output.json"` | — | 生成文件的持久化输出路径，使用 `/workspace/...` |
| python_script_path | STRING | 是 | `""` | — | Python 脚本路径；脚本必须定义 process(input_file_path, output_file) |
| overwrite_output | BOOLEAN | 是 | `false` | — | 是否允许覆盖已存在的输出文件 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| output_file_path | STRING | Python 脚本写入完成后的文件路径。 |
