# Python File Processor 节点

![alt text](/imgs/PythonFileProcessorNode/PythonFileProcessorNode.png)

## 1.1 功能概述

执行定义了 process(input_file_path, output_file) 的 Python 脚本。脚本向 output_file 写入文本或字节数据，节点返回生成文件的路径。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|
| file_path | STRING | 是 | 输入文件路径。 |
| output_file_path | STRING | 是 | 生成文件的输出路径。默认值：/workspace/output.json |
| python_script_path | STRING | 是 | Python 脚本路径；脚本必须定义 process(input_file_path, output_file)。 |
| overwrite_output | BOOLEAN | 是 | 是否允许覆盖已存在的输出文件。默认值：false |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| output_file_path | STRING | Python 脚本写入完成后的文件路径。 |

## 1.4 Workflow JSON 定义

已脱敏的单节点 workflow 示例见 [workflow/PythonFileProcessorNode/PythonFileProcessorNode.json](../../workflow/PythonFileProcessorNode/PythonFileProcessorNode.json)。

## 1.5 运行 Workflow

~~~bash
export PYROMIND_API_KEY=<your-api-key>
python -m pyromind_sdk.test_run_workflow_cli workflow/PythonFileProcessorNode/PythonFileProcessorNode.json --pretty
~~~
