# Save Anything To File 节点

![alt text](/imgs/SaveAnythingToFileNode/SaveAnythingToFileNode.png)

## 1.1 功能概述

Save any provided content to a file path.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| file_path | STRING | 是 | 加载或保存的文件路径 |
| content | * | 否 | 已加载文件内容。默认值：空字符串 |
| append | BOOLEAN | 否 | Append。默认值：false |
| ensure_parent | BOOLEAN | 否 | Ensure parent。默认值：true |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| saved_file_path | STRING | 已保存文件路径 |
| written_bytes | INT | 写入字节数 |
