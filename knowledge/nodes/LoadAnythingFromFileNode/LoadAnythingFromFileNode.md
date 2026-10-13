# Load Anything From File 节点

![alt text](/imgs/LoadAnythingFromFileNode/LoadAnythingFromFileNode.png)

## 1.1 功能概述

Load content from a file path.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| file_path | STRING | 是 | 加载或保存的文件路径 |
| encoding | STRING | 否 | Encoding。默认值：`utf-8` |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| loaded_file_path | STRING | 已加载文件路径 |
| content | * | 已加载文件内容 |
| read_bytes | INT | 读取字节数 |
