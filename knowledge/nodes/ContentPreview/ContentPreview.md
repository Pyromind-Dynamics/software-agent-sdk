# Content Preview 节点

![alt text](/imgs/ContentPreview/ContentPreview.png)

## 1.1 功能概述

按照指定文件格式展示输入内容。可用于将文本类内容按 `txt`、`json`、`xml` 或 `python` 格式预览。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| content | * | 是 | 要展示的内容 |
| file_format | STRING | 是 | 预览文件格式。默认值：`txt` Options: txt, json, xml, python |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| content | * | 原始预览内容 |
