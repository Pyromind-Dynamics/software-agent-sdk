# Download File 节点

![alt text](/imgs/DownloadURLNode/DownloadURLNode.png)

## 1.1 功能概述

从 URL 下载文件到本地工作区路径。节点支持分片并行 Range 下载与断点续传，严格校验 HTTP 206 部分内容响应，并将分片合并为单个输出文件。

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| url | STRING | 是 | 远程文件 URL。 |
| output_path | STRING | 是 | 合并后文件写入的本地路径。 |
| expected_sha256 | STRING | 否 | 可选的预期 SHA-256 校验值。设置后会对下载文件进行校验。 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| path | STRING | 下载完成后的本地文件路径。 |
