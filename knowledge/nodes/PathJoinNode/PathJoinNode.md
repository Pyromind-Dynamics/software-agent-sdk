# Join Path 节点

![alt text](/imgs/PathJoinNode/PathJoinNode.png)  

## 1.1 功能概述

路径拼接工具节点：将 `base_path` 与 `subpath` 拼接为完整的路径字符串，供下游节点使用。
`base_path` 与 `subpath` 可来自上游节点输出，也可在节点配置中直接填写
- **典型场景**：将工作区根目录（如 `/workspace/`）与相对子目录（如 `models`、`datasets/gsm8k`）组合，生成模型、数据集或输出文件的完整路径

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| base_path | STRING | 是 | 基础路径（目录前缀）。默认值：`""`。示例：`/workspace/` |
| subpath | STRING | 是 | 待拼接的子路径（相对路径或目录名）。默认值：`""`。示例：`models` |
| environment | ENV | 否 | 可选的环境变量配置。默认值：- |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| joined_path | STRING | 拼接后的完整路径。例如 `base_path=/workspace/`、`subpath=models` 时输出 `/workspace/models` |
