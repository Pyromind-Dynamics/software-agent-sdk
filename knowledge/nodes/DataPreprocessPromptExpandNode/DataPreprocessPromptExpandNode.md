# Prompt Augmentation 节点

![alt text](/imgs/DataPreprocessPromptExpandNode/DataPreprocessPromptExpandNode.png)

## 1.1 功能概述

Run data_preprocess prompt-expand pipeline and emit output path.

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| input_path | STRING | 是 | 未声明 | — | Input path |
| output_path | STRING | 是 | 未声明 | — | 持久化输出目录或文件路径，使用 `/workspace/...` |
| api_base | STRING | 是 | 未声明 | — | Api base |
| model | STRING | 是 | 未声明 | — | Model |
| api_key | STRING | 是 | 未声明 | — | Api key |
| user_prompt | STRING | 否 | 未声明 | — | User prompt |
| user_prompt_column | STRING | 否 | `""` | — | User prompt column |
| format_lang | STRING | 否 | `"en"` | — | Format lang |
| system_prompt | STRING | 否 | `""` | — | System prompt |
| max_tokens | INT | 否 | `1024` | — | 每次 API 调用的最大 token 数 |
| thinking | BOOLEAN | 否 | `false` | — | Thinking |
| limit | INT | 否 | `0` | — | 校验预览的最大样本数（0 表示不限制） |
| concurrency | INT | 否 | `1` | — | Concurrency |
| num_variations | INT | 否 | `1` | — | Num variations |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| augmentation_path | STRING | Prompt 扩展输出路径 |
