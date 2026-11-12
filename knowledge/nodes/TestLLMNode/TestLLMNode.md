# LLM Test 节点

![alt text](/imgs/TestLLMNode/TestLLMNode.png)  

## 1.1 功能概述

Test Sglang Node：测试 sglang endpoint 的节点

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 默认值 | 约束 | 描述 |
|---|---|---|---|---|---|
| endpoint | STRING | 是 | `"http://localhost:30000"` | — | 字符串 |
| prompt | STRING | 是 | `"Hello, how are you?"` | — | 字符串 |
| max_tokens | INT | 是 | `100` | 最小 1；最大 4096 | 整数 |
| temperature | FLOAT | 是 | `0.7` | 最小 0.0；最大 2.0 | 浮点数 |

## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|---|---|---|
| result | STRING | 字符串 |
