# LLM Test 节点

![alt text](/imgs/TestLLMNode/TestLLMNode.png)  

## 1.1 功能概述

Test Sglang Node：测试 sglang endpoint 的节点

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| endpoint | STRING | 是 | 字符串。默认值：`http://localhost:30000` |
| prompt | STRING | 是 | 字符串。默认值：`Hello, how are you?` |
| max_tokens | INT | 是 | 整数。默认值：`100` |
| temperature | FLOAT | 是 | 浮点数。默认值：`0.7` |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| result | STRING | 字符串 |
