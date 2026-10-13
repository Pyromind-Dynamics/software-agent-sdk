# URL Check 节点

![alt text](/imgs/CheckUrlNode/CheckUrlNode.png)  

## 1.1 功能概述

Check URL 节点：检查URL是否可用

## 1.2 输入类型

| 参数 | 数据类型 | 必填 | 描述 |
|------|---------|------|------|
| url | STRING | 是 | 需要测试的url地址。默认值：`http://localhost:3000` |
| max_retries | INT | 是 | 最大重试次数。默认值：`30` |
| retry_interval | INT | 是 | 重试间隔s。默认值：`2` |
## 1.3 输出类型

| 参数 | 数据类型 | 描述 |
|------|---------|------|
| endpoint | STRING | Successed/Failed |
