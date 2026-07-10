---
name: debug-workflow
description: >-
  当用户要求对当前 Pyromind 工作流进行测试/调试/debug/试跑（在平台上真实执行一次并根据报错修复）
  时使用此技能。用户可能会用"测试"“test”“调试”“debug”“试跑”等任意说法表达同一个诉求：把工作流提交
  到平台真实跑一次。驱动"触发测试 → 读取报错 → 局部修改 → 再次测试"的循环，直到通过或达到最大重试次数。
triggers:
- debug
- test
- 测试
- 测试工作流
- 测试一下
- test workflow
- run the workflow
- 调试工作流
- 调试一下
- 试跑
- 运行一下工作流
- debug workflow
---

# 测试 / 调试 Pyromind 工作流

## 概述

用户说"测试”“test”“调试”“debug”“试跑”这个工作流，指的都是同一件事：调用 `debug_workflow`
工具，把当前工作目录下的 `workflow.py` 提交到 Pyromind 平台**真实执行**一次，并阻塞等待结果
（通常 30 秒到 2 分钟）。这个工具只负责"触发并拿到这一次的结果"，不会自动重试或修改文件——
生成→测试→读报错→改代码→再测试的整个循环由你（LLM）驱动。

**`workflow.py` 是声明式 DSL，不是可执行的 Python 脚本。** 它借用 Python 语法描述节点及节点
间的连线，但只是数据/配置的声明，本地并没有对应的运行时去执行它：

- 不要尝试用 `execute_bash` 运行 `python workflow.py`，也不要给它添加 `import` 语句
- 不要凭 Python 语法/解释器语义去"推演"它会怎么跑、会不会报错——它不是那样被执行的
- **真正让它跑起来的唯一方式就是调用 `debug_workflow`**；测试通过与否、报错信息都以这个
  工具的返回为准。报错来自平台侧节点的真实执行，定位方式是对照 `error_log` 与相关节点文档
  （`<知识库绝对路径>/nodes/<NodeType>/<NodeType>.md`），而不是分析 Python 调用栈

**`debug_workflow` 和 `validate_workflow_dsl` 是两个不同的工具，报错含义不要混淆：**
`validate_workflow_dsl`（`generate-workflow-dsl` 技能里生成/修改后会自动调用）做的是几秒
内返回的静态结构校验（节点/端口/类型/DAG），不会真正执行；`debug_workflow` 才是提交到平台
**真实执行**一次。`debug_workflow` 返回 `failed` 时的 `error_log` 是运行时报错，通常是
`validate_workflow_dsl` 覆盖不到的问题（例如数据集/模型不可达、显存不足、训练脚本报错等）。

## 前置条件

调用 `debug_workflow` 前，确认当前工作目录下已经存在 `workflow.py`（例如刚通过
`generate-workflow-dsl` 技能生成，或用户已有工作流）。如果不存在，先生成工作流，不要直接
调用 `debug_workflow`。

如果 `workflow.py` 是本轮刚生成/修改、还没有经过 `validate_workflow_dsl` 校验，先调用一次
`validate_workflow_dsl`：结构性问题几秒内就能发现并修复，没必要每次都用一次 30 秒到 2 分钟
的真实执行去发现同样的问题。`validate_workflow_dsl` 通过之后，再调用 `debug_workflow` 进入
下面的真实测试循环。

## 循环步骤

1. 调用 `debug_workflow`（可选带一句 `note` 说明这次改了什么，仅用于你自己的会话记录）
2. 根据返回的 `status` 处理：
   - **`passed`**：工作流已测试通过。用一两句话告诉用户测试成功，然后停止，不要再调用
     `debug_workflow`。
   - **`failed`**：这是平台真实执行产生的运行时报错（不是静态语法检查），`error_log` 字段
     包含具体的报错信息（通常含文件名、行号、异常类型）。
     - 仔细阅读 `error_log`，定位报错对应的具体节点/参数
     - 使用 `apply_patch` **只修改导致报错的那几行**，不要重写整个 `workflow.py`
     - 修改后再次调用 `debug_workflow`
   - **`timeout`**：平台在等待时间内没有返回结果。可以直接再次调用 `debug_workflow` 重试，
     不需要先修改文件。
   - **`error`**：本次调用没有真正提交到平台（例如 `workflow.py` 不存在，或已经用完所有
     测试次数）。不要再调用 `debug_workflow`；按 `error_log`/文本提示处理（生成文件或
     停止并报告）。
3. 观测中的 `attempt`/`max_attempts` 字段是工具自己维护的计数器（当前上限为 10 次），
   达到上限后工具会直接返回 `status="error"` 并拒绝再执行。看到这种情况时，如实向用户说明
   已达到最大重试次数、当前仍存在的报错内容，不要继续调用 `debug_workflow`。

## 修改原则

- 每一轮只做**最小必要修改**：只改报错指向的节点参数或字段，不要顺带重构其它节点
- 不要因为一次报错就大幅改变工作流结构（例如更换节点类型），除非 `error_log` 明确指出
  当前节点类型/组合不支持
- 如果同一个报错连续出现两次以上，说明上一次的修改没有真正解决问题，重新读一遍
  `error_log` 和相关节点文档（`<知识库绝对路径>/nodes/<NodeType>/<NodeType>.md`），
  不要重复同样的修改

## 与用户的沟通

- 每一轮测试之后，用一两句话简要同步进展（第几次尝试、通过还是失败、失败原因概述），
  不需要把完整的 `error_log` 逐字贴给用户
- 循环期间不要频繁提问打断用户；只有达到最大次数、或报错明显超出你能修复的范围（例如提示
  平台侧资源/权限问题）时才停下来询问用户下一步怎么做
