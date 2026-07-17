---
name: debug-workflow
description: >-
  仅当用户要求在不修改节点、模型或参数的前提下，对当前 Pyromind 工作流进行测试/调试/debug/试跑
  （在平台上真实执行一次并根据报错修复）时使用。若请求同时包含换模型、改参数、改结构或生成
  workflow，使用 generate-workflow-dsl，本轮不得调用 workflow_debug。调用 `workflow_debug`
  工具（内部委托 `run_workflow(test_mode=true)`；原 `debug_workflow` / `pyromind_debug` 已停用，
  也不要再直接调 `run_workflow` 的 test_mode）。
  用户可能会用"测试"“test”“调试”“debug”“试跑”等任意说法表达同一个诉求：把工作流提交到平台
  真实跑一次。驱动"触发测试 → Kafka/callback 终态自动续跑会话 → 失败则按报错内容局部修改
  再测；成功则简短通知用户并等待后续输入"。
triggers:
- debug
- test
- 测试
- 测试工作流
- 测试一下
- test workflow
- 调试工作流
- 调试一下
- 试跑
- 试跑工作流
- debug workflow
---

# 测试 / 调试 Pyromind 工作流

边界：请求同时要求修改配置与“跑一下/看看效果”时，先用 `generate-workflow-dsl` 完成 DSL 修改，
本轮停止；不要调用 `workflow_debug`。本 Skill 只接收不含配置修改的显式测试/调试请求。

## 概述

> **迁移说明：** 原 `debug_workflow` 工具（`pyromind_debug` 包）已从 Agent 工具列表移除。
> 测试/调试/试跑统一使用 **`workflow_debug`**（见
> `openhands-tools/openhands/tools/workflow_debug/`）。该工具内部调用
> `run_workflow(..., test_mode=True)`，Agent **不要**再直接对 `run_workflow` 传
> `test_mode=true`。

用户说"测试”“test”“调试”“debug”“试跑"这个工作流，指的都是同一件事：调用
**`workflow_debug`**，把当前工作目录下的 `workflow.py` DSL 提交到 Pyromind 平台**真实
test 执行**一次（平台侧附加 `execution_mode=test`）。

这是**异步提交**：工具在平台接受任务后返回 `task_id`、初始状态（如 `Pending`/`Running`）
和 `keep_ui_lock=true`。提交时 `out_id` 为 `agent1#debug#<conversation_id>`（正式
`run_workflow` 为 `agent1#<conversation_id>`）。平台终态经 **Kafka**
（`StudioWorkflowNotifyHandler` → `deliver_run_workflow_status(..., auto_run=True)`）
注入 `<system_reminder>` 并**自动继续本会话**。

> **重要：** Kafka 消息是通用的。只有带 `debug#` 标记（即由 **`workflow_debug`** 发起）的
> 任务，才执行下面「成功仅提示并等待用户 / 失败重生成 DSL 再测」的约定；正式
> `run_workflow` 终态用通用 resume 指引，不要套用本技能的成功等待逻辑。

## 工具与参数

```python
workflow_debug(
    dsl=<workflow.py 全文>,   # 必填：DSL 源码字符串，不是文件路径
    name="workflow",          # 可选
    note="optional",          # 可选：本轮改动说明
)
```

调用前用 `file_editor` 读取 `workflow.py`，将全文传入 `dsl`。不要用 bash/Python 本地执行
`workflow.py`（声明式 DSL）。

## 与 `validate_workflow_dsl` 的区别

| 工具 | 作用 | 何时使用 |
|------|------|----------|
| `validate_workflow_dsl` | 静态结构校验 | 每次写入/修改 `workflow.py` 后 |
| `workflow_debug` | 平台真实 test；Kafka callback 终态 + auto_run | 用户要求测试/调试/试跑 |
| `run_workflow` | 正式运行/发布（无 `keep_ui_lock`） | 用户要求正式运行/发布 |

## 前置条件

- 工作目录已有 `workflow.py`；否则先生成，不要直接 `workflow_debug`
- 若本轮刚改过且未校验，先 `validate_workflow_dsl`，通过后再 `workflow_debug`

## 循环步骤

1. 读取 `workflow.py`，调用 `workflow_debug(dsl=<全文>)`
2. 提交失败（`is_error` / `Failed`/`Error`）：按 observation 处理，勿盲目重试
3. 提交成功：告知用户已提交、正在等待平台结果；**等待 Kafka callback**（会自动续跑）
4. 收到终态 `<system_reminder>` 后：

   | 终态 | 下一步 |
   |------|--------|
   | **Succeeded** | **仅简短告知用户：本次 test 工作流成功**，然后**停止并等待用户后续输入**；不要再调 `workflow_debug`，也不要主动继续改 DSL |
   | **Failed / Error** | 运行错误多半说明 **DSL 有问题**。根据 `error_log` **重新生成或修复** `workflow.py`（必要时先 `validate_workflow_dsl`），再调用 `workflow_debug` **继续测试** |
   | Terminated | 向用户说明；按 `error_log` 决定是否修复后再测 |
   | Pending / Running | 仍在跑；等待 callback，勿重复提交同一轮 |

5. `attempt`/`max_attempts`（上限 10）：达上限后如实告知用户，停止再调 `workflow_debug`

## 失败时如何改 DSL

- 仔细读 `error_log`，对照 `knowledge/nodes/<NodeType>/<NodeType>.md`
- 优先按报错修复或**按错误语义重生成相关节点/整份 DSL**（use.md：可能需要重新生成）
- 改完校验通过后必须再 `workflow_debug`，形成闭环
- 同一报错连续两轮无进展时，重读文档后再换修法，不要机械重复同一修改

## 成功时如何沟通

- **仅当终态来自 `workflow_debug` 任务时**：只提示一句「本次 test 工作流成功」，然后
  **等待用户下一步指示**（不要自动发布、不要继续改文件、不要再调 `workflow_debug`）
- 若终态来自正式 `run_workflow`，按通用会话指引继续，不要套用本条「仅提示并等待」

## 与用户的沟通（进行中）

- 提交后：说明已提交并等待平台结果
- 失败修复循环中：一两句进展即可，不必整段粘贴 `error_log`
- 仅在达最大次数或明显是平台资源/权限问题时再向用户确认下一步
