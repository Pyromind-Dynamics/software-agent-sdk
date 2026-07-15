---
name: debug-workflow
description: >-
  当用户要求对当前 Pyromind 工作流进行测试/调试/debug/试跑（在平台上真实执行一次并根据报错修复）
  时使用此技能。调用 `run_workflow` 工具并设置 `test_mode=true`（原 `debug_workflow` /
  `pyromind_debug` 已停用）。用户可能会用"测试"“test”“调试”“debug”“试跑”等任意说法表达同一个
  诉求：把工作流提交到平台真实跑一次。驱动"触发测试 → 等待 callback 终态 → 读取报错 → 局部修改
  → 再次测试"的循环，直到通过或达到最大重试次数。
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

## 概述

> **迁移说明：** 原 `debug_workflow` 工具（`pyromind_debug` 包）已从 Agent 工具列表移除。
> 测试/调试/试跑统一使用 **`run_workflow(test_mode=true)`**（见
> `openhands-tools/openhands/tools/workflow/run_workflow.py`）。

用户说"测试”“test”“调试”“debug”“试跑”这个工作流，指的都是同一件事：调用 **`run_workflow`**
工具，并在 `RunWorkflowAction` 中设置 **`test_mode=true`**，把当前工作目录下的 `workflow.py`
DSL 提交到 Pyromind 平台**真实 test 执行**一次（平台侧附加 `execution_mode=test`）。

与正式运行（`run_workflow(test_mode=false)`）相同，这是**异步提交**：工具在平台接受任务后
返回 `task_id`、初始状态（如 `Pending`/`Running`）和 `keep_ui_lock=true`；终态结果通过平台
callback 以 `<system_reminder>` 注入会话并触发 Agent 继续处理。

生成→测试→读报错→改代码→再测试的整个循环由你（LLM）驱动；`run_workflow` 每次调用只负责提交
一次 test run。

## 工具与参数

使用 **`run_workflow`**（不是已停用的 `debug_workflow`）：

```python
run_workflow(
    dsl=<workflow.py 全文>,   # 必填：DSL 源码字符串，不是文件路径
    test_mode=True,           # 必填：测试/调试/试跑必须为 true
    name="workflow",          # 可选：提交到平台的工作流名称
    note="optional",          # 可选：本轮改动说明，仅写入会话历史
)
```

调用前用 `file_editor` 读取 `workflow.py`，将全文传入 `dsl`。

**`workflow.py` 是声明式 DSL，不是可执行的 Python 脚本。** 它借用 Python 语法描述节点及节点
间的连线，但只是数据/配置的声明，本地并没有对应的运行时去执行它：

- 不要尝试用 `execute_bash` 运行 `python workflow.py`，也不要给它添加 `import` 语句
- 不要凭 Python 语法/解释器语义去"推演"它会怎么跑、会不会报错——它不是那样被执行的
- **真正让它跑起来的方式是 `run_workflow(test_mode=true)`**；测试通过与否、报错信息以
  callback 终态和 observation 中的 `error_log` 为准。报错来自平台侧节点的真实执行，定位方式
  是对照 `error_log` 与相关节点文档
  （`knowledge/nodes/<NodeType>/<NodeType>.md`），而不是分析 Python 调用栈

## 与 `validate_workflow_dsl` 的区别

**`run_workflow(test_mode=true)` 和 `validate_workflow_dsl` 是两个不同的工具，报错含义不要混淆：**

| 工具 | 作用 | 何时使用 |
|------|------|----------|
| `validate_workflow_dsl` | 静态结构校验（节点/端口/类型/DAG），秒级返回 | 每次写入/修改 `workflow.py` 后（`generate-workflow-dsl` 技能） |
| `run_workflow(test_mode=true)` | 提交平台真实 test 执行，异步 callback 返回终态 | 用户明确要求测试/调试/试跑时（本技能） |
| `run_workflow(test_mode=false)` | 正式异步运行/发布 | 用户要求运行/发布/执行生产任务时（非本技能） |

`validate_workflow_dsl` 通过只说明结构合法；`run_workflow(test_mode=true)` 终态 `Failed`/`Error`
时的 `error_log` 是运行时报错（数据集不可达、显存不足、训练脚本报错等）。

## 前置条件

调用 `run_workflow(test_mode=true)` 前，确认当前工作目录下已经存在 `workflow.py`（例如刚通过
`generate-workflow-dsl` 技能生成，或用户已有工作流）。如果不存在，先生成工作流，不要直接调用
`run_workflow`。

如果 `workflow.py` 是本轮刚生成/修改、还没有经过 `validate_workflow_dsl` 校验，先调用一次
`validate_workflow_dsl`：结构性问题几秒内就能发现并修复，没必要每次都用平台真实执行去发现
同样的问题。`validate_workflow_dsl` 通过之后，再调用 `run_workflow(test_mode=true)` 进入下面的
真实测试循环。

## 循环步骤

1. 读取 `workflow.py`，调用 `run_workflow(dsl=<全文>, test_mode=true)`（可选 `note` 说明
   这次改了什么，仅用于你自己的会话记录）
2. 若提交 observation 的 `status` 为 `Failed`/`Error` 或 `is_error=true`：说明任务未成功提交，
   按 `error_log`/文本提示处理，不要盲目重复重试。
3. 提交成功后告知用户任务已提交、正在等待平台结果；等待 platform callback（会话会收到
   `<system_reminder>` 描述终态）。根据终态处理：

   | 终态 | 含义 | 下一步 |
   |------|------|--------|
   | `Succeeded` | test run 通过 | 告诉用户测试成功，停止，不再调用 `run_workflow(test_mode=true)` |
   | `Failed` / `Error` | 平台真实运行时报错 | 读 `error_log` → `apply_patch` 最小修改 → 再测 |
   | `Terminated` | 工作流被停止 | 向用户说明，按 `error_log` 决定是否重试 |
   | `Pending` / `Running` | 仍在运行 | 等待 callback，不要重复提交同一轮测试 |

4. `Failed`/`Error` 处理细节：
   - 仔细阅读 `error_log`，定位报错对应的具体节点/参数
   - 使用 `apply_patch` **只修改导致报错的那几行**，不要重写整个 `workflow.py`
   - 修改后再次调用 `run_workflow(test_mode=true)`

5. 观测中的 `attempt`/`max_attempts` 是工具维护的计数器（当前上限 10 次）。达到上限后工具会
   返回 `status="Error"` 并拒绝再执行。如实向用户说明已达最大重试次数，不要继续调用
   `run_workflow(test_mode=true)`。

## 修改原则

- 每一轮只做**最小必要修改**：只改报错指向的节点参数或字段，不要顺带重构其它节点
- 不要因为一次报错就大幅改变工作流结构（例如更换节点类型），除非 `error_log` 明确指出
  当前节点类型/组合不支持
- 如果同一个报错连续出现两次以上，说明上一次的修改没有真正解决问题，重新读一遍
  `error_log` 和相关节点文档（`knowledge/nodes/<NodeType>/<NodeType>.md`），
  不要重复同样的修改

## 与用户的沟通

- 提交 test run 后先告知用户任务已提交、正在等待平台结果；收到终态后再用一两句话同步进展
  （通过还是失败、失败原因概述），不需要把完整的 `error_log` 逐字贴给用户
- 循环期间不要频繁提问打断用户；只有达到最大次数、或报错明显超出你能修复的范围（例如提示
  平台侧资源/权限问题）时才停下来询问用户下一步怎么做
