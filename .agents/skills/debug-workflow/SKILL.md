---
name: debug-workflow
description: >-
  仅当用户要求在不修改节点、模型或参数的前提下，对当前 Pyromind 工作流进行测试/调试/debug/试跑
  （在平台上真实执行一次并根据报错修复）时使用。若请求同时包含换模型、改参数、改结构或生成
  workflow，使用 generate-workflow-dsl，本轮不提交调试。使用固定 test mode 执行，保留次数限制与回调。
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

只处理未同时要求更换模型、节点或参数的显式测试请求。请求含配置修改时，
使用 generate-workflow-dsl 完成 DSL 修改并结束本轮；本轮不得调用 workflow_debug，
也不得通过 SDK 提交测试任务。

## 入口与准备

启用 SDK 时，读取[SDK 提交与回调](../workflow-execution/references/sdk-execution.md)，
调用保留的 `validate_workflow_dsl(test_mode=True)`，直接读取其模型可见 content 返回的
`validated_workflow_path`，文件内是实际测试图；以当前 `workflow_debug` 关联直接调用 SDK。
不要从内部 `raw_response` 搬写 JSON，也不将测试图重新转换为 DSL。
没有路径或出现 `artifact_unavailable` 时停止 SDK 提交，即使 `valid=true`。
轻量 helper 由执行层自动安装，
预算路径由 `PYROMIND_DEBUG_STATE_PATH` 提供。缺少这些条件且
`workflow_debug` 仍列出时，使用该已有入口。
`dsl_path` 缺省为 `public_data/workflow_canvas/workflow.py`；从真实文件读取 DSL。

先确认当前 DSL 已存在；测试提交前先用上述测试模式校验。不要用本地
bash/Python 执行声明式 `workflow.py`。正式运行使用 workflow-execution，
不能把测试成功当成已正式发布。

## 提交与回调

`submit_debug_workflow` 固定平台 `execution_mode=test`，受理后更新调用方提供的
`DebugBudget.load(...)`；helper 在共享文件锁内持久化，沿用旧调试计数。
使用当前环境提供的调试关联；不要另建计数器。预算保存失败但带实际任务 ID 时只修预算，不再创建任务。
提交成功返回 `task_id` 和初始状态；告知已提交后结束本轮，
等待 Kafka/callback 自动续跑。Pending/Running 时不重复提交，也不在 terminal 轮询等待。

超时、断连或未取得任务 ID 时，先核对回调和平台，不自动重投。
已终止的失败调试经修复后属于新尝试，继续消耗本轮预算。

收到对应调试任务的终态后：

| 终态 | 下一步 |
|---|---|
| Succeeded | 简短告知本次 test 成功，等待用户后续输入；不自动发布或继续修改 DSL。 |
| Failed / Error | 读取实际错误和节点文档，局部修复或重生成相关 DSL，校验后按用户已授权的调试范围再测。 |
| Terminated | 说明终止结果，结合错误决定是否需要修复或用户决策。 |

每轮最多接受 10 次调试任务。达到上限后停止；只有用户明确要求继续调试时才重置预算。
旧 Tool 路径使用其 `reset_round=true` 开始新一轮；SDK 路径更新并保存同一共享预算记录。
同一报错连续两轮没有进展时，重新核对相关节点文档，避免机械重复。

正式运行的终态按 workflow-execution 处理；不要套用调试任务“成功后等待”的规则。
反馈只报告实际状态与关键证据，不整段粘贴日志。
