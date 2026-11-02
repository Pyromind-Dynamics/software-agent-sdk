# task 9680：Eval 恢复、Manager 通知与前端展示排查

排查日期：2026-11-02。以下时间均为北京时间；SQL 和事件文件中的原始 UTC 时间已加 8 小时。

## 结论

会话 `1b68a6d227b7497f8b9e4aaff9ed6595` 的任务完成链路实际走通：HTTP callback 和 Kafka 都收到成功状态，Runtime 恢复 Eval 生成并登记分析报告，随后唤醒 Manager，Manager 完成最终回复。没有发现本次任务漏唤醒或漏通知。

前端上下两张 Eval 卡片对应两个不同的原生运行：准备提交与恢复后分析报告。它们不是同一次运行的重复状态。排查时浏览器和最终快照中两张卡均为已完成。

存在两个确定的实现缺陷：托管委派没有完整登记展示元数据；`df_check_progress` 对合法 Eval 请求的归属校验使用了缺失的 submissions 字段。另有 Manager 重复探查和核查导致的行为差异。

## 事实链

| 时间 | 事实 | 证据 |
|---|---|---|
| 19:11:40 | 第一轮 Eval 启动，负责准备与提交 | Product event seq 151；native run `5f4f0d20-6a73-4e3d-9764-008b9a0f8538` |
| 19:22:21 | 平台任务 9680 提交 | Product event seq 338 |
| 19:22:21.723 | Kafka running 入库，最终 applied | SQL runtime_messages：source=kafka、task_id=9680 |
| 19:22:37 | 第一轮 Eval 已完成，工作转为等待平台 | Product event seq 345；execute_task checkpoint 为 waiting_platform |
| 19:23:55.840 | HTTP callback succeeded 入库，最终 applied | SQL runtime_messages：source=callback |
| 19:23:56.094 | Kafka succeeded 入库，最终 applied | SQL runtime_messages：source=kafka |
| 19:24:20.840 | 查询对账也确认 succeeded | SQL runtime_messages：source=query |
| 19:24:58.988 | 校验产物后生成 generate_report 工作项 | SQL runtime_work_items |
| 19:24:59 | 第二轮 Eval 启动分析 | Product event seq 349；native run `222f6029-ee5f-4504-b0ae-5034efa8c9f0` |
| 19:30:25 | 报告登记就绪，创建 manager_turn | Product event seq 370；SQL runtime_work_items |
| 19:30:26 | Manager 开始处理回传结果 | Product event seq 371 |
| 19:31:01 | 第二轮 Eval 的展示终态写入 | Product event seq 379；原生进程在 19:30:45 结束 |
| 19:34:47 | Manager 最终回复完成 | Product event seq 892/893 |

`execute_task`、`generate_report`、`manager_turn` 三个队列项均为 completed，attempts 均为 1；Manager checkpoint 为 delivered=true。实验执行记录最终 state=succeeded、report_state=ready。

HTTP callback 比 Kafka 成功消息早约 0.25 秒，本次首先推进状态的是 callback。不能把本次分析的启动单独归因于 Kafka，但 Kafka 成功消息确实被持久化并应用。

19:23:56 到 19:24:59 之间经过查询对账和产物读取校验；19:24:59 到 19:30:25 为分析/登记报告阶段。19:30:25 到 19:30:26 的 Manager 唤醒没有显著延迟。

当前聊天没有可读的挂载服务终端；Kafka 的确认依据是持久化 inbox，而非直接读到服务终端日志。

## 两张 Eval 卡片的原因

`DelegatedExecution.prepare()` 在 report 阶段通过统一 control 端口恢复已有子会话。pi-subagents 的 resume 返回新 run ID，因此产生第二个 operation。前端按时间线逐项展示，这个顺序本身正确。

问题在展示信息：

1. `harness-adapter/pi-runtime/src/subagent-host.ts:207` 的托管 `control()` 直接调用原生 RPC，没有像普通 subagent 工具路径那样完整关联 task 和 previous_run_id。
2. 同文件的 `refresh()` 尝试从 `status.steps[0].task` 恢复任务说明；实际 v3 status 中是 description，且已被插件脱敏为 `[prompt redacted]`，不存在 task。不能通过恢复原始 prompt 来修复展示。
3. 两个 operation 的 `details.subagent` 均缺少 task，第二个还缺 previous_run_id，所以都显示“子 Agent · eval / 未记录任务说明”。
4. `minimal-chat-frontend/src/subagentPresentation.ts` 的 grouping 只将委派/控制工具归到对应卡片，没有将 resume 的两个运行展示为一个逻辑任务的两个阶段。

从 19:24:59 到 19:31:01，“上方已完成、下方运行中”符合两次运行的事实，但没有解释二者关系。最终快照中两个 operation 均 completed；排查时浏览器也均显示已完成，未复现永久停留运行中的现象。

建议：Adapter 为托管 spawn/resume 保存安全的任务摘要和 previous_run_id，通过通用 ProductEvent details 暴露；如需显示 phase/task 标题，由 Product Runtime 提供通用展示信息，Adapter 不解释场景业务。前端显示“准备与提交 → 等待平台 → 分析报告”，可在一个逻辑任务卡内保留两次运行的轨迹。保留独立 run ID 和终态，不通过重排卡片或把旧运行改回 running 掩盖阶段关系。

## df_check_progress 的确定性缺陷

`pyromind_runtime/application/experiment_execution.py:205` 通过 `row.get("submissions", [])` 获取收据并校验 output_dir。

但 row 来自 `SqlExperimentExecutionStore.role_scope()`；后者返回 `_run()` 查询的 experiment_task_runs 原始行，不包含关联的 external_task_submissions。只有 `list()` 装配了 submissions。

因此 receipts 恒为空，本次正确的 output_dir 也被拒绝，返回 `progress belongs to another execution`。9650 的预期会话同样遭遇此错误；它不是本次事件唤醒失败，也不是恢复会话导致凭据或归属变化。

建议：在 Store 的受信归属和租约校验范围内获取当前执行的提交收据，再校验 output_dir。不得删除归属校验或允许任意目录。补充合法目录通过、其他执行目录拒绝、过期租约拒绝的测试。

## 与 d95eb2fa 的行为差异

两者都采用准备提交、等待平台、恢复 Eval 分析、Manager 汇总的链路。9650 的 inbox 记录只有 query；9680 则有 callback、Kafka 和 query。

1b68a6d2 中，Manager 在委派前读取 Skill、探查模型与数据并限定评分口径；从用户请求到委派约 6 分 57 秒。d95eb2fa 从请求到委派约 5 秒。这不符合本次用户要求“Manager 只委派 Eval，Eval 自主探查和选择评分口径”。

1b68a6d2 的 Manager 在收到已登记的结果后又调用 12 次工具，包括 df_check_progress、6 次 terminal、get_storage_url、上下文与计划工具；出现路径错误、字段假设错误及脚本报错。d95eb2fa 的对应阶段只调用 get_experiment_execution、update_experiment_context 两个工具，约 23 秒完成。

当前 Manager Profile 允许直接探查、执行和报告，delegation 为 optional。它解释了模型为何容易扩展职责，但不能取代用户明确的委派要求。建议在 Profile/场景 Skill 中用简短职责契约表达：用户明确委派的工作由子角色负责；Manager 接收已核验事实，按需读取登记报告后汇总，有实质证据矛盾时再核查。避免为这一次任务添加触发词特判，也无需修改共享 Tool 行为。

两个会话的 provider/base_url 也不同：d95eb2fa 使用 OpenRouter，1b68a6d2 使用平台 OpenAI-compatible 网关；模型名称均指向 DeepSeek v4.1 Flash。该差异需要受控对比才能判断是否影响延迟或行为，本次证据不足以归因为网关。

## 建议实施顺序与验证

1. 修复 progress 收据查询，恢复合法进度/报告入口读取。
2. 补齐托管 spawn/resume 的展示元数据；前端展示接续关系与阶段，恢复后仍能读到相同信息。
3. 收紧 Profile/场景 Skill 的委派与结果汇总职责，减少重复执行探查。

上述改动应保持公开 API 和旧持久化数据兼容；新 details 字段可选，旧记录仍可逐 run 展示。Harness 原生关系只在 Adapter 内转换，Runtime 与前端不导入 Pi 类型。

验证应覆盖单次平台提交、多源成功事件去重、恢复后分析及 Manager 单次通知、跨 Profile 拒绝和 conversation 隔离；运行 `uv run pytest tests/pyromind_runtime`、Pi Adapter 对应测试，以及前端测试与构建。

## 后续实施与验证

用户复核范围后已实施以下修复：

- Runtime Store 的 `role_scope()` 在同一受信事务中装配当前执行的提交收据；原有 owner、角色、执行和租约检查保持不变。
- Product Runtime 通过统一 control 请求传递任务标题和阶段。Pi Adapter 使用原生 resume 继续执行，仅增加展示信息与前序运行关联，并将白名单展示字段保存在会话自己的 `pi/subagent-display.json`，重建后可恢复。该文件不保存 prompt、轨迹、凭据或生命周期状态。
- 前端根据明确的前序运行关系归并卡片，主卡显示最新运行的阶段与状态，此前运行的终态、轨迹和输出仍可展开查看。不关联没有证据的独立任务；旧记录缺少关联信息时保持逐 run 展示。
- experiment-planning Skill 采用 Manager 与任意已启用子角色通用的分工、证据交接和按需核查原则，不为 Eval 增加专用协作流程。

层与依赖：业务收据与阶段由 Product Runtime 负责，原生运行关系由 Adapter 转换，前端只消费通用 operation details。未新增 Harness 或 Agent/场景分支；公开 API 与数据库结构不变。新增展示字段可选，旧客户端和旧快照继续可读。

修复前的回归复现为合法目录被拒绝，其他隔离检查通过；修复后五项均通过。Runtime 全量测试 692 通过、8 项环境条件跳过；Pi Adapter 全量 219 通过，最终事件顺序调整后相关 8 项再次通过；前端 43 项通过、构建成功。补充了通用事件边界字段透传、原生重启恢复后摘要/阶段/前序关系、跨会话与跨 Profile 拒绝、过期租约和前端归并回归。浏览器预览已核验当前运行与此前轨迹的展开层级。

未重放历史事件或重新提交 GPU 任务。历史会话原有运行元数据不被猜测补写。
