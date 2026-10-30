# EchoMind 步骤 2：实现记录

更新时间：2026-10-30。**Eval 准备阶段已实际接通；完整步骤 2 baseline 闭环尚未完成。**

目标仍为：确认计划 → Manager 委派 Eval → 准备评测 → 用户确认 → 平台执行 → baseline 产物 → Eval 报告 → Manager 建议。
Init 保留在实验图中并连接 Data，任务自由命名。平台提交不确定时停住核对，不改造 Studio 接口。

## 当前已实现

- Runtime 领域：执行请求、评测提案、通用完成条件与执行确认/拒绝命令。
- Runtime 用例：已确认计划和角色权限校验，目标阶段启用校验；执行绑定不可变计划版本。
- Runtime 端口：执行存储、平台能力和工作队列。应用层不导入 SQLAlchemy、服务端或 Harness 实现。
- SQL 工作队列：入队去重、角色串行、领取、续租、隔离代次、检查点、退避重试、显式过期核对。
- Worker：可注入处理器、续租失败取消、关闭保留恢复证据；异常后不自动重放外部副作用。
- SQL 执行账本：活动任务唯一性、执行提案、确认事务、平台提交意图、不确定提交恢复、Inbox、产物及 Outbox。
- 确认事务锁定计划当前版本；相同确认/拒绝命令支持数据库回放。Runtime 继续使用文件命令账本校验命令 ID 与请求的一致性。
- 平台观测可早于关联落库；非终态不能覆盖终态，冲突终态保留待核实。
- 产物登记和报告入队在同一事务中；评测结果与报告状态分开。
- 请求凭据及带凭据 URL 在新增持久化边界被拒绝。
- 统一 SessionSpec 支持可信 delegated_roles；Pi 宿主按配置生成角色文件，校验角色/工具/Skill，并按角色验证报告归属。默认 search/general_purpose 行为保持兼容。
- 执行记录通过统一 API 的 experiment-executions 查询和可选 experiment_executions 快照字段返回，保留原计划版本关联，拒绝跨用户和跨 Profile 访问。
- Outbox 随服务生命周期发布 experiment_execution.changed 刷新事件；写入事件失败时保留待投递记录，重试使用稳定事件 ID。
- 前端展示真实执行状态、指标、产物路径及报告状态；按计划版本显示阶段状态，历史执行不自动作为当前计划成果。
- SQL 控制用例已增加取消、终止跟踪和经可信验证器核对后绑定任务。尚未向产品命令或模型工具开放这些控制入口。

过期工作只有在确认旧执行已经停止后才允许重试；不能确定时继续占用角色。
`submit_execution` 禁止通过普通队列重试入口重新提交。

## 数据库迁移

新增 Alembic revision `0002_execution_queue`，承接 `0001_experiment_plans`，创建七张表：

- experiment_task_runs
- agent_role_sessions
- runtime_work_items
- external_task_submissions
- runtime_inbox
- experiment_artifacts
- runtime_outbox

迁移使用冻结的 schema 定义，不导入未来版本的存储模型。schema 一致性测试同时比较计划表和执行表。
迁移仍由已有启动命令执行：

```sh
uv run python -m pyromind_agent_server.plan_migrations
```

2026-10-30 尝试真实 MariaDB 集成测试时，服务器在连接握手阶段返回 OperationalError 2013；
后续带 5 秒读超时的 SELECT 1 检查同样失败。本轮未创建临时测试库，也未更新应用库。
本地迁移与事务测试使用 SQLite；它不是生产数据库回退。

后续连接恢复后，独立临时 MariaDB 测试库的迁移重复执行、双连接并发入队去重和互斥领取测试通过，临时库已删除。
应用库已执行到 `0002_execution_queue`，随后只读核对七张执行表均存在。应用库未写入模拟实验数据。

## 集成状态与后续工作

配置计划数据库后，执行存储和只读执行查询已接入 ConversationRuntime，Outbox 发布器随服务启动与关闭。
新建 EchoMind 会话已启用 Eval 委派准备：Manager 使用 request_experiment_task，持久队列通过统一委派端口调用 pi-subagents。
Eval 在独立子进程中读取声明输入、校验评测配置，生成待用户确认的执行提案。
旧会话保持其持久化工具白名单，未静默升级；使用新增能力需新建 EchoMind 会话。
**平台提交、Kafka 与报告处理器仍未接通，不会将准备阶段当作 baseline 已执行。**

以下是继续开发的必需工作，不属于已完成能力：

1. Eval 角色、原生 Skills、受控准备工具、启动/查询/恢复/停止桥接及原生自动唤醒抑制已完成。后续增加报告阶段的业务处理器与用户控制入口。
2. Manager 的用户消息与报告通知进入同一持久串行入口，并接入服务生命周期、凭据重新授权和旧会话显式升级。
3. 使用既有评测构建器完成真实输入冻结、受控提交、查询、停止和人工核对；现有 SQL 产物登记方法只能由可信验证器调用，不能直接向模型暴露。
4. 接入独立 Kafka consumer group，关闭自动提交；Inbox 落库后提交 offset，查询兜底并处理冲突终态。
5. 接入 Eval 报告及 Manager 通知；Outbox 刷新发布已完成，仍需补充业务通知、失败收尾和平台轮询。
6. 在已有执行查询 API、快照和只读展示基础上，接入前端执行确认/取消/核对控制，并补显式能力升级。
7. 继续补真实 MariaDB 回滚故障测试、真实 Pi 业务角色恢复、Kafka 故障注入和前端端到端验收。

本轮没有修改 Pi、pi-subagents、AgentGenome 的版本或原 Pyromind 执行链路。

## 验证入口

```sh
uv run pytest tests/pyromind_runtime/test_execution_queue.py
uv run pytest tests/pyromind_runtime
```

并发数据库测试必须指向独立测试库：

```sh
EXPERIMENT_EXECUTION_TEST_DATABASE_URL='mysql+pymysql://USER:PASSWORD@HOST:PORT/TEST_DATABASE' \
  uv run pytest tests/pyromind_runtime/test_execution_queue.py -k mariadb
```

该测试验证两连接并发入队去重、互斥领取与已提交检查点读取。不配置测试库时明确跳过。
完整步骤 2 的验收标准仍以已确认技术方案为准，不能用底层单元测试替代真实 baseline 验收。

本轮增量验证：Runtime 621 项通过、6 项跳过；Pi 210 项通过；前端 40 项通过，构建通过；
所有已修改文件的 pre-commit 检查通过。前端 lint 仅有原有的两个 Fast Refresh 警告。
真实 MariaDB 双连接测试单独运行并通过，不包含在未配置数据库时跳过的测试计数中。
这些结果覆盖当前增量，尚不代表完整 Init baseline 业务链路通过验收。


## Eval 实际接入

角色定义位于 agent_profiles/echomind_eval.py；配置归属于 EchoMind，通用宿主不包含场景判断。
生成的会话文件为 pi/agent/agents/eval.md，包含 tools、extensions、skills、skillPath，关闭父级/全局上下文和 Skill 自动继承。
挂载 experiment-evaluation 与 avi-baseline-evaluation 两个 Skill，同时保留宿主读取白名单。

当前 Eval 工具：read、read_experiment_resource、submit_experiment_evaluation、get_experiment_execution。
读取工具仅接受当前任务声明的资源 ID。提案提交复用已有 DatasetMapping/EvaluationConfig 校验，并重新计算已声明输入的文件指纹；过期租约、其他执行或错误角色的写入被拒绝。
submit_experiment_evaluation 的实际效果为保存 awaiting_confirmation 提案，不调用平台创建接口。
模型版本字段是待审核配置的一部分，完整模型产物冻结与平台提交仍须由后续平台执行处理器核验。

队列只领取已装配的准备工作类型，不会误消费尚未实现的提交/报告任务。
启动前保存请求关联，取得原生运行身份后保存检查点；响应丢失时查询原关联，不重复启动。
原生运行身份与状态由 Adapter 转为通用委派响应，Runtime 不依赖插件的 runId 或关联前缀。
工作项恢复保留原角色占用，原工作接管前其他同角色工作不能抢跑。凭据只保留在活动会话内存，重新授权后可继续处理等待授权的工作。

使用方式：新建 EchoMind 会话，确认计划后要求 Manager 为 Init 中的基模任务准备评测。
计划资源中的待核验文件需提供可读取的绝对 Storage 路径。准备完成可在执行记录中查看提案；缺输入会明确记录 needs_input。

新增真实 Pi 集成测试使用本地模拟模型，验证独立 Eval 子进程、原生 Skill 目录、实际业务工具、SQL 提案、查询、恢复与停止终态；确认完成通知未绕过队列启动 Manager。
测试不代表真实模型规划质量或平台 GPU 评测已经验收。


本次 Eval 接入验证：完整 Runtime 626 项通过、6 项跳过；Pi 210 项通过；前端 40 项通过，构建通过。
最后的委派与队列定向回归为 25 项通过、1 项跳过。真实子进程测试覆盖原生 Skill 注入、实际受控工具调用、输入指纹复核、保存提案、继续已有会话与停止终态。
所有模型请求使用本地模拟服务，不消耗真实平台评测资源；未把模拟模型测试作为业务评测质量或 GPU baseline 验收。
