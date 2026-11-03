# EchoMind 实验执行与 Eval

本页描述当前实验模式，替代早期“确认 Graph → 生成执行提案”的流程。旧提案和计划接口保留读取兼容，不用于新任务的执行授权。

## 执行链路

用户描述目标 → Manager 直接探查、分析和执行，或按需委派 Eval。委派时沿用原链路：Eval 探查、写配置、调用原评测工具 → Runtime 等待平台并验证产物 → 恢复 Eval 分析并登记报告 → Manager 收到一次持久通知。

参考 Graph 是可选、可变的方向信息。任务独立于 Graph 和计划版本；Graph 变化不重跑任务。`intent=execute` 按用户授权执行，`intent=prepare` 禁止平台提交。仅启用 Eval，Dataset、Train 尚未开放。

## 复用与分层

- Pi 0.87.0 执行 Manager 和 Eval；pi-subagents 0.78.0 负责子角色配置加载、启动、状态、中断、继续和停止。通过公开扩展工厂和 RPC 接入，不修改插件源码。
- Adapter 关闭受管子任务的原生父级自动唤醒；SQL 队列统一串行投递结果，避免重复回答。非受管会话保留原行为。
- AgentGenome 0.6.60 保留原集成，不作为 baseline 成功的前提。
- Eval 复用 `inference-evaluation` Skill、`df_submit_evaluation`、`df_check_progress` 和既有 PipelineRuns、DSL、评分、报告实现。没有新增一套评分执行器。
- Runtime 只处理归属、委派、状态、幂等、结果证据和持久队列。Harness 差异均在 Adapter 内。场景规则、Prompt、白名单放 Profile 与 Skill。

Manager 的平台工具和 Skill 白名单以 `a2c02d30` 父提交的完整开发能力为基准恢复，从会话开始即可使用，不依赖计划确认。删除 `read_skill`，统一使用 `read` 读取 Skill 和业务文件；EchoMind 不开放通用 `subagent`，仅保留可选的 Eval 委派。Pyromind 的 `search`、`general_purpose` 和原链路不变。模型可见的委派 schema 必须包含任务正文；旧 `task_id/expected_revision` 请求仍由兼容 API 接受。Eval 保持角色独立的工具白名单，无法修改全局方向或启动其他角色。

参考计划通过既有 `ExperimentContext` 增加可选 `tasks`，复用原 `ExperimentTask` 及输入输出引用校验。`update_experiment_context` 是完整替换，使用 sequence 做并发控制；阶段参考图仍允许回环，不参与执行调度。旧 context 缺少 tasks 时按空数组读取，SQL 表结构不变。

Manager 的 `read_experiment_artifact(run_id, name)` 复用既有 Runtime/HTTP 产物读取逻辑，验证会话归属、登记产物和 SHA-256。报告登记规范化任务内的相对路径；Storage 超时或 502/503/504 最多读取三次，退避 0.5、1 秒，认证和路径错误不重试。故障不会触发新的 GPU 任务。

EchoMind 配置版本升为 4，已有会话在空闲且无执行中的受管工作时通过原配置机制更新 Prompt、工具和 Skill；不重写历史计划、不自动启动任务。改动分别位于 Profile/场景 Skill、通用 Domain/Application 和 Adapter；依赖方向不变。公开协议仅增量扩展 tasks 和模型报告读取工具，旧正式计划 API 与历史数据保持兼容。

所有角色工具校验白名单和执行作用域。原生写入、编辑和终端在执行前校验 SQL 工作项及租约代次；终端沿用既有权限检查，不能因为 Manager 没有终端而错误拒绝 Eval。

## 持久化、等待与恢复

实验 SQL 数据库支持 SQLite / MariaDB，沿用七张业务表和 Alembic 迁移，详见 [数据库说明](experiment-database.md)。聊天与原评测账本保留原存储。

- 工作项按逻辑角色串行；Manager 用户消息优先于后台通知。等待平台释放 Agent 执行槽。
- 评测操作先登记稳定标识，再进入既有工具的预留、冻结和提交链路。重复工具调用查询原提交。
- 提交响应不明进入 `submission_uncertain`，禁止自动重提。仅报告失败时重试报告，不重跑 GPU。
- 先持久化 Kafka Inbox，再提交 offset；独立消费组，不依赖已加载会话。早到、重复和冲突消息保留并归并。Kafka 不可连接时，对已知平台任务继续查询对账。
- 平台成功不等于有效 baseline。读取并校验 metrics、report、predictions 的数量、格式和覆盖关系，再展示指标并排队生成解释报告。
- 执行结果与报告状态分开。报告登记验证文件可读性与内容摘要，再通过 Outbox 投递 Manager 摘要。
- 关服中断子 Agent 使用插件的可恢复中断；用户主动取消使用停止。恢复先检查原生运行，不把进程消失当作可以重复提交的证明。
- Manager 通知保存稳定关联。通知回答中断可在原会话继续，不重复追加原通知；已完成回答通过原生回执去重。
- 服务重启后，打开会话的认证请求重新提供凭据并恢复等待任务。Cookie、Token、Authorization 不写入 SQL、队列、Prompt 或会话恢复文件。

前端展示任务状态、等待原因、真实指标和可校验报告；参考 Graph 放在折叠辅助区。队列摘要只公开安全的类别、状态、尝试次数和等待原因，不公开工作项原始载荷。

参考阶段可以通过实验上下文的可选 `stage_executions`（阶段 ID → TaskRun ID 列表）关联已有执行，支持先执行、事后整理参考图。Runtime 校验这些执行属于当前会话；该字段只影响展示，不修改执行快照，也不触发任务。实验参考图没有关联记录时显示“未关联执行”，不能据此判断“尚未执行”。旧上下文省略此字段仍可直接读取，无需数据库迁移；旧正式计划的版本展示保持原行为。

## 本地真实验收

复制 `.env.test.example` 为 `.env.test.local`，填写 `PYROMIND_TEST_COOKIE` 和 `PYROMIND_TEST_X_CLUSTER`。开发环境可填写账号 ID、测试模型与数据的 Storage 路径及小样上限。文件已加入 Git 忽略规则，建议权限 `600`。

该文件只由本地验收客户端读取，作为正常 HTTP 请求头发送。不会把某个测试账号注入服务器全局鉴权。引号中的 `$`、`#` 等字符按原值读取，不执行 shell。

```sh
./start_inference.sh
uv run python scripts/echomind_baseline.py probe
uv run python scripts/echomind_baseline.py run --timeout 1800
# 网络响应丢失或重启后，跟踪同一会话，避免重复发起：
uv run python scripts/echomind_baseline.py watch --conversation CONVERSATION_ID
```

Agent 模型默认使用正在运行的服务配置（即启动脚本的配置）；`.env.test.local` 的 LLM 字段只用于可选覆盖。被评测模型与 Agent 模型分别配置。默认最多四个样本，不授权训练或部署。

客户端在发送命令前打印会话和命令标识。遇到提交待核对时停住；不会通过重新调用 `run` 自动重试。只有真实指标、报告可读取且后台处理完成，才输出 `baseline_artifacts_verified=true`。

## 验证

```sh
uv run pytest tests/pyromind_runtime
cd harness-adapter/pi-runtime && npm test
# 在 minimal-chat-frontend：npm test && npm run build
```

真实 Pi 集成测试覆盖独立角色、真实工具 schema、原生继续、完整 Runtime 重建后等待平台恢复、报告登记与 Manager 单次通知；平台调用使用测试替身，因此不能替代真实 GPU 验收。Kafka 故障注入覆盖落库失败不确认、offset 提交失败重放；真实 broker 和 MariaDB 网络验收需可连接的部署环境。

### 真实小样验收记录

2026-11-02 重新打开并核验了此前完成的实验：

- 会话：`d95eb2fa655b40038134b08b84a625cc`。
- TaskRun：`e10857aacca0413e805d26b488c396e7`；平台任务：`9650`。
- 基模：`/models/Qwen/Qwen3-VL-4B-Instruct`；验证集：`pcb_avi_recheck_0823_0828/val.jsonl`。
- 实际请求及评测均为 4 条，无 API/评分执行异常；执行成功、报告就绪，四个工作项均完成。分类准确率为 0；这只证明四条小样的结果，不代表完整验证集效果。
- 预测、指标、评分报告及 Eval 解释报告已登记；Manager 已根据结果完成总结。本次重开没有再次提交评测。
- Sandbox 空闲回收后，报告通过当前请求凭据直接从持久 Storage 读取，无须启动 Agent、重新创建容器或提供模型密钥。报告 SHA-256 与首次登记一致。

本地核验副本在 `workspace/acceptance/step2-baseline/`：`report.md`、`metrics.json` 和 `verification.json`，不包含请求凭据。原始 SQL 元数据与 Storage 文件仍是事实源。

本轮修复涉及 Adapter 文件读取、子角色恢复时绑定新的受信工作租约，以及 SQL Store 对多次提交的核对/取消选择；公开 API 和数据库结构不变。跨用户、跨 Profile 的校验仍先于产物读取，旧执行者的过期租约仍被拒绝。

真实 Pi 恢复测试包含准备过程中关闭服务和等待平台时关闭服务两种情况，均保持单次评测提交、恢复后生成报告与 Manager 单次通知。真实 MariaDB 已在独立临时库通过 48 项事务、并发及迁移测试，临时库已清理，现有实验数据库未修改。Kafka 当前只能建立 TCP 连接，Consumer 仍无法完成协议连接；本次真实 baseline 使用平台查询完成对账，真实 Kafka 端到端验收尚待 broker 配置或连接恢复。

最终本地回归：Runtime 与原评测提交工具共 736 项通过、8 项环境条件跳过；Pi Adapter 218 项通过；前端 41 项通过并成功构建。MariaDB 上述 48 项为单独配置临时数据库运行的专项结果。
