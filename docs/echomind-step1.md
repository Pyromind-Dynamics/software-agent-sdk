# EchoMind 步骤 1：Manager 与实验计划

本步支持对话规划、计划提案、用户确认、不可变版本、历史查询与刷新恢复。
默认参考图为 Init → Data → Train → Eval，Inference、Tracer 是未启用占位。
阶段与任务的业务含义由 EchoMind Manager Profile 和 avi-single-turn Skill 提供；
共享 Runtime 只校验结构、引用、依赖环及访问权限。任务名称自由填写。

本步产物是计划，不执行评测、数据处理或训练。后续子 Agent 和任务队列不在本步实现中。

## 运行

在 software-agent-sdk 根目录安装锁定依赖：

```sh
uv sync --frozen
```

为计划配置 MariaDB（数据库需事先创建，支持 MariaDB 11.8）：

```sh
export EXPERIMENT_PLAN_DATABASE_URL='mysql+pymysql://USER:PASSWORD@HOST:3306/DATABASE?charset=utf8mb4'
uv run python -m pyromind_agent_server.plan_migrations
```

凭据仅通过部署环境提供，不写入仓库或计划。迁移使用独立的
`experiment_plan_schema_version` 版本表，并创建：

- `experiment_projects`：项目归属及当前版本指针。
- `experiment_plan_proposals`：不可变提案、确认状态、确认命令 ID。
- `experiment_plan_revisions`：确认后的不可变计划快照。

本机 `start_inference.sh` 已配置为读取 `.env.experiment-plans.local`，
然后在启动服务前自动执行计划迁移。连接信息只保存在该忽略文件中（权限 600），
显式设置的 EXPERIMENT_PLAN_DATABASE_URL 优先。当前应用库为 `echomind_experiments`。

按现有方式启动 `pyromind_agent_server`；保持已有模型、鉴权和执行环境配置。
新 EchoMind 会话由入口配置选择 Pi，Pyromind 的默认 Harness 配置保持原行为。
没有配置计划数据库时，新 EchoMind 会话创建返回 503，不使用文件或 SQLite 作为生产回退。
修改数据库连接后需重启服务。

前端在 minimal-chat-frontend 中执行 `npm ci`、`npm run dev`，进入 `?chat=echomind`。
旧页面保留在 `?chat=echomind-legacy`，原 REST 接口保持兼容；旧项目不自动迁移。

## 交互与协议

新会话使用 `/api/v2/echomind/conversations`，复用统一会话路由工厂、
ConversationRuntime、快照和 SSE。原 Pyromind 入口继续使用原路径。

Manager 工具：

| 工具 | 行为 |
|---|---|
| create_experiment_plan | 生成并持久化初始提案，不创建项目 |
| revise_experiment_plan | 基于 expected_revision 提交完整的新计划；修改待确认提案时携带 source_proposal_id |
| get_experiment_plan | 读取当前计划、待确认提案及历史版本摘要；revision 参数读取指定版本 |

Manager 仅开放 read 和上述三个计划工具。Pi 的模型可见工具、实际调用及 Skill
读取均受会话白名单约束；子 Agent 和历史经验插件的执行工具不向本步 Manager 开放。
Pi 0.87.0、pi-subagents 0.78.0、AgentGenome 0.6.60 的现有依赖保持不变。

用户确认使用统一 ProductCommand：

```json
{
  "type": "confirm_experiment_plan",
  "command_id": "a-client-generated-unique-id",
  "proposal_id": "server-returned-proposal-id",
  "expected_revision": 0
}
```

拒绝使用 `reject_experiment_plan`，携带 command_id、proposal_id。
这些命令由前端用户交互提交，不向 Manager 提供确认工具。
同一确认重试应复用 command_id；首次确认以一个事务创建项目和版本 1。
旧提案、版本冲突和被拒绝的提案不能覆盖当前计划。

快照新增可选字段 `experiment_plan`，与原有 Agent 工作清单 `plan`、
DSL 工作流 `current_workflow` 分开。计划变化通过 `experiment_plan.changed`
事件提示刷新，事件不携带另一份计划事实源。
数据库已提交后，SSE 或文件命令回执写入失败不撤销计划提交；重连、刷新和查询从数据库读取。

历史计划详情接口：

```text
GET /api/v2/echomind/conversations/{conversation_id}/experiment-plans?revision=1
```

所有访问都校验当前用户与持久化 Profile 归属，客户端不能通过请求体覆盖入口配置。
旧的无归属记录保留原入口归属，不被默认认作 EchoMind。

## 实现归属与兼容

- Runtime domain：通用 WorkflowPlan、Stage、Task、输入引用、提案与版本模型。
- Runtime application / ports：计划用例、确认编排和 ExperimentPlanStore 端口。
- Runtime infrastructure：SQLAlchemy 存储；同步 SQL 操作由用例在线程边界调用。
- Server：可信 Profile 配置、MariaDB 连接、Alembic 迁移、HTTP 鉴权与依赖装配。
- Harness Adapter：接收会话配置、暴露工具代理、转换统一事件；不直接访问计划数据库。
- 前端：通用 ProductClient、SSE Hook、只读计划图和确认组件；EchoMind 入口只注入地址、标题与本地状态命名空间。

Pyromind 业务提示已移到服务端默认 Profile；Runtime 和 Pi Adapter 不再无条件注入它。
现有接口保持原路径，命令/事件/快照按增量方式扩展。空的 experiment_plan 字段不输出，
原有严格客户端仍可读取旧会话快照。原项目文件及 project_request.json 不作为新流程事实源。

MariaDB 使用事务、项目行锁与按会话归属生成的 GET_LOCK 命名锁。
命名锁覆盖尚无项目行的首次提案和首次确认；不同服务实例确认同一提案仍只产生一个项目和版本。
SQLite 仅用于自动化测试中的存储替身，不在服务端配置入口启用。

## 验证

```sh
uv run pytest tests/pyromind_runtime
uv run pytest tests/agent_server/test_echomind_router.py tests/agent_server/test_agent_profile_conv_start.py
```

Pi 子目录 `harness-adapter/pi-runtime`：

```sh
npm run check
npm test
```

前端根目录：

```sh
npm run build
npm run lint
npm test
```

真实 MariaDB 并发确认测试需指向独立测试库：

```sh
export EXPERIMENT_PLAN_TEST_DATABASE_URL='mysql+pymysql://USER:PASSWORD@HOST:3306/TEST_DATABASE?charset=utf8mb4'
uv run pytest tests/pyromind_runtime/test_experiment_plans.py -k mariadb
```

此测试应用计划迁移，并使用两条独立连接并发确认首次提案，验证只有一个项目和一个版本，
清理它自己创建的记录。未配置测试连接时明确跳过。

自动化覆盖结构与依赖环校验、确认幂等、提案替换/拒绝、版本冲突、SQL 事务回滚、
通知失败后的事实源读取、Profile 隔离及真实 Pi 进程的计划工具调用与会话恢复。
本地模拟模型测试验证调用链路，不代表付费业务模型的规划质量已验收。

浏览器验收使用临时测试数据库和本地模拟模型，验证生成草稿、创建项目、对话修改、
确认第二版、刷新恢复和查看第一版。阶段全部显示尚未执行。

本次实际连接 MariaDB 11.8.6 创建应用库和上述表；在另建的临时数据库中验证了
并发首次确认、过期版本拒绝、真实事务回滚与旧版本恢复。临时测试库已清理，
应用库的业务表保持空表，等待用户通过界面首次确认。
