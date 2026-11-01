# 实验数据库：SQLite / MariaDB

实验管理、执行账本和持久队列共用 SQLAlchemy 2 Store 与 Alembic 迁移。
数据库差异只在基础设施层处理，不改变 Runtime 用例、API、Profile 权限或 Pi 插件。
聊天历史、Pi 持久化和独立的产品版本账本保持原存储方式。

## 启动配置

在 `start_inference.sh` 修改一个字段即可切换，默认 SQLite：

```bash
export EXPERIMENT_DB_BACKEND="${EXPERIMENT_DB_BACKEND:-sqlite}"
```

也可以只对这次启动覆盖：

```bash
./start_inference.sh
EXPERIMENT_DB_BACKEND=mariadb ./start_inference.sh
```

- SQLite 默认文件：`workspace/experiments/experiments.sqlite3`，相对仓库根目录。
  可用 `EXPERIMENT_SQLITE_PATH` 覆盖。目录、文件和表由启动迁移创建。
- MariaDB 沿用 `EXPERIMENT_PLAN_DATABASE_URL`；未显式配置时才加载
  `.env.experiment-plans.local`，可用 `EXPERIMENT_PLAN_ENV_FILE` 改变配置文件位置。
- SQLite 模式不读取 MariaDB 配置文件，不连接局域网数据库。
- 数据库切换后使用另一份数据，不会自动复制或同步两个数据库。
- 直接调用服务端、不经过脚本的旧调用方仍可单独提供数据库 URL；均未配置时，
  保留原先关闭实验存储的行为。非法配置、连接或迁移失败会明确报错，不自动回退。

单独执行迁移：

```bash
EXPERIMENT_DB_BACKEND=sqlite uv run python -m pyromind_agent_server.plan_migrations
```

迁移前停止使用该数据库的服务与 Worker。SQLite 使用本地文件、WAL、外键检查和
5 秒忙等待；短写事务先执行 `BEGIN IMMEDIATE`。MariaDB 使用原有行锁、租约和
`SKIP LOCKED`。数据库锁等待不会重放模型调用或平台提交。

## 七张业务表

| 表 | 数据 |
|---|---|
| experiment_projects | 实验归属、上下文、当前版本指针 |
| experiment_plan_documents | 用 kind 区分提案和版本；内容存 attributes |
| experiment_task_runs | 执行状态；attributes.artifacts 按名称存产物元数据 |
| agent_role_sessions | 角色执行槽、租约、隔离代次 |
| runtime_work_items | 队列、重试、检查点 |
| external_task_submissions | 外部提交意图、幂等与平台任务关联 |
| runtime_messages | direction 区分 in/out；消息内容存 attributes |

另外保留 `experiment_plan_schema_version` 迁移版本表。旧提案、确认、历史查询及
公开返回结构保持兼容。原 Artifact ID、URI、摘要与元数据迁入 TaskRun JSON；
通过所属记录锁保护并发更新，已登记内容仍不可覆盖。消息 ID 在收取、投递两个
命名空间内分别唯一，因此 Inbox/Outbox 原本相同的 ID 不会互相覆盖。

`0004_compact_storage` 先复制、逐条核验，再提交副本并删除五张旧表。
MariaDB DDL 可能独立提交，迁移按步骤检测已完成的结构，支持中断后继续；内容冲突
会停止迁移。历史迁移未改写，旧库升级与新库初始化使用同一条迁移链。

## 验证

```bash
uv run pytest tests/pyromind_runtime/test_experiment_database.py
uv run pytest tests/pyromind_runtime
```

共享契约包含 SQLite 与可选 MariaDB 参数；真实 MariaDB 验证使用独立测试库：
`EXPERIMENT_DATABASE_TEST_URL` 用于并发及 JSON 契约，
`EXPERIMENT_MIGRATION_TEST_URL` 必须指向另一份空测试库，用于旧结构升级。
未提供时明确跳过，不连接应用库。测试覆盖旧历史、产物、待处理消息迁移、
重复迁移、删除中断恢复、确认事务回滚、并发领取和并发 JSON 更新。
