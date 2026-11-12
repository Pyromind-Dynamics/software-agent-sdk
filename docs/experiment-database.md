# 实验数据库：SQLite / MariaDB

Project 和创建前的准备记录共用 SQLAlchemy 2 Store 与 Alembic 迁移。
数据库差异只在基础设施层处理，不改变 Runtime 用例、API、Profile 权限或 Pi 插件。
聊天历史、Pi 持久化和独立的产品版本账本保持原存储方式。

## 启动配置

在 `start_inference.sh` 修改一个字段即可切换，默认 MariaDB：

```bash
export EXPERIMENT_DB_BACKEND="${EXPERIMENT_DB_BACKEND:-mariadb}"
```

也可以只对这次启动覆盖：

```bash
./start_inference.sh
EXPERIMENT_DB_BACKEND=sqlite ./start_inference.sh
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

迁移前备份并停止使用该数据库的服务。SQLite 使用本地文件、WAL、外键检查和
5 秒忙等待；短写事务先执行 `BEGIN IMMEDIATE`。MariaDB 使用事务和行锁。

## 两张业务表

| 表 | 数据 |
|---|---|
| experiment_projects | 项目归属、cluster、创建命令 ID、workflow_plan_path |
| experiment_preparations | 准备计划、入口任务、审阅状态、revision、Project 关联 |

另外保留 `experiment_plan_schema_version` 迁移版本表。
`workflow_plan_path` 相对于所属会话工作区，新项目写入
`public_data/workflowPLAN.json`。文件创建、原生子 Agent、平台任务账本保持原有流程。
现有 Project 返回结构中的接受计划与版本复用 preparation，不再在 Project 表复制。

`0008_retire_experiment_state` 提取原 creation_metadata.command_id，
删除 Project 的 current_revision、context、context_sequence、creation_metadata，
并删除六张旧表：experiment_plan_documents、experiment_task_runs、
agent_role_sessions、runtime_work_items、external_task_submissions、runtime_messages。
Project 和 preparation 行保留；被删的旧执行数据不再恢复，必须事先备份。
MariaDB DDL 可独立提交，迁移支持中断后重试。历史迁移不改写，
新库和旧库使用同一条迁移链，最终只留下两张业务表。

## 验证

```bash
uv run pytest tests/pyromind_runtime/test_experiment_storage_migration.py
uv run pytest tests/pyromind_runtime
```

迁移测试覆盖 SQLite 与可选 MariaDB。`EXPERIMENT_MIGRATION_TEST_URL`
必须指向空的独立测试库；未配置时跳过 MariaDB，不连接应用库。
测试覆盖新库、旧结构升级、重复迁移、保留计划与归属、并发创建及事务回滚。
