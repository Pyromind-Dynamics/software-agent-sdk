# Pyromind 三表版本管理

本期提供对话中的保存、历史查询、差异比较、版本选择、新建轮次和失败恢复。一个会话绑定一个项目；首次保存或执行时创建，纯查询不会创建空项目。没有新增前端面板或自动训练状态机。

## 存储与架构

- `projects`：项目身份、会话及用户绑定、`context_json`、并发 `revision`。轮次、当前选中版本和选择历史存在 JSON 中。
- `runs`：每次实际提交一行。`spec_json` 固定本次轮次、输入版本和有效定义；`source_json` 关联现有请求、平台任务及 `retry_of`。不复制任务状态和日志。
- `artifact_versions`：产物系列、版本号、生产 Run、登记幂等键、状态及 `manifest_json`。manifest 包含摘要、Storage 引用、父版本和登记恢复信息。

实际 DDL 在 `pyromind-runtime/pyromind_runtime/infrastructure/sqlite_version_store.py`。Pydantic 模型在 `domain/versions.py`，用例在 `application/versions.py`，持久化及快照通过 `ports/versions.py` 访问。Server 装配服务，两个 Harness 仅包装通用工具，不向上暴露原生类型。

现有会话、事件和外部任务继续使用原文件存储；公开命令、快照及 SSE 协议没有变化。新数据库只增加版本元数据，不迁移历史会话，也不会在 fork 后继承旧项目绑定。

默认位置为 `<conversations_dir 的父目录>/product-versions/versions.sqlite3`，可设置 `PYROMIND_VERSION_DB_PATH`。使用本机持久化磁盘、WAL、每连接外键校验、5 秒 busy timeout 和 FULL 同步；不支持共享网络文件系统上的多机写入。

数据库迁移通过 `PRAGMA user_version`，当前为 1。备份调用 `SQLiteVersionStore.backup(path)`，不要只复制正在使用的数据库主文件。`snapshot-staging` 保存上传重试所需的本地快照，也需要持久化。当前不做版本和暂存文件回收。

## 通过对话使用

Pi 和 OpenHands 均提供 `product_versions` 工具，入口是 `ConversationRuntime.version_call`。

保存示例：

```json
{
  "action": "save",
  "command_id": "save-training-script-1",
  "arguments": {
    "kind": "code",
    "logical_key": "training-script",
    "source": {"storage_ref": "/project/train.py"}
  }
}
```

`source` 支持 Storage 路径，或 `text` 与 `filename`。`kind` 支持 dataset、annotation、code、workflow、config、model、evaluation、trajectory、report。

| action | arguments |
|---|---|
| `list` | `{}`；返回项目 revision、版本和执行记录 |
| `save` | `kind`、`logical_key`、`source`；可附带 `parents`、`summary`、`producer_run_id`、`output_role` |
| `compare` | `left_version_id`、`right_version_id` |
| `select` | `slot`、`version_id`、`expected_revision`，可选 `reason` |
| `new_iteration` | `goal`、`expected_revision` |
| `recover` | `{}`；重试本项目未完成的快照登记 |

每次操作使用稳定、唯一的 `command_id`，重试时复用。选择或新建轮次前先查询 revision；冲突后重新查询，不能盲目覆盖。

选中版本可通过工作流路径字段的 `version://<slot>` 使用。已有执行引用固定的版本 ID，不跟随指针变化。已识别的 `input_path`、`model_path`、`base_model` 字段也会读取对应选择；任意 shell 命令中的字符串不会自动改写。

代码、工作流与配置比较返回文本差异。数据和模型比较返回 manifest 与来源。评测只有数据、评分规则和推理配置引用相同且均为 `ready` 时，才返回 `comparable=true`；这些引用分别放在 summary 的 `dataset_version_id`、`scoring_version_id`、`inference_config_version_id` 中。

## 快照和执行

登记流程为：插入 `pending` → 事务外保存独占快照 → 提交 `ready`。失败保留同一版本 ID 和版本号，状态为 `failed`；恢复复用原快照目录。失败版本号不回收。最终版本禁止修改和删除。

远端快照位于 `/.pyromind-agent/<conversation_id>/versions/<version_id>/`。文件内容先上传，manifest 最后上传；本地持久化暂存保证重试上传相同字节。每次选用、消费或比较冻结内容时校验摘要。

支持的自动执行入口：

- Product 工作流准备及确认：保存原始 DSL（提供时）和实际工作流定义；确认时再次检查输入快照。
- 旧版正式工作流、数据清洗及数据准备工具：通过 Adapter 注入提交回调，复用 Product Runtime 的持久化提交路径。调试提交保留原语义。
- 内置推理评测：额外保存并实际使用固定的评测脚本、数据映射配置和评分配置；重试关联独立 Run。选定输入改变时必须新建完整评测。

完成事件携带 `completion_result.artifacts` 时，按实际清单自动登记输出。每项结构与 `save.arguments` 相同，生产 Run 和输入来源由系统补齐。内置推理评测还会检查已知的 `report.json`。没有可靠输出清单的训练回调不推测 checkpoint 路径，可以随后通过 `save` 显式登记并关联 Run。

通用工作流固化有效定义和原始 DSL；定义中任意外部脚本、环境依赖不会被假定为已完整复制。当前平台实现支持最大 32 MiB 的独立文件快照，文本比较上限为 2 MiB。数据集、标注及模型的 Storage 导入保守登记为 `reference_only`，因为现有平台接口尚不能证明外部媒体、基模等依赖均已固定；不能将这类记录选为可恢复快照。`FileArtifactSnapshots` 是用于挂载存储及测试的目录快照实现，不代表平台已具备通用数据/模型依赖快照。

## 恢复与事务边界

SQLite 使用短写事务分配编号、更新选择和登记版本。文件复制、平台调用不放进数据库事务。

原有请求文件和任务登记仍是执行事实来源。平台提交结果不确定时保留 `uncertain`，不自动重复提交；缺少 SQLite 关联时只补登记。服务启动扫描待恢复项目，会话重新连接后使用当前凭据继续快照。无凭据时保留 pending/failed，不持久化 Cookie、Authorization 或临时签名 URL。

恢复执行前，已有输出的 pending/failed 登记必须处理完成，避免覆盖尚未保存的内容。重复完成事件使用相同登记键。跨项目的父版本、执行输入、生产 Run 和已知评测引用均被拒绝。

RSIGym 仅提供设计参考，本实现未复制其代码，也未增加其依赖。

## 验证

```bash
uv run pytest tests/pyromind_runtime
uv run pytest tests/tools/workflow/test_task_submission.py tests/tools/workflow/test_run_workflow.py tests/tools/pyromind_cleaning/test_pyromind_cleaning.py tests/tools/data_preparation/test_df_submit_pipeline.py tests/tools/data_preparation/test_inference_submit.py
```

版本专项测试覆盖并发编号、幂等冲突、隔离与外键、版本不可变、选择 CAS、快照校验、失败恢复、重复回调、平台提交不确定性，以及两种 Harness 的共用工具语义。
