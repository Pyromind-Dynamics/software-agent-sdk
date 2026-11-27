# SDK 提交与回调

系统消息明确启用 SDK 时，在当前 terminal 使用 `"$PYROMIND_SDK_PYTHON"`。
身份、API 地址和集群由当前用户的执行环境提供；不要抄写凭据到脚本、配置或报告。
SDK Main 默认停止注入 `df_submit_pipeline`、`df_submit_evaluation`、`submit_workflow`、
`workflow_debug`、`run_dataset_cleaning`、`edp_render`、`edp_submit`、`edp_aggregate`
八个业务提交入口，不按历史任务重新加入。未启用 SDK 的部署按当前实际清单使用原入口。
任务状态与节点日志也通过 SDK 查询，`inspect_workflow_task` Tool 不再注入。
需要查询或处理失败回调时，按需读取[SDK 任务检查](sdk-inspection.md)。

准备数据处理 Pipeline 和提交已校验工作流时，分别使用以下导入。

```python
from pyromind_common.platform.preparation import prepare_pipeline
from pyromind_common.platform.workflows import submit_workflow
```

## 准备实际工作流

声明式 `workflow.py` 是输入定义，不是可在本地执行的 Python 程序。生成画布的真实路径为
`public_data/workflow_canvas/workflow.py`；已有 Storage 定义使用真实 `/workspace/...` 路径。
保留服务端节点及资源校验，并提交校验返回的实际图。

`pyromind_common.platform.workflows` 提供普通函数：

| 函数 | 输入和职责 |
|---|---|
| `prepare_workflow(validator, dsl=..., name=...)` | 使用调用方已鉴权的 `httpx.Client` 校验 DSL 和资源，返回实际图，不创建任务 |
| `validate_resources(validator, workflow)` | 检查将要提交的实际图，不预留资源 |
| `submit_workflow(client, workflow, out_id=..., name=...)` | 提交前在内存绑定 runtime 环境，使用原生 SDK client 创建一次，返回原生响应 |
| `submit_workflow_batch(client, workflows, out_id=...)` | 逐项创建，首个失败即停止；异常的 accepted 保留所有已受理响应 |
| `prepare_debug_workflow(workflow)` | 复制实际图并固定 `execution_mode=test` |
| `submit_debug_workflow(client, workflow, out_id=..., budget=..., validated=True)` | 提交已校验的固定测试图；也可提供已鉴权 validator，仅受理成功后消耗预算 |

执行层将这些纯函数打成 `pyromind-platform-helpers` wheel，正常安装到当前 SDK 解释器，
版本随源码内容变化。它使用 `pyromind_common.platform` 导入路径，只依赖 SDK、HTTP 客户端、
Pydantic 和 filelock，不要求安装完整 Host `pyromind-common`、Runtime、Tool 或模型网关依赖。
只安装 `pyromind-sdk==0.1.17` 不等于具备 helper；不要 import Host 源码或自行修改 Python 路径。
`validator` 的 base URL 指向平台 `workflows/` 路由，鉴权由调用方显式提供。
没有可用的校验身份时，继续使用 `validate_workflow_dsl`；不要假定 SDK Access Key
能够替代旧校验路由的登录 Cookie。SDK 本地 `DslValidator` 不能替代这两项服务端校验。
SDK Main 的校验 Tool 在成功后保存实际图，模型可见的 content 提供
`validated_workflow_path` 和 `sha256`。直接读取该路径，不复制或重写图；
`raw_response.data.workflow` 是内部观测字段，不代表模型已经取得图内容。
只有 `valid=true`、没有 artifact path 或出现 `artifact_unavailable` 时停止 SDK 提交。
含请求凭据、实际配置 secret 或签名 URL 的图可能无法安全保存；不要删改图绕过检查。

## 场景准备函数

`pyromind_common.platform.preparation` 的 `LocalFiles(root)` 读取调用方工作目录；
`StagingDirectory(local, storage)` 将本地冻结目录对应到真实 Storage 位置。
在已挂载的 Storage 上，用 `StagingDirectory.mounted("/workspace/.../<run UUID>")`
创建同一目录的本地与 Storage 映射。新任务的 UUID 同时用于目录和 `run_id`；
续跑使用原 UUID 和原目录。OS 本地目录需要先发布冻结文件，普通准备函数不自动上传。
远端已挂载 Storage 时直接写入或复制到真实 `/workspace/...` 目录，核对大小及 SHA-256 后
再提交；不调用上传/下载中转 Tool。进度、报告和文件交付按
[挂载进度参考](../../data-processing/references/storage-progress.md) 处理。本机没有挂载时，
按实际清单使用可用的发布入口，不能把本机目录当成 Storage。

| 准备函数 | 当前边界 |
|---|---|
| `prepare_cleaning(options, files=..., staging=..., runtime_dir=..., run_id=...)` | 冻结清洗脚本、运行时和输入约定；不提交或登记任务 |
| `prepare_pipeline(options, files=..., staging=..., runtime_dir=..., run_id=...)` | 支持 none/text/vision，冻结输入和执行修订；严格续跑复用冻结资产，兼容续跑要求 reuse_assessment；保留 Dataset Recipe |
| `prepare_edp_render(options, files=..., staging=..., runtime_dir=..., run_id=..., source_schema=...)` | 冻结模板，保留列引用预检；不执行分片任务 |
| `prepare_edp_aggregate(options, staging=..., runtime_dir=..., run_id=...)` | 冻结聚合脚本；调用方先确认各分片终态 |
| `pyromind_common.platform.environment.prepare_edp_shard(options, manifest=..., staging=..., runtime_dir=..., run_id=..., platform_env=..., cluster=...)` | 冻结分片 runner/profile/runtime/wheels，返回工作流及 manifest、run ID、输出目录；不创建任务 |
| `pyromind_common.platform.evaluation.stage_evaluation(options, files=..., staging=..., runtime_dir=..., run_id=..., frozen_reward=..., selection=...)` | 无 HTTP 地冻结推理、评分、报告和恢复输入，返回 `StagedEvaluation`；独立测试校验冻结选择证据 |

`options` 来自 `pyromind_common.platform.scene_options` 的 `PipelineOptions`、
`CleaningOptions`、`RenderOptions`、`AggregateOptions`；评测使用
`pyromind_common.platform.evaluation_core.EvaluationOptions`，不需要导入旧 Tool 模型。
EDP 分片使用 `pyromind_common.platform.environment.EdpSubmitOptions`。
`run_id` 使用该运行的 UUID，
`runtime_dir` 指向该 Skill 实际运行库。不要把其他场景的运行库目录代入。
独立 test 的 `selection` 使用 `evaluation_core.ValidationSelection`：
`selection_run_id`、train/validation/test 的 `FrozenDataset(path, sha256)`、
精确 checkpoint 的 `ValidationCheckpoint(path, kind, base_model)`，以及验证时冻结的
Reward 或评分脚本/规格。调用方先从已授权 Project 的版本、训练分析和报告取得这些证据，
不能只用自报 task ID 证明归属。准备函数校验独立 test、选定模型和完整测试范围，
冻结 `validation-selection.json`；resume 复用该文件，不重新选模。
Pipeline 的准备修订只记录冻结输入，不代表平台已接受一次提交。提交结果未确认时先核对
回调或平台；不要调用 resume 创建新修订来重投。确定需要继续执行后才准备新的 resume 修订。

评测默认调用 `stage_evaluation`，以返回的 `workflow_path` 相对于工作区的路径和 `name`
调用保留的 `validate_workflow_dsl`。直接读取其 content 返回的
`validated_workflow_path`，其中保存的服务端实际图才是提交图。
用 `pyromind_common.platform.scene_core.validate_pipeline_topology` 检查该图，
传 `require_inference=staged.requires_inference`，沿用评测的推理→CPU 评分拓扑约束。
`prepare_evaluation(..., validator=...)` 是已有校验登录身份时的便捷包装；不假设 SDK
Access Key 可以替代旧校验路由的 Cookie。

清洗、Pipeline 和 EDP 准备返回原始业务图；仍须执行服务端资源预检。
用 SDK 已有的 `from pyromind_sdk.client.workflow import DslConverter`，调用
`DslConverter().to_python(graph)` 并保存为工作区内的 DSL 文件，再调用保留的
`validate_workflow_dsl(dsl_path=相对路径, name=graph["name"])`。使用 content 提供的
`validated_workflow_path` 文件内的实际校验图，
不要直接把未检查资源的原始图交给 create。当前清洗/EDP 单节点图没有额外顶层执行参数；
含其他执行语义的图不能假定 DSL 转换会保留所有字段。
发布所需文件、完成服务端校验后，用普通提交函数调用 SDK 创建一次任务。

### Pipeline 准备示例

`prepare_pipeline` 直接返回含 `name`、`nodes`、`edges` 的工作流字典。
小样通过并获准全量后，先生成覆盖完整目标数据的 Manifest，核对记录总数与图片根目录。
`mode="full"` 使用传入的全部记录，输入路径应指向刚核对的全量 Manifest。
图片相对路径按 Manifest 所在目录解析，所引用的图片和冻结资产都要对平台任务可见。

将下面脚本保存为 `public_data/data-preparation/prepare_pipeline.py`。
三个参数依次为全量 Manifest、Storage 运行目录的父目录、已验证的小样 Pipeline 脚本。
模型 profile 沿用小样的视觉网关配置，用户改过网关时按下一节在准备前重建。

```python
import json
import sys
from pathlib import Path
from uuid import uuid4

from pyromind_sdk.client.workflow import DslConverter
from pyromind_common.platform.preparation import (
    LocalFiles,
    StagingDirectory,
    prepare_pipeline,
)
from pyromind_common.platform.scene_options import PipelineOptions

workspace = Path.cwd()
manifest = Path(sys.argv[1])
run_id = uuid4()
staging = StagingDirectory.mounted(Path(sys.argv[2]) / str(run_id))
graph = prepare_pipeline(
    PipelineOptions(
        script_path=sys.argv[3],
        input_path=str(manifest),
        model_profile="vision",
        output_schema="structured",
        mode="full",
    ),
    files=LocalFiles(workspace),
    staging=staging,
    runtime_dir=workspace / ".agents/skills/data-processing/scripts/preparation",
    run_id=run_id,
)
dsl_path = Path("public_data/data-preparation") / f"pipeline-{run_id}.py"
(workspace / dsl_path).parent.mkdir(parents=True, exist_ok=True)
(workspace / dsl_path).write_text(DslConverter().to_python(graph), encoding="utf-8")
print(json.dumps({
    "run_id": str(run_id),
    "output_dir": str(staging.local),
    "dsl_path": str(dsl_path),
    "name": graph["name"],
}))
```

用 `"$PYROMIND_SDK_PYTHON" public_data/data-preparation/prepare_pipeline.py`
执行并传入实际三个参数。准备完成后，使用打印的 `dsl_path` 和 `name`
调用 `validate_workflow_dsl`。取得 `validated_workflow_path` 后，按下面的 SDK
创建示例读取该文件，Pipeline 的 `out_id` 选择
`json.loads(os.environ["PYROMIND_TASK_CONTEXTS"])["data_preparation"]`。
Project 场景使用对应 task/attempt 的 `out_ids["data_preparation"]`。

冻结错误包含实际本地文件路径。新的运行目录需要同时改变本地目录和 Storage 映射，
`StagingDirectory.mounted` 会完成这两个值的配对。只有 Storage 字符串变化时，
文件仍会写入旧本地目录。记录准备返回的 UUID 与路径，修正校验或提交参数后继续使用
已准备的图；每次准备成功仅表示冻结资产生成，平台受理以真实 task ID 为准。

### 模型环境与视觉网关

`prepare_pipeline` 从 `PYROMIND_DATAFLOW_PROFILES` 选择 `text` 或 `vision`，`none`
不读取模型配置。模型 profile 的公开配置指纹写入环境声明，准备与提交之间发生漂移会拒绝
提交；轮换密钥不改变公开配置。不得自行修改声明或移除指纹绕过检查。

用户指定视觉网关时，沿用小样 `gateway` 的非密钥配置，将整个 vision profile 替换，
不能把用户网关地址与原平台密钥混用。以下代码只修改当前 Python 进程内存：

```python
import json
import os
from pathlib import Path

gateway = json.loads(Path("public_data/gateway.json").read_text())
profiles = json.loads(os.environ["PYROMIND_DATAFLOW_PROFILES"])
profiles["vision"] = {
    "DF_API_URL": gateway["api_url"],
    "DF_API_BASE_URL": gateway["api_url"].removesuffix("/chat/completions"),
    "DF_MODEL_NAME": gateway["model"],
    "DF_API_KEY": os.environ[gateway["api_key_env"]],
}
os.environ["PYROMIND_DATAFLOW_PROFILES"] = json.dumps(profiles)
```

密钥只取已授权环境变量；`gateway.json` 只含 `api_url`、`model`、`api_key_env`。
同一进程完成准备、已鉴权校验和提交时只构造一次。经校验 Tool 跨到新 Python 进程时，
提交脚本须按相同公开配置和当前授权密钥重建 profile；terminal 不自动保留上次进程的内存
修改。小样使用同一 gateway，并在修改网关后重新验证。

### EDP 分片

先用 `select_edp_manifests(options, shard_index=bytes)` 选择 manifest；单片 `manifest`
模式不传 index，`shards` 模式先从实际 Storage 取得小型索引。默认一片，最多五片；
增加并发仍须用户授权。每片使用独立 run UUID，在 manifest 同级创建
`StagingDirectory(local, storage=<manifest父目录>/<run UUID>)`，并调用 `prepare_edp_shard`。
`runtime_dir` 是 `.agents/skills/data-processing/scripts/edp`。

准备结果 `PreparedEdpShard` 包含 `workflow`、`manifest`、`run_id`、`output_dir`、
`resumed`。逐片保留这些信息，并按顺序对应 SDK 真实任务 ID；不要只保留最后一片。
恢复时给 `EdpSubmitOptions.run_id` 原 UUID，使用原目录与冻结文件；不要生成新目录，
冻结的 runner/profile/runtime 会做 hash 校验。Worker 使用 API key，不传请求 Cookie/JWT；
纯环境 profile 不要求 LLM，解题 profile 按声明提供 LLM 环境。

完成每片的文件发布和服务端校验后，可用 `submit_workflow_batch` 逐片创建。
捕获 `WorkflowBatchSubmissionError`，保存其 `accepted` 中每个响应的 task ID，
按准备顺序对应分片；失败后后续分片没有提交，不能把整批重新提交。
等各片终态后读取 `<output_dir>/run/verdicts.jsonl` 与 traces，再选择聚合输入。

### 提交时的凭据边界

本地冻结图、脚本、报告与 Project 记录不包含密钥。含 `{{pyromind.runtime:...}}`
环境声明的校验图必须使用普通 `submit_workflow` 或 `submit_workflow_batch`，不能直接
`studio.create`：原生 SDK 不会解析这些声明。helper 复制图，在 SDK POST 前才将当前
环境值绑定到内存请求，原冻结图保持不变。

最终 Studio POST 沿用平台既有 command 环境，含 Worker 需要的值，平台可能保存任务定义。
这不是新 Secret 服务，不代表平台侧不落盘。不要打印、保存或返回绑定后的请求图；
只返回真实任务 ID、状态和无密钥的运行元数据。

## 选择当前关联上下文

`PYROMIND_TASK_CONTEXTS` 是 JSON 对象，按 `workflow_run`、`workflow_debug`、
`data_preparation` 等任务类型提供服务端生成的 `out_id`。按实际场景取值；没有所需类型时
报告上下文缺口，不自行拼造会话、用户或关联签名。

Project 任务读取 `PYROMIND_PROJECT_EXECUTIONS`：它是包含
`project_id`、`task_id`、`attempt`、`out_ids` 的 JSON 列表。选择已获授权的 Project
任务和当前 attempt，再从该项的 `out_ids` 选择任务类型。不要用普通会话关联替代
Project 关联；新业务尝试需要新的 attempt 上下文，旧 attempt 的迟到回调不代表新尝试完成。
训练任务如果有 `blocked_by`，先完成其中列出的 Baseline 任务的报告登记。
登记后从新的终端调用读取 `out_ids`，取得该训练任务的提交关联再提交。
有 `waiting_for` 的任务等待所列平台任务的回调，同次尝试暂时不会取得新的提交关联。
其他节点可以继续已获授权的工作和报告登记。没有可推进的工作时结束本轮。
分片提交保留每片真实 `task_id`，不只保存最后一片。
执行计划内的模型评测时，使用该评测任务当前尝试的 `out_ids["evaluation"]`，让工作台能将推理回执关联到对应的模型和验证集。

核验产物后更新 `public_data/workflowPLAN.json`。保留获批的 `plan.tasks`，在
`runtime.tasks[计划任务ID]` 记录当前 `attempt`、实际 `platform_task_id` 和
`verified_outputs`（按声明的产物名映射到绝对路径）。产物齐备后将该记录的
`business_status` 设为 `completed`。工作台按任务 ID 和执行轮次登记交付。

`out_id` 仅用于关联，不是平台提交幂等键，也不是平台权限凭据。可信平台回调会核验
服务端签名关联，然后补齐任务事实并通知原 Main；普通 HTTP 手工回调不能据此认领未知任务。
SDK 的 `get_task()` 当前不返回 owner 或 out_id，查到任务存在不能作为归属证明。

## 使用原生 SDK 创建一次

将校验 Tool content 返回的 `validated_workflow_path` 作为脚本首个参数，
直接读取已保存的实际图。不要自行指定一个新文件名并搬写图内容。
Project 场景将 `out_id` 的选择改为上面的 Project task/attempt 项。

```python
import json
import os
import sys
from pathlib import Path

from pyromind_sdk import PyroMindAPIClient
from pyromind_common.platform.workflows import submit_workflow

validated_workflow_path = sys.argv[1]
workflow = json.loads(Path(validated_workflow_path).read_text())
out_id = json.loads(os.environ["PYROMIND_TASK_CONTEXTS"])["workflow_run"]
with PyroMindAPIClient(max_retries=0) as client:
    task = submit_workflow(client, workflow, out_id=out_id)
    print(json.dumps({"task_id": task.task_id, "status": task.status}))
```

示例中的 `submit_workflow` 是普通 Python 函数，直接调用 SDK，不经过旧 Tool 或统一提交
代理。含环境声明的图必须走此函数；不要调用 helper 后再次调用 `client.studio.create()`。

超时、断连或响应解析失败且没有任务 ID 时，提交结果未确认，不能自动重投。
`submit_workflow` 用 `WorkflowSubmissionUncertain` 表达此情况；原生 SDK 异常也按同一规则处理。
明确的参数、鉴权或资源拒绝可按错误修正。取得 `task_id` 后保存该真实 ID，告知已提交并
结束当前轮次等待回调，不在 terminal 阻塞轮询。

## Project 绑定与业务验收

平台创建与 Project 元数据更新是两个动作。沿用现有 Project task/attempt、执行事件及
元数据更新入口；绑定失败只重试绑定，保留真实任务 ID，不再次创建平台任务。
对于计划内的 Train，Runtime 会根据签名回调的任务号，在项目所有者的登录态下从 Studio 读取实际执行图并归档。回调时缺少登录态，恢复原会话后可补登记同一任务。训练报告与工作流分别保存。
SDK 查询、文件操作、停止任务不要求先创建 Project 或业务 execution。
Dataset Recipe 的 task、实际工作流、代码、冻结配方及输出元数据在可信回调确认归属后，
通过已有 `product_versions` 的 `bind_execution_metadata` 动作单独登记；不是提交前置
条件，登记失败也不重新 create。SDK `get_task` 成功不能代替可信回调的归属确认。

先读取无密钥的实际校验图与冻结的 `dataset-recipe.json`。以下是对话中 Tool 调用的参数
示意，`safe_validated_graph` 和 `frozen_recipe` 是已读取的 JSON 对象，不在 terminal
执行此调用：

```python
product_versions(
    action="bind_execution_metadata",
    command_id=f"bind-sdk-{task_id}",
    arguments={
        "task_id": task_id,
        "workflow": safe_validated_graph,
        "dataset_recipe": frozen_recipe,
        "output_dir": output_dir,
    },
)
```

需要保存代码时再传可选 `code`。成功返回 `status="metadata_bound"`；同一事实重试保持
相同 `command_id` 和参数，已登记任务不允许改写成另一份工作流、Recipe 或输出。
不要传提交时绑定了凭据的请求图，也不要因登记暂时失败而重新创建平台任务。

平台 Succeeded 仅说明作业完成。收到回调后检查报告、数据和模型产物，再更新业务结果；
不要在提交时把 Project 阶段标为完成，也不要用最后一条任务状态覆盖其他分片的结果。

## 调试预算

先调用保留的 `validate_workflow_dsl(test_mode=True)`。直接读取其 content 返回的
`validated_workflow_path`，文件中是已固定测试模式并通过资源校验的图；
不要重新转 DSL、复制重写图，或把普通正式图配上 `validated=True`。
缺少路径或出现 `artifact_unavailable` 时停止 SDK 提交，即使 `valid=true`。

`PYROMIND_DEBUG_STATE_PATH` 指向当前执行工作区
`public_data/.pyromind/workflow-debug-state.json`，沿用旧
`pyromind_workflow_attempts` 计数。使用 `DebugBudget.load(Path(...))` 读取；
`submit_debug_workflow` 在文件锁内重新读取、提交并原子保存，恢复与切回旧入口不重置预算。
Project 调试仍按上面的 task/attempt 规则选择对应 `out_ids`，不改用普通会话关联。

```python
import json
import os
import sys
from pathlib import Path

from pyromind_sdk import PyroMindAPIClient
from pyromind_common.platform.workflows import (
    DebugBudget,
    DebugBudgetPersistenceError,
    submit_debug_workflow,
)

validated_workflow_path = sys.argv[1]
workflow = json.loads(Path(validated_workflow_path).read_text())
budget = DebugBudget.load(Path(os.environ["PYROMIND_DEBUG_STATE_PATH"]))
out_id = json.loads(os.environ["PYROMIND_TASK_CONTEXTS"])["workflow_debug"]
with PyroMindAPIClient(max_retries=0) as client:
    try:
        task = submit_debug_workflow(
            client, workflow, out_id=out_id, budget=budget, validated=True
        )
    except DebugBudgetPersistenceError as error:
        print(json.dumps({"task_id": error.response.task_id, "budget_saved": False}))
        raise SystemExit(1)
    print(json.dumps({"task_id": task.task_id, "attempt": budget.accepted}))
```

预算保存失败但响应已带任务 ID 时，保留该真实 ID，只修复预算记录，不再 create。
预算路径缺失、已有记录损坏或达到 10 次上限时停止，不自行另建预算文件或归零。
只有用户明确要求继续调试时，才对同一共享记录开始新一轮。

部署回退使用 `OH_PLATFORM_SDK_ENABLED=false`；旧实现继续保留，只恢复原授权范围内的
Tool。恢复入口和读取回执都不意味着重新提交已有任务。
