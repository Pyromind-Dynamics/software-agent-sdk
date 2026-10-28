# 持续 checkpoint 评测

训练负责保存模型，业务评分由独立的推理和 CPU 评测工作流执行。已有 eval_step_n.jsonl
仍可走 CPU 评分入口；新选模任务不依赖它。所有改动位于 software-agent-sdk。

## 分层与调用

- `pyromind-runtime/domain` 定义注册请求、训练观察、候选状态和选模任务状态。
- `pyromind-runtime/application/CheckpointEvaluationCoordinator` 负责稳定性判断、串行调度、
  重试、收尾和调用既有 `select_checkpoint`。通过平台端口访问外部系统。
- `harness-adapter/checkpoint_evaluation.py` 提供共享平台实现，复用现有推理 DSL 构建器、
  LoRA 合并、平台任务提交、Storage 和训练产物发现；不需要活跃的 Harness 会话。
- Server 生命周期启动 `CheckpointScheduler`，每 60 秒调用 `ConversationRuntime`。
  调度器使用文件锁，每个会话存储只允许一个服务进程驱动轮询。其他进程启动会明确报错。
- Pi/OpenHands 使用同一个 `checkpoint_evaluations` 工具及 Runtime 接口；前端继续消费现有
  外部任务和 notice 事件。子任务终态不会触发 Agent 轮次，父选模任务最终完成时通知一次。

依赖保持 `Server -> Runtime`、`Server -> harness-adapter -> Runtime`；Runtime 不导入
FastAPI、OpenHands、Pi 或服务端。评分脚本只在已有 CPU 评测进程执行。

## 使用入口

Agent 在训练前准备好验证集配置和评价规则，在训练正式提交成功取得 task_id 后调用
`checkpoint_evaluations(action="register")`。已提交训练也可补挂。训练节点 ID 必须明确，
输出目录从实际执行图解析；若尚未生成则轮询继续解析，不根据当前编辑器 DSL 猜测。
工具参数示例见 `.agents/skills/inference-evaluation/references/checkpoint-selection.md`。

同一入口提供 get、cancel、reconcile；HTTP 对应：

`POST /api/v2/pyromind/conversations/{conversation_id}/checkpoint-evaluations`

HTTP body 为 `{action, request_id, arguments}`。register 的 arguments 包含
`training_task_id`、`training_node_id`、`template`、可选 `selection`。HTTP template 直接
使用 JSON 配置及脚本文本；Agent 工具还支持工作区配置和脚本路径。request_id 是 UUID，
重试同一注册必须复用。不同配置使用新的 ID。get/cancel 的 arguments 为 `{run_id}`。
reconcile 使用 `{run_id, step, task_id}`，校验平台任务名称和工作流输出目录后恢复关联。

注册只建立此轮持续选模任务。后续每个 checkpoint 的工作流使用独立 run_id/output_dir，
模板、验证数据、评分脚本、生成设置与资源配置保持相同，仅替换模型引用。
验证集 JSONL 和执行脚本复制到任务的 frozen 目录，媒体内容冻结指纹并在提交前与结果
接收时核对；媒体变化的结果不参加排名。数据集和模型路径按 Storage 路径解释。

## 轮询与结束

每个选模任务最多运行一个子评测。checkpoint 的必要文件和索引完整、非空，且文件名、
大小、纳秒修改时间连续三次轮询不变，才可进入就绪状态。W&B 实际训练 step 超过该
checkpoint 或训练已经终止时，稳定检查通过后可提交；W&B 缺失、延迟或未推进时，
稳定窗口延长至五分钟。W&B `_step` 不作为训练 step；只读取真实 global_step 指标。

训练成功、失败或停止后，继续补扫十分钟，再关闭候选发现，等待已提交和排队的评测。
缺失、未稳定、失败或被训练清理的模型均有覆盖记录；选模结论是成功可比候选中的最佳。
不会因业务阈值达标或 W&B 指标下降提前停止训练，也不会自动开启下一轮训练。

cancel 取消未提交候选，并停止已运行的评测，查询确认终态后结束；训练任务独立。
明确属于权重未就绪的加载失败最多补试一次，且需要重新经过稳定窗口。
评测成功但产物暂不可见时最多等待十分钟，再记录失败。其他失败不自动重跑 GPU。

## 持久化、恢复和兼容

选模任务与子评测复用 Product Store 的 PipelineRun，候选状态存放在父运行结果内，
不改变既有持久化 schema。旧记录和已有预测评分入口继续使用原格式。

候选运行 ID 由父运行、checkpoint 引用和有界尝试次数确定；平台提交前保存候选与子运行。
提交超时或进程中断留下的不明结果不会自动重提。已知任务 ID 可从子运行记录恢复，
其余使用 reconcile 显式关联，无法核实时保留待处理状态。

Cookie、Authorization 和 W&B API key 不写入 Product Store；只保存模型、数据、评分
及运行状态。进程重启后没有临时凭据的任务进入 waiting_credentials；同一用户通过
snapshot、命令或选模入口提供新凭据后继续。任务状态和工作区按 conversation_id 隔离。

独立 test 入口同时支持旧评分批次和新持续选模任务。新任务必须 completed，test 与训练、
验证数据的路径和指纹必须独立，且只能使用选定 checkpoint 及冻结评价规则。test 不参与排名。

默认使用 PYROMIND_SKILLS_PATH 或随代码分发的 .agents/skills。打包二进制使用其资源根；
观察进程可通过 PYROMIND_CHECKPOINT_OBSERVER_PYTHON 指定 Python，二进制默认查找 python3。

## 验证和运行限制

覆盖状态机、串行评测、配置冻结、真实 CPU 评分、训练终态补扫、W&B 缺失、分片不完整、
重启、凭据刷新、取消、任务关联、重复回调、独立 test、架构边界及 HTTP 鉴权。

完整回归命令：

```sh
uv run pytest tests/pyromind_runtime \
  tests/skills/test_training_evaluation.py tests/skills/test_inference_evaluation.py \
  tests/tools/data_preparation/test_inference_submit.py \
  tests/agent_server/test_evaluation_runs_api.py
```

另运行所有变更文件的 pre-commit，以及 inference-evaluation/generate-workflow-dsl 的
quick_validate。历史回归夹具已对齐当前独立评测工具 API 和文件读取提示；架构测试未放宽。

本地使用替身覆盖平台调用和模型请求，CPU 评分运行真实脚本。未执行真实平台/GPU 联调。
就绪检查仍是启发式；训练提前删除 checkpoint 或中途修改媒体会产生覆盖缺口。
部署前需验证服务端挂载与训练 Storage 的可见性、平台终态及权重加载错误信息。
