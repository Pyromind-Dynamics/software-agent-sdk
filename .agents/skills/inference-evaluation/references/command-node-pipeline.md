# 平台推理与评测节点执行契约

## 节点选择

| 场景 | 平台节点 | 资源 |
|---|---|---|
| Storage 模型需要临时启动 | `VLLMInference` → `CustomCommandCPUNode` | GPU 节点提供 endpoint，CPU 节点执行评测 |
| 已有可访问 inference endpoint | `CustomCommandCPUNode` | `command`、`cpu`、`memory` |

Storage 模型必须拆成两个职责明确的节点：`VLLMInference` 使用平台维护的 vLLM 镜像加载模型，
`CustomCommandCPUNode` 请求上游 endpoint，执行固定 Rubric 评分并生成报告。不要使用
`CustomCommandNode` 或不存在的 `CustomCommandGPUNode` 在评测脚本里再次启动 vLLM。

这个选择适用于所有文本、图像、多模态和结构化输出测试集。不要把 `VLLMInference` 连接到
`ModelEvalApiNode`，也不要通过 `MetricsConfigBuilderNode` 或 `MetricsConfigBuilderCustomNode` 绕过
Agent 制定的 Rubric。平台没有所需节点或 CPU 节点不支持 `param` 输入时应停止，而不是选择另一条
评测架构。

## 执行模式

两节点链路的端到端评测必须正式运行。`workflow_debug` 会以 `test_mode=true` 提交，平台可能让
`VLLMInference` 返回仅用于连线校验的占位值；CPU 节点收到该值后无法请求真实模型服务。因此：

- 只生成或修改工作流时，写完 DSL 后仅执行本地门禁和 `validate_workflow_dsl()`。
- 用户明确要求实际评测或生成报告时，使用正式 `run_workflow`，不设置 `test_mode`。
- 当前 Agent 没有正式运行工具时，让用户在 Studio 使用正式运行入口，不回退到
  `workflow_debug`。

正式运行时，CPU 节点消费同一工作流中 `VLLMInference` 输出的内部 endpoint，不需要外部部署的
API Key。不要为了避开 test mode 而把公开 endpoint 的 Key 写入 `command` 或 `param`。

## Storage 模型的两节点 DSL

`VLLMInference` 默认使用 `gpu_product="NVIDIA-L40S"`、`gpu_count=1`、`port=3000`。
`max_model_len` 来自用户配置或模型约束，未提供时使用节点默认值。不要在 CPU command 中写
`127.0.0.1`：两个节点是不同 Pod，必须消费节点输出的 endpoint。

CPU 节点用 `param=inference.endpoint` 建立依赖并接收 endpoint。平台把这个输入注入为小写 shell
变量 `param`，命令必须直接使用 `--endpoint "$param"`。不要改成大写 `$PARAM`，也不要使用
`${PARAM:-${param:-}}` 等嵌套 shell fallback；节点运行器只对直接的 `$param` 引用执行输入替换。
生成后必须调用 `validate_workflow_dsl()` 验证端口绑定。

```python
inference = VLLMInference(
    id=1,
    model_path="/workspace/models/checkpoints/<model>",
    port=3000,
    max_model_len=8192,
    gpu_count=1,
    gpu_product="NVIDIA-L40S",
)

evaluation = CustomCommandCPUNode(
    id=2,
    command=(
        "python3 /workspace/.pyromind-agent/<conversation_id>/evaluate_inference.py "
        "--endpoint \"$param\" --model <served-model-name> "
        "--model-reference /workspace/models/checkpoints/<model> "
        "--dataset-path /workspace/datasets/<dataset>/eval.jsonl "
        "--dataset-config /workspace/.pyromind-agent/<conversation_id>/"
        "inference_dataset_config.json "
        "--evaluation-config /workspace/.pyromind-agent/<conversation_id>/"
        "inference_evaluation_config.json "
        "--output-dir /workspace/outputs/<unique-run-id> --limit 0"
    ),
    cpu=4,
    memory=32,
    param=inference.endpoint,
)
```

CPU 命令的 `--model` 使用用户或 endpoint 契约明确提供的 served model name。未提供时固定使用
`default`；不能把 `model_path` 的 basename、checkpoint 目录名或报告中的模型引用当作 served model
name。`--model-reference` 只用于报告记录模型来源，不参与 API 路由。

## Storage 路径映射

上传工具返回的是 Storage 逻辑路径。例如：

```text
/.pyromind-agent/<conversation_id>/evaluate_inference.py
```

命令节点的容器把 Storage 挂载在 `/workspace`，因此 shell command 中必须转换成：

```text
/workspace/.pyromind-agent/<conversation_id>/evaluate_inference.py
```

两个上传配置文件使用相同转换。若上传工具已经返回 `/workspace/...`，保持原值，不能重复增加前缀。
模型、数据集和输出目录同样使用 `/workspace/...` 容器路径。裸 `/.pyromind-agent/...` 只适合表示
Storage 对象，不能作为命令容器中的文件路径。

`output_dir` 是运行状态命名空间，不只是报告目录。模型、数据集、配置或评测脚本变化时必须使用新的
唯一目录，避免旧 `run_manifest.json` 与新配置冲突或覆盖旧报告。只有恢复同一评测的中断任务时才复用
原目录：`predictions.partial.jsonl` 中已有非空 prediction 的样本会跳过，无 prediction 的超时、连接或
endpoint 错误会重新请求；Rubric 未通过但已有 prediction 的样本仍视为推理已完成，不重复计费。

评测脚本不安装、升级或卸载 vLLM、Transformers、Torch 等推理依赖。模型运行时完全由
`VLLMInference` 镜像负责；CPU 节点只需要 Python 标准库和已暂存的评测脚本。DSL 中的 `command`
使用静态字符串字面量，以便生成前门禁能够确定性验证实际执行入口和参数。

## 已部署 endpoint 命令

```text
python3 /workspace/.pyromind-agent/<conversation_id>/evaluate_inference.py \
  --endpoint <openai-compatible-endpoint> \
  --model <served-model-name> \
  --model-reference <deployment-or-checkpoint-id> \
  --dataset-path <storage-jsonl-path> \
  --dataset-config /workspace/.pyromind-agent/<conversation_id>/inference_dataset_config.json \
  --evaluation-config /workspace/.pyromind-agent/<conversation_id>/inference_evaluation_config.json \
  --output-dir <storage-output-dir> \
  --limit 0
```

如果需要 API Key，使用 `--api-key-env <ENV_NAME>`，并由平台环境/Secret 提供该变量；命令和 JSON
中不出现 Key 值。若 `CustomCommandCPUNode` 的实时契约没有提供 Secret/环境变量注入能力，不生成
带鉴权的外部 endpoint 工作流；改用正式运行的内部两节点链路，或报告平台缺少安全注入契约。

## 输出

脚本在 `output_dir` 生成：

- `predictions.jsonl`：input、GT、prediction、Rubric score/evidence 和逐条错误
- `metrics.json`：通过率、聚合指标、请求数、成功 prediction 数和成功评分数
- `evaluation_report.html`：独立 HTML 报告
- `run_manifest.json` 与 `predictions.partial.jsonl`：断点续跑和运行隔离

`CustomCommandCPUNode.result` 是命令日志与最终路径 JSON，不是报告文件本身。报告保存在 Storage
`output_dir`。

运行脚本不调用 Rubric Planner。正常情况下每个样本只向被测 endpoint 发送一次 inference 请求，
随后由 Python 按配置中的固定 Rubric 评分。
