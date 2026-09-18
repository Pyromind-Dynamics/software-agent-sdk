# 平台命令节点执行契约

## 节点选择

| 场景 | 平台节点 | 资源 |
|---|---|---|
| Storage 模型需要临时启动 | `CustomCommandNode` | `command`、`cpu`、`memory`、`gpu_count`、`gpu_product` |
| 已有可访问 inference endpoint | `CustomCommandCPUNode` | `command`、`cpu`、`memory` |

`CustomCommandNode` 是知识库中的 GPU 命令节点真实名称；不要使用不存在的
`CustomCommandGPUNode`。它执行 shell 命令并把 stdout/stderr 合并为 `result`。

这个选择适用于所有文本、图像、多模态和结构化输出测试集。不要插入 `VLLMInference` 后再连接
`ModelEvalApiNode`，也不要通过 `MetricsConfigBuilderNode` 或 `MetricsConfigBuilderCustomNode` 绕过
Agent 制定的 Rubric。平台没有所需命令节点时应停止，而不是选择另一条评测架构。

## GPU 命令

脚本在同一 GPU 容器中启动 vLLM 和完成评测，避免跨 Pod 使用 `127.0.0.1`：

`CustomCommandNode` 默认使用 `gpu_product="NVIDIA-L40S"`。只有用户明确指定其他型号，或
`validate_workflow_dsl()` 的实时契约不支持 L40S 时才更改。

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

```text
python3 /workspace/.pyromind-agent/<conversation_id>/evaluate_inference.py \
  --model-path <storage-model-path> \
  --model default \
  --dataset-path <storage-jsonl-path> \
  --dataset-config /workspace/.pyromind-agent/<conversation_id>/inference_dataset_config.json \
  --evaluation-config /workspace/.pyromind-agent/<conversation_id>/inference_evaluation_config.json \
  --output-dir <storage-output-dir> \
  --gpu-count <same-as-node-gpu-count> \
  --max-model-len <context-length> \
  --limit 0
```

`output_dir` 必须包含本次配置或运行的唯一标识，不复用历史运行目录，避免旧 `run_manifest.json` 与
新配置冲突或覆盖旧报告。

命令节点的 `gpu_count` 必须与脚本的 `--gpu-count` 一致。脚本用它配置 vLLM tensor parallel。
如果镜像没有安装 `vllm`，节点会失败并输出明确的模块加载错误；不要在命令中临时 `pip install`。
DSL 中的 `command` 使用静态字符串字面量，以便生成前门禁能够确定性验证实际执行入口和参数。

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
中不出现 Key 值。

## 输出

脚本在 `output_dir` 生成：

- `predictions.jsonl`：input、GT、prediction、Rubric score/evidence 和逐条错误
- `metrics.json`：通过率、聚合指标、请求数、成功 prediction 数和成功评分数
- `evaluation_report.html`：独立 HTML 报告
- `run_manifest.json` 与 `predictions.partial.jsonl`：断点续跑和运行隔离

`CustomCommandNode.result` 是命令日志与最终路径 JSON，不是报告文件本身。报告保存在 Storage
`output_dir`。

运行脚本不调用 Rubric Planner。正常情况下每个样本只向被测 endpoint 发送一次 inference 请求，
随后由 Python 按配置中的固定 Rubric 评分。
