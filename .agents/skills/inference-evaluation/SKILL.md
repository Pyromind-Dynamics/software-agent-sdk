---
name: inference-evaluation
description: >-
  为 Pyromind 设计并生成 Agent 制定 Rubric 的通用模型推理评测流程：预览带 ground truth 的测试集，
  由当前 Agent 固化任务适配的评价指标，再通过平台已有 VLLMInference 和 CustomCommandCPUNode
  分别提供推理 endpoint、执行确定性评分并生成 HTML 报告。所有 inference 测试、GT 对比或评测报告请求均使用此流程；
  不创建自定义节点，不使用 Benchmark By Api 替代；只有用户明确要求实际运行评测时才正式提交工作流。
---

# 通用 Inference Rubric 评测

当前 Agent 负责理解测试集和制定 Rubric；被测模型只负责生成 prediction。运行脚本读取已固化的
Rubric 做评分和报告，不在运行时生成检查项，不允许被测模型充当 Rubric Planner。

## 强制架构

- Storage 模型：DSL 只包含一个 `VLLMInference` 和一个 `CustomCommandCPUNode`。
  `VLLMInference` 默认申请一张 `NVIDIA-L40S` 并输出 OpenAI-compatible endpoint；CPU 节点通过
  `param=inference.endpoint` 接收该 endpoint，并在 command 中直接使用 `--endpoint "$param"`，只执行
  请求、Rubric 评分和报告生成。不得把 `$param` 改成大写变量或嵌套 shell fallback。只有用户明确指定
  或实时平台契约不支持 L40S 时才改用其他型号。
- 已部署 endpoint：DSL 只包含一个 `CustomCommandCPUNode`，直接请求已有服务。
- 禁止使用 `CustomCommandNode` 自行启动 vLLM；禁止用 `ModelEvalApiNode`、
  `MetricsConfigBuilderNode`、`MetricsConfigBuilderCustomNode` 或 Benchmark By Api 代替本流程。
- `inference_evaluation_config.json.mode` 必须为 `agent_rubric`，并包含 Agent 写出的非空
  `rubrics`；禁止写 `planner_prompt` 或运行时 Planner 配置。
- 如果平台没有所需命令节点，停止并报告缺失契约，不创建新 NodeType。

## 执行模式与鉴权

- 生成、修改或校验评测流程时只产出并校验 DSL，不提交工作流。
- 用户明确要求“实际运行评测”“生成报告”或“正式运行”时，使用正式 `run_workflow` 提交，不能调用
  `workflow_debug`，也不能传 `test_mode=true`。`VLLMInference` 的 test/debug 执行可能只返回校验占位值，
  不是可供下游 CPU 节点请求的真实 endpoint。
- 如果当前 Agent 没有正式 `run_workflow` 工具，完成校验后让用户从 Studio 的正式运行入口启动；不得
  回退到 `workflow_debug` 模拟端到端推理评测。
- Storage 模型的两节点工作流使用 `VLLMInference` 输出的内部 endpoint，不需要公开部署的 API Key，
  也不得把外部部署 Key 拼入 CPU 节点命令。
- 已部署 endpoint 需要鉴权时，命令只能写 `--api-key-env <ENV_NAME>`，并由平台 Secret/环境变量注入。
  如果实时节点契约没有安全的 Secret 注入能力，停止并说明缺失契约；不得要求用户把 Key 写入 DSL、
  JSON、`param` 或命令文本。

## 固定执行顺序

1. 当前画布存在时，只读取一次 `public_data/workflow_canvas/workflow.py`。
2. 读取 [references/input-contract.md](references/input-contract.md) 和
   [references/rubric-authoring.md](references/rubric-authoring.md)。需要生成命令节点 DSL 时，再读取一次
   [references/command-node-pipeline.md](references/command-node-pipeline.md)。
3. 使用 `preview_dataset` 预览测试集：先确认实际文件，再对目标 JSON/JSONL 取代表性样本；确认字段、
   模态、GT 结构和输出约束。对每个媒体字段至少验证一个实际文件路径。媒体值已经是相对 JSON/JSONL
   目录的路径（例如 `images/a.jpg`）时，`inference_dataset_config.json` 中必须完全不存在
   `media_base_dir` 键，不能写空串，也不得再次写入数据集目录。用
   `JSONL 父目录 / 样本媒体路径` 解析并预览成功后才能继续；若解析结果重复出现数据集目录片段，立即
   修正配置。不得从路径名或 PCB 示例推断任务。
4. 当前 Agent 按 Rubric 范式选择 2～5 个互不重复的评价维度，写入
   `public_data/inference_evaluation_config.json`；同时写入字段映射
   `public_data/inference_dataset_config.json`。
5. 上传前先运行配置预检；失败时修正配置，不上传无效版本：
   `python3 "$PYROMIND_SKILLS_PATH/inference-evaluation/scripts/validate_pipeline.py" --configs-only public_data/inference_dataset_config.json public_data/inference_evaluation_config.json`。
6. 直接运行 staging 命令，不读取或改写脚本：
   `python3 "$PYROMIND_SKILLS_PATH/inference-evaluation/scripts/stage_runtime.py" public_data/evaluate_inference.py`。
7. 上传评测脚本和两个配置。每个新会话都必须上传本会话刚通过预检的配置，不能复用旧会话的
   `/.pyromind-agent/<conversation_id>/inference_dataset_config.json`。只有上传工具返回 Storage 绝对路径后
   才生成 DSL。上传结果
   `/.pyromind-agent/...` 是 Storage 逻辑路径；写入命令参数时必须转换为容器挂载路径
   `/workspace/.pyromind-agent/...`。
8. 按节点参考模板直接写 DSL。Storage 模型使用 `VLLMInference`（默认
   `gpu_product="NVIDIA-L40S"`、`gpu_count=1`、`port=3000`）连接
   `CustomCommandCPUNode`；未提供 served model name 时使用 `default`，不得从 checkpoint 目录名推断。
   已有 endpoint 只使用 CPU 节点。运行本地门禁，再调用
   `validate_workflow_dsl()`。仅根据用户要求或校验错误修正资源枚举。默认到此停止；仅在用户明确要求
   实际运行评测时，按“执行模式与鉴权”正式提交。

## Rubric 制定原则

- Rubric 来自当前 Agent 对测试集字段、GT、提示词和代表性样本的分析，不来自 inference endpoint。
- 优先使用确定性算子验证 JSON、字段值、数量、数值范围和 bbox IoU；这些规则必须能由代码复现。
- 每项只评价一个目标，具有唯一 `name`、清晰 `criterion`、正数 `weight` 和完整 `evaluator`。
- 会让结果失去业务意义的核心项设置 `required: true`；必需项失败时，即使加权总分达标也不通过。
- 权重总和不强制为 1，运行时会归一化。关键任务正确性应高于格式或表述质量。
- 不添加测试集没有要求的风格偏好，不把同一错误重复计入多个高权重指标。
- 语义判断只有在用户提供独立 Judge endpoint 时才能使用；不得默认让被测模型自评。

## 行为边界

- 不得用 `read`、`grep`、`rg`、`cat` 或终端读取
  `inference-evaluation/scripts/stage_runtime.py`、`evaluate_inference.py`、`validate_pipeline.py`。
- 不得进入宿主机仓库搜索 DSL、节点类、SDK 实现或转换器源码。节点知识与
  `validate_workflow_dsl()` 的实时契约是唯一依据。
- 不重复读取同一知识文件，不为寻找实现细节遍历 `.agents`、`knowledge` 或源码树。
- 引用文件使用 `read` 直接读取 SKILL 中的已知路径；不要用 terminal `ls` 枚举 Skill 目录。
- 原样执行本 Skill 给出的 Python 命令；不要添加绝对宿主机 `cd`、内联环境变量、`env -i` 或
  heredoc。若两次终端调用返回相同的 `WORKSPACE_SANDBOX_UNAVAILABLE`、`Argument list too long`
  或 `Operation not permitted`，立即停止终端探测并报告 Agent Server terminal backend 故障；不得再用
  `echo`、`ls`、`cp`、绝对宿主机路径或更短命令探测阈值。
- 不伪造 `/.pyromind-agent/...` 上传路径；只对上传工具实际返回的路径增加 `/workspace` 容器挂载
  前缀。命令中不得出现裸 `/.pyromind-agent/...`，也不在命令或 JSON 中写明文 Secret。

## 门禁与成功标准

写完 DSL 和配置后运行：

```text
python3 "$PYROMIND_SKILLS_PATH/inference-evaluation/scripts/validate_pipeline.py" \
  public_data/workflow_canvas/workflow.py \
  public_data/inference_dataset_config.json \
  public_data/inference_evaluation_config.json
```

门禁失败必须修改后重跑。报告只有在 `total`、`request_count`、`successful_predictions` 和
`evaluated_cases` 均大于 0 时有效。交付时说明字段映射、Agent 制定的 Rubric、节点资源、输出目录和
DSL 校验结果；未正式提交时不得声称 GPU 推理已经运行。用户要求实际生成报告时使用正式
`run_workflow`，不得交给 `debug-workflow`。
