---
name: inference-evaluation
description: >-
  为 Pyromind 设计并生成 Agent 制定 Rubric 的通用模型推理评测流程：预览带 ground truth 的测试集，
  由当前 Agent 固化任务适配的评价指标，再通过平台已有 CustomCommandNode/CustomCommandCPUNode
  执行推理、确定性评分和 HTML 报告。所有 inference 测试、GT 对比或评测报告请求均使用此流程；
  不创建自定义节点，不使用 Benchmark By Api 替代，不执行正式工作流。
---

# 通用 Inference Rubric 评测

当前 Agent 负责理解测试集和制定 Rubric；被测模型只负责生成 prediction。运行脚本读取已固化的
Rubric 做评分和报告，不在运行时生成检查项，不允许被测模型充当 Rubric Planner。

## 强制架构

- Storage 模型：DSL 只包含一个 `CustomCommandNode`，默认申请 `NVIDIA-L40S`，在同一 GPU
  容器启动 vLLM 并执行评测。只有用户明确指定或实时平台契约不支持 L40S 时才改用其他型号。
- 已部署 endpoint：DSL 只包含一个 `CustomCommandCPUNode`，直接请求已有服务。
- 禁止使用 `VLLMInference`、`ModelEvalApiNode`、`MetricsConfigBuilderNode`、
  `MetricsConfigBuilderCustomNode` 或 Benchmark By Api 代替本流程。
- `inference_evaluation_config.json.mode` 必须为 `agent_rubric`，并包含 Agent 写出的非空
  `rubrics`；禁止写 `planner_prompt` 或运行时 Planner 配置。
- 如果平台没有所需命令节点，停止并报告缺失契约，不创建新 NodeType。

## 固定执行顺序

1. 当前画布存在时，只读取一次 `public_data/workflow_canvas/workflow.py`。
2. 读取 [references/input-contract.md](references/input-contract.md) 和
   [references/rubric-authoring.md](references/rubric-authoring.md)。需要生成命令节点 DSL 时，再读取一次
   [references/command-node-pipeline.md](references/command-node-pipeline.md)。
3. 使用 `preview_dataset` 预览测试集：先确认实际文件，再对目标 JSON/JSONL 取代表性样本；确认字段、
   模态、GT 结构和输出约束。不得从路径名或 PCB 示例推断任务。
4. 当前 Agent 按 Rubric 范式选择 2～5 个互不重复的评价维度，写入
   `public_data/inference_evaluation_config.json`；同时写入字段映射
   `public_data/inference_dataset_config.json`。
5. 上传前先运行配置预检；失败时修正配置，不上传无效版本：
   `python "$PYROMIND_SKILLS_PATH/inference-evaluation/scripts/validate_pipeline.py" --configs-only public_data/inference_dataset_config.json public_data/inference_evaluation_config.json`。
6. 直接运行 staging 命令，不读取或改写脚本：
   `python "$PYROMIND_SKILLS_PATH/inference-evaluation/scripts/stage_runtime.py" public_data/evaluate_inference.py`。
7. 上传评测脚本和两个配置。只有上传工具返回 Storage 绝对路径后才生成 DSL。上传结果
   `/.pyromind-agent/...` 是 Storage 逻辑路径；写入命令参数时必须转换为容器挂载路径
   `/workspace/.pyromind-agent/...`。
8. 按命令节点参考模板直接写 DSL，默认 `gpu_product="NVIDIA-L40S"`，运行本地门禁，再调用
   `validate_workflow_dsl()`。仅根据用户要求或校验错误修正资源枚举；不执行正式工作流。

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
  或 `Operation not permitted`，停止终端探测并报告 Agent Server terminal backend 故障。
- 不伪造 `/.pyromind-agent/...` 上传路径；只对上传工具实际返回的路径增加 `/workspace` 容器挂载
  前缀。命令中不得出现裸 `/.pyromind-agent/...`，也不在命令或 JSON 中写明文 Secret。

## 门禁与成功标准

写完 DSL 和配置后运行：

```text
python "$PYROMIND_SKILLS_PATH/inference-evaluation/scripts/validate_pipeline.py" \
  public_data/workflow_canvas/workflow.py \
  public_data/inference_dataset_config.json \
  public_data/inference_evaluation_config.json
```

门禁失败必须修改后重跑。报告只有在 `total`、`request_count`、`successful_predictions` 和
`evaluated_cases` 均大于 0 时有效。交付时说明字段映射、Agent 制定的 Rubric、节点资源、输出目录和
DSL 校验结果；不得声称 GPU 推理已经运行。用户随后明确要求试跑时，交给 `debug-workflow`。
