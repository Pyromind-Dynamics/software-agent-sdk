---
name: generate-workflow-dsl
description: >-
  当用户想在 Pyromind 平台上用自己的数据训练/微调模型（典型形式是贴一个 storage 数据集
  相对路径说"帮我用这些数据训练一个模型"）、要求生成或修改训练工作流、跑基线评测
  （benchmark），或单独要求校验当前工作流 DSL 时使用此技能。流程：先用 preview_dataset
  了解数据 → 引导先跑基模 bench 基线（用户可跳过）→ 默认生成 SFT 工作流并按决策链自动
  配参 → 写入 workflow.py 后必须通过 validate_workflow_dsl 校验，报错则修复后重新校验。
triggers:
- 训练一个模型
- 训练模型
- 微调
- fine-tune
- train a model
- 生成工作流
- generate workflow
- 训练工作流
- training workflow
- workflow DSL
- benchmark
- 基线评测
- 校验工作流
- 检查工作流
- validate workflow
---

# 生成 Pyromind 工作流 DSL

## 概述

典型用户不会说"给我一个 SFT 工作流"。他们通常是把一批（格式往往比较乱的）训练数据传到
storage，把相对路径贴进对话框，说"帮我用这些数据训练一个模型"；也可能顺带描述自己的行业
背景和想落地的业务场景。你的职责是把这类模糊需求落成可运行的工作流 DSL。

SFT/DPO/GRPO 等后训练方法的原理你已经掌握，不要向用户科普，也不要依赖用户说出这些术语。
注意力放在四件事上：数据格式判断、节点选型、参数决策、DSL 正确性。

如果用户只是要求校验现有 workflow.py（没有生成/修改诉求），直接跳到"校验循环"。

## 工作流程

```
1. 数据理解      preview_dataset 看清格式/条数/长度/是否多模态
2. 基线评测引导  建议先用基模跑 bench（用户可明确跳过）
3. 生成训练工作流 默认 SFT；数据明摆着是偏好对/仅 prompt 才用 DPO/GRPO
4. 自动配参      数据(N,L) → 模型类型 → 规模 → LoRA/Full → GPU
5. 写入与校验    写 workflow.py → validate_workflow_dsl 循环直到通过
```

### 第 1 步：数据理解

用户贴了 storage 数据路径时，先调用 `preview_dataset` 工具查看真实数据，再决定节点和
字段映射参数，不要凭路径名猜格式。需要记下三个配参输入：条数 N、P95 序列长度 L、是否
含图片/视频。

根据样本行判断字段映射节点：

| 数据形态 | 节点 |
|----------|------|
| `messages` 数组（多轮对话，可含 tool_calls） | DatasetConfigBuilderMessageNode |
| 独立的 prompt / response 字段 | DatasetConfigBuilderTextNode |
| 含图片/视频字段 | DatasetConfigBuilderVisionNode |
| chosen/rejected 偏好对 | DPO 数据（字段映射同上，加 rejected_field） |

用户描述了行业背景和业务场景时，把这些信息用于指标推荐和基模选择，不要只依赖数据集
反推。信息不足时针对性地问一两个问题，不要发问卷。

### 第 2 步：基线评测引导（生成训练工作流之前）

在生成训练工作流之前，先建议用户用基模（主要是 Qwen 系列）跑一个 bench 基线：部署基模
推理服务，按业务场景推荐指标，从用户数据里采样一部分跑出基线分数，训练后用同一套评测
对比才能说明训练效果。用户明确说不需要/跳过时，直接进入第 3 步，不要反复推销。

bench 工作流结构（完整模板见 [references/example-workflows.md](references/example-workflows.md) 示例5）：

```
基模(CloneAndCacheModel) → VLLMInference 部署
用户数据(PathJoinNode → DatasetToJsonlNode 如需) → DatasetConfigBuilderNode
两路汇入 ModelEvalApiNode（max_samples 控制采样条数）+ 指标配置
```

指标选择，按业务场景推荐：

| 场景 | entry |
|------|-------|
| 数学/数值答案 | `compute_gsm8k`（默认） |
| 分类/精确匹配 | `compute_accuracy` |
| 翻译/受约束生成 | `compute_bleu` |
| 摘要/长文本生成 | `compute_rouge_l` |

- 现成指标用 `MetricsConfigBuilderNode`，entry 必须填完整枚举值
  `examples/eval_metrics_common.py:<上表函数名>`，如
  `examples/eval_metrics_common.py:compute_gsm8k`
- 现成指标都不合适时（如工具调用正确率、业务自定义打分），用 `MetricsConfigBuilderCustomNode`：
  1. 在工作区写好指标 py 文件，函数签名
     `fn(gt_text, pred_text, sample, *, metrics_name=None) -> dict | None`，
     返回 dict 中含与指标同名的键、值为 0~1 分数
  2. 调用 `upload_file_to_pyromind` 工具上传，拿到形如 `/workspace/script/agent/acc.py`
     的 storage 路径
  3. entry 填 `<storage路径>:<函数名>`，例如 `/workspace/script/agent/acc.py:acc_func`

### 第 3 步：生成训练工作流

- **默认生成 SFT 工作流**。只有数据明摆着是 DPO 格式（chosen/rejected 偏好对）或 GRPO
  格式（仅 prompt + 可程序化验证的答案/reward），才生成对应类型；拿不准时选 SFT，并向
  用户说明一句判断依据。
- 基模用 `CloneAndCacheModel`，可选值只有：
  - 纯文本：`Qwen/Qwen3-0.6B`、`Qwen/Qwen3-1.7B`、`Qwen/Qwen3-4B`
  - 多模态：`Qwen/Qwen3-VL-2B-Instruct`、`Qwen/Qwen3-VL-4B-Instruct`
- 用户要用列表之外的开源模型时，改用 `DownloadAndCacheModel`（Download Model 节点）从
  huggingface/modelscope 下载。
- 以 [references/example-workflows.md](references/example-workflows.md) 里最接近的示例为
  模板改参数，不要从零拼节点。用户 storage 数据 + messages 格式的 SFT（最常见场景）直接
  用示例6。

### 第 4 步：自动配参

决策链：**数据集(N, L) → 模型类型(LLM/VL) → 模型规模 → LoRA/Full → GPU 资源**。

生成训练工作流前先读 [references/parameter-decision.md](references/parameter-decision.md)，
按表填 lora_rank、batch_size、learning_rate、epoch 等参数。速记原则：

- LoRA 是默认，Full 只在数据量大且有深度对齐需求且资源够时用
- VL 模型：batch 和 lr 在 LLM 基础上减半，显存需求升一档
- DPO lr ≈ SFT 的 5%~10%；GRPO lr ≈ SFT 的 10%~20%
- 资源不够时依次：Full→LoRA、降 rank、batch 减半 accum 翻倍，**不要先降 lr**

### 第 5 步：写入与校验

1. 用 `apply_patch` 把 DSL 写入当前工作目录的相对路径 `workflow.py`（修改已有工作流也
   只编辑这个路径）。不要手写会话目录的长绝对路径；如用 `file_editor` 也传 `workflow.py`。
   不要只口头说已生成——必须实际调用工具创建或修改文件。
2. 每次写入/修改后立即进入下方"校验循环"。
3. 校验通过后正常结束：不要调用额外发布工具，也不要主动调用 `run_workflow(test_mode=true)`
   （那是用户明确要求"测试/调试/试跑"时由 **debug-workflow** 技能通过 `run_workflow` test
   模式处理）。若用户要求**正式运行/发布**工作流，使用 `run_workflow(test_mode=false)`（默认）。
   后端会在本轮结束后自动把 workflow.py 同步给前端。

## DSL 语法

`workflow.py` 是**声明式 DSL**，只是借用 Python 语法描述节点及连线，不能本地执行，也不要
按 Python 运行时语义推理它的行为。

```python
# workflow: <工作流名称>

variable_name = NodeType(
    id="<唯一节点ID>",
    param1="静态值",
    param2=another_variable.output_port,  # 引用上游节点输出端口
)
```

规则：文件开头 `# workflow: <name>` 注释；每个节点一个变量赋值；`id` 必须唯一（数字
字符串）；被引用的节点必须先定义；静态值支持字符串/数字/浮点（如 `1e-4`）/布尔。

## 节点与知识库检索

检索优先级：**内置示例模板 → 直接读相关节点文档 → 必要时宽泛 grep**。

- 节点速查表和常用参数见 [references/node-reference.md](references/node-reference.md)
- 节点契约文档路径固定：`<知识库绝对路径>/nodes/<NodeType>/<NodeType>.md`（知识库绝对
  路径见路由提示）。已确定要用的节点直接读其文档，只读会实际用到或参数不确定的
- grep 只用于补充检索，用宽泛关键词（`SFT`、`DPO`、`GRPO`、`dataset`、`reward` 或精确
  NodeType），不要用完整标题、`^###`、`# workflow: ...` 这类依赖格式的 pattern
- 只有用户问平台概念、Studio 操作、SDK 脚本写法时，才检索知识库的 `basic/`、`studio/`、
  `sdk/`、`jupyterlab/` 子目录

## 校验循环

`workflow.py` 每次被创建或修改后，立即用 `validate_workflow_dsl` 校验（不传 `dsl` 参数，
工具自己读当前工作目录的 `workflow.py`），不要在未校验的情况下结束：

1. `valid == true`：通过。`warnings` 非空也算通过，结束语里提一句即可，不要为消除警告
   继续改
2. `valid == false`：`errors` 是结构化列表，含 `code`、`message`、`node_id`/`node_type`/
   `edge_id`/`field` 定位信息，`detail` 里的 `node_code`/`target_node_code`/
   `source_node_code` 是对应的原始 DSL 语句。用这些字段直接定位，用 `apply_patch` 只修改
   报错指向的那几处，不要因为一条报错重写整个文件；改完回到步骤 1
3. 调用本身失败（网络错误、非 2xx、JSON 解析失败）不代表工作流有问题：如实告知用户校验
   服务暂时不可用，不要当成 DSL 错误去"修复"
4. **最多重试 5 轮**（自己计数）。5 轮后仍不通过，或同一 `code`/`node_id` 连续两轮没消失，
   停止重试并如实说明剩余错误

`validate_workflow_dsl`（静态结构校验，秒级返回）和 `run_workflow`（提交平台异步执行）职责不同：

| 工具 | 模式 | 职责 |
|------|------|------|
| `validate_workflow_dsl` | — | 每次写入后必须调；只做静态结构校验 |
| `run_workflow` | `test_mode=true` | 平台真实 test 执行；属于 **debug-workflow** 技能；callback 返回终态 |
| `run_workflow` | `test_mode=false`（默认） | 正式运行/发布；非生成技能默认行为 |

前者每次写入后必须调；后者只在用户明确要求"测试/调试/试跑"（test 模式）或"运行/发布"
（正式模式）时用。报错来源要分清再修。原 `debug_workflow` / `pyromind_debug` 已停用，统一走
`run_workflow`。

## 参考文件

- [references/parameter-decision.md](references/parameter-decision.md)：配参决策链的档位表与推荐超参
- [references/node-reference.md](references/node-reference.md)：节点速查表与常用节点参数
- [references/example-workflows.md](references/example-workflows.md)：6 个示例工作流模板
