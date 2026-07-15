---
name: generate-workflow-dsl
description: >-
  当用户提供已清洗的 Storage 相对路径、平台/远程数据集标识，或尚未提供数据，要求在
  Pyromind 平台生成或修改 SFT/DPO/GRPO 训练工作流、跑 benchmark，或校验当前
  workflow.py 时使用。支持预览用户上传数据；缺少数据时用场景匹配的平台测试集占位并
  继续生成。不负责数据清洗；参数由用户诉求、当前数据和资源决定，模板值仅作兜底。
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

数据准备和清洗发生在生成训练工作流之前，当前能力不负责清洗或格式转换。数据入口可以是
用户上传到 Storage 的已清洗数据相对路径、数据集标识，也可以暂时缺省；职责是把它们落成
可运行的工作流 DSL，并清楚标注测试集占位。

SFT/DPO/GRPO 等后训练方法的原理你已经掌握，不要向用户科普，也不要依赖用户说出这些术语。
注意力放在四件事上：数据格式判断、节点选型、参数决策、DSL 正确性。

如果用户只是要求校验现有 workflow.py（没有生成/修改诉求），直接跳到"校验循环"。

## 工作流程

```
1. 数据路由      上传路径先预览；数据集标识直接使用；缺省时选测试集占位
2. 基线评测引导  建议先用基模跑 bench（用户可明确跳过）
3. 生成训练工作流 默认 SFT；数据明摆着是偏好对/仅 prompt 才用 DPO/GRPO
4. 自动配参      数据(N,L) → 模型类型 → 规模 → LoRA/Full → GPU
5. 写入与校验    写 workflow.py → validate_workflow_dsl 循环直到通过
```

### 第 1 步：数据路由

按输入类型路由，不要把下面的已知数据集误当作唯一允许的数据入口：

1. **用户上传的 Storage 相对路径**（如 `datasets/my_data/train.jsonl`）：先调用一次
   `preview_dataset`，以真实样本确定字段映射、条数 N、P95 长度 L 和是否多模态。若输入是
   目录，使用 preview 返回的 `preview_file_path` 选定实际训练文件。工作流里用
   `PathJoinNode(base_path="/workspace/", subpath=<相对文件路径>)` 接入；数据必须已清洗且可
   直接训练，不添加 `DatasetToJsonlNode`、`DatasetValidatorNode` 等清洗或格式转换节点。
2. **数据集标识**：不调用 `preview_dataset`。下面是已知可直接使用的数据，不限制用户使用
   其他远程数据集；其他 `organization/dataset` 标识使用 `DownloadAndCacheDataset`，字段
   映射优先依据用户提供的数据说明，必要时只追问关键字段。

   | 节点 | 参数 | 已知数据集 |
   |------|------|------------|
   | CloneAndCacheDataset | `dataset` | `Writer/omniact`、`openai/gsm8k`、`pyromind/alpaca-gpt4-llm-demo`、`pyromind/geometry-vqa-vlm-demo`、`pyromind/self-cognition` |
   | DownloadAndCacheDataset | `dataset_name` | `pyromind/easyhard-24k`、`pyromind/agentic-tool-call-dataset-12k` |

   `CloneAndCacheDataset.dataset` 当前枚举以表中五项为准。其中三个 `pyromind/*` 数据集是
   平台测试集：`geometry-vqa-vlm-demo` 用于多模态，另外两个用于 LLM 场景；
   `openai/gsm8k` 可用于数学 benchmark。
3. **用户未提供数据**：告诉用户正式训练仍需提供已清洗数据，但不要停下来等待回复；先用
   测试集占位并继续生成完整工作流：文本 SFT/默认场景用 `pyromind/self-cognition`，文本
   DPO 用 `pyromind/alpaca-gpt4-llm-demo`，多模态/VLM 用
   `pyromind/geometry-vqa-vlm-demo`。最终回复明确说明用了哪个占位测试集、后续应替换哪里。

若用户上传的数据 preview 后无法直接训练，说明当前不负责数据清洗，并按第 3 种情况用测试
集继续生成占位工作流；用户明确要求停止时除外。

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

bench 工作流结构（完整模板见 [references/example-workflows.md](references/example-workflows.md) 示例6）：

```
基模(CloneAndCacheModel) → VLLMInference 部署
已清洗评测数据(PathJoinNode/CloneAndCacheDataset/DownloadAndCacheDataset) → DatasetConfigBuilderNode
两路汇入 ModelEvalApiNode（max_samples 控制采样条数）+ 指标配置
```

指标选择，按业务场景推荐：

| 场景 | entry |
|------|-------|
| 数学/数值答案 | `compute_gsm8k`（默认） |
| 分类/精确匹配 | `compute_accuracy` |
| 翻译/受约束生成 | `compute_bleu` |
| 摘要/长文本生成 | `compute_rouge_l` |

- 现成指标用 `MetricsConfigBuilderNode`，entry 必须填裸函数名枚举值，如
  `compute_gsm8k`；不要加 `examples/eval_metrics_common.py:` 前缀
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
- SFT/DPO/GRPO 的数据配置默认接入 `DatasetExtraConfigBuilderNode`；`max_seq_length` 默认
  `4096`，GRPO collator 默认使用
  `train.data.default_vision_grpo_collate:create_grpo_collate_fn`。训练节点的
  `thinking_as_input_ratio` 是可选参数，默认 `0`。
- 使用 WandB 时，`WandbConfigBuilderNode` 必须提供 `wandb_api_key`（Secret 名）和
  `wandb_project`，`wandb_name` 可选；训练节点本身的 `wandb_config` 输入仍是可选的。
- 以 [references/example-workflows.md](references/example-workflows.md) 里最接近的示例为
  拓扑骨架，不要从零拼节点。参数名、输入端口和输出端口必须以知识库节点 schema 为准；
  示例与 schema 冲突时以 schema 为准。

### 第 4 步：自动配参

决策链：**数据集(N, L) → 模型类型(LLM/VL) → 模型规模 → LoRA/Full → GPU 资源**。

生成训练工作流前先读 [references/parameter-decision.md](references/parameter-decision.md)，
按表填 lora_rank、batch_size、learning_rate、epoch 等参数。速记原则：

参数优先级：**用户明确要求 > 修改任务中未要求改变的现有有效值 > 当前数据与资源按决策表
算出的值 > 示例模板值**。不要因为套模板而覆盖用户要求或无关的现有配置；需要覆盖模板值时，
整组参数保持决策一致，并在最终回复中说明数据/资源依据。

- LoRA 是默认，Full 只在数据量大且有深度对齐需求且资源够时用
- VL 模型：batch 和 lr 在 LLM 基础上减半，显存需求升一档
- DPO lr ≈ SFT 的 5%~10%；GRPO lr ≈ SFT 的 10%~20%
- 资源不够时依次：Full→LoRA、降 rank、batch 减半 accum 翻倍，**不要先降 lr**
- 按优先级链选出第一个满足模态、效果、资源和用户约束的方案后停止扩展候选；只记录最终
  选择与一个被排除方案的关键理由

### 第 5 步：写入与校验

1. 新建 `workflow.py` 或整体重写全文件时用 `apply_patch`（patch 格式以工具描述为准），
   路径只传当前工作目录下的相对路径 `workflow.py`。修改已有文件（追加/替换节点块）时，
   先用 `file_editor` 读取当前内容，默认用 `str_replace` 做唯一匹配的最小替换。不要手写
   会话目录的长绝对路径，也不要只口头说已生成。若 patch 失败一次，不再重试；重新读取
   文件确认实际状态后切换到 `str_replace`。
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

检索优先级：**运行时校验结果 → [平台契约覆盖层](references/platform-contract-overrides.md)
→ `<知识库绝对路径>/workflow_template_xyflow/` 中对应的当前模板 → 直接读相关节点文档 →
本 Skill 示例 → 必要时宽泛 grep**。使用 GPU、Metrics、Reward 或 WandB 时必须先读覆盖层；
它专门修正可能被上游同步覆盖的知识库契约。

- 节点速查表和常用参数见 [references/node-reference.md](references/node-reference.md)
- 节点契约文档路径固定：`<知识库绝对路径>/nodes/<NodeType>/<NodeType>.md`（知识库绝对
  路径见路由提示）。已确定要用的节点直接读其文档，只读会实际用到或参数不确定的
- 修改已有 `workflow.py` 前做两张内部清单：①需求验收项，把每个明确动作映射为可观察
  结果（“展示/预览返回内容”必须有 Preview 类节点消费结果端口；“评测”必须有指标、样本
  上限和结果输出）；②图差分，列出必须保留/新增/删除的节点与连线。修改后先核对验收项，
  再核对图差分，最后校验完整 DSL。静态校验通过不等于需求验收通过；未要求改变的有效模型、
  参数、节点和连线保持原样
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
   `source_node_code` 是对应的原始 DSL 语句。用这些字段直接定位，优先用 `str_replace` 只
   修改报错指向的唯一片段，不要因为一条报错重写整个文件；改完回到步骤 1
3. 调用本身失败不代表工作流有问题。若 `retryable == true`（SSL、timeout、429/5xx 等传输
   失败），短暂退避后最多重试 2 次；仍失败则调用 `dsl_to_xyflow` 做本地语法、引用和建图
   检查。转换成功不等于平台完整校验通过，最终回复必须同时说明主校验不可用和降级结果。
   `retryable == false` 时不重试，也不要把传输/服务错误当成 DSL 错误去修改文件
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
- [references/platform-contract-overrides.md](references/platform-contract-overrides.md)：易漂移的平台枚举与 Secret 约定
- [references/node-reference.md](references/node-reference.md)：节点速查表与常用节点参数
- [references/example-workflows.md](references/example-workflows.md)：6 个示例工作流模板
