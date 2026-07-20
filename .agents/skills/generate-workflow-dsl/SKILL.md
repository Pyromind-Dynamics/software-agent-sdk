---
name: generate-workflow-dsl
description: >-
  为 Pyromind 生成、修改或校验模型训练与评测工作流。用户用自然语言要求“用这份数据训练模型”、
  微调、基线评测、SFT/DPO/GRPO、修改画布、换模型跑 benchmark/看看效果或检查 workflow.py
  时使用；负责预览 Storage 数据、推断训练阶段、整组决定参数、生成并上传自定义
  Metrics/Reward、绑定阶段产物并校验 Python DSL。不执行正式训练，不在训练生成场景自动清洗数据。
---

# 生成 Pyromind 工作流

把工作流视为少量阶段模板的组合。先证明数据和参数可用，再写 DSL；不要从完整示例反推需求。

## DSL 与资料边界

- `workflow.py` 直接声明节点实例，不写 import，也不能当普通 Python 在 Agent 本地执行。
- 本 Skill 只生成并校验 DSL；数据处理、Benchmark、训练和推理由 Pyromind 平台节点执行。
- 数据、模型和训练产物通过输出端口绑定；禁止用 Agent 本地 terminal 替代平台运行时或查找数据副本。
- Benchmark 最小骨架是数据配置 → 模型入口 → VLLM → Metric → Eval。
- 每个 `MetricsConfigBuilderNode` 只输出一个 `metrics_config`；契约未声明的组合方式不得猜测。
- Reference 是按缺失事实选择的索引，不是阅读清单；同一轮不得重复读取同一路径。
- Skill 与 reference 足以生成初稿时不查 `knowledge/`。只有校验明确暴露未覆盖的平台契约时，
  才能围绕该错误定向查询一次。

## 资料索引

| reference | 读取时机 |
|---|---|
| `references/data-routing.md` | 判断数据源、preview 结果、训练格式、字段映射或训练类型 |
| `references/workflow-contracts.md` | 完整生成或组合阶段时查拓扑、节点参数、端口、枚举和平台覆盖项 |
| `references/parameter-decision.md` | 训练数值参数的整组决策或训练 OOM 调整 |
| `references/custom-python-assets.md` | 内置 Metrics/Reward 不适用，需要生成、上传并回填 Python 入口 |

调用格式为 `skills_read(skill_name="generate-workflow-dsl", path="references/...")`。

## 执行状态机

### 0. 先判定局部修改

若请求只改已有节点参数，或把一个节点替换为输出端口兼容的单节点，走快路径：

- 保留原变量名、节点 ID、下游连线和所有未被点名的参数；只写需求图差分。
- 跳过数据画像、阶段选择和整组配参；不调用 preview，不读取 reference 或 `knowledge/`。
- 只有缺少新节点契约或校验返回结构错误时，才读取一份最相关的 reference，然后修改并进入第 8 步。

模型入口规则供快路径和完整生成共用：

- `Qwen/Qwen3-0.6B`、`Qwen/Qwen3-1.7B`、`Qwen/Qwen3-4B`、
  `Qwen/Qwen3-VL-2B-Instruct`、`Qwen/Qwen3-VL-4B-Instruct` 使用
  `CloneAndCacheModel(model=..., target_path="/workspace/models/")`。
- 用户指定其他开源模型时使用
  `DownloadAndCacheModel(modelname=..., cache_dir="/workspace/models/<org>/<model>", download_source="huggingface")`；
  它与 Clone 都输出 `model_path`，不得改下游绑定。
- 将 `qwen3.5-2b` 规范化为 `Qwen/Qwen3.5-2B`，缓存到
  `/workspace/models/Qwen/Qwen3.5-2B`；用户指定其他来源时再覆盖默认 Hugging Face。

### 1. 锁定目标

- 仅校验现有工作流：直接进入第 8 步，不改变有效配置。
- 明确要求 Benchmark：只组合评测所需阶段。
- 要求训练或修改训练工作流：继续执行；用户不需要说出 SFT/DPO/GRPO。

### 2. 建立数据画像

- Storage 相对路径：按 `data-routing.md` 获取或复用数据画像。
- 平台预置或外部数据集标识：选择 Clone/Download，不调用 `preview_dataset`。
- 未提供数据：先索要 Storage 路径；只有用户明确要求演示/模板时才使用测试集。
- 内部记录数据源、实际训练文件、N、P95 长度 L、字段、模态和样本形态，不向用户展示冗长清单。

若数据不满足 `data-routing.md` 的训练格式，停止生成，指出缺失字段并给目标 JSONL 样例。
禁止调用 `run_dataset_cleaning`，也禁止静默改用测试集。

### 3. 选择阶段

- 用户明确指定阶段时遵从；仅修改已有训练工作流时保留原阶段。否则，有 chosen/rejected
  偏好对时选 DPO；存在 response/assistant 示范目标时选 SFT；仅有 prompt 加 ground truth
  或 reward 时才考虑 GRPO。
- 可程序验证只说明 reward 可构造，不能把监督样本改判为 GRPO；未明确阶段且数据同时支持
  SFT/GRPO 时默认 SFT。自动选择 GRPO 还要求当前模型是 SFT/指令 checkpoint 或上游 SFT
  产物；缺少该条件时说明冷启动缺口。用户明确要求基模 GRPO 时遵从并提示风险。
- 完整生成或组合阶段时读取一次 `workflow-contracts.md`，按其中契约落图，不复制整套案例。
- 用户要求训练时本轮仍生成训练 DSL。用户未说跳过基线时，在最终回复建议用同一数据切分和
  指标先跑基模 Benchmark；这不是生成门禁。

### 4. 决定契约与枚举

使用第 3 步已读取的 `workflow-contracts.md`，不要为了参数、端口或枚举再次读取其他资料。
局部修改只有缺少新节点契约时才读取该文件一次。

### 5. 整组配参

阶段锁定后，需要自动决定或调整训练数值时读取 `parameter-decision.md`，一次性确定 max sequence length、
batch、grad accumulation、learning rate、epoch、LoRA rank、max steps 和 num generations。

参数优先级：**用户明确要求 > 修改任务中已有有效值 > 数据与资源决策 > 模板兜底值**。
不要只改一个相互依赖参数，也不要用模板覆盖无关配置。

### 6. 准备自定义资产

阶段锁定后，只有内置 Metric/Reward 不满足业务目标时才读取 `custom-python-assets.md`。必须先在工作区写脚本、
校验接口、调用 `upload_file_to_pyromind` 成功取得 Storage 路径，再把
`<storage_path>:<function>` 写入 DSL。上传失败就停止，不得伪造路径。

### 7. 写入或局部修改

- 文件固定为 `public_data/workflow_canvas/workflow.py`；新建或整体生成使用 `apply_patch`。
- 修改前先读取现有文件，列出内部“需求验收项”和“图差分”，仅替换相关节点与连线。
- 所有上游产物都通过输出端口绑定。不得把 SFT、Merge、GRPO、Inference 之间的模型路径写死。
- WandB 仅写 Secret 名；不得把 API Key、Cookie、集群凭证或其他明文 Secret 写入 DSL。

### 8. 校验闭环

每次写入后立即调用 `validate_workflow_dsl`，不传 `dsl` 参数：

1. `valid=true`：结束；warnings 只在最终回复简述。
2. `valid=false`：按 `code`、`node_id`、`field` 和 `detail.*_node_code` 做唯一片段最小修改。
3. `retryable=false`（包括 401）：立即停止，不重复校验，也不得用 terminal 探测凭证。
4. `retryable=true`：最多重试两次；仍失败则停止并说明平台校验未完成。
5. 最多修改五轮；同一错误连续两轮不消失时停止并报告。

生成 Skill 不调用 `workflow_debug`、`run_workflow` 或 `run_dataset_cleaning`。只有不含配置修改的
显式调试请求才转用 `debug-workflow`；“换模型/改参数后跑或看效果”本轮只修改并校验 DSL，正式
执行由前端/平台触发。

## 最终回复

简述数据入口与字段映射、所选阶段、基模、关键参数及校验结果。对训练请求，除非用户已明确
跳过，补一句使用同一指标跑基模 Benchmark 的建议；不要复述完整 DSL。
