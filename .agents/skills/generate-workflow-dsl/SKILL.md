---
name: generate-workflow-dsl
description: >-
  为 Pyromind 生成、修改或校验工作流 Python DSL，包括单节点工作流和模型训练。
  用于修改画布、检查 workflow.py、SFT/DPO/GRPO 或其他平台节点工作流；
  基模 Benchmark、训后效果评测、GT 对比和报告统一使用 inference-evaluation。
  负责按需读取节点契约、绑定产物并校验 DSL。不执行正式工作流，不在训练生成场景自动清洗数据。
---

# 生成 Pyromind 工作流

从用户目标和已有资产出发，只补足影响当前工作流正确生成的信息缺口。优先复用有效数据与配置，额外准备由实际输入约束或用户要求决定。

本 Skill 生成、修改并校验 DSL。独立模型效果评测交给 `inference-evaluation`；执行训练默认衔接同验证集的基模与训后评测，明确只生成或不评测时遵从。不在这里新建 Metric/Eval 链。
训练生成负责准备最终模型产物，LoRA 评测前需合并；不默认增加试用半成品的独立部署阶段。

## 数据与训练决策

已有数据、字段映射和有效配置足以落图时，直接生成。需要了解数据时按信息缺口查看样本、目录或统计，复用仍适用的读取结果。完整数据画像、processor/collator 实测和依赖安装不是生成 DSL 的前置条件；用户要求精确分析或实际兼容性问题需要时再做。

数据形态和 Builder 映射见 `references/workflow-contracts.md`，来源及挂载路径见 `references/data-routing.md`。训练节点支持的原始数据直接映射字段；合并文件或转换格式应有实际输入约束或用户需求作为依据。缺少数据源、必要监督信号或节点契约时指出具体缺口，不用占位数据或猜测补齐。

阶段优先遵从用户指定或已有工作流。未指定时，偏好对选择 DPO，示范答案选择 SFT，仅有 prompt 加 ground truth/reward 时考虑 GRPO；同时支持 SFT/GRPO 时默认 SFT。自动选择 GRPO 需要指令/SFT checkpoint 或上游 SFT 产物；用户明确要求基模 GRPO 时遵从并说明冷启动风险。

参数优先级：**用户明确要求 > 当前工作流有效值 > 用户提供的同任务已用配置 > 数据、模型和资源推理 > 平台默认值**。只调整本次要求或有证据不兼容的部分，区分估算与实测，不为生成工作流强求精确长度或容量结论。需要分析已有训练效果时使用 `training-analysis`。

局部修改先读取现有文件，保留变量名、节点 ID、下游连线及未涉及的配置，仅修改相关节点和连线；仅校验时直接调用校验工具。文件不存在时说明缺失。

## DSL 与资料边界

- 文件固定为 `public_data/workflow_canvas/workflow.py`；直接声明节点实例，不写 import，不当普通 Python 在 Agent 本地执行。
- 数据处理、训练和推理由 Pyromind 平台节点执行；上游数据、模型及训练产物通过输出端口绑定，不写死下游产物路径，不用 Agent 本地运行替代平台节点。
- SFT/DPO/GRPO 的 `wandb_config`、`accelerate_config` 当前按必填校验，连接对应 Builder 输出。WandB 仅写 Secret 名，不写明文凭据。
- 内置 Metric/Reward 不满足目标时才读取 `custom-python-assets.md`；脚本校验并上传成功后，使用真实返回的 Storage 路径填写入口，上传失败停止，不伪造路径。
- 节点契约以 `validate_workflow_dsl` 的实时结果为准。缺少契约时先查下列 reference，再从 `knowledge/index.md` 定向读取节点原文；已有信息足够时直接落图。

## 资料索引

| reference | 用途 |
|---|---|
| `references/data-routing.md` | 数据来源、文件与目录入口、Storage 和训练环境的路径语义 |
| `references/workflow-contracts.md` | 数据字段映射、模型入口、节点参数、输出端口和平台约束 |
| `references/custom-python-assets.md` | 自定义 Metric/Reward 的接口、上传和入口绑定 |

Reference 路径相对本 `SKILL.md` 所在目录；按 `<available_skills>` 的位置用 `read` 打开。
`knowledge/` 是会话根下的顶层只读知识库，使用 `knowledge/index.md`，不要拼到 Skill 目录下。
按缺失事实读取资料，复用已读内容，不用 terminal `ls/find` 探索 Skill 或知识库。

## 校验与交付

每次写入后调用 `validate_workflow_dsl()`，缺省校验工作区 `workflow.py`，不传 DSL 全文：

1. `valid=true`：完成；最终回复简述 warnings。
2. `valid=false`：按 `code`、`node_id`、`field` 和 `detail.*_node_code` 最小修复。平台契约漂移以返回结果为准；生成任务只修改当前 DSL，不编辑 Skill 或知识库。
3. `retryable=false`（包括 401）：停止，不重复校验或探测凭据。`retryable=true` 最多重试两次，仍失败则说明平台校验未完成。
4. 最多修改五轮；同一错误连续两轮未消失则报告阻塞。GPU 枚举或必填端口不确定时，可用只含目标节点与最小依赖的 DSL 做一次校验探测，不用 `run_workflow` 探测。

生成任务不调用 `workflow_debug`、`run_workflow` 或 `run_dataset_cleaning`。无配置修改的显式调试请求使用 `debug-workflow`；要求运行时，校验通过后转入 `workflow-execution`，上传实际生成的文件并按其约定提交，由 Copilot 确认。

最终简述所选节点、关键输入、参数依据和校验结果；说明影响执行的假设及未验证项，不复述完整 DSL。训练效果由独立评测报告判断。
