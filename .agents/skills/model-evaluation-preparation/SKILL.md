---
name: model-evaluation-preparation
description: 收集数据准备、训练和评测信息，生成可审阅的业务阶段计划，由用户创建 Project 后启动。
---

# 实验准备

[Checklist](checklist.json) 声明本场景的信息要求，get_preparation 返回当前缺口和依据引用。
用户已经给出的信息直接使用；用 inspect_resource 读取数据前缀、目录或模型元信息，
结合只读知识检索补齐事实。记录读取范围，不能将预览当作全量验证。
探查失败或存在歧义时用 ask_user；参数和技术实现不必逐项询问。

用 update_preparation 保存完整 facts 和 expected_revision；resolved 信息使用
get_preparation 返回的 message:/operation: 或 inspect_resource 的 evidence_ref。
数据位置变化后重新检查字段及评分依据。未知、冲突项保留缺口，不猜测。
先保存完整 Checklist，再用 set_preparation_plan 提交计划与 expected_revision。
计划使用 resources、stages、tasks、edges。节点与任务只能选择 get_preparation 中
policy.workflow_nodes 的定义：stage.kind 引用节点 id，task.kind 引用该节点的任务 id。
稳定的 stage.id / task.id 标识实例，不能拿自由生成的阶段名称替代节点类型。
基模评测、评分、报告与基线解读都归在一个 Init stage（kind=init），由 init 角色执行。
当前基模评测目标只需要这一个 Init 节点，不额外拆出“基线解读”或 Eval 节点。
训练后候选评测使用 Eval；Dataset、Train 及其他节点沿用库内定义。
不新增业务阶段或任务类型；实例 ID 保持稳定，并遵守节点库的实例数量限制。
只选目标需要的步骤；Init、Dataset、Train、Eval 分别使用 init、dataset、train、eval。
其他未实现节点只能 disabled + placeholder。
workflowPLAN 是简洁的业务节点计划：只记录任务引用、必要输入输出和已确认约束。
任务 description 简述目的与交付物，脚本、工具调用顺序和技术实现留到执行时确定。
entry_task_id 引用 plan.tasks 的首个任务，role_id 选择首个 Stage 对应的角色。
任务 inputs 用 resource_id 引用 plan.resources；Runtime 自动解析资源并交接已确认事实，
不重复填写 facts 或另一份任务正文。任务 description 说明目标、约束和交付物，
评测执行复用 inference-evaluation；数据与训练复用对应角色的现有 Skill。

计划在对话中展示，可继续讨论修改。ready 后由用户点击“创建并启动”；
不要调用旧计划确认或直接提交工具。创建后读取实验上下文并按结果推进，
委派时指定稳定 stage_id。结果以验证后的产物为据，报告重试沿用已有预测。

## 数据与训练链路

只选用户目标需要的已有节点。数据清洗用 Dataset，训练用 Train，候选评测用 Eval；
基模评测和解读仍合并为一个 Init。缺少模型或评分不妨碍独立数据准备，但涉及评测或
训练的首个任务必须具备实际必需输入；后续任务可使用上游明确声明的产物引用。
Dataset 在需要用户审阅小样、补充信息或处理阻塞时直接调用 ask_user。
用 presentation 组织真实证据、统计、校验及必要图片（绝对 Storage 路径）。
用户回答直接返回发问角色。后续执行遵守冻结任务范围和平台工具的提交校验。
Train 复用现有工作流提交确认，完成后由 Manager 把实际模型产物交给 Eval。

子 Agent 用 Markdown 回复交付报告路径。Manager 用 read 打开报告，核对实际产物后将
Storage 路径传入后续 task.inputs。已登记的产物也可通过 task.artifact_refs 引用。
训练准备交付可直接消费的 messages JSONL，在报告里分别列出训练集、可选验证集和统计路径。
将 Dataset 交付的训练、验证文件分别绑定给 Train，不另建一份冻结数据。
格式转换与模型批处理按 data-processing 的对应范式执行。
