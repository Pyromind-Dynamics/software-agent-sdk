---
name: model-evaluation-preparation
description: 收集数据准备、训练和评测信息，生成可审阅的业务阶段计划，由用户创建 Project 后启动。
---

# 实验准备

[Checklist](checklist.json) 声明本场景的信息要求，get_preparation 返回缺口和依据引用。
用户已提供的信息直接使用；用 inspect_resource 预览 Storage，用 read 读取 knowledge/
下的文档，不把逻辑路径拼成 terminal 挂载路径。预览不代表全量验证。
用 update_preparation 增量记录有依据的新事实，未提交的项保留；未知项不猜测。
提问答复后使用 ask_user 返回的 preparation_revision；Checklist 补齐后提交计划。
计划实例 ID 保持稳定。每个 task.stage_id 引用所属 stage.id，task.kind 来自该节点的任务目录；
edges 连接 stage ID，任务间产物引用用 task.inputs。
基模评测、评分、报告与基线解读都归在一个 Init stage（kind=init），由 init 角色执行。
当前基模评测目标只需要这一个 Init 节点，不额外拆出“基线解读”或 Eval 节点。
训练后候选评测使用 Eval；Dataset、Train 及其他节点沿用库内定义。
不新增业务阶段或任务类型；实例 ID 保持稳定，并遵守节点库的实例数量限制。
只选目标需要的步骤；Init、Dataset、Train、Eval 分别使用 init、dataset、train、eval。
其他未实现节点只能 disabled + placeholder。
workflowPLAN 是简洁的业务节点计划：只记录任务引用、必要输入输出和已确认约束。
任务 description 简述目的与交付物，脚本、工具调用顺序和技术实现留到执行时确定。
entry_task_id 选择启动任务，role_id 选择该任务所属节点允许的角色。
入口 inputs 用 resource_id 引用 plan.resources 的实际位置；Runtime 交接已记录事实。
例如归属 `baseline` 节点的入口任务：
`{"id":"measure","kind":"bench","stage_id":"baseline","name":"基线评测","inputs":[{"name":"dataset","resource_id":"data"}]}`。

计划在对话中展示，可继续讨论修改。ready 后由用户点击“创建并启动”；
不要调用旧计划确认或直接提交工具。创建后 Manager 用原生文件工具维护
public_data/workflowPLAN.json，并通过原生 subagent 委派和继续已有子会话。
计划记录业务判断，平台成功不等于业务完成；是否推进由 Manager 结合证据决定。

## 数据与训练链路

只选用户目标需要的已有节点。数据清洗用 Dataset，训练用 Train，候选评测用 Eval；
基模评测和解读仍合并为一个 Init。缺少模型或评分不妨碍独立数据准备，但涉及评测或
训练的首个任务必须具备实际必需输入；后续任务可使用上游明确声明的产物引用。
Dataset 在需要用户审阅小样、补充信息或处理阻塞时直接调用 ask_user。
用 presentation 组织真实证据、统计、校验及必要图片（绝对 Storage 路径）。
用户回答直接返回发问角色。后续执行遵守冻结任务范围和平台工具的提交校验。
Train 复用现有工作流提交确认，完成后由 Manager 把实际模型产物交给 Eval。

子 Agent 用 Markdown 回复交付报告路径。Manager 用 read 打开报告，核对实际产物后将
Storage 路径传入后续任务说明。报告重试和继续原任务复用原子会话及已有产物。
训练准备交付可直接消费的 messages JSONL，在报告里分别列出训练集、可选验证集和统计路径。
将 Dataset 交付的训练、验证文件分别绑定给 Train，不另建一份冻结数据。
格式转换与模型批处理按 data-processing 的对应范式执行。
