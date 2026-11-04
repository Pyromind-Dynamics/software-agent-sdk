---
name: model-evaluation-preparation
description: 收集通用模型评测信息，生成可审阅的阶段计划，由用户创建 Project 后启动评测。
---

# 模型评测准备

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
基模评测、评分、报告与基线解读都归在一个 Init stage（kind=init），由 eval 角色执行。
当前基模评测目标只需要这一个 Init 节点，不额外拆出“基线解读”或 Eval 节点。
训练后候选评测使用 Eval；Dataset、Train 及其他节点沿用库内定义。
不新增业务阶段或任务类型；实例 ID 保持稳定，并遵守节点库的实例数量限制。
只选目标需要的步骤；当前只启用由 eval 角色执行的节点，其他未实现节点只能 disabled + placeholder。
workflowPLAN 是简洁的业务节点计划：只记录任务引用、必要输入输出和已确认约束。
任务 description 简述目的与交付物，脚本、工具调用顺序和技术实现留到执行时确定。
entry_task_id 引用 plan.tasks 的首个任务，role_id=eval。
任务 inputs 用 resource_id 引用 plan.resources；Runtime 自动解析资源并交接已确认事实，
不重复填写 facts 或另一份任务正文。任务 description 说明目标、约束和交付物，
执行时复用 inference-evaluation。

计划在对话中展示，可继续讨论修改。ready 后由用户点击“创建并启动”；
不要调用旧计划确认或直接提交工具。创建后读取实验上下文并按结果推进，
委派时指定稳定 stage_id。结果以验证后的产物为据，报告重试沿用已有预测。
