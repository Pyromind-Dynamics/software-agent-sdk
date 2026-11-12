---
name: model-evaluation-preparation
description: 创建 Project 前，核对数据、训练或评测需求，补齐影响执行的信息并提交可审阅 workflowPLAN；不执行平台任务。
---

# 实验准备

## 信息是否足够

以用户目标决定需要确认的信息，不要求无关项：

- 目标与范围：这次交付什么、处理哪些数据，用户明确的资源或操作限制。
- 数据：实际位置、格式、可用字段与媒体角色；训练集和验证集的用途。
- 训练：基座来源、监督输入输出约定、预期模型产物。
- 评测：被评模型或预测、评测输入、GT 或评分依据；评分不能把答案混入模型输入。

用户已提供的信息直接使用。用 read、terminal 查看数据，用 read 读取 knowledge/ 文档；
Storage 路径使用 /workspace/<path>，不把 knowledge/ 拼成 Storage 挂载路径。
有限预览不能说明全量数据已经验证。能查证的信息自行获取；无法确定、影响执行的
信息或业务选择用 ask_user 确认。presentation 是对象，不是 JSON 字符串。
缺少依据时不猜测，不把历史产物当作本轮确认或执行授权。

## 可审阅计划

信息足够后调用 set_preparation_plan(plan, entry_task_id)；无需保存 Checklist、
证据 ID、ready 标记或版本号。只规划用户需要的业务阶段，不提前执行。

[节点目录](workflow-nodes.json) 是可选节点、任务类型和角色的唯一来源。
stage.kind 选择节点 id，task.kind 选择该节点的任务 id，task.stage_id 引用 stage.id。
Init 合并基模评测与基线解读；Eval 用于训练后候选评测。未实现节点不启用。
实例 ID 保持稳定，edges 连接 stage ID；下游任务用 task_id/output_name 引用上游产物。

计划写明目标、真实输入、已确认约束、任务目的与交付物；实现细节留到执行时决定。
entry_task_id 选择首先执行的任务，它必须通过 resource_id 引用有实际位置的资源。
重要信息放入 constraints 或任务 description，不再复制另一份 facts。

最小示例（数据准备）：

```json
{
  "plan": {
    "goal": "整理训练数据并交付校验报告",
    "resources": [{"id": "data", "name": "原始数据", "storage_path": "/workspace/datasets/source.jsonl"}],
    "stages": [{"id": "prepare", "kind": "dataset", "name": "数据准备"}],
    "tasks": [{
      "id": "clean", "kind": "clean_label", "stage_id": "prepare", "name": "整理数据",
      "description": "按已确认约定整理数据，交付 messages JSONL 和统计报告。",
      "inputs": [{"name": "dataset", "resource_id": "data"}],
      "outputs": [{"name": "train_messages", "description": "训练文件与报告路径"}]
    }]
  },
  "entry_task_id": "clean"
}
```

提交成功后沿用聊天中的计划预览，用户可继续修改；只有用户点击“创建并启动”才开始工作。
创建后按通知把接受的计划写入 public_data/workflowPLAN.json，用原生 subagent 委派。
核对报告和真实产物再推进后续阶段；继续或重试复用原子会话及已有产物。
凭据只用当前运行环境，不写入计划、数据或报告。
