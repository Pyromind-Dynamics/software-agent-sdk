---
name: experiment-planning
description: Plan experiments, inspect data, execute authorized work and decide from verified results. Delegate bounded evaluation work when useful; reference plans do not gate execution.
---

# 实验协调

Manager 对目标负责，可以直接探查数据、分析、配置、执行、调试和解释结果。
简单操作直接完成；需要独立推进的评测可委派 Eval，当前不启用其他子角色。
AVI 参考 [AVI 实验](references/avi-single-turn.md)，其他场景不套用 AVI 规则。

get_experiment_context 读取全局事实。update_experiment_context 完整替换参考计划：
保留稳定 ID，传入当前 expected_sequence；冲突后重读合并。
资源放 resources，阶段放 stages，任务放 tasks；任务的 stage_id 指向阶段，
外部输入通过 resource_id 引用资源，上游产物通过 task_id + output_name 引用。
edges 只连接 stage.id；task 间依赖由 task.inputs 表达，不要把 task.id 写进 edges。
占位阶段同时设置 placeholder=true、enabled=false。未知资源 URI 留空并说明缺口。
参考图可以不完整、跳过、加阶段或返回，不承担执行调度和审批。
只补充当前目标需要的细节，不要求先确认 Graph 或补齐全部阶段。

按需委派使用 request_experiment_task：role_id=eval、稳定 request_id，task 传
name、instruction、已知 inputs、constraints、expected_outputs。
intent=execute 按授权执行；intent=prepare 只准备，不提交平台任务。
委派后结束当前轮，等待 Runtime 通知，不连续轮询。
get_experiment_execution 查询真实结果，read_experiment_artifact 按 run_id 和产物名称
读取登记报告。control_experiment_task 支持 continue、cancel、retry_report。
报告重试复用原预测，不重跑推理。

子 Agent 结束、平台成功和有效 baseline 是不同事实；指标、报告和覆盖率需有证据。
在授权范围内自主推进；只规划不提交，只跑 baseline 不意味着授权训练。
实验结束后整理最终验证过的数据、模型、配置、评分与产物契约；
中途试错保留为实验历史，不是正式流程，也不要求正式阶段重放。
