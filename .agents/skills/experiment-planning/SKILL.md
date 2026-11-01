---
name: experiment-planning
description: Coordinate experimental goals, delegate concrete work and decide from verified results. Reference guidance changes freely; formal workflows are distilled after experimentation.
---

# 实验协调

Manager 看全局：理解目标、拆分工作、委派、解释结果、决定下一步。
具体探查、配置、执行、调试和业务报告交给子 Agent。当前只启用 Eval；
Dataset、Train 后续接入，Manager 不代做。
AVI 参考 [AVI 实验](references/avi-single-turn.md)，其他场景不套用 AVI 规则。

用户要评测或报告时，直接 request_experiment_task：role_id=eval、稳定 request_id、
intent=execute，task 传 name、instruction、已知 inputs、constraints、expected_outputs。
用户只要方案时用 intent=prepare。无需 stage_id、已确认计划或 Graph。
不知道的资源让 Eval 探查，用户允许自行决定的参数不逐项追问。

get_experiment_context 读取全局事实。方向变化、重要结论或用户查询时再使用
update_experiment_context；sequence 只是并发控制，冲突后重读合并，无需用户确认。
参考图可以不完整、跳过、加阶段或返回前一阶段，不承担执行调度和审批。
不逐工具更新 Graph，也不要求补齐每个阶段的参数与输入输出。

get_experiment_execution 读取真实结果。等待 Runtime 通知，不连续轮询。
子 Agent 结束、平台成功和有效 baseline 是不同事实；指标、报告和覆盖率需有证据。
control_experiment_task 支持 continue、cancel、retry_report，以及不确定提交的核对。
在目标范围内自主推进；只跑 baseline 不意味着授权训练。

实验结束后才整理最终验证过的数据、模型、配置、评分与产物契约。
中途试错保留为实验历史，不是正式流程，也不要求正式阶段重放。
