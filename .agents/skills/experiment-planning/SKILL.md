---
name: experiment-planning
description: Coordinate experiments and delegated work, execute within authorization and decide from verified results. Reference plans do not gate execution.
---

# 实验协调

Manager 对目标负责，可以直接探查数据、分析、配置、执行、调试和解释结果。
简单操作直接完成；独立工作按当前可用角色的能力委派。
用户明确指定分工时遵守该边界；委派应包含目标、已知事实、约束和交付要求，
将范围内的探查与实现选择留给子 Agent，不为委派重复完成同一工作。
子 Agent 交接结论、证据与产物位置、验证结果和未解决事项。
Manager 根据交付证据推进或汇总；只有证据缺失、相互矛盾或影响后续决策时才补充核查，
需要继续原任务时复用已有执行及产物。
AVI 参考 [AVI 实验](references/avi-single-turn.md)，其他场景不套用 AVI 规则。

get_experiment_context 读取全局事实。update_experiment_context 完整替换参考计划：
保留稳定 ID，传入当前 expected_sequence；冲突后重读合并。
资源放 resources，阶段放 stages，任务放 tasks；任务的 stage_id 指向阶段，
外部输入通过 resource_id 引用资源，上游产物通过 task_id + output_name 引用。
占位阶段同时设置 placeholder=true、enabled=false。未知资源 URI 留空并说明缺口。
参考图可以不完整、跳过、加阶段或返回，不承担执行调度和审批。
只补充当前目标需要的细节，不要求先确认 Graph 或补齐全部阶段。
展示阶段结果时，可用 stage_executions 将阶段 ID 关联到已有 TaskRun ID；
事后整理参考图也可关联，不改写执行快照，不按阶段名或角色名猜测执行状态。

按需委派使用 request_experiment_task：role_id 选当前启用的角色、稳定 request_id，task 传
name、instruction、已知 inputs、constraints、expected_outputs。
intent=execute 按授权执行；intent=prepare 只准备，不提交平台任务。
委派后可继续独立工作；需要子任务结果时等待 Runtime 通知，不连续轮询。
get_experiment_execution 查询真实结果，read_experiment_artifact 按 run_id 和产物名称
读取登记报告。control_experiment_task 支持 continue、cancel、retry_report。
报告重试复用已有产物，不重新提交外部任务。

子 Agent 结束、外部任务成功和交付可用是不同事实；完成条件以验证结果和产物为据。
在授权范围内自主推进；只规划不提交，只跑 baseline 不意味着授权训练。
实验结束后整理最终验证过的数据、模型、配置、评分与产物契约；
中途试错保留为实验历史，不是正式流程，也不要求正式阶段重放。
