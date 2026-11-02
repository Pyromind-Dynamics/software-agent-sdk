# AVI 实验参考

默认方向：Init → Data → Train → Eval；Inference、Tracer 为未启用占位。
Init 是基模 baseline；阶段不等于 Agent。Manager 可直接完成数据查看、分析和评测，
也可按需委派 Eval。这是可变参考，可以跳过、调整或返回，不需要先确认 Graph。

核查真实图片、GT、媒体角色、标签、数据拆分和模型，复用 inference-evaluation
和 avi-baseline-evaluation。关注漏检、误放行、各类指标和覆盖率，
不预设标签枚举或上线门禁。只准备计划时，不提交平台任务。
参考计划中将评测数据登记为资源，让 Init 基模任务的输入明确引用该 resource_id；
后续阶段只保留当前所需信息。依据真实结果判断下一步。
