# AVI 实验参考

默认方向：Init → Data → Train → Eval；Inference、Tracer 为未启用占位。
Init 是基模 baseline，由 Eval 子 Agent 执行；阶段不等于 Agent。
这是可变参考，可以跳过、调整或返回。只评测时直接委派 Eval，
不准备数据处理、训练阶段的详细参数，不更新或确认 Graph 后才执行。

Eval 核查真实图片、GT、媒体角色、标签、数据拆分和模型，运行现有评测能力。
关注漏检、误放行、各类指标和覆盖率，不预设标签枚举或上线门禁。
Manager 据真实结果判断下一步，不直接探查逐条数据或提交训练。
