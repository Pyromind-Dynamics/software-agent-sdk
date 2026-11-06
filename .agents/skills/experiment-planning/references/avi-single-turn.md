# AVI 实验参考

默认方向：Dataset → Train → Inference → Evaluator。阶段不等于 Agent，Manager 可以直接
完成数据查看、配置、调试和执行，也可按需委派 Eval。这是可变参考，可以跳过、调整或返回，
不需要先确认 Graph。

用户提供明确的数据位置后，再核查真实图片、GT、媒体角色、标签、数据拆分和模型。
Dataset 使用 data-processing 能力完成校验、清洗、转换和划分；Train 通过
generate-workflow-dsl 或训练能力创建平台工作流；Inference 生成预测；Evaluator 使用
inference-evaluation 或 AVI 的确定性 rubric 产物。

AVI 常见输入是每条样本三张图（defect、diff、gt）、GT 和确定性评分脚本。只把这些作为
当前数据的实际约束，不把它们写成通用节点规范。关注漏检、误放行、各类指标和覆盖率，
不预设标签枚举或上线门禁。只准备计划时，不提交平台任务。依据真实结果判断下一步。
