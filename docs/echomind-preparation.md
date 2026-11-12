# EchoMind 准备、Project 与原生委派

## 准备阶段

Manager 使用 model-evaluation-preparation Skill 判断信息是否足够。
read、terminal 负责数据与知识探查，ask_user 补充无法查证的信息或用户决策。
Checklist 只在 Skill 中引导判断，不保存逐项答案、证据目录或完成进度。
唯一的准备工具 set_preparation_plan(plan, entry_task_id) 接收完整可审阅计划；
任务输入引用 plan.resources，edges 连接 stage.id，入口角色由节点目录推导。

workflow-nodes.json 声明可用节点、任务类型和角色，Runtime 仅校验计划结构、
引用与角色权限；信息是否充分由 LLM 判断。Init、Dataset、Train、Eval 保持独立配置。
提交成功设为 ready，发出原 preparation.changed 并展示计划，不启动任务。
新消息或用户答复撤销 ready；Adapter 在模型请求前取得 Runtime 修订号，
提交时通过受信上下文传递，拒绝过期模型请求，LLM 无需维护版本。

前端只展示计划预览与创建按钮，按 phase 和 review_pending 判断就绪状态。
SQL 只保存计划、入口任务、审阅状态和版本；快照额外提供节点目录。
历史会话已清空，不再接收旧 Checklist 字段，也不保留 facts/missing 展示适配。
本次同步更新前后端契约，不修改数据库表或运行中的平台任务。

## Project 创建

用户提交 create_project(command_id, expected_preparation_revision)。
SQL Store 在单个事务中保存 Project 和用户接受的准备快照。重复创建返回同一 Project。
事务失败整体回滚，创建后的工具策略在空闲边界激活。
Runtime 通过幂等 Project 通知把接受的计划和入口任务交给 Manager。
已确认业务信息放在计划约束及任务描述中，不复制另一份事实。
Manager 用普通 write 创建 public_data/workflowPLAN.json，再委派入口任务。

Project 创建保存计划，原生插件保存子 Agent 的会话和运行记录。
旧 SQL TaskRun、角色租约、持久执行队列、报告队列和实验协调工具已经移除。

## 执行与交付

Manager 使用原生 subagent 启动角色，并使用 status、steer、resume、stop 跟进。
原生 run ID 和子会话关系由 Adapter 处理。同一子会话的恢复轮次在阶段视图归为一次
执行，各轮消息和工具轨迹完整保留。独立启动的子会话单独显示。

子 Agent 保留各自专业工具与 Skill，完成探查、平台操作和报告。
Manager 先读已有报告，根据证据决定业务完成、补充分析或下一步。
证据不足时继续原子会话，并复用已有平台任务和产物。
workflowPLAN 由 Manager 通过文件工具维护业务判断。
原生轮次结束、报告交付和平台任务终态分别记录，业务完成由 Manager 判断。

Runtime 保留平台授权、确认、提交账本、回调关联、取消和通知恢复。
相同会话中，同一路径和冻结内容的重复工作流提交复用已有回执，涵盖新开子会话。
提交结果不确定时保留回执供核对。子 Agent 的原生停止与远端平台取消分别处理。
子 Agent 通过平台停止工具确认远端取消结果。

Adapter 校验报告归属，使用既有文件写入通道交付报告。
前端消费通用 snapshot 和事件，维持阶段工作台与 Manager 对话的现有展示。

## 隔离与验证

准备记录、会话、文件、事件和平台关联按 user/profile/conversation 隔离。
请求凭据仅在当前授权上下文使用。Harness 专属 RPC、运行记录和恢复关系留在 Adapter。
数据库迁移历史用于已有安装的 schema 升级，当前运行只使用 Project 和准备记录。

运行 uv run pytest tests/pyromind_runtime、Pi Runtime npm test，
以及前端 npm test 和 npm run build。覆盖准备门槛、幂等创建、跨 Profile 拒绝、
原生四角色委派、续跑归并、报告交付、重复提交、回调和重启恢复。
平台替身验证应用链路，真实平台任务另行验收。
