# EchoMind 准备、Project 与原生委派

## 准备阶段

Manager 使用场景 Skill 的 checklist.json 收集事实、资源和执行约束。
read、write、edit、terminal、ask_user 负责探查和交互，inspect_resource
使用当前请求凭据读取 Storage。resolved facts 必须引用 message:/operation: 或
resource: 依据，update_preparation(expected_revision, facts) 增量保存事实。
set_preparation_plan(expected_revision, plan, entry_task_id, role_id) 提交完整计划。
任务输入引用 plan.resources，edges 连接 stage.id。

workflow_nodes 声明可用节点、任务类型和角色，Runtime 校验结构、引用与角色权限。
Init、Dataset、Train、Eval 保持独立角色配置。准备信息在用户审阅期间可继续修改。
新消息撤销 ready 并推进修订号，创建按钮使用最新准备版本。

## Project 创建

用户提交 create_project(command_id, expected_preparation_revision)。
SQL Store 在单个事务中保存 Project 和用户接受的准备快照。重复创建返回同一 Project。
事务失败整体回滚，创建后的工具策略在空闲边界激活。
Runtime 通过幂等 Project 通知把接受的计划、入口任务和事实交给 Manager。
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
