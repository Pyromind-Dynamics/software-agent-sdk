# EchoMind 准备、Project 与阶段执行

新会话使用通用模型评测 Checklist：先收集事实，再生成 workflowPLAN，在对话中审阅修改，
用户点击“创建并启动”后创建 SQL Project 和首个匹配角色的 TaskRun，继续沿用同一会话。
新会话支持独立 Init、Dataset、Train、Eval 角色；已有会话保留原策略与 Project 身份。

## 分层和契约

Domain 定义 PreparationPolicy/View、ProjectView 和 create_project 命令；Application
校验信息依据、计划与角色并协调创建；SQL Store 在单个事务中写入 Project、冻结的
准备快照、TaskRun 和队列。Server 负责 Profile 装配、Storage 只读适配与 HTTP。
pi-subagents 继续管理原生子角色生命周期；Adapter 转换其状态、轨迹及受信执行关联。
Runtime 和前端均不按 Harness 或场景名称分支。

Checklist 策略在会话创建时固定；工具能力可在空闲时随 Profile 配置升级，保留准备事实、
计划和 Project 状态。默认未配置的旧会话不会被路由升级为准备模式。
Checklist 声明来自场景 Skill 的 checklist.json，Runtime 只解释通用字段、类型、依赖和依据。
Skill 根目录沿用 PYROMIND_SKILLS_PATH；默认兼容源码与 PyInstaller 布局，构建包包含
准备清单和角色所需 Skill，并保持原有目录结构。
准备阶段开放 read/write/edit、terminal、检索、inspect_resource、get/update_preparation
和 ask_user，可创建 Sandbox 探查数据、运行分析脚本及准备配置文件。检索工具使用同一
会话执行环境；inspect_resource 直接通过当前请求凭据读取 Storage 前缀或目录。
用户创建 Project 后才启动业务任务和委派执行。

v2 会话 snapshot 增加可选 preparation、project；新增 GET /{conversation_id}/preparation
和 GET /{conversation_id}/project。事件 preparation.changed、project.changed 触发快照刷新。
旧会话不输出这些字段或事件。旧正式计划与文件型 Project 继续由原兼容链路读取。

模型用 update_preparation(expected_revision, facts) 保存完整事实。
resolved facts 必须携带有效的 message:/operation: 或 resource: 依据；先保存完整事实，
再用 set_preparation_plan(expected_revision, plan, entry_task_id, role_id) 提交完整计划。
edges 连接 stage.id，任务依赖通过 task.inputs 引用；每次提交均需提供入口 task.id 和 role_id。
Runtime 从已保存事实和 plan.tasks 的入口任务构造委派，解析资源引用，并将接受的事实
交给子角色；模型无需重复生成 facts、任务正文和已解析输入。内部旧完整更新请求保持兼容。
创建按钮提交 create_project(command_id, expected_preparation_revision)，仅用户命令开放，
不提供模型工具。返回 Project 和 initial_run_id；表示已排队，平台结果随后上报。

## 存储、恢复和兼容

迁移 0005_preparation 增加 experiment_preparations 和 Project.creation_metadata，
支持 SQLite/MariaDB。准备记录按 user/profile/conversation 隔离，创建前没有 Project ID。
同一准备会话重复创建返回同一 Project 和 TaskRun。事务失败全部回滚；SQL 提交后
策略更新失败时，打开快照或 Worker 执行前重新激活策略，不重复提交平台任务。
新用户消息撤销 ready 并推进修订号，旧按钮不能创建过期方案。

委派冻结 project_id、stage_id、plan_sequence 和阶段快照。业务 TaskRun ID、原生
子 Agent ID 与 ProductEvent.run_id 分别使用；operation.execution 显式关联 TaskRun。
每个阶段可有多次执行，每次执行可以继续原子角色或进入报告轮次。
活动阶段禁止删除；重命名和后续改图不改写既有输入或结果。完成后的阶段被删除，
前端仍使用冻结快照展示历史。平台成功、产物验证和报告状态保持独立。

请求凭据只在当前授权上下文中使用，不进入 Checklist、快照元数据、队列或子角色关联。

## 验收

运行 uv run pytest tests/pyromind_runtime、Pi Runtime npm test，及前端 npm test/build/lint。
本期覆盖准备门槛、依据与依赖失效、幂等与事务回滚、阶段绑定、旧会话回归和跨 Profile 拒绝。
真实 Pi 集成测试同时覆盖旧实验模式、新准备模式、准备中重启和平台等待中重启。
平台使用测试替身时，只证明应用链路，不作为真实 GPU 小样验收证据。


## 节点库与简洁计划（Profile v8）

场景声明文件的 workflow_nodes 是可选节点与任务的唯一配置来源。基模推理、评分、报告与基线解读统一属于一个 Init stage（kind=init），使用独立 init 角色；bench 与 analysis 可以是同阶段的任务。训练后候选评测属于 Eval。Dataset 与 Train 已启用独立角色，Inference、Tracer、RawData 继续保留禁用占位。Init 的实例上限为 1，不能将基线解读拆成第二个 Init。

计划实例的 stage.id / task.id 与节点/任务类型 kind 分开保存。Manager 从声明中选择 kind，Runtime 在准备提交和创建后的上下文更新时拒绝库外类型、错误任务归属及超限实例，并检查委派角色。共享层没有场景名或节点类型分支。旧策略未声明节点库时保留原语义，旧计划省略 kind 字段即可读取；不改写正在审阅的旧草稿或历史 TaskRun。

创建前在聊天中渲染简洁的 workflowPLAN：只展示库内节点与真实连线，任务和资源详情默认折叠。创建后使用同一节点展示并按稳定 Stage ID 对齐轨迹。旧会话使用原策略；使用节点库约束需新建会话。


## 子角色执行

子角色是主 Agent 现有能力的子集。Init 与 Eval 初始使用相同工具、Skill，但分别保存
角色配置。Dataset 使用 data-processing，Train 使用 generate-workflow-dsl、
workflow-execution、training-analysis；不继承另一个 Agent 的 Profile。可选工具由
DelegatedRole.optional_tools 声明，Adapter 按执行环境能力开放；平台沙箱使用已有
Storage 挂载发布文件，不为上传另建执行环境。

节点声明的 required_fields 为当前计划增加必需信息：仅 Dataset 的计划不要求评测模型或
评分依据；启用 Init/Eval 时必须有模型与评分依据，Train 必须有基模。Runtime 在 ready
校验这些字段，不能因清洗场景放宽了通用必填项而让评测计划缺少执行输入。

PreparationPolicy.execution_policies 是服务端装配的通用执行策略，按角色声明提交工具、
产物契约与人工检查点；Runtime 将它冻结到 TaskRun，模型不能通过委派参数覆盖。
旧任务没有该字段时继续使用原评测契约，无需 SQL 表迁移。新增状态和 JSON 字段均为增量。

Dataset 可用 ask_user 展示真实小样，或询问缺失输入和阻塞决策。用户回答后，
原子 Agent 继续当前轮次。正式提交经过冻结任务范围、提交工具和平台校验。
历史任务如显式冻结了人工检查点策略，保留原审批和继续链路。新任务使用直接提问。

Train 的 submit_workflow 复用现有 permission.requested / permission_response；
等待时 TaskRun 为 waiting_confirmation，用户确认后才创建提交账本并执行冻结工作流。
平台任务、外部回调与 TaskRun 关联，评测的指标、预测和覆盖范围由 Runtime 校验。
所有子 Agent 通过 Pi 原生完成通知返回 Markdown，详细证据和产物路径写入任务工作区
的报告文件。Manager 用 read 打开报告并核对实际产物，再把 Storage 路径传入下游
task.inputs。本地数据处理完成后标记为已交付报告，平台任务状态按真实回执流转。
报告重试复用已有产物。

Manager 将报告中核对过的实际 Storage 路径传入下游 task.inputs。对 Runtime 已校验的
平台评测文件，可用 task.artifact_refs 指定 input_name、task_run_id 和 artifact_name。
Runtime 检查同一用户、Profile、会话、Project 和报告就绪状态，解析并冻结 URI、
摘要与来源。
Manager 按用户接受的范围推进 Dataset → Train → Eval，不新增 DAG 调度服务。

前端继续复用 ChatPanel 呈现子角色 message/tool call，以及原 UserInputCard 呈现确认。
不新增“任务与报告”栏，也不把小样、全量和报告拆成业务节点。


## Project 待确认侧栏

创建前不展示待确认侧栏：Checklist、workflowPLAN 预览与提问仍在对话中。
存在 SQL Project 后，pending_inputs 与 pending_permissions 统一放到右侧待确认区，
Manager 聊天与阶段日志保留过程记录，不重复渲染可操作确认表单。
每条提问可以携带 Agent 编写的 `presentation: {title?, content}`；content 复用 Chat 的
text、image、image_url 内容块。text 支持 Markdown、表格与文件链接，图片可引用 HTTPS
地址或绝对 Storage 路径。前端复用 ContentList，不解析任务结果，不识别清洗前后、样本
字段或对比顺序；这些内容由 Manager 根据真实证据组织。未携带 presentation 的旧请求
继续显示原提问，不从检查点拼装专用模板。

Manager 与受管子 Agent 共用 ask_user。启用会话提问能力后，子 Agent 可提交
presentation 和 questions，回答沿 input_response 回到发问工具。子任务标识由 Adapter
根据当前受管执行关联填写。Project 创建后所有提问统一显示在侧栏。创建前提问留在 Chat。

提问无需样本、检查点或 Manager 转达。每条请求按会话与请求 ID 独立处理，取消子 Agent
提问只结束对应等待。执行时仍检查冻结任务范围、工具权限、平台参数与产物。历史冻结任务
保留其显式提交策略，已有检查点引用不限制提问显示或答复。

Storage 图片使用新增的通用会话读取接口：
`GET /{conversation_id}/pending-inputs/{request_id}/media/{content_index}/{image_index}`。
路由只调用 ConversationRuntime；Runtime 校验用户、Profile、会话与待处理请求，从持久化
presentation 取出图片引用，通过可选 ResourceMediaReader 端口读取。请求不接收任意路径。
Storage 采用本次请求鉴权，单图限 8 MB，只返回 JPEG/PNG/WebP 数据预览；签名 URL 和
请求凭据不进入展示文档或队列。请求解决后不可再通过该接口读取。读取失败可逐图重试。
旧 checkpoint media 接口保留兼容，本期前端不再使用。

新会话通过提示和工具 schema 使用 presentation；历史会话保持
原策略。新增协议字段可选且缺失时不序列化，沿用 input.requested、快照和恢复链路，
无需 SQL 迁移。改动归属共享展示协议、Runtime 只读用例与端口、Server 装配和 Profile
提示；Harness 仅在 Adapter 内传递字段，没有新增调度器或业务专属前端。


## Manager 引导原子会话

Manager 根据上下文需求选择委派方式。独立任务使用新 request_id 调用
request_experiment_task，开始空白子会话。修改、解释或排查已有工作时使用
control_experiment_task(action=continue, run_id, message)，沿用原 TaskRun、Stage 和子会话。
相同 command_id 重试复用已有引导记录。

continue 接受所有业务状态与历史计划任务。运行中的子会话通过 Adapter 转为原生 steer，
已结束的子会话通过原生 resume 继续。引导保存到既有持久队列，实时消息有独立的消费通道，
使正在等待子会话结束的执行 Worker 不阻塞引导。子会话恰好结束时，消息转入同一角色的
继续轮次。找不到原子会话时报告上下文不可用，由 Manager 明确选择新任务。

引导本身不重置执行结果、报告状态、提交账本或检查点。子 Agent 的后续工具调用沿用原
TaskRun 的执行范围与平台校验。平台提交结果不确定时仍可讨论和排查，实际重提在提交工具
中核对。HTTP、ProductCommand 与前端沿用现有协议，新增队列种类与可选 payload 字段，
无需数据库迁移。Harness 的 steer/resume 转换只发生在 Adapter 内。
