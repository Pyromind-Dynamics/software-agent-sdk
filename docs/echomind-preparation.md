# EchoMind 准备、Project 与阶段执行

新会话使用 Preparation Checklist：先收集事实，再生成 workflowPLAN，在对话中审阅修改。
用户点击“创建 Project”后，SQL Project 成为唯一业务身份；创建动作不自动启动节点。

当前模型评测入口会在创建 SQL Project 的同时创建首个 Eval TaskRun。该行为只属于
现有评测链路和旧会话兼容；训练飞轮 Project 创建只创建 Project 和 WorkingGraph，
不自动启动节点。

已有会话保留原准备策略与 Project 身份。新会话使用通用实验规划策略，
Dataset、Train、Inference、Evaluator 是可调整的首套参考节点。

## 与 WorkingGraph / StageRun 的接缝

统一身份链为：

```text
SQL Project（唯一 Project ID）
  -> WorkingGraph（project_id 与 SQL Project 相同）
  -> StageRun（节点级主运行）
  -> TaskRun 0..N（确定性执行明细）
  -> external_task_id（平台任务）
```

- SQL Project 是身份、权限、创建事务和 TaskRun 的权威来源。
- 本地 `graph.json` 只承载开发态 WorkingGraph，不是第二个 Project。
- SQL 提交成功后，Server 使用冻结的 Preparation plan，按相同 `project_id`
  幂等物化 WorkingGraph。
- Graph 缺失时按原 Project ID 补齐；重复 `create_project`、刷新和重启都不能生成
  第二个 Project。
- 创建 Project 不自动启动节点；用户启动节点后才创建 StageRun。
- StageRun 增加可选 `task_run_id`，TaskRun 增加可选 `stage_run_id`。
- TaskCorrelation 需要携带 `project_id / stage_run_id / task_run_id`，使平台 callback
  能回到准确节点；旧记录允许字段为空。
- 新 EchoMind 会话不再通过旧 `confirm_experiment_plan` 创建文件型 Project。

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
EchoMind Profile 不加载 AgentGenome，也不提供历史经验查询工具；准备事实只来自用户消息、
用户明确给出的数据路径和必要澄清。Workbench 与其他既有入口保持自己的 AgentGenome 配置。
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


## 通用节点库与简洁计划（Profile v9）

EchoMind 的规划策略来自 `experiment-planning/checklist.json`。workflow_nodes 是计划
可引用的业务节点目录，不是固定流水线。首套参考包含 Dataset、Train、Inference、
Evaluator，节点与任务可以从目录中选择、组合、禁用或调整标题和配置；AVI 只是其中一个
reference，不定义通用节点规范。

`model-evaluation-preparation` 只服务模型评测入口，不再作为 EchoMind 训练飞轮的通用
规划契约。创建 Project 只物化当前计划，不自动创建入口 TaskRun。当前节点由用户在 Project
工作台单独启动，Dataset 使用 Agent-led development，其余节点先使用确定性执行器契约壳。

计划实例的 stage.id / task.id 与节点/任务类型 kind 分开保存。Manager 从声明中选择 kind，Runtime 在准备提交和创建后的上下文更新时拒绝库外类型、错误任务归属及超限实例，并检查委派角色。共享层没有场景名或节点类型分支。旧策略未声明节点库时保留原语义，旧计划省略 kind 字段即可读取；不改写正在审阅的旧草稿或历史 TaskRun。

创建前在聊天中渲染简洁的 workflowPLAN：只展示库内节点与真实连线，任务和资源详情默认折叠。创建后使用同一节点展示并按稳定 Stage ID 对齐轨迹。旧会话使用原策略；使用节点库约束需新建会话。
