# EchoMind 步骤 1：Manager 规划与 Graph 创建

## 目标

用户只通过 EchoMind Manager 对话描述目标。Manager 先读取通用的
`experiment-planning` Skill，再按需加载匹配的场景 reference，通过计划工具形成
可审阅的 `WorkflowPlan`。用户在前端计划面板确认后，平台立即创建或更新
Project/Graph，并在同一个 EchoMind 会话中开放完整开发能力。

## 唯一入口

```text
/api/v2/echomind/conversations
```

前端：

```text
/?chat=echomind&userId=<debug-user-id>
```

不存在第二个 EchoMind Chat，也不存在 legacy 入口。

## 规划阶段能力

Tool：

```text
read
terminal
create_experiment_plan
revise_experiment_plan
get_experiment_plan
```

Skill：

```text
experiment-planning
```

Plan Proposal 是草稿，不是 Project。规划阶段的 Agent 不能：

- 写工作流文件；
- 创建 Sandbox；
- 提交平台任务；
- 伪造 Project、Graph、Run 或 Artifact；
- 在未确认时声称资源已经存在。

## 确认语义

用户点击确认后，前端提交：

```json
{
  "type": "confirm_experiment_plan",
  "command_id": "...",
  "proposal_id": "...",
  "expected_revision": 0
}
```

服务端处理顺序：

1. 校验 `expected_revision`；
2. 在 SQL 中确认 Proposal，生成或推进 PlanRevision；
3. 使用 Revision.project_id 绑定文件型 Project/Graph；
4. 激活完整开发 Tool/Skill allowlist；
5. 更新 Pi session，在下一次 prompt 前重启 runner；
6. 返回 Revision 并通知前端刷新。

同一 `command_id` 重试是幂等的。首次确认创建 Project，后续确认更新同一个
Project/Graph。

## Graph

AVI reference 首期固定：

```text
dataset -> inference -> evaluator -> train
```

Graph 从已确认 `WorkflowPlan` 的 stages、tasks、resources 和 edges 生成。后端不按
Skill 名选择场景模板，也不维护 AVI 专属路由。
Project/Graph 不再读取 requirements 文件。

Dataset 节点绑定 Agent 开发模式，其余节点首期使用 Contract Probe 壳，确保
Stage Run、状态、错误和 Artifact 接口先跑通。

## 前端

EchoMind 页面包含：

- 会话历史；
- 计划草稿、Diff、确认、拒绝和历史 Revision；
- Project ID 与真实 Graph；
- 节点启动/运行；
- Run、错误和 Artifact 查看。

确认后右侧读取：

```text
GET /api/echomind/projects/{project_id}
```

计划中的阶段图用于规划审阅，Project Graph 用于开发态执行和验证，两者不再
混用事实源。

## 不再保留

- `/api/echomind/conversations`；
- `ECHOMIND_CHAT_PROFILE`；
- `project_request.json`；
- `POST /api/echomind/projects`；
- `?chat=echomind-legacy`；
- 旧 EchoMind 会话迁移。

## 验证

```bash
uv run pytest -q \
  tests/pyromind_runtime/test_experiment_plans.py \
  tests/pyromind_runtime/test_product_api.py \
  tests/agent_server/test_echomind_router.py
```

```bash
cd /Users/zhangpeng/PycharmProjects/minimal-chat-frontend
npm test && npm run build && npm run lint
```
