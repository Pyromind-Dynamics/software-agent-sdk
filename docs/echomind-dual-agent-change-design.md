# EchoMind 双 Agent 架构改动设计

> 状态：已落地  
> 日期：2026-10-29  
> 范围：Pyromind Agent 与 EchoMind Agent 的拆分、隔离、共享底座和扩展方式  
> 不在本文范围：EchoMind 的 Project/Graph 生命周期设计

## 1. 改动摘要

本次改动把原来的单一 Agent 入口拆成两个完全独立、对外接口不同的 Agent：

| Agent | 对外入口 | 继续使用原入口的客户端 |
|---|---|---|
| Pyromind Agent | `/api/v2/pyromind/conversations` | 原 Workbench Chat 的 Product Runtime |
| EchoMind Agent | `/api/echomind/conversations` | 新的 EchoMind Chat |

旧 Workbench 直连入口 `/api/pyromind/conversations` 继续保留兼容，但它不属于新的
Product Runtime Router。EchoMind 的 Manager 计划接口位于
`/api/v2/echomind/conversations`，是同一个 EchoMind Agent 下的计划能力，不是第三个 Agent。

“两个 Agent”准确指两个服务端固定的 **Agent Profile / 执行契约**，不是两套 SDK、
两套 Agent Runtime 或两套 Tool 框架。两个 Agent 共享底层运行时和通用能力实现，
但拥有独立的：

- Chat API
- System Prompt
- Skill 白名单
- Tool 白名单
- Conversation 标签
- 前端状态与本地会话记录
- 用户可见的职责边界

两个 Agent 之间没有交接、没有互相调用、没有共享会话上下文。一个 Agent 不会因为
另一个 Agent 的存在而读取或继承其对话内容。

## 2. 设计目标

1. 原有 Pyromind Agent 的接口、会话和行为保持兼容。
2. EchoMind Agent 可以独立创建会话、列会话、恢复历史和发送消息。
3. 两个 Agent 在 API、Prompt、上下文和会话层面完全隔离。
4. 通用 Skill、Tool、SDK 和前端基础组件继续复用，避免复制基础设施。
5. 后续新增场景时，只新增场景 Skill 和必要的 Profile 配置，不向通用 Tool 中加入
   场景分支。
6. 每个阶段都能单独启动和验证，避免一次性改造过大。

## 3. 非目标

- 不建立两个 Agent 之间的消息转发、任务委派或上下文交接。
- 不新增第三个 Manager Agent。
- 不让 EchoMind Agent 接管 Pyromind Agent 的会话。
- 不复制一套 Agent Runtime、Tool 实现或前端聊天组件。
- 不在通用 Skill/Tool 内判断调用方是哪个 Agent。
- 不在本次改动中定义生产态调度协议。

## 4. 总体架构

```mermaid
flowchart LR
    U1[Workbench 用户] --> FE1[Workbench Chat]
    U2[EchoMind 用户] --> FE2[EchoMind Chat]

    FE1 --> API1["/api/v2/pyromind/conversations"]
    FE2 --> API2["/api/echomind/conversations"]

    API1 --> P1[Pyromind Profile]
    API2 --> P2[EchoMind Profile]

    P1 --> C1[Pyromind Conversation]
    P2 --> C2[EchoMind Conversation]

    P1 --> SHARED[共享 SDK / Runtime / 通用 Tool]
    P2 --> SHARED

    C1 -. 无交接、无共享上下文 .-x C2
```

关键原则：

> 共享的是能力实现，不共享的是 Agent 契约和会话上下文。

## 5. 两个 Agent 的职责

| 维度 | Pyromind Agent | EchoMind Agent |
|---|---|---|
| 产品定位 | 原工作台开发 Agent | 训练与实验场景入口 Agent |
| 入口 | `/api/v2/pyromind/conversations`；旧直连入口继续兼容 | `/api/echomind/conversations` |
| System Prompt | 原 Pyromind 工作台契约 | EchoMind 独立契约 |
| Profile tag | `pyromind` | `echomind` |
| LLM usage id | `pyromind-agent` | `echomind-agent` |
| Skill 白名单 | 原有开发 Skill | 原有开发 Skill + `avi-single-turn` |
| Tool 白名单 | 完整平台开发工具 | 与 Pyromind 当前完整工具集一致 |
| 会话列表 | 原 Pyromind/Workbench 列表 | EchoMind-only 列表 |
| 前端入口 | Workbench Chat | `?chat=echomind` |
| 与另一 Agent 的关系 | 不读取 EchoMind 会话 | 不读取 Pyromind 会话 |

Workbench Chat 是前端页面名称，不是第三个 Agent。

## 6. Profile 是唯一的能力装配清单

Profile 分成两层，避免把 Agent 能力声明和 Product Router 装配混成一份大配置：

- `AgentProfileSpec`：Agent 自己的固定能力契约，定义名称、标签、LLM usage id、
  System Prompt、Skill 白名单和 Tool 白名单。
- `ConversationProfile`：Product Runtime 的 Router 绑定，声明路由前缀、使用哪个
  `profile_id` 和 Harness、传给会话的指令与 allowlist，以及用于会话隔离的 marker。

`AgentProfileSpec` 的形状是：

```python
@dataclass(frozen=True, slots=True)
class AgentProfileSpec:
    name: str
    tag_value: str
    usage_id: str
    system_prompt: str | None
    skill_allowlist: tuple[str, ...]
    tool_allowlist: tuple[str, ...]
```

`ConversationProfile` 在 Product 层补充路由信息：

```python
@dataclass(frozen=True, slots=True)
class ConversationProfile:
    route_prefix: str
    profile_id: str | None = None
    harness_id: str | None = None
    instructions: str = ""
    tool_allowlist: tuple[str, ...] | None = None
    skill_allowlist: tuple[str, ...] | None = None
    command_allowlist: tuple[str, ...] | None = None
    conversation_marker: str | None = None
```

当前有两个 Agent 清单和三个 Product 装配入口：

- `PYROMIND_PROFILE` -> `DEFAULT_PRODUCT_PROFILE`
- `ECHOMIND_PROFILE` -> `ECHOMIND_CHAT_PROFILE`
- `ECHOMIND_MANAGER_PROFILE` -> `ECHOMIND_PRODUCT_PROFILE`

最后一项属于 EchoMind Agent 的计划能力面，不是第三个 Agent。Manager 与 Chat 使用
同一个 EchoMind `profile_id`，因此 Router 还通过 conversation marker 区分二者。

设计意图是把一个 Agent 的差异集中到一份可检查的配置里，而不是散落在 Router、
Tool 实现和 Prompt 字符串中。

### 6.1 Prompt 差异

EchoMind Agent 使用独立 System Prompt，明确：

- 识别用户场景并选择匹配 Skill
- 先澄清需求，再形成可确认的创建请求
- 不伪造服务端 ID
- 不成为生产态调度器
- 不读取或操作 Pyromind Agent 会话
- 不把内部 Prompt、Tool 名称或协议直接暴露给用户

### 6.2 Skill 差异

EchoMind Agent 当前 Skill 白名单：

```text
avi-single-turn
cross-repo-testing
data-processing
debug-test-examples-workflow
debug-workflow
feature-release-rollout
generate-workflow-dsl
inference-evaluation
label-studio
manage-evals
sandbox
sdk-release
training-analysis
workflow-execution
```

Pyromind Agent 使用同一组通用开发 Skill，但不包含 `avi-single-turn`。

`avi-single-turn` 是 EchoMind 的场景配方，负责 AVI 场景的需求澄清和场景资源引用。
它不负责修改通用 Tool，也不要求平台通用能力感知 AVI 业务。

### 6.3 Tool 差异

当前两个 Profile 的 Tool 白名单完全一致。EchoMind Agent 保留原 Agent 的完整平台
开发能力，包括：

- 文件、终端和 Skill 工具
- Workflow 生成、校验、调试和提交工具
- Sandbox 创建、终端、文件读写和删除工具
- Storage 与数据处理工具
- 训练分析、失败诊断和 Label Studio 工具

当前选择“Tool 白名单一致”是有意的：

- 两个 Agent 的差别首先是产品契约和上下文，不是能力是否可用。
- EchoMind Agent 仍需执行开发期调试和远端 Sandbox 验证。
- Tool 行为必须保持通用；如果未来确实需要不同行为，应新增通用 Tool 能力，
  而不是在旧 Tool 内按 Agent 名称分支。

`tool_allowlist` 使用 Product Runtime/Pi 执行面的工具名，由新会话创建时传给
Pi session。旧 `/api/pyromind/conversations` 的 OpenHands 直连构建路径继续装配其
原有完整工具集，不使用这份 Pi 工具清单做二次过滤。

## 7. 服务端构建与路由

Product Runtime 通过同一个 Router 工厂生成互相独立的执行入口：

```text
app.include_router(create_product_router(DEFAULT_PRODUCT_PROFILE))
app.include_router(create_product_router(ECHOMIND_PRODUCT_PROFILE))
app.include_router(create_product_router(ECHOMIND_CHAT_PROFILE))
app.include_router(echomind_projects_router)
```

共享 Router 工厂只复用请求校验、命令、SSE、快照和鉴权装配；每个入口使用自己的
`ConversationProfile`，不会因此共享会话或 Agent 上下文。

### 7.1 Pyromind Product Runtime 入口

Workbench Chat 使用：

```text
POST /api/v2/pyromind/conversations
GET  /api/v2/pyromind/conversations
GET  /api/v2/pyromind/conversations/{conversation_id}/snapshot
POST /api/v2/pyromind/conversations/{conversation_id}/commands
GET  /api/v2/pyromind/conversations/{conversation_id}/events
```

旧 Workbench OpenHands 直连入口 `/api/pyromind/conversations` 及其 messages 接口
继续兼容，不迁移已有会话。

### 7.2 EchoMind Chat 入口

```text
POST /api/echomind/conversations
GET  /api/echomind/conversations
GET  /api/echomind/conversations/{conversation_id}/snapshot
POST /api/echomind/conversations/{conversation_id}/commands
GET  /api/echomind/conversations/{conversation_id}/events
GET  /api/echomind/conversations/{conversation_id}/project-request
```

EchoMind 客户端不需要调用 Pyromind 的 Chat 接口来列会话、恢复历史或发送消息。

### 7.3 EchoMind Manager 计划入口

```text
POST /api/v2/echomind/conversations
GET  /api/v2/echomind/conversations/{conversation_id}/experiment-plans
```

这是 EchoMind Agent 的版本化计划能力，不是第三个 Agent，也不与 EchoMind Chat
互相读取会话。

### 7.4 Project 入口

```text
POST /api/echomind/projects
GET  /api/echomind/projects/{project_id}
POST /api/echomind/projects/{project_id}/nodes/{node_id}/start
POST /api/echomind/projects/{project_id}/nodes/{node_id}/run
GET  /api/echomind/projects/{project_id}/runs
```

Project 创建必须由用户显式确认；Project 创建后才生成 Graph。

### 7.5 共用构建函数，不共用执行契约

旧 Workbench 直连路径可以复用 `_build_agent_with_profile`，但必须显式传入自己的
Profile：

```python
agent, secrets = _build_agent_with_profile(
    http_request,
    request=request,
    profile=PYROMIND_PROFILE,
    ...
)
```

共享构建函数只负责装配通用能力，不负责让两个 Agent 共享上下文。

## 8. Conversation 隔离

Runtime 的 `profile_id` 负责第一层隔离。由于 EchoMind Manager 与 EchoMind Chat
共用同一个 `profile_id`，Product Router 还会在 conversation 目录写入 marker：

```text
<conversations>/<conversation_id>/.agent-profile
```

当前 marker：

| 入口 | Runtime profile_id | conversation marker |
|---|---|---|
| Pyromind Product Runtime | `None` | 无；通过“无 marker”兼容旧会话 |
| EchoMind Chat | `echomind` | `echomind` |
| EchoMind Manager | `echomind` | `echomind-manager` |

旧 Workbench 直连会话继续使用 `app=pyromind`、`agentprofile=pyromind` 的标签兼容；
没有 marker 的 Product Runtime 会话按 Pyromind 处理。带正确 marker、但历史元数据
缺少 `profile_id` 的 EchoMind 会话会在首次访问时补齐 `profile_id`。

服务端在每个受限入口同时校验 Runtime profile 和 Router marker：

- Pyromind Product Router 只接受无 marker 的 Pyromind 会话。
- EchoMind Chat 只接受 `echomind` marker。
- EchoMind Manager 只接受 `echomind-manager` marker。
- EchoMind Project API 只读取 `echomind` marker 会话中的 Project request。
- 跨 Profile 请求返回 `404`，不暴露另一个 Agent 的会话是否存在。

隔离规则：

1. 不共享 `conversation_id`。
2. 不共享事件流。
3. 不共享 conversation workspace。
4. 不共享后台任务上下文。
5. 不把一个 Agent 的对话摘要注入另一个 Agent。
6. 不允许一个 Profile 的消息 API 操作另一个 Profile 的会话。

## 9. 前端设计

前端共享展示组件，例如消息气泡、Tool 调用卡片、事件流 Hook 和状态栏；但不共享
Agent 会话状态。

### 9.1 原 Workbench Chat

- 继续使用 Pyromind Chat API。
- 原会话列表、工作流画布和调试交互保持不变。
- 不读取 EchoMind 本地会话 key。

### 9.2 EchoMind Chat

- 独立页面入口：`/?chat=echomind&userId=<debug-user-id>`
- 独立 API 客户端和契约模块。
- 独立 EchoMind-only 会话列表。
- 会话 ID 按用户持久化。
- 刷新后恢复当前 EchoMind 会话、历史事件和对应业务状态。
- 左侧会话列表显示标题、状态、短 ID 和更新时间。
- “New”只切换到新会话状态，不删除旧会话。

共享 UI 组件不等于共享会话状态。两个页面的 conversation id、事件数组和本地存储
key 必须分开。

## 10. 扩展方式

新增一个 EchoMind 场景时，推荐顺序：

1. 新增场景 Skill，例如 `.agents/skills/<scenario>/SKILL.md`。
2. 在 EchoMind Profile 的 `skill_allowlist` 中登记该 Skill。
3. 在 Prompt 或 Skill 中定义场景澄清项和业务边界。
4. 优先复用现有通用 Tool。
5. 只有通用能力确实缺失时，才新增通用 Tool。
6. 不修改 Pyromind Profile 的场景白名单。

禁止模式：

```python
# 不推荐
if agent_profile == "echomind" and scenario == "avi":
    return avi_special_case()
```

推荐模式：

```text
通用 Tool 接收标准输入，产生标准输出
场景差异由 Skill、Profile 白名单和 Prompt 承担
```

## 11. 兼容性

- 新 Workbench 使用 `/api/v2/pyromind/conversations`。
- 旧 `/api/pyromind/conversations` 接口继续存在。
- 原 Workbench Chat 的会话和历史不受 EchoMind 改动影响。
- Pyromind Profile 兼容历史会话：没有 Profile 标签或标签为 `workbench` 的旧会话
  仍视为 Pyromind 会话。
- EchoMind 会话不会进入 Pyromind 会话列表。
- EchoMind 消息接口不会接受 Pyromind 会话。
- 当前测试明确约束两个 Profile 的 Tool 白名单一致，EchoMind 只额外获得场景 Skill。

## 12. 风险与控制

| 风险 | 控制措施 |
|---|---|
| 两个入口复制过多路由代码 | 复用底层构建/服务函数；保留独立对外契约 |
| 共用 Tool 被写入 Agent/场景分支 | Code Review 明确禁止；差异放在 Profile/Skill |
| EchoMind 改动污染原 Agent | 独立 Router、Profile、Prompt、Conversation tag 和测试 |
| 前端共享状态导致串会话 | 独立 API、存储 key 和 EchoMind-only 会话列表 |
| 新增场景导致核心 Skill 膨胀 | 一个场景一个 Skill，通过 Profile 装配 |
| Profile 清单与实际加载不一致 | 测试校验 Skill/Tool allowlist 和会话隔离 |

## 13. 验证标准

### 架构

- 两个 Agent 使用不同 API 入口。
- 两个 Agent 使用不同 System Prompt 和 Profile tag。
- 两个 Agent 的 Conversation 不互相出现在对方列表中。

### 隔离

- EchoMind 消息接口拒绝 Pyromind 会话。
- Pyromind 消息接口拒绝 EchoMind 会话。
- 跨 Profile 请求返回 `404`。
- 一个 Agent 的对话内容不会进入另一个 Agent 的系统上下文。

### 能力

- EchoMind Agent 可以使用完整平台开发 Tool。
- EchoMind Agent 额外加载场景 Skill。
- Sandbox 工具在 EchoMind Profile 中可用。
- 通用 Tool 不包含 EchoMind 或场景专属分支。

### 回归

- 原 Pyromind Chat 创建、消息、事件流和工作流能力不变。
- 前端两个入口分别构建、启动和刷新恢复。
- EchoMind 会话列表只显示 EchoMind 会话。

## 14. 渐进式实施

| 阶段 | 内容 | 独立验证 |
|---|---|---|
| P0-1 | 抽取 `AgentProfileSpec`、`ConversationProfile` 和两个 Agent 清单 | Profile 加载、Skill/Tool 白名单 |
| P0-2 | 新增 EchoMind Router 和 Conversation tag | 创建会话、跨 Profile 404 |
| P0-3 | 新增 EchoMind 会话列表、详情和历史接口 | 列表隔离、刷新恢复 |
| P0-4 | 前端新增 EchoMind API 客户端和独立入口 | 创建、发送、切换、刷新 |
| P0-5 | 加入首个场景 Skill | Skill 选择、需求澄清、场景资源 |
| P1 | 按场景逐步补齐可执行能力 | 每个阶段独立启动验证 |

## 15. 代码索引

| 责任 | 文件 |
|---|---|
| Profile 契约 | `openhands-agent-server/openhands/agent_server/agent_profiles/spec.py` |
| Pyromind Profile | `openhands-agent-server/openhands/agent_server/agent_profiles/pyromind.py` |
| EchoMind Profile | `openhands-agent-server/openhands/agent_server/agent_profiles/echomind.py` |
| EchoMind Manager Profile | `openhands-agent-server/openhands/agent_server/agent_profiles/echomind_manager.py` |
| Product Router 与隔离 | `pyromind-agent-server/pyromind_agent_server/api/router.py` |
| Product Profile 装配 | `pyromind-agent-server/pyromind_agent_server/profiles.py` |
| Router 注册 | `pyromind-agent-server/pyromind_agent_server/bootstrap.py` |
| 共用 Agent 构建 | `openhands-agent-server/openhands/agent_server/pyromind_router.py` |
| EchoMind Project API | `pyromind-agent-server/pyromind_agent_server/api/echomind_projects.py` |
| EchoMind 前端 | `minimal-chat-frontend/src/EchoMindChat.tsx` |
| EchoMind API 契约 | `minimal-chat-frontend/src/echomindContract.ts` |
| 双 Agent 测试 | `tests/agent_server/test_echomind_router.py` |

## 16. 结论

本次改动的核心不是复制出第二个 Agent Runtime，而是把 Agent 的产品职责提升为服务端
可配置、可校验的执行契约：

- 两个 Agent 独立进出，不交接上下文。
- 通用能力继续复用，避免基础设施重复。
- 场景差异进入 Skill 和 Profile，不进入通用 Tool。
- 原 Pyromind Agent 保持兼容，EchoMind Agent 可以独立演进。

后续新增场景时，应继续遵循：

> 新场景优先增加 Skill，不改造共享 Tool；两个 Agent 可以共享能力，但不能共享会话。
