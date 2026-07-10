# run_workflow 工作流调度逻辑说明

`run_workflow` 调用了 Pyromind-sdk，并将**当前会话的 id 作为备注内容传递给 task**，调用 Pyromind 系统是**异步启动工作流**的，会直接返回 `task_id` 和当前工作流状态。此时当前会话的这个 `run_workflow` 就直接返回，但是需要按照一定的格式通知用户：工作流正在运行中；当前对话虽然结束了，但是 **Web 页面还需要锁定**，等工作流结束后会继续工作。

集成方会自行接入 **Kafka**，接收工作流完成的消息通知，然后通过**之前写入 task 的会话 id**，找到之前的会话，然后**重新启动**之前触发这个工作流的会话，继续做后续的事情。

---

## 实现方案

### 1. 核心设计：conversation_id 随 task 走完全链路

| 阶段 | conversation_id 的作用 |
|------|------------------------|
| **提交** | 写入 Pyromind task 的备注 / metadata（与 `task_id` 一并持久化到平台） |
| **Kafka 回传** | 平台将 task 备注中的 `conversation_id` 带入状态变更消息 |
| **回调** | Consumer 用 `conversation_id` 定位 agent-server 上的原会话并重启 Agent |
| **内存 registry** | 可选补充：Debug 阻塞等待、同进程快速 lookup；**非**跨重启的主路由依据 |

这样即使 agent-server 重启、Kafka consumer 与提交不在同一进程，只要 Kafka 消息里带有提交时写入的 `conversation_id`，仍能找回会话并继续。

### 2. 总体链路

```
Agent 调用 run_workflow（当前 conversation_id = C）
    │
    ▼
构造 TrainingTaskCreateRequest
    ├─ workflow: xyflow
    └─ remark / metadata: conversation_id = C   ◄── 提交时写入
    │
    ▼
studio.create() ──► 平台返回 task_id + Pending/Running
    │
    ├─► [可选] registry.register(task_id, conversation_id=C, ...)
    ├─► RunWorkflowObservation 返回 Agent（含 task_id、运行中提示）
    ├─► 会话本轮结束（Agent FINISHED / IDLE）
    └─► Web 前端锁定 UI（等待工作流终态）

        ... 平台异步执行 task（备注中仍携带 C）...

Kafka 工作流状态变更消息
    ├─ task_id
    ├─ status
    └─ conversation_id = C   ◄── 来自 task 备注，由平台/Kafka 透传
    │
    ▼
你的 Kafka Consumer
    │
    ▼
deliver_run_workflow_status(task_id, status, conversation_id=C, ...)
    │
    ├─► ConversationService.get_event_service(C)  找回原会话
    ├─► 终态：注入 conversation（system_reminder）
    ├─► auto_run=True：重新启动该会话的 Agent，继续后续工作
    └─► Web 前端解锁，展示终态与 Agent 后续回复
```

### 3. 提交阶段（run_workflow 工具内）

| 步骤 | 说明 |
|------|------|
| 读取 conversation_id | `str(conversation.id)` |
| DSL → xyflow | 复用 `DslToXyflowExecutor` |
| test_mode | `True` 时附加 `execution_mode: test`（Debug 试跑） |
| **写入 task 备注** | 将 `conversation_id` 写入 `TrainingTaskCreateRequest` 的 remark / metadata 字段（字段名以 Pyromind SDK 为准） |
| studio.create | 异步提交，平台持久化 task + 备注 |
| register（可选） | 同进程内写入 `RunWorkflowResultBroker`，供 Debug 阻塞 wait 使用 |
| 返回 observation | `status=Pending`（或平台初始状态）、`task_id`、用户可见文案 |

**提交请求伪代码**：

```python
conversation_id = str(conversation.id)

request = TrainingTaskCreateRequest(
    name=workflow_name,
    workflow=workflow_xyflow,
    remark=conversation_id,  # 或 metadata={"conversation_id": conversation_id}，以 SDK 字段为准
)
response = client.studio.create(request)
```

**返回给 Agent / 用户的 observation 文案格式（建议）**：

```text
工作流已提交到 Pyromind 平台，正在运行中。

- task_id: {task_id}
- conversation_id: {conversation_id}
- status: {status}
- 当前对话本轮已结束；工作流完成后 Agent 将自动在本会话继续。

请勿关闭页面，界面将保持锁定直至运行结束。
```

Web 前端可读取 observation 中的 `task_id`、`status`，或订阅 WebSocket 会话状态，进入「工作流运行中」锁定态。

### 4. 终态回调阶段（Kafka → 找回会话 → 重启）

| 步骤 | 说明 |
|------|------|
| Kafka 消息 | **必须**能解析出 `conversation_id`（来自 task 备注）；另含 `task_id`、`status`；失败时含 `error_log` |
| Consumer | 解析后调用 `deliver_run_workflow_status(conversation_id=..., ...)` |
| **找回会话** | `await conversation_service.get_event_service(UUID(conversation_id))`；找不到则 `unknown_conversation` |
| resolve | 更新 broker；若存在 Debug 阻塞 waiter 则唤醒 |
| **重启会话** | 终态 + `wait_mode=async`：注入 `<system_reminder>` + **`auto_run=True`** → `send_message(..., run=True)` |
| 前端 | 收到终态事件与新 Agent 消息后解锁 UI |

**终态**（触发重启对话）：`Succeeded`、`Failed`、`Error`、`Terminated`。  
**非终态**（`Pending`、`Running`）：不重启 Agent（可选更新进度，默认不推以免刷屏）。

**conversation_id 解析优先级**：

1. Kafka 消息中的 `conversation_id`（**主路径**，来自 task 备注）
2. 内存 registry 按 `task_id` lookup（同进程 fallback）
3. 均缺失 → 返回 `unknown_task`，无法重启会话

### 5. Web 页面锁定语义

| 状态 | 前端行为 |
|------|----------|
| 收到 `run_workflow` 提交成功 observation | 锁定输入/发送；展示「工作流运行中」+ `task_id` |
| 工作流执行中 | 保持锁定；依赖 Kafka 终态（或可选轮询平台） |
| Kafka 终态 + 同会话 auto_run 完成 | 解锁；展示终态摘要与 Agent 后续消息 |

锁定是 **Portal/前端产品行为**；SDK 通过 observation 文案 + WebSocket 事件驱动前端，不在 tool 内直接操作 UI。

### 6. Debug 链路差异（同一工具）

对 Agent 而言始终调用 `run_workflow`；Debug 与普通运行的差异：

| 维度 | 普通运行 | Debug 试跑 |
|------|----------|------------|
| 触发 | 用户要求运行工作流 | 用户要求 debug/调试/试跑当前画布工作流 |
| `test_mode` | `False` | `True` |
| `wait_mode` | `async`（提交即返回） | `block`（提交后返回 task_id，再阻塞等终态） |
| task 备注 | 同样写入 `conversation_id` | 同样写入 `conversation_id` |
| 用户可见 | 提交 observation + 页面锁定 | **同样立即返回 task_id**；tool 阻塞至终态后继续 debug 闭环 |
| 终态后 | Kafka → 按 `conversation_id` 重启 + `auto_run=True` | 同轮 `broker.wait` 返回终态 observation；Agent 读 `error_log` 修复后再次调用 |

Debug **不靠**独立 conversation 类型识别，靠**用户意图 + debug-workflow skill 路由**；Router 注入 `test_mode=True` 与 `wait_mode=block`（不由 LLM 自选）。

Debug 链路下 Kafka 仍会带 `conversation_id`；若同进程内已有 `broker.wait`，`resolve` 优先唤醒 waiter，**不再**对该次终态做 async 投递，避免重复。

### 7. 模块职责

| 模块 | 路径 | 职责 |
|------|------|------|
| 提交与 observation | `run_workflow.py` | 备注写入 `conversation_id`、`studio.create`、格式化 observation |
| Broker + 同步 callback | `workflow/run_workflow_broker.py`（待建） | Debug 阻塞 wait；可选 registry fallback |
| 会话找回与重启 | `agent_server/run_workflow_callback.py`（待建） | 按 `conversation_id` 找会话、`deliver_run_workflow_status`、auto_run |
| 可选 HTTP | `pyromind_router` webhook | `POST /api/pyromind/workflow/callback`（跨进程 consumer） |
| Kafka Consumer | **集成方自建** | 解析消息（含 task 备注中的 `conversation_id`）→ 调 `deliver_run_workflow_status` |

### 8. Kafka 消息格式

平台/Kafka 应透传提交时写入 task 的 `conversation_id`：

```json
{
  "task_id": "platform-task-123",
  "status": "Succeeded",
  "error_log": null,
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "updated_at": "2026-07-09T11:00:00Z"
}
```

`conversation_id` 与提交时 `TrainingTaskCreateRequest.remark`（或 metadata）一致，是**重启原会话的主键**。

### 9. 状态映射

```text
平台 / Kafka          →  RunWorkflowObservation.status
─────────────────────────────────────────────────────
succeeded, success    →  Succeeded
pending               →  Pending
running               →  Running
failed                →  Failed
error                 →  Error
terminated, stopped   →  Terminated
```

### 10. 已确认决策

| 项 | 决策 |
|----|------|
| 会话关联方式 | **提交时将 conversation_id 写入 task 备注**；Kafka 回传后按 id 找回会话 |
| 终态后是否重启 Agent | **是**，`auto_run=True` |
| Debug 识别方式 | 用户意图 + skill 路由；`test_mode=True` |
| Debug 是否返回 task_id | **是**，提交成功后立即返回 |
| 当前对话轮次 | 提交后本轮结束；终态到达后**在同一 conversation** 自动开启新轮次 |

### 11. 实施顺序

1. `run_workflow.py` — `studio.create` 请求携带 `conversation_id` 备注 + 格式化 observation
2. `run_workflow_broker.py` — Debug 阻塞 wait + 可选 registry
3. `run_workflow_callback.py` — 按 `conversation_id` 找回会话 + 注入事件 + auto_run
4. Router — 注入 `wait_mode` / `test_mode`
5. 平台/Kafka — 保证状态消息透传 task 备注中的 `conversation_id`
6. 集成方 Kafka consumer → 调用 `deliver_run_workflow_status`
7. Portal — 提交锁定 / 终态解锁 UI

---

## 方法定义（API）

以下方法为 SDK 侧约定接口；Kafka Consumer 与 agent-server 按此调用。

### 数据类型

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

RunWorkflowStatus = Literal[
    "Succeeded", "Pending", "Running", "Failed", "Error", "Terminated"
]

WaitMode = Literal["async", "block"]

CallbackOutcome = Literal[
    "resolved_blocked",       # Debug：broker.wait 被唤醒
    "delivered_async",        # 按 conversation_id 找回会话并 auto_run
    "unknown_task",           # 无 task_id 对应信息
    "unknown_conversation",   # conversation_id 无效或会话不在本实例
    "duplicate_terminal",     # 重复终态，已忽略
    "ignored_non_terminal",   # 非终态，不重启会话
]


@dataclass(frozen=True)
class RunWorkflowTaskRegistration:
    """同进程 registry 条目（Debug 阻塞 + fallback）。"""

    task_id: str
    conversation_id: str
    wait_mode: WaitMode
    test_mode: bool
    attempt: int
    max_attempts: int
    registered_at: datetime


@dataclass(frozen=True)
class RunWorkflowResult:
    """终态结果。"""

    task_id: str
    status: RunWorkflowStatus
    error_log: str | None = None
    conversation_id: str | None = None


@dataclass(frozen=True)
class RunWorkflowCallbackResult:
    """callback 方法返回值。"""

    outcome: CallbackOutcome
    task_id: str
    normalized_status: RunWorkflowStatus | None
    conversation_id: str | None
```

### 工具层 — `openhands.tools.workflow.run_workflow_broker`

```python
def normalize_platform_status(raw: str) -> RunWorkflowStatus:
    """将平台 / Kafka 原始 status 归一化为 RunWorkflowObservation 枚举值。"""


def build_task_remark(conversation_id: str) -> str:
    """
    构造写入 Pyromind task 的备注内容。

    当前约定：remark 即为 conversation_id 字符串（或与平台约定的 JSON）。
    平台需在 Kafka 状态消息中原样或解析后回传 conversation_id。
    """


class RunWorkflowResultBroker:
    def register(
        self,
        registration: RunWorkflowTaskRegistration,
    ) -> None:
        """studio.create 成功后调用；供 Debug wait 与 registry fallback。"""

    def resolve(
        self,
        task_id: str,
        status: RunWorkflowStatus,
        error_log: str | None = None,
    ) -> bool:
        """终态到达。若有 block waiter 则唤醒并返回 True。"""

    def wait(
        self,
        task_id: str,
        timeout: float,
    ) -> RunWorkflowResult | None:
        """Debug 链路阻塞等待终态；超时返回 None。"""

    def lookup(self, task_id: str) -> RunWorkflowTaskRegistration | None:
        """按 task_id 查 registry（conversation_id fallback）。"""


def get_run_workflow_result_broker() -> RunWorkflowResultBroker:
    """进程内单例 broker。"""


def resolve_run_workflow_status(
    *,
    task_id: str,
    status: str,
    error_log: str | None = None,
    conversation_id: str | None = None,
    updated_at: datetime | None = None,
) -> RunWorkflowCallbackResult:
    """
    同步核心 callback。

    1. normalize_platform_status(status)
    2. 若 conversation_id 为空，尝试 registry.lookup(task_id)
    3. broker.resolve(task_id, ...) — block 模式优先
    4. async 投递由 deliver_run_workflow_status 完成（需 agent-server）

    conversation_id 优先来自 Kafka（task 备注透传），registry 仅为 fallback。
    """
```

### Agent-server 层 — `openhands.agent_server.run_workflow_callback`

```python
async def deliver_run_workflow_status(
    *,
    task_id: str,
    status: str,
    error_log: str | None = None,
    conversation_id: str | None = None,
    updated_at: datetime | None = None,
    auto_run: bool = True,
) -> RunWorkflowCallbackResult:
    """
    Kafka consumer 推荐入口（async）。

    1. resolve_run_workflow_status(...) — 含 broker 唤醒
    2. 若 outcome == resolved_blocked：返回（Debug 同轮已处理）
    3. 解析 conversation_id（Kafka > registry）
    4. event_service = await conversation_service.get_event_service(UUID(conversation_id))
       - 找不到 → unknown_conversation
    5. 终态 + async：build_run_workflow_terminal_reminder → MessageEvent
    6. auto_run=True：event_service.send_message(..., run=True) 重启该会话 Agent
    7. 前端 WebSocket 收到事件后解锁 UI
    """


async def resume_conversation_after_workflow(
    *,
    conversation_id: UUID,
    task_id: str,
    status: RunWorkflowStatus,
    error_log: str | None = None,
    auto_run: bool = True,
) -> None:
    """
    按 conversation_id 找回原会话并注入终态提醒、可选 auto_run。

    从 deliver_run_workflow_status 拆出的核心「重启会话」步骤。
    """


def build_run_workflow_terminal_reminder(
    *,
    task_id: str,
    status: RunWorkflowStatus,
    error_log: str | None = None,
) -> str:
    """构造注入 LLM 的 system_reminder 文本。"""


def build_run_workflow_submission_user_text(
    *,
    task_id: str,
    conversation_id: str,
    status: RunWorkflowStatus,
) -> str:
    """提交成功后 RunWorkflowObservation.text：含 task_id、conversation_id、锁定提示。"""
```

### 提交层 — `run_workflow.py` 内挂钩

```python
def _execute_run(
    self,
    *,
    client: PyroMindAPIClient,
    conversation: BaseConversation,
    workflow_json: dict,
    workflow_name: str,
    test_mode: bool = False,
    attempt: int,
) -> RunWorkflowObservation:
    conversation_id = str(conversation.id)
    remark = build_task_remark(conversation_id)

    request = TrainingTaskCreateRequest(
        name=workflow_name,
        workflow=workflow_xyflow,
        remark=remark,  # 字段名以 Pyromind SDK 为准
    )
    response = client.studio.create(request)
    task_id = response.task_id

    get_run_workflow_result_broker().register(
        RunWorkflowTaskRegistration(
            task_id=task_id,
            conversation_id=conversation_id,
            wait_mode=self.wait_mode,
            test_mode=test_mode,
            attempt=attempt,
            max_attempts=self._max_attempts,
            registered_at=datetime.now(UTC),
        )
    )

    submission_text = build_run_workflow_submission_user_text(
        task_id=task_id,
        conversation_id=conversation_id,
        status=normalize_platform_status(response.status),
    )

    if self.wait_mode == "block":
        result = broker.wait(task_id, timeout=self.timeout_seconds)
        # 构造终态 RunWorkflowObservation ...

    return RunWorkflowObservation.from_text(
        text=submission_text,
        status=normalize_platform_status(response.status),
        task_id=task_id,
        attempt=attempt,
        max_attempts=self._max_attempts,
        is_error=False,
    )
```

### 可选 HTTP Webhook — `pyromind_router`

```python
class PyromindWorkflowCallbackRequest(BaseModel):
    task_id: str
    status: str
    error_log: str | None = None
    conversation_id: UUID  # 来自 task 备注，Kafka 透传


@pyromind_debug_webhook_router.post("/workflow/callback", response_model=Success)
async def pyromind_workflow_callback(
    request: PyromindWorkflowCallbackRequest,
) -> Success:
    """跨进程 consumer 调用；内部 await deliver_run_workflow_status(...)."""
```

### Kafka Consumer 调用示例（集成方）

```python
async def on_kafka_message(msg: dict) -> None:
    # conversation_id 来自提交 task 时写入的 remark，由平台/Kafka 透传
    await deliver_run_workflow_status(
        task_id=msg["task_id"],
        status=msg["status"],
        error_log=msg.get("error_log"),
        conversation_id=msg["conversation_id"],
        updated_at=parse_datetime(msg.get("updated_at")),
        auto_run=True,
    )
```

---

*文档版本：2026-07-09（conversation_id 随 task 备注透传）*
