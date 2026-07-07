# Workflow Canvas Event Snapshot Design

## 背景

当前前端和 Agent 对话时，前端会传入：

1. `workflowDslData`：当前工作流 DSL 字符串
2. 用户输入文本

`workflowDslData` 表示用户本次输入时看到的工作流快照，也就是 `in` 快照。`out` 快照需要等 Agent 输出 workflow 类型事件后才能得到。

前端页面实际渲染可能使用 xyflow/canvas 数据结构，但快照存储层不关心 xyflow，也不负责 DSL 与 xyflow 的相互转换：

- 前端发送消息前：将当前 xyflow/canvas 状态转换为 DSL 字符串
- 后端快照存储：只保存 DSL 字符串
- 前端恢复快照时：拿到 DSL 字符串后自行转换为 xyflow/canvas 并渲染

当前框架里的消息和输出本质上都是事件，每个事件都有自己的 `event.id`：

- 用户输入会生成 `source="user"` 的 `MessageEvent`
- Agent 输出会生成 `source="agent"` 的事件
- 如果 Agent 输出事件的 `eventType` 是 workflow，则该事件本身可以绑定 `out` 快照

因此快照绑定模型应从“按一轮对话 messageId 绑定 in/out”调整为“按框架事件 id 绑定快照”。

## 目标

最终语义：

```text
用户输入 MessageEvent.id      -> in 快照 DSL
Agent workflow 输出 Event.id  -> out 快照 DSL
```

前端加载历史消息时，每条消息都有自己的 `event.id`，可以直接用这个 `event.id` 查询是否存在快照。

对于一条具体消息：

- 如果它是用户输入消息，查询到的是输入时的 `in` 快照 DSL
- 如果它是 Agent workflow 输出消息，查询到的是输出后的 `out` 快照 DSL
- 如果它不是 workflow 相关事件，查询不到快照是正常结果

## 职责边界

### 后端快照存储负责

1. 保存 DSL 字符串快照
2. 绑定 `eventId -> versionId`
3. 区分快照角色：`in` 或 `out`
4. 按 `eventId` 查询快照
5. 按 `versionId` 查询版本
6. 文件形式持久化，保证幂等写入

### 后端快照存储不负责

1. 不保存 xyflow/canvas 渲染数据
2. 不做 DSL 与 xyflow/canvas 的互转
3. 不解析 DSL 语义
4. 不判断当前前端应该展示哪个版本
5. 不维护会话当前 active version
6. 不执行 undo/redo 状态切换

## 核心设计

### 快照版本

`WorkflowCanvasVersion` 只保存完整 DSL 快照，不表达事件关系。

字段建议：

```python
class WorkflowCanvasVersion(BaseModel):
    session_id: str = Field(alias="sessionId")
    version_id: str = Field(alias="versionId")
    version_no: int = Field(alias="versionNo")
    workflow_dsl_data: str = Field(alias="workflowDslData")
    summary: str | None = None
    feature: Any | None = None
    created_by: str | None = Field(default=None, alias="createdBy")
    created_at: datetime = Field(default_factory=utc_now, alias="createdAt")
    is_deleted: bool = Field(default=False, alias="isDeleted")
```

说明：

- `workflowDslData` 使用字符串接收和存储
- 不再使用 `workflowData/canvasData`
- 不再使用 `workflowJsonData`
- 一期保存完整 DSL 快照，不做增量 diff

### 事件快照绑定

新增或替换原来的 message 版本关系模型：

```python
class WorkflowCanvasEventSnapshot(BaseModel):
    session_id: str = Field(alias="sessionId")
    event_id: str = Field(alias="eventId")
    snapshot_role: Literal["in", "out"] = Field(alias="snapshotRole")
    version_id: str = Field(alias="versionId")
    parent_user_message_event_id: str | None = Field(
        default=None,
        alias="parentUserMessageEventId",
    )
    event_type: str | None = Field(default=None, alias="eventType")
    feature: Any | None = None
    created_at: datetime = Field(default_factory=utc_now, alias="createdAt")
```

说明：

- `eventId` 是框架内真实事件 id
- `snapshotRole = "in"` 时，`eventId` 通常是用户输入 `MessageEvent.id`
- `snapshotRole = "out"` 时，`eventId` 通常是 Agent workflow 输出事件 id
- `parentUserMessageEventId` 只在 `out` 场景建议记录，用于追溯该 workflow 输出基于哪条用户输入生成
- 前端按消息查快照时只需要 `eventId`

## 文件存储结构

仍然使用会话目录下的文件存储：

```text
{conversation_dir}/workflow_canvas/state.json
```

建议 schema：

```json
{
  "schemaVersion": 2,
  "nextVersionNo": 3,
  "versions": {
    "v000001": {
      "sessionId": "s1",
      "versionId": "v000001",
      "versionNo": 1,
      "workflowDslData": "workflow ...",
      "summary": "用户输入时快照",
      "createdBy": "workflow_canvas_snapshot_hook",
      "createdAt": "2026-07-07T10:00:00Z",
      "isDeleted": false
    }
  },
  "eventSnapshots": {
    "user_event_1": {
      "sessionId": "s1",
      "eventId": "user_event_1",
      "snapshotRole": "in",
      "versionId": "v000001",
      "createdAt": "2026-07-07T10:00:00Z"
    },
    "agent_event_9": {
      "sessionId": "s1",
      "eventId": "agent_event_9",
      "snapshotRole": "out",
      "versionId": "v000002",
      "parentUserMessageEventId": "user_event_1",
      "eventType": "workflow",
      "createdAt": "2026-07-07T10:00:05Z"
    }
  }
}
```

幂等键：

```text
sessionId + eventId
```

同一个事件只允许绑定一个快照。

## 前端对话入参

当前 Pyromind 发送消息接口已经包含 `workflow_dsl`：

```http
POST /api/pyromind/conversations/{conversation_id}/messages
```

请求体：

```json
{
  "text": "帮我加一个通知节点",
  "workflow_dsl": "workflow ...",
  "run": true
}
```

字段说明：

- `workflow_dsl` 可选
- `workflow_dsl = null` 时，不保存 `in` 快照
- `workflow_dsl = ""` 时，表示画布为空，也需要保存空 DSL 快照
- 前端负责在发送前将当前 xyflow/canvas 转换成 DSL 字符串
- 通用 `/api/conversations/{conversation_id}/events` 接口不需要为本能力新增入参

现有 Pydantic 结构：

```python
class PyromindSendMessageRequest(BaseModel):
    text: str = Field(description="The user's message text.")
    workflow_dsl: str | None = Field(
        default=None,
        description="DSL of the workflow currently on the canvas.",
    )
    run: bool = True
```

## Hook 方案

快照保存逻辑不直接写进主流程，而是封装成内部 hook。

建议新增：

```text
openhands-agent-server/openhands/agent_server/workflow_canvas_snapshot_hook.py
```

核心职责：

1. 用户消息创建后保存 `in` 快照 DSL
2. Agent workflow 输出事件产生后保存 `out` 快照 DSL
3. 忽略无快照、非 workflow、重复写入等场景
4. 调用 `FileWorkflowCanvasStore` 完成文件持久化

示例结构：

```python
class WorkflowCanvasSnapshotHook:
    def __init__(self, store: FileWorkflowCanvasStore):
        self.store = store

    def on_user_message_created(
        self,
        *,
        event_id: str | None,
        snapshot: WorkflowCanvasSnapshotInput | None,
    ) -> None:
        if event_id is None or snapshot is None:
            return
        self.store.save_event_snapshot(
            SaveWorkflowCanvasEventSnapshotRequest(
                eventId=event_id,
                snapshotRole="in",
                workflowDslData=snapshot.workflow_dsl_data,
                summary=snapshot.summary,
                createdBy="workflow_canvas_snapshot_hook",
            )
        )

    def on_agent_event(
        self,
        *,
        event: Event,
        parent_user_message_event_id: str | None,
    ) -> None:
        snapshot = extract_workflow_snapshot(event)
        if snapshot is None:
            return
        self.store.save_event_snapshot(
            SaveWorkflowCanvasEventSnapshotRequest(
                eventId=event.id,
                snapshotRole="out",
                workflowDslData=snapshot.workflow_dsl_data,
                parentUserMessageEventId=parent_user_message_event_id,
                eventType="workflow",
                summary=snapshot.summary,
                createdBy="workflow_canvas_snapshot_hook",
            )
        )
```

## 触发点

### 触发点 1：保存 in 快照

位置：

```text
openhands-agent-server/openhands/agent_server/event_service.py
EventService.send_message()
```

当前流程：

```python
await event_service.send_message(
    message,
    run=request.run,
    extended_content=[reminder] if reminder else None,
    workflow_dsl_snapshot=request.workflow_dsl,
)
```

`EventService.send_message()` 内部在用户 `MessageEvent` 创建后、Agent run 前保存 in 快照：

```python
await loop.run_in_executor(
    None,
    conversation.send_message,
    message,
)

if workflow_dsl_snapshot is not None:
    await loop.run_in_executor(
        None,
        self._save_pyromind_workflow_input_snapshot_sync,
        workflow_dsl_snapshot,
    )
```

说明：

- `LocalConversation` 已经在用户 `MessageEvent` 被 emit 时更新 `state.last_user_message_id`
- `_save_pyromind_workflow_input_snapshot_sync()` 通过 `last_user_message_id` 读取用户输入事件 id
- `workflow_dsl=None` 时不保存 in
- `workflow_dsl=""` 时保存空 DSL 快照

### 触发点 2：保存 out 快照

位置：

```text
openhands-agent-server/openhands/agent_server/event_service.py
EventService._emit_pyromind_workflow_if_dirty_sync()
```

当前 Pyromind workflow 输出已经由 `_emit_pyromind_workflow_if_dirty_sync()` 统一生成：

```python
event = ConversationStateUpdateEvent(
    key=PYROMIND_WORKFLOW_EVENT_KEY,
    value=observation.model_dump(mode="json"),
)
```

在这个事件被写入会话后，直接用该事件自己的 `event.id` 保存 out 快照：

```python
conversation._on_event(event)
self._workflow_canvas_snapshot_hook().save_out_snapshot(
    event_id=event.id,
    workflow_dsl_data=observation.workflow,
    parent_user_message_event_id=parent_user_message_event_id,
    summary=observation.summary,
)
```

说明：

- out 快照绑定的是 Agent workflow 输出事件自己的 `event.id`
- DSL 来源是现有 `WorkflowFileObservation.workflow`
- `parentUserMessageEventId` 只用于追溯，不影响前端按 `event.id` 查询快照
- 非 Pyromind 会话、非 dirty workflow、没有 `workflow.py` 时不会保存 out

## Workflow 输出识别

当前不需要从普通 Agent 文本中猜测 workflow 输出，服务端已有明确事件：

```python
ConversationStateUpdateEvent(
    key=PYROMIND_WORKFLOW_EVENT_KEY,
    value=WorkflowFileObservation(...).model_dump(mode="json"),
)
```

判断条件：

- `event.key == PYROMIND_WORKFLOW_EVENT_KEY`
- `event.value` 可解析为 `WorkflowFileObservation`
- `WorkflowFileObservation.workflow` 是要保存的 DSL 字符串

快照存储层不从 xyflow/canvas 或普通文本反推 DSL。

## API 设计

### 保存事件快照

内部 hook 可以直接调用 store；仍建议保留 HTTP API，方便调试、补偿写入和前端兜底。

```http
POST /conversations/{conversation_id}/workflow-canvas/event-snapshots
```

请求：

```json
{
  "eventId": "user_event_1",
  "snapshotRole": "in",
  "workflowDslData": "workflow ...",
  "summary": "用户输入时快照"
}
```

out 请求：

```json
{
  "eventId": "agent_event_9",
  "snapshotRole": "out",
  "workflowDslData": "workflow ...",
  "parentUserMessageEventId": "user_event_1",
  "eventType": "workflow",
  "summary": "Agent workflow 输出快照"
}
```

返回：

```json
{
  "sessionId": "s1",
  "eventId": "agent_event_9",
  "snapshotRole": "out",
  "versionId": "v000002",
  "versionNo": 2,
  "workflowDslData": "workflow ...",
  "parentUserMessageEventId": "user_event_1",
  "eventType": "workflow",
  "createdAt": "2026-07-07T10:00:05Z"
}
```

### 查询单个事件快照

```http
GET /conversations/{conversation_id}/workflow-canvas/events/{event_id}/snapshot
```

返回：

```json
{
  "sessionId": "s1",
  "eventId": "agent_event_9",
  "snapshotRole": "out",
  "versionId": "v000002",
  "versionNo": 2,
  "workflowDslData": "workflow ..."
}
```

没有快照时返回 404。

### 批量查询事件快照

历史消息列表建议使用 batch，避免 N+1。

```http
POST /conversations/{conversation_id}/workflow-canvas/event-snapshots/batch
```

请求：

```json
{
  "eventIds": ["user_event_1", "agent_event_9", "agent_event_10"]
}
```

返回：

```json
{
  "snapshots": {
    "user_event_1": {
      "eventId": "user_event_1",
      "snapshotRole": "in",
      "versionId": "v000001",
      "versionNo": 1
    },
    "agent_event_9": {
      "eventId": "agent_event_9",
      "snapshotRole": "out",
      "versionId": "v000002",
      "versionNo": 2
    }
  }
}
```

没有快照的 event 不返回，或返回 `null`，二选一即可。建议不返回缺失项，响应更轻。

### 查询指定版本

```http
GET /conversations/{conversation_id}/workflow-canvas/versions/{version_id}
```

### 查询版本列表

```http
GET /conversations/{conversation_id}/workflow-canvas/versions
```

## 幂等和冲突

幂等规则：

1. 同一个 `sessionId + eventId` 只能绑定一条快照记录
2. 重复写相同 `snapshotRole + workflowDslData`，返回已有版本
3. 重复写同一个 `eventId` 但 `snapshotRole` 不同，返回冲突
4. 重复写同一个 `eventId` 且 DSL 数据不同，返回冲突
5. 非 workflow Agent 事件不保存 out
6. 用户消息没有 `workflow_dsl` 时不保存 in

错误码建议：

```text
DUPLICATE_WORKFLOW_CANVAS_EVENT_SNAPSHOT
WORKFLOW_CANVAS_EVENT_SNAPSHOT_NOT_FOUND
WORKFLOW_CANVAS_VERSION_NOT_FOUND
```

## 前端加载流程

历史消息加载：

1. 前端加载会话事件列表
2. 提取所有 `event.id`
3. 调用 batch 快照接口
4. 将返回的快照元数据挂到对应消息上

用户点击某条消息回滚：

1. 取该消息 `event.id`
2. 查询该 event 绑定的快照
3. 使用返回的 `workflowDslData`
4. 前端将 DSL 转换成 xyflow/canvas 并恢复画布

对于用户输入消息：

```text
event.id -> in 快照 DSL
```

对于 Agent workflow 输出消息：

```text
event.id -> out 快照 DSL
```

## 后续实现清单

1. 修改 `WorkflowCanvasVersion` 字段：只保留 `workflowDslData`
2. 删除或替换 `MessageWorkflowCanvasVersion` 概念
3. 新增 `WorkflowCanvasEventSnapshot` 模型
4. store 从 `save_message_versions()` 改成 `save_event_snapshot()`
5. state 文件结构从 `messageVersions` 改成 `eventSnapshots`
6. router 从 `/message-versions`、`/messages/{id}` 改成 `/event-snapshots`、`/events/{event_id}/snapshot`
7. 新增 batch 查询接口
8. 复用 `PyromindSendMessageRequest.workflow_dsl` 作为 in 快照来源
9. 新增 `WorkflowCanvasSnapshotHook`
10. `EventService.send_message()` 在用户消息创建后触发 in hook
11. `EventService._emit_pyromind_workflow_if_dirty_sync()` 生成 workflow 事件后触发 out hook
12. 测试覆盖：
    - 用户消息保存 in DSL
    - Agent workflow 输出保存 out DSL
    - 非 workflow Agent 输出不保存
    - 同 event 重复写相同 DSL 幂等
    - 同 event 重复写不同 DSL 冲突
    - 单 event 查询
    - batch 查询
    - 版本列表和指定版本查询
