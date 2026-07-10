# Workflow Canvas Event Snapshot Design

## 背景

当前前端和 Agent 对话时，前端会传入：

1. `workflow_xyflow`：当前工作流的 xyflow JSON
2. 用户输入文本

`workflow_xyflow` 表示用户本次输入时看到的工作流快照，也就是 `in` 快照的前端结构。后端收到后通过 `pyromind_sdk.client.workflow.DslConverter` 转成 DSL，写入 `workflow.py` 并提供给 Agent 处理。

Agent 输出 workflow 时，输出源仍然是 `workflow.py` 中的 DSL。后端在推送给前端前再将 DSL 转成 xyflow JSON，供前端直接渲染。

快照存储同时保存两份结构：

- `workflowDslData`：后端和 Agent 使用的 DSL 字符串
- `workflowXyflowData`：前端展示/恢复使用的 xyflow JSON

约束：主流程里的 xyflow JSON 不进入 Agent 上下文，Agent 只看到转换后的 DSL / `workflow.py`，避免上下文膨胀。DSL 与 xyflow 的自动互转由后端服务代码直接调用 converter helper 完成，不依赖 Agent 调用 tool。

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

每个快照版本同时可携带 xyflow JSON：

```text
event.id -> workflowDslData + workflowXyflowData
```

前端加载历史消息时，每条消息都有自己的 `event.id`，可以直接用这个 `event.id` 查询是否存在快照。

对于一条具体消息：

- 如果它是用户输入消息，查询到的是输入时的 `in` 快照 DSL + xyflow
- 如果它是 Agent workflow 输出消息，查询到的是输出后的 `out` 快照 DSL + xyflow
- 如果它不是 workflow 相关事件，查询不到快照是正常结果

## 职责边界

### 后端快照存储负责

1. 保存 DSL 字符串快照
2. 保存对应的 xyflow JSON 快照
3. 绑定 `eventId -> versionId`
4. 区分快照角色：`in` 或 `out`
5. 按 `eventId` 查询快照
6. 按 `versionId` 查询版本
7. 文件形式持久化，保证幂等写入

### 后端快照存储不负责

1. 不解析 DSL 语义
2. 不判断当前前端应该展示哪个版本
3. 不维护会话当前 active version
4. 不执行 undo/redo 状态切换

## 核心设计

### 快照版本

`WorkflowCanvasVersion` 保存完整 DSL 快照和对应 xyflow 快照，不表达事件关系。

字段建议：

```python
class WorkflowCanvasVersion(BaseModel):
    session_id: str = Field(alias="sessionId")
    version_id: str = Field(alias="versionId")
    version_no: int = Field(alias="versionNo")
    workflow_dsl_data: str = Field(alias="workflowDslData")
    workflow_xyflow_data: dict[str, Any] | None = Field(
        default=None,
        alias="workflowXyflowData",
    )
    summary: str | None = None
    feature: Any | None = None
    created_by: str | None = Field(default=None, alias="createdBy")
    created_at: datetime = Field(default_factory=utc_now, alias="createdAt")
    is_deleted: bool = Field(default=False, alias="isDeleted")
```

说明：

- `workflowDslData` 使用字符串接收和存储
- `workflowXyflowData` 使用 JSON 对象接收和存储
- 不再使用 `workflowData/canvasData`
- 不再使用 `workflowJsonData`
- 一期保存完整快照，不做增量 diff

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
  "schemaVersion": 1,
  "nextVersionNo": 3,
  "versions": {
    "v000001": {
      "sessionId": "s1",
      "versionId": "v000001",
      "versionNo": 1,
      "workflowDslData": "workflow ...",
      "workflowXyflowData": {
        "name": "workflow",
        "nodes": [],
        "edges": []
      },
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

当前 Pyromind 发送消息接口只接收 `workflow_xyflow`：

```http
POST /api/pyromind/conversations/{conversation_id}/messages
```

请求体：

```json
{
  "text": "帮我加一个通知节点",
  "workflow_xyflow": {
    "name": "workflow",
    "nodes": [],
    "edges": []
  },
  "run": true
}
```

字段说明：

- `workflow_xyflow` 可选
- `workflow_xyflow = null` 或不传时，不保存 `in` 快照
- `workflow_xyflow.nodes = []` 且 `workflow_xyflow.edges = []` 时，表示画布为空，后端转换为空 DSL 快照
- `workflow_dsl` 不再作为请求入参接受，调用方继续传该字段会被请求模型拒绝
- 后端负责将 xyflow 转为 DSL，再同步到 `workflow.py`
- 通用 `/api/conversations/{conversation_id}/events` 接口不需要为本能力新增入参

现有 Pydantic 结构：

```python
class PyromindSendMessageRequest(BaseModel):
    text: str = Field(description="The user's message text.")
    workflow_xyflow: dict[str, Any] | None = Field(
        default=None,
        description="xyflow JSON of the workflow currently on the canvas.",
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

1. 用户消息创建后保存 `in` 快照 DSL + xyflow
2. Agent workflow 输出事件产生后保存 `out` 快照 DSL + xyflow
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
                workflowXyflowData=snapshot.workflow_xyflow_data,
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
                workflowXyflowData=snapshot.workflow_xyflow_data,
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
    workflow_dsl_snapshot=workflow_dsl,
    workflow_xyflow_snapshot=request.workflow_xyflow,
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
        workflow_xyflow_snapshot,
    )
```

说明：

- `LocalConversation` 已经在用户 `MessageEvent` 被 emit 时更新 `state.last_user_message_id`
- `_save_pyromind_workflow_input_snapshot_sync()` 通过 `last_user_message_id` 读取用户输入事件 id
- `workflow_xyflow=None` 时不保存 in
- 空 xyflow 画布转换成 `workflow_dsl=""` 后保存空 DSL 快照
- `workflow_xyflow_snapshot` 只进入快照存储，不注入 Agent 上下文

### 触发点 2：保存 out 快照

位置：

```text
openhands-agent-server/openhands/agent_server/event_service.py
EventService._emit_pyromind_workflow_if_dirty_sync()
```

当前 Pyromind workflow 输出已经由 `_emit_pyromind_workflow_if_dirty_sync()` 统一生成：

```python
workflow_xyflow = convert_dsl_to_xyflow(observation.workflow, name=observation.name)
observation = observation.model_copy(update={"xyflow": workflow_xyflow})
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
    workflow_xyflow_data=workflow_xyflow,
    parent_user_message_event_id=parent_user_message_event_id,
    summary=observation.summary,
)
```

说明：

- out 快照绑定的是 Agent workflow 输出事件自己的 `event.id`
- 事件 value 面向前端携带 `xyflow`
- Agent 上下文只通过 `WorkflowFileObservation.to_llm_content` 看到轻量文本摘要，不携带 xyflow JSON
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
2. 重复写相同 `snapshotRole + workflowDslData + workflowXyflowData`，返回已有版本
3. 重复写同一个 `eventId` 但 `snapshotRole` 不同，返回冲突
4. 重复写同一个 `eventId` 且 DSL 或 xyflow 数据不同，返回冲突
5. 非 workflow Agent 事件不保存 out
6. 用户消息没有可转换的 workflow 快照时不保存 in

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
3. 使用返回的 `workflowXyflowData` 恢复画布
4. 后端 fork 分支时使用 `workflowDslData` 重建 `workflow.py`

对于用户输入消息：

```text
event.id -> in 快照 DSL + xyflow
```

对于 Agent workflow 输出消息：

```text
event.id -> out 快照 DSL + xyflow
```

## 后续实现清单

1. 修改 `WorkflowCanvasVersion` 字段：保存 `workflowDslData` 和 `workflowXyflowData`
2. 删除或替换 `MessageWorkflowCanvasVersion` 概念
3. 新增 `WorkflowCanvasEventSnapshot` 模型
4. store 从 `save_message_versions()` 改成 `save_event_snapshot()`
5. state 文件结构从 `messageVersions` 改成 `eventSnapshots`
6. router 从 `/message-versions`、`/messages/{id}` 改成 `/event-snapshots`、`/events/{event_id}/snapshot`
7. 新增 batch 查询接口
8. 使用 `PyromindSendMessageRequest.workflow_xyflow` 作为 in 快照来源，不接受 `workflow_dsl` 入参
9. 新增 `WorkflowCanvasSnapshotHook`
10. `EventService.send_message()` 在用户消息创建后触发 in hook
11. `EventService._emit_pyromind_workflow_if_dirty_sync()` 生成 workflow 事件后触发 out hook
12. 测试覆盖：
    - 用户消息保存 in DSL + xyflow
    - Agent workflow 输出保存 out DSL + xyflow
    - 非 workflow Agent 输出不保存
    - 同 event 重复写相同 DSL + xyflow 幂等
    - 同 event 重复写不同 DSL 冲突
    - 单 event 查询
    - batch 查询
    - 版本列表和指定版本查询
