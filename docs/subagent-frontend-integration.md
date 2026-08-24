# Subagent 前端对接说明

## 1. 目标

服务端提供统一的阻塞式 `subagent` 工具，当前支持两种类型：

| 类型 | 用途 | 权限 |
|---|---|---|
| `search` | 查询 Pyromind 知识库 | 只读 |
| `general_purpose` | 执行复杂、多步骤工作区任务 | 读写 |

前端需要把主 Agent 发起的 Subagent 调用展示为独立任务卡，并在任务完成后更新同一张卡片。

## 2. 事件链路

Subagent 沿用现有 Conversation WebSocket 和历史事件接口，不需要新增前端 API。

```text
主 Agent 生成 subagent tool call
  → 父会话发送 ActionEvent
  → 子 Agent 在独立会话中阻塞运行
  → 父会话发送 ObservationEvent
  → 主 Agent读取 handoff 并继续处理
  → 主 Agent发送最终 MessageEvent
```

子 Agent 的中间事件不会进入父会话事件流。前端不需要轮询子会话，也不需要读取子 Agent 的逐步思考和工具调用。

## 3. ActionEvent

识别条件：

```ts
event.kind === 'ActionEvent'
  && event.tool_call.name === 'subagent'
```

示例：

```json
{
  "kind": "ActionEvent",
  "id": "action-id",
  "timestamp": "2026-08-18T10:00:00Z",
  "source": "agent",
  "tool_name": "subagent",
  "tool_call": {
    "id": "call-id",
    "name": "subagent",
    "arguments": "{\"type\":\"search\",\"task\":\"查询 SFT 文档\"}"
  }
}
```

`tool_call.arguments` 可能是 JSON 字符串，也可能已经是对象，前端需要兼容两种格式。

```ts
export type SubAgentType = 'search' | 'general_purpose';

export interface SubAgentArguments {
  type: SubAgentType;
  task: string;
}
```

## 4. ObservationEvent

前端通过 `action_id` 将 Observation 配对到原来的 Action：

```ts
observationEvent.action_id === actionEvent.id
```

不要依赖 Action 和 Observation 在事件数组中相邻。并行工具调用、WebSocket 重连和历史事件重放都可能让它们不相邻。

示例：

```json
{
  "kind": "ObservationEvent",
  "id": "observation-id",
  "timestamp": "2026-08-18T10:00:31Z",
  "source": "environment",
  "tool_name": "subagent",
  "tool_call_id": "call-id",
  "action_id": "action-id",
  "observation": {
    "kind": "SubAgentObservation",
    "task_id": "task_00000001",
    "status": "completed",
    "type": "search",
    "child_conversation_id": "child-conversation-id",
    "is_error": false,
    "to_llm_content": [
      {
        "type": "text",
        "text": "SFT 训练说明……\n来源：knowledge/studio/sft-training.mdx"
      }
    ]
  }
}
```

建议类型定义：

```ts
export interface SubAgentObservation extends BaseObservation {
  kind: 'SubAgentObservation';
  task_id: string;
  status: 'completed' | 'error';
  type: SubAgentType;
  child_conversation_id?: string | null;
}
```

Subagent 是阻塞式工具。父会话收到 Observation 时，任务已经进入终态；运行中状态由前端根据“存在 ActionEvent，但没有对应 ObservationEvent”判断。

## 5. 状态判断

```ts
const observation = observationsByActionId.get(action.id);

const isRunning = !observation;
const isError = Boolean(observation?.observation.is_error)
  || observation?.observation.status === 'error';
const isCompleted = Boolean(observation) && !isError;
```

最终 handoff 文本优先从 `to_llm_content` 读取，兼容回退到 `content`：

```ts
const content = observation.observation.to_llm_content
  ?? observation.observation.content
  ?? [];

const resultText = content
  .filter((item) => item.type === 'text')
  .map((item) => item.text)
  .join('\n');
```

## 6. 事件配对

先遍历全部 Observation 建立索引，再渲染 Action：

```ts
const observationsByActionId = new Map<string, ObservationEvent>();

for (const event of events) {
  if (event.kind === 'ObservationEvent' && event.action_id) {
    observationsByActionId.set(event.action_id, event);
  }
}

for (const event of events) {
  if (
    event.kind === 'ActionEvent'
    && event.tool_call.name === 'subagent'
  ) {
    renderSubAgentCard({
      action: event,
      observation: observationsByActionId.get(event.id),
    });
  }
}
```

该方式同时适用于实时 WebSocket 事件和重新加载后的历史事件。

## 7. UI 建议

### 7.1 类型文案

```ts
const typeLabels: Record<SubAgentType, string> = {
  search: '知识检索',
  general_purpose: '通用任务',
};
```

### 7.2 默认卡片

```text
子 Agent · 知识检索                    已完成
查询 SFT 训练相关文档

已完成 · 31s                         查看详情
```

### 7.3 运行中

`search`：

```text
正在查询知识库索引……
```

`general_purpose`：

```text
正在处理多步骤任务……
```

### 7.4 展开详情

展开区域建议显示：

- 完整 `task`
- 最终 handoff 或错误文本
- `task_id`
- `child_conversation_id`
- Action 到 Observation 的耗时

`child_conversation_id` 默认只作为诊断信息展示。除非后端提供相应权限和跳转接口，否则不要将它做成可点击链接。

## 8. Subagent 卡片与主 Agent 回答

Subagent 卡片和主 Agent 最终回答是两种不同内容：

```text
Subagent 卡片：展示委托任务、运行状态和最终 handoff
主 Agent 消息：展示主 Agent 整理后的用户答案
```

卡片主体中的 `task` 是主 Agent 传给子 Agent 的任务说明，不是子 Agent 的最终回答。最终 handoff 应在 Observation 到达后显示在详情区域。

## 9. 去重原则

- 一个 `ActionEvent.id` 对应一张卡片。
- Observation 到达后更新原卡片，不创建第二张卡片。
- 不要根据 `task` 文本去重。
- 不要合并不同 `ActionEvent.id` 的调用。

服务端提示词要求主 Agent 在一轮中最多发起一次 `search` Subagent 调用，但不同 Action ID 仍代表真实的独立执行。前端不能通过隐藏卡片代替服务端调用控制。

## 10. 错误与异常

以下任一条件成立时显示失败状态：

```ts
observation.observation.is_error === true
  || observation.observation.status === 'error'
```

如果 Observation 不是 `SubAgentObservation`，沿用现有通用工具错误展示逻辑，不要强制转换类型。

WebSocket 断线重连后，应使用完整历史事件重新建立 `observationsByActionId`，而不是依赖组件内的临时状态恢复任务结果。

## 11. 验收清单

1. 收到 `search` Action 后立即显示“知识检索 · 运行中”。
2. 收到 `general_purpose` Action 后立即显示“通用任务 · 运行中”。
3. Observation 到达后更新原卡片，不新增第二张卡片。
4. `completed` 显示成功状态和最终 handoff。
5. `error` 或 `is_error=true` 显示失败状态和错误信息。
6. 刷新页面并重新加载历史事件后，Action 与 Observation 仍能正确配对。
7. 多个不同 Action ID 分别展示，不按相似 task 文本隐藏。
8. 主 Agent 后续 MessageEvent 继续使用普通回答气泡展示。
9. 前端不轮询或展开子会话内部事件。

## 12. 代码位置

服务端契约：

- `openhands-agent-server/openhands/agent_server/pyromind_subagent.py`

参考前端实现：

- `src/components/SubAgentBlock.tsx`
- `src/components/ChatPanel.tsx`
- `src/types.ts`
