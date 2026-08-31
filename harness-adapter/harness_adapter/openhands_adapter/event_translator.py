from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, JsonValue
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.events import HarnessEvent

from openhands.agent_server.models import ServerErrorEvent
from openhands.agent_server.pyromind_constants import PYROMIND_WORKFLOW_EVENT_KEY
from openhands.sdk.event import (
    ActionEvent,
    AgentErrorEvent,
    Condensation,
    ConversationStateUpdateEvent,
    Event,
    InterruptEvent,
    MessageEvent,
    ObservationEvent,
    PauseEvent,
    StreamingDeltaEvent,
    UserRejectObservation,
)
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.llm import ImageContent, TextContent
from openhands.sdk.utils.redact import sanitize_dict


@dataclass(slots=True)
class TranslationState:
    session_id: str
    run_id: str | None = None
    streaming_message_id: str | None = None
    pending_actions: dict[str, ActionEvent] = field(default_factory=dict)
    pending_permission_id: str | None = None
    last_status: str | None = None
    last_usage: JsonObject | None = None
    external_tasks: dict[str, JsonObject] = field(default_factory=dict)
    suppress_workflow_events: int = 0

    def begin_command(self, command_id: str) -> None:
        self.run_id = command_id

    def ensure_run_id(self) -> str:
        if self.run_id is None:
            self.run_id = uuid4().hex
        return self.run_id


def translate_event(
    state: TranslationState,
    event: Event,
) -> tuple[HarnessEvent, ...]:
    if isinstance(event, MessageEvent):
        return _translate_message(state, event)
    if isinstance(event, StreamingDeltaEvent):
        return _translate_delta(state, event)
    if isinstance(event, ActionEvent):
        state.pending_actions[event.tool_call_id] = event
        category = "subagent" if event.tool_name == "subagent" else "tool"
        return (
            _event(
                state,
                event,
                "operation.started",
                {
                    "operation_id": event.tool_call_id,
                    "name": event.tool_name,
                    "category": category,
                    "arguments": _tool_arguments(event),
                    "thought": _content(event.thought),
                },
            ),
        )
    if isinstance(event, ObservationEvent):
        state.pending_actions.pop(event.tool_call_id, None)
        payload: JsonObject = {
            "operation_id": event.tool_call_id,
            "name": event.tool_name,
            "output": _content(event.observation.to_llm_content),
            "details": _json_value(event.observation.model_dump(mode="json")),
        }
        output = [
            _event(
                state,
                event,
                "operation.failed"
                if event.observation.is_error
                else "operation.completed",
                {
                    **payload,
                    **(
                        {"error_code": "tool_execution_failed"}
                        if event.observation.is_error
                        else {}
                    ),
                },
            )
        ]
        if event.tool_name == "update_plan":
            plan = getattr(event.observation, "plan", None)
            if isinstance(plan, list):
                output.append(
                    _event(
                        state,
                        event,
                        "plan.updated",
                        {
                            "steps": [
                                step.model_dump(mode="json")
                                if isinstance(step, BaseModel)
                                else _json_value(step)
                                for step in plan
                            ],
                            "explanation": getattr(
                                event.observation, "explanation", None
                            ),
                        },
                        event_id=f"{event.id}:plan",
                    )
                )
        submitted = _external_task_submission(state, event)
        if submitted is not None:
            output.append(submitted)
        return tuple(output)
    if isinstance(event, UserRejectObservation):
        state.pending_actions.pop(event.tool_call_id, None)
        return (
            _event(
                state,
                event,
                "operation.failed",
                {
                    "operation_id": event.tool_call_id,
                    "name": event.tool_name,
                    "output": [{"type": "text", "text": event.rejection_reason}],
                    "error_code": "permission_denied",
                },
            ),
        )
    if isinstance(event, AgentErrorEvent):
        state.pending_actions.pop(event.tool_call_id, None)
        return (
            _event(
                state,
                event,
                "operation.failed",
                {
                    "operation_id": event.tool_call_id,
                    "name": event.tool_name,
                    "output": [{"type": "text", "text": event.error}],
                    "error_code": "agent_tool_error",
                },
            ),
        )
    if isinstance(event, ConversationStateUpdateEvent):
        return _translate_state_update(state, event)
    if isinstance(event, Condensation):
        return (
            _event(
                state,
                event,
                "context_compaction.updated",
                {
                    "status": "completed",
                    "compaction_id": event.id,
                    "summary": event.summary,
                },
            ),
        )
    if isinstance(event, (ConversationErrorEvent, ServerErrorEvent)):
        return (
            _event(
                state,
                event,
                "notice.raised",
                {
                    "severity": "error",
                    "code": event.code,
                    "message": event.detail,
                },
            ),
            _event(
                state,
                event,
                "status.changed",
                {"status": "error"},
                event_id=f"{event.id}:status",
            ),
        )
    if isinstance(event, (PauseEvent, InterruptEvent)):
        return (
            _event(
                state,
                event,
                "status.changed",
                {"status": "paused"},
            ),
        )
    return ()


def _translate_message(
    state: TranslationState,
    event: MessageEvent,
) -> tuple[HarnessEvent, ...]:
    if event.source not in {"user", "agent"}:
        return ()
    role = "user" if event.source == "user" else "assistant"
    message_id = event.id
    if role == "assistant" and state.streaming_message_id is not None:
        message_id = state.streaming_message_id
        state.streaming_message_id = None
        return (
            _event(
                state,
                event,
                "message.completed",
                {
                    "message_id": message_id,
                    "role": role,
                    "content": _content(event.llm_message.content),
                },
            ),
        )
    content = _content(event.llm_message.content)
    return (
        _event(
            state,
            event,
            "message.started",
            {"message_id": message_id, "role": role, "content": content},
        ),
        _event(
            state,
            event,
            "message.completed",
            {"message_id": message_id, "role": role, "content": content},
            event_id=f"{event.id}:completed",
        ),
    )


def _translate_delta(
    state: TranslationState,
    event: StreamingDeltaEvent,
) -> tuple[HarnessEvent, ...]:
    if event.content is None or event.content == "":
        return ()
    run_id = state.ensure_run_id()
    message_id = state.streaming_message_id
    output: list[HarnessEvent] = []
    if message_id is None:
        message_id = f"{run_id}:assistant"
        state.streaming_message_id = message_id
        output.append(
            _event(
                state,
                event,
                "message.started",
                {"message_id": message_id, "role": "assistant", "content": []},
                event_id=f"{event.id}:started",
            )
        )
    output.append(
        _event(
            state,
            event,
            "message.delta",
            {"message_id": message_id, "text": event.content},
        )
    )
    return tuple(output)


def _translate_state_update(
    state: TranslationState,
    event: ConversationStateUpdateEvent,
) -> tuple[HarnessEvent, ...]:
    output: list[HarnessEvent] = []
    status_value: object | None = None
    stats_value: object | None = None
    external_tasks_value: object | None = None
    if event.key == "full_state" and isinstance(event.value, dict):
        status_value = event.value.get("execution_status")
        stats_value = event.value.get("stats")
        external_tasks_value = event.value.get("active_long_tasks")
    elif event.key == "execution_status":
        status_value = event.value
    elif event.key == "stats":
        stats_value = event.value
    elif event.key == "active_long_tasks":
        external_tasks_value = event.value
    elif event.key == "context_condensation" and isinstance(event.value, dict):
        output.append(
            _event(
                state,
                event,
                "context_compaction.updated",
                _json_object(event.value),
                event_id=f"{event.id}:compaction",
            )
        )
    elif event.key == PYROMIND_WORKFLOW_EVENT_KEY:
        if state.suppress_workflow_events:
            state.suppress_workflow_events -= 1
            return ()
        workflow = _workflow_payload(event.value, event.id)
        if workflow is not None:
            output.append(
                _event(
                    state,
                    event,
                    "workflow.updated",
                    workflow,
                    event_id=f"{event.id}:workflow",
                )
            )

    if status_value is not None:
        status = str(status_value)
        if status != state.last_status:
            state.last_status = status
            output.append(
                _event(
                    state,
                    event,
                    "status.changed",
                    {"status": status},
                    event_id=f"{event.id}:status",
                )
            )
    usage = _usage_payload(stats_value)
    if usage is not None and usage != state.last_usage:
        state.last_usage = usage
        output.append(
            _event(
                state,
                event,
                "usage.updated",
                usage,
                event_id=f"{event.id}:usage",
            )
        )
    output.extend(_external_task_updates(state, event, external_tasks_value))
    if str(status_value) == "waiting_for_confirmation":
        permission = _permission_event(state, event)
        if permission is not None:
            output.append(permission)
    return tuple(output)


def _external_task_submission(
    state: TranslationState,
    event: ObservationEvent,
) -> HarnessEvent | None:
    kinds = {
        "df_submit_pipeline": "data_preparation",
        "run_dataset_cleaning": "data_cleaning",
        "workflow_debug": "workflow_debug",
    }
    kind = kinds.get(event.tool_name)
    if kind is None or event.observation.is_error:
        return None
    details = event.observation.model_dump(mode="json")
    task_id = details.get("task_id")
    run_id = details.get("run_id") or task_id
    output_dir = details.get("output_dir")
    identifiers = (task_id, run_id)
    if not all(isinstance(value, str) and value for value in identifiers):
        return None
    assert isinstance(task_id, str)
    assert isinstance(run_id, str)
    if kind != "workflow_debug" and not (isinstance(output_dir, str) and output_dir):
        return None
    status = _external_task_status(details.get("status"))
    payload: JsonObject = {
        "task_id": task_id,
        "kind": kind,
        "run_id": run_id,
        "status": status,
        "output_dir": output_dir if isinstance(output_dir, str) else None,
        "attempt": details.get("attempt"),
        "max_attempts": details.get("max_attempts"),
        "keep_ui_lock": bool(details.get("keep_ui_lock", False)),
        "submitted_at": event.timestamp,
        "updated_at": event.timestamp,
        "resume_pending": False,
    }
    state.external_tasks[task_id] = payload
    return _event(
        state,
        event,
        "external_task.submitted",
        payload,
        event_id=f"{event.id}:external-task:{task_id}:submitted",
    )


def _external_task_updates(
    state: TranslationState,
    event: ConversationStateUpdateEvent,
    value: object,
) -> tuple[HarnessEvent, ...]:
    if not isinstance(value, list):
        return ()
    output: list[HarnessEvent] = []
    for item in value:
        if isinstance(item, BaseModel):
            item = item.model_dump(mode="json")
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id")
        if not isinstance(task_id, str):
            continue
        previous = state.external_tasks.get(task_id)
        if previous is None:
            continue
        status = _external_task_status(item.get("status"))
        if previous.get("status") == status:
            continue
        updated: JsonObject = {
            **previous,
            "status": status,
            "updated_at": event.timestamp,
            "resume_pending": False,
        }
        state.external_tasks[task_id] = updated
        terminal = status in {"succeeded", "failed", "terminated", "stopped"}
        output.append(
            _event(
                state,
                event,
                "external_task.completed" if terminal else "external_task.updated",
                updated,
                event_id=f"{event.id}:external-task:{task_id}:{status}",
            )
        )
    return tuple(output)


def _external_task_status(value: object) -> str:
    normalized = str(value or "pending").strip().lower()
    return {
        "success": "succeeded",
        "succeeded": "succeeded",
        "error": "failed",
        "failed": "failed",
        "terminated": "terminated",
        "stopped": "stopped",
        "running": "running",
        "pending": "pending",
    }.get(normalized, "pending")


def _permission_event(
    state: TranslationState,
    event: Event,
) -> HarnessEvent | None:
    if state.pending_permission_id is not None:
        return None
    permission_id = f"{state.ensure_run_id()}:confirmation"
    state.pending_permission_id = permission_id
    actions = tuple(state.pending_actions.values())
    description = "Allow the pending OpenHands operation?"
    if actions:
        description = "Allow pending operations: " + ", ".join(
            action.summary or action.tool_name for action in actions
        )
    return _event(
        state,
        event,
        "permission.requested",
        {
            "permission_id": permission_id,
            "operation_ids": [action.tool_call_id for action in actions],
            "description": description,
            "choices": ["allow_once", "deny"],
        },
        event_id=f"{event.id}:permission",
    )


def _workflow_payload(value: object, version: str) -> JsonObject | None:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if not isinstance(value, dict):
        return None
    workflow = value.get("workflow")
    if not isinstance(workflow, str):
        return None
    canvas = value.get("xyflow")
    return {
        "resource_id": "pyromind_workflow",
        "version": version,
        "dsl": workflow,
        "canvas": _json_value(canvas) if isinstance(canvas, dict) else None,
    }


def _event(
    state: TranslationState,
    source: Event,
    event_type: str,
    payload: JsonObject,
    *,
    event_id: str | None = None,
) -> HarnessEvent:
    data: dict[str, object] = {
        "event_id": event_id or source.id,
        "source_event_id": source.id,
        "session_id": state.session_id,
        "type": event_type,
        "run_id": state.run_id,
        "payload": payload,
    }
    try:
        data["occurred_at"] = datetime.fromisoformat(source.timestamp)
    except ValueError:
        pass
    return HarnessEvent.model_validate(data)


def _tool_arguments(event: ActionEvent) -> JsonValue:
    raw = event.tool_call.arguments
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {"raw": raw}
    return _json_value(sanitize_dict(raw))


def _content(content: object) -> list[JsonValue]:
    output: list[JsonValue] = []
    if not isinstance(content, (list, tuple)):
        return output
    for block in content:
        if isinstance(block, TextContent):
            output.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            image: JsonObject = {
                "type": "image",
                "image_urls": list(block.image_urls),
            }
            output.append(image)
    return output


def _usage_payload(value: object) -> JsonObject | None:
    if not isinstance(value, dict):
        return None
    metrics = value.get("usage_to_metrics")
    if not isinstance(metrics, dict):
        return None
    input_tokens = output_tokens = cached_tokens = 0
    cost_usd = 0.0
    has_cost = False
    for metric in metrics.values():
        if not isinstance(metric, dict):
            continue
        cost = metric.get("accumulated_cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            cost_usd += float(cost)
            has_cost = True
        token_usage = metric.get("accumulated_token_usage")
        if not isinstance(token_usage, dict):
            continue
        input_tokens += _non_negative_int(token_usage.get("prompt_tokens"))
        output_tokens += _non_negative_int(token_usage.get("completion_tokens"))
        cached_tokens += _non_negative_int(token_usage.get("cache_read_tokens"))
    payload: JsonObject = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
    }
    if has_cost:
        payload["cost_usd"] = cost_usd
    return payload


def _non_negative_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _json_value(value: object) -> JsonValue:
    return json.loads(json.dumps(value, default=str))


def _json_object(value: object) -> JsonObject:
    normalized = _json_value(value)
    return normalized if isinstance(normalized, dict) else {"value": normalized}
