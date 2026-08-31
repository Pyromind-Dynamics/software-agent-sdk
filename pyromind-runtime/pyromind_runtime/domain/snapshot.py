from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.content import ContentBlock, JsonObject


type ConversationStatus = Literal[
    "idle",
    "running",
    "paused",
    "waiting_for_confirmation",
    "waiting_for_external_task",
    "finished",
    "error",
    "stuck",
    "deleting",
    "closed",
]


class TimelineMessage(ContractModel):
    kind: Literal["message"] = "message"
    item_id: str = Field(min_length=1)
    started_seq: int = Field(ge=1)
    completed_seq: int | None = Field(default=None, ge=1)
    role: Literal["user", "assistant", "system"]
    content: tuple[ContentBlock, ...] = ()
    status: Literal["streaming", "completed", "failed"] = "streaming"
    run_id: str | None = None


class TimelineOperation(ContractModel):
    kind: Literal["operation"] = "operation"
    item_id: str = Field(min_length=1)
    started_seq: int = Field(ge=1)
    completed_seq: int | None = Field(default=None, ge=1)
    name: str = Field(min_length=1)
    category: Literal["tool", "subagent", "observation"] = "tool"
    status: Literal["running", "completed", "failed"] = "running"
    thought: tuple[ContentBlock, ...] = ()
    arguments: JsonValue = Field(default_factory=dict)
    output: tuple[ContentBlock, ...] = ()
    details: JsonValue = None
    error_code: str | None = None
    run_id: str | None = None


class WorkflowState(ContractModel):
    resource_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    dsl: str
    canvas: JsonObject | None = None


class TimelineWorkflow(ContractModel):
    kind: Literal["workflow"] = "workflow"
    item_id: str = Field(min_length=1)
    started_seq: int = Field(ge=1)
    workflow: WorkflowState


class TimelineCompaction(ContractModel):
    kind: Literal["compaction"] = "compaction"
    item_id: str = Field(min_length=1)
    started_seq: int = Field(ge=1)
    status: Literal["started", "completed", "skipped", "failed"]
    summary: str | None = None


class TimelineNotice(ContractModel):
    kind: Literal["notice"] = "notice"
    item_id: str = Field(min_length=1)
    started_seq: int = Field(ge=1)
    severity: Literal["info", "warning", "error"]
    code: str = Field(min_length=1)
    message: str


type TimelineItem = Annotated[
    TimelineMessage
    | TimelineOperation
    | TimelineWorkflow
    | TimelineCompaction
    | TimelineNotice,
    Field(discriminator="kind"),
]


class PlanStep(ContractModel):
    step: str
    status: Literal["pending", "in_progress", "completed"]


class PlanState(ContractModel):
    explanation: str | None = None
    steps: tuple[PlanStep, ...]


class PendingPermission(ContractModel):
    permission_id: str = Field(min_length=1)
    operation_ids: tuple[str, ...] = ()
    description: str = ""
    choices: tuple[Literal["allow_once", "deny"], ...] = (
        "allow_once",
        "deny",
    )


class UsageState(ContractModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class ExternalTaskState(ContractModel):
    task_id: str = Field(min_length=1)
    kind: Literal["data_cleaning", "data_preparation", "workflow_debug"]
    run_id: str | None = None
    status: Literal[
        "pending", "running", "succeeded", "failed", "terminated", "stopped"
    ]
    output_dir: str | None = None
    attempt: int | None = Field(default=None, ge=0)
    max_attempts: int | None = Field(default=None, ge=1)
    keep_ui_lock: bool = False
    submitted_at: str
    updated_at: str
    resume_pending: bool = False
    error_summary: str | None = None


class ConversationSnapshot(ContractModel):
    schema_version: int = Field(default=1, ge=1)
    conversation_id: str = Field(min_length=1)
    through_seq: int = Field(default=0, ge=0)
    status: ConversationStatus = "idle"
    capabilities: HarnessCapabilities
    timeline: tuple[TimelineItem, ...] = ()
    current_workflow: WorkflowState | None = None
    plan: PlanState | None = None
    pending_permissions: tuple[PendingPermission, ...] = ()
    compaction_status: Literal["started", "completed", "skipped", "failed"] | None = (
        None
    )
    usage: UsageState = Field(default_factory=UsageState)
    external_tasks: tuple[ExternalTaskState, ...] = ()


class ConversationSummary(ContractModel):
    conversation_id: str = Field(min_length=1)
    status: ConversationStatus
    through_seq: int = Field(ge=0)
    title: str | None = None
