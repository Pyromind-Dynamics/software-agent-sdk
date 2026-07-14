from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import httpx
from pydantic import BaseModel, Field
from rich.text import Text

from openhands.sdk.tool.registry import register_tool
from openhands.sdk.tool.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


PRE_VALIDATE_URL = (
    "https://pre-api-portal.pyromind.ai/std2/studio_api/api/agent/workflows/"
    "dsl/validate"
)
PROD_VALIDATE_URL = (
    "https://api-portal.pyromind.ai/std2/studio_api/api/agent/workflows/dsl/validate"
)
PYROMIND_VALIDATE_AUTH_COOKIE_SECRET = "PYROMIND_VALIDATE_AUTH_COOKIE"
PYROMIND_VALIDATE_HEADERS_STATE_KEY = "pyromind_validate_workflow_dsl_headers"
_PROD_APP_ENVS = {"prod", "production", "online"}


def _default_validate_url() -> str:
    app_env = os.getenv("APP_ENV", "dev").strip().lower()
    if app_env in _PROD_APP_ENVS:
        return PROD_VALIDATE_URL
    return PRE_VALIDATE_URL


DEFAULT_VALIDATE_URL = _default_validate_url()


_ISSUE_DETAIL_DESCRIPTION = (
    "Additional backend metadata for locating or debugging the issue. "
    "DSL parse details can include line, column, and raw_error. "
    "Workflow location details can include location, raw_message, source, "
    "and target. DSL-to-xyflow source mapping can include target_node_line "
    "and source_node_line as 1-based line numbers in the original DSL. "
    "node_code is the DSL statement for issue.node_id. target_node_code is "
    "the DSL statement for the affected target node, usually the same as "
    "node_code for node-level errors. source_node_code is the DSL statement "
    "for the source node in edge or type validation errors. "
    "Edge and type validation details can include source_node_id, "
    "target_node_id, source_handle, target_handle, available_outputs, "
    "available_inputs, source_types, and target_types."
)


class WorkflowValidationIssue(BaseModel):
    code: str | None = Field(
        default=None,
        description=(
            "Machine-readable validation issue code, such as DSL_PARSE_FAILED, "
            "NODE_NOT_FOUND, EDGE_HANDLE_INVALID, or TYPE_INCOMPATIBLE."
        ),
    )
    level: str | None = Field(
        default=None,
        description=(
            "Issue severity from the validator. Backend values are error or warning."
        ),
    )
    workflow_id: str | None = Field(
        default=None,
        description="Workflow id associated with the validation issue.",
    )
    node_id: str | None = Field(
        default=None,
        description="Workflow node id associated with the issue, when available.",
    )
    node_type: str | None = Field(
        default=None,
        description="Workflow node type associated with the issue, when available.",
    )
    node_name: str | None = Field(
        default=None,
        description="Workflow node display name associated with the issue.",
    )
    edge_id: str | None = Field(
        default=None,
        description="Workflow edge id associated with the issue, when available.",
    )
    field: str | None = Field(
        default=None,
        description=(
            "Affected field or location path, such as nodeType, source, target, "
            "nodes[1], or a parameter name."
        ),
    )
    message: str | None = Field(
        default=None,
        description="Human-readable validation issue message from the backend.",
    )
    source: str | None = Field(
        default=None,
        description=(
            "Validation layer that produced the issue. Backend values are dsl, "
            "xyflow, or k8s."
        ),
    )
    detail: dict[str, Any] = Field(
        default_factory=dict,
        description=_ISSUE_DETAIL_DESCRIPTION,
    )


class ValidateWorkflowDslAction(Action):
    dsl: str | None = Field(
        default=None,
        description=(
            "Pyromind workflow Python DSL source code to validate (not a file path). "
            "Pass the declarative workflow script text the agent generated or "
            "edited. If omitted, the tool reads the saved `workflow.py` contents "
            "from the active conversation workspace."
        ),
    )
    name: str = Field(default="workflow", description="Workflow name.")

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Validate workflow DSL: ", style="bold blue")
        content.append(self.name)
        return content


class ValidateWorkflowDslObservation(Observation):
    success: bool | None = Field(
        default=None,
        description=(
            "Top-level BizResponse.success from the validation API. True means the "
            "API request was handled; check valid for workflow correctness."
        ),
    )
    valid: bool | None = Field(
        default=None,
        description=(
            "data.valid from WorkflowValidationResult. True means the DSL passed "
            "SDK and platform workflow validation."
        ),
    )
    workflow_id: str | None = Field(
        default=None,
        description="data.workflow_id from WorkflowValidationResult.",
    )
    errors: list[WorkflowValidationIssue] = Field(
        default_factory=list,
        description=(
            "Blocking validation issues from data.errors. Non-empty errors mean "
            "valid is false."
        ),
    )
    warnings: list[WorkflowValidationIssue] = Field(
        default_factory=list,
        description=(
            "Non-blocking validation issues from data.warnings. Warnings can be "
            "present even when valid is true."
        ),
    )
    message: str | None = Field(
        default=None,
        description="Top-level BizResponse.message from the validation API.",
    )
    error_code: str | None = Field(
        default=None,
        description="Top-level BizResponse.error_code from the validation API.",
    )
    raw_response: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Full raw BizResponse JSON returned by the validation API, including "
            "success, data, message, and error_code."
        ),
    )
    retryable: bool = Field(
        default=False,
        description=(
            "Whether retrying the same validation request may recover from a "
            "transient transport failure. True for request errors and 408/429/5xx."
        ),
    )
    failure_stage: (
        Literal["transport", "dsl_parse", "sdk_schema", "platform_schema"] | None
    ) = Field(
        default=None,
        description=(
            "Stage that failed: transport, DSL parsing, SDK workflow schema, or "
            "platform workflow schema. None when validation passes or no stage "
            "can be determined."
        ),
    )

    @property
    def visualize(self) -> Text:
        content = Text()
        if self.is_error:
            content.append("Workflow DSL validation failed", style="bold red")
        elif self.valid is True:
            content.append("Workflow DSL is valid", style="bold green")
        elif self.valid is False:
            content.append("Workflow DSL is invalid", style="bold yellow")
        else:
            content.append("Workflow DSL validation result", style="bold blue")
        if self.workflow_id:
            content.append(f" ({self.workflow_id})")
        if self.errors:
            content.append(f"\nErrors: {len(self.errors)}")
        if self.warnings:
            content.append(f"\nWarnings: {len(self.warnings)}")
        return content


TOOL_DESCRIPTION = """Validate Pyromind workflow DSL through the platform API.

Use this tool when you need to check whether a workflow DSL is valid or inspect
structured validation errors. The API validates DSL syntax first, converts DSL
to xyflow, runs SDK workflow schema validation, then runs the platform
XyflowParser/XyflowValidator.

The tool returns platform issue codes such as DSL_PARSE_FAILED,
SDK_NOT_AVAILABLE, WORKFLOW_SCHEMA_INVALID, NODE_TYPE_UNKNOWN, NODE_NOT_FOUND,
EDGE_SOURCE_NOT_FOUND, EDGE_TARGET_NOT_FOUND, EDGE_HANDLE_INVALID,
TYPE_INCOMPATIBLE, MISSING_REQUIRED_INPUT, EMPTY_REQUIRED_INPUT,
ENUM_VALUE_INVALID, PARAMETER_OUT_OF_RANGE, DAG_CYCLE_DETECTED,
PRIMITIVE_NODE_INVALID, and NODE_DEFINITION_UNRESOLVED.

The API response is a BizResponse wrapper: success, data, message, and
error_code. On success, data is a WorkflowValidationResult with valid,
workflow_id, errors, and warnings. Each issue includes code, level, workflow_id,
node_id, node_type, node_name, edge_id, field, message, source, and detail.
The issue detail object preserves backend location metadata such as location,
target_node_line, node_code, target_node_code, source_node_id, target_node_id,
available_inputs, available_outputs, source_types, and target_types.
For DSL source mapping, target_node_line/source_node_line are 1-based line
numbers in the original DSL; node_code is the DSL statement for node_id;
target_node_code is the DSL statement for the affected target node; and
source_node_code is the DSL statement for the source node in edge/type errors.
When fixing invalid DSL, prefer detail.node_code, detail.target_node_code, and
detail.source_node_code over xyflow-only fields because they point back to the
original DSL statements.

The observation also returns retryable and failure_stage. Retry only when
retryable=true; deterministic DSL/schema errors must be fixed instead of retried.

Note: a successful tool call can still return valid=false. In that case, use
the returned node_id, node_type, edge_id, field, and detail line information to
fix the workflow DSL.
"""


class ValidateWorkflowDslExecutor(ToolExecutor):
    def __init__(
        self,
        endpoint_url: str | None = None,
        headers: Mapping[str, str] | None = None,
        secret_headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._endpoint_url = endpoint_url or _default_validate_url()
        self._headers = dict(headers or {})
        self._secret_headers = dict(secret_headers or {})
        self._timeout = timeout

    def __call__(
        self,
        action: ValidateWorkflowDslAction,
        conversation: BaseConversation | None = None,
    ) -> ValidateWorkflowDslObservation:
        try:
            dsl = self._resolve_dsl(action, conversation)
        except ValueError as exc:
            return ValidateWorkflowDslObservation.from_text(
                text=str(exc),
                is_error=True,
            )

        headers = {
            "accept": "*/*",
            "content-type": "application/json",
            **self._headers,
        }
        try:
            headers.update(self._resolve_conversation_headers(conversation))
            headers.update(self._resolve_secret_headers(conversation))
        except ValueError as exc:
            return ValidateWorkflowDslObservation.from_text(
                text=str(exc),
                is_error=True,
            )
        try:
            response = httpx.post(
                self._endpoint_url,
                headers=headers,
                json={"name": action.name, "dsl": dsl},
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            return ValidateWorkflowDslObservation.from_text(
                text=(
                    "Failed to call workflow DSL validation API: "
                    f"{type(exc).__name__}: {exc}"
                ),
                is_error=True,
                retryable=True,
                failure_stage="transport",
            )

        if response.status_code >= 400:
            return ValidateWorkflowDslObservation.from_text(
                text=(
                    "Workflow DSL validation API returned HTTP "
                    f"{response.status_code}: {_truncate_response_text(response.text)}"
                ),
                is_error=True,
                retryable=_is_retryable_http_status(response.status_code),
                failure_stage="transport",
            )

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            return ValidateWorkflowDslObservation.from_text(
                text=(f"Workflow DSL validation API returned invalid JSON: {exc.msg}"),
                is_error=True,
                failure_stage="transport",
            )

        if not isinstance(payload, dict):
            return ValidateWorkflowDslObservation.from_text(
                text="Workflow DSL validation API returned a non-object JSON payload.",
                is_error=True,
                failure_stage="transport",
            )

        return _observation_from_payload(payload)

    def _resolve_dsl(
        self,
        action: ValidateWorkflowDslAction,
        conversation: BaseConversation | None,
    ) -> str:
        if action.dsl is not None:
            return action.dsl
        if conversation is None:
            raise ValueError(
                "Cannot read workflow.py without an active conversation. "
                "Pass `dsl` explicitly or call this tool from a conversation."
            )

        workspace = cast(Any, conversation).workspace
        workflow_path = Path(workspace.working_dir) / "workflow.py"
        if not workflow_path.is_file():
            raise ValueError(
                f"Cannot validate workflow DSL: {workflow_path} does not exist."
            )
        return workflow_path.read_text(encoding="utf-8")

    def _resolve_secret_headers(
        self,
        conversation: BaseConversation | None,
    ) -> dict[str, str]:
        secret_headers = dict(self._secret_headers)
        if conversation is not None:
            state = cast("ConversationState", conversation.state)
            secret_registry = state.secret_registry
            if secret_registry.get_secret_value(PYROMIND_VALIDATE_AUTH_COOKIE_SECRET):
                secret_headers.setdefault(
                    "cookie", PYROMIND_VALIDATE_AUTH_COOKIE_SECRET
                )
        if not secret_headers:
            return {}
        if conversation is None:
            raise ValueError(
                "Cannot resolve validation API header secrets without an active "
                "conversation."
            )

        resolved: dict[str, str] = {}
        state = cast("ConversationState", conversation.state)
        secret_registry = state.secret_registry
        for header_name, secret_name in secret_headers.items():
            value = secret_registry.get_secret_value(secret_name)
            if not value:
                raise ValueError(
                    f"Secret '{secret_name}' required for validation API header "
                    f"'{header_name}' was not found."
                )
            resolved[header_name] = value
        return resolved

    def _resolve_conversation_headers(
        self,
        conversation: BaseConversation | None,
    ) -> dict[str, str]:
        if conversation is None:
            return {}

        state = cast("ConversationState", conversation.state)
        headers = state.agent_state.get(PYROMIND_VALIDATE_HEADERS_STATE_KEY)
        if not isinstance(headers, dict):
            return {}
        return {
            str(name): str(value)
            for name, value in headers.items()
            if value is not None
        }


class ValidateWorkflowDslTool(
    ToolDefinition[ValidateWorkflowDslAction, ValidateWorkflowDslObservation]
):
    """Tool for validating Pyromind workflow DSL through the platform API."""

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        endpoint_url = str(params.pop("endpoint_url", _default_validate_url()))
        headers = params.pop("headers", None)
        secret_headers = params.pop("secret_headers", None)
        timeout = float(params.pop("timeout", 30.0))
        if params:
            names = ", ".join(sorted(params))
            raise ValueError(f"ValidateWorkflowDslTool got unknown params: {names}")
        if not endpoint_url.strip():
            raise ValueError("endpoint_url must be a non-empty string")
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if headers is not None and not isinstance(headers, dict):
            raise ValueError("headers must be a dictionary when provided")
        if secret_headers is not None and not isinstance(secret_headers, dict):
            raise ValueError("secret_headers must be a dictionary when provided")

        normalized_headers = (
            {str(k): str(v) for k, v in headers.items()} if headers else None
        )
        normalized_secret_headers = (
            {str(k): str(v) for k, v in secret_headers.items()}
            if secret_headers
            else None
        )

        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=ValidateWorkflowDslAction,
                observation_type=ValidateWorkflowDslObservation,
                executor=ValidateWorkflowDslExecutor(
                    endpoint_url=endpoint_url,
                    headers=normalized_headers,
                    secret_headers=normalized_secret_headers,
                    timeout=timeout,
                ),
                annotations=ToolAnnotations(
                    title="validate_workflow_dsl",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
            )
        ]


def _observation_from_payload(
    payload: dict[str, Any],
) -> ValidateWorkflowDslObservation:
    success = payload.get("success")
    data = payload.get("data")
    if success is not True:
        return ValidateWorkflowDslObservation.from_text(
            text=_format_api_failure(payload),
            is_error=True,
            success=success if isinstance(success, bool) else None,
            message=_optional_str(payload.get("message")),
            error_code=_optional_str(payload.get("error_code")),
            raw_response=payload,
            failure_stage="transport",
        )
    if not isinstance(data, dict):
        return ValidateWorkflowDslObservation.from_text(
            text="Workflow DSL validation API response is missing object field 'data'.",
            is_error=True,
            success=True,
            raw_response=payload,
            failure_stage="transport",
        )

    valid = data.get("valid")
    workflow_id = _optional_str(data.get("workflow_id"))
    errors = _parse_issues(data.get("errors"))
    warnings = _parse_issues(data.get("warnings"))
    text = _format_validation_summary(valid, workflow_id, errors, warnings)

    return ValidateWorkflowDslObservation.from_text(
        text=text,
        success=True,
        valid=valid if isinstance(valid, bool) else None,
        workflow_id=workflow_id,
        errors=errors,
        warnings=warnings,
        message=_optional_str(payload.get("message")),
        error_code=_optional_str(payload.get("error_code")),
        raw_response=payload,
        failure_stage=(_infer_failure_stage(errors) if valid is not True else None),
    )


def _is_retryable_http_status(status_code: int) -> bool:
    return status_code in {408, 429} or status_code >= 500


def _infer_failure_stage(
    errors: list[WorkflowValidationIssue],
) -> Literal["dsl_parse", "sdk_schema", "platform_schema"]:
    if any(
        issue.code == "DSL_PARSE_FAILED" or issue.source == "dsl" for issue in errors
    ):
        return "dsl_parse"
    if any(
        issue.code in {"SDK_NOT_AVAILABLE", "WORKFLOW_SCHEMA_INVALID"}
        for issue in errors
    ):
        return "sdk_schema"
    return "platform_schema"


def _format_api_failure(payload: dict[str, Any]) -> str:
    message = _optional_str(payload.get("message")) or "unknown API failure"
    error_code = _optional_str(payload.get("error_code"))
    if error_code:
        return f"Workflow DSL validation API failed with {error_code}: {message}"
    return f"Workflow DSL validation API failed: {message}"


def _format_validation_summary(
    valid: Any,
    workflow_id: str | None,
    errors: list[WorkflowValidationIssue],
    warnings: list[WorkflowValidationIssue],
) -> str:
    status = "valid" if valid is True else "invalid"
    parts = [f"Workflow DSL is {status}."]
    if workflow_id:
        parts.append(f"workflow_id={workflow_id}.")
    if errors:
        parts.append(f"errors={len(errors)}.")
        parts.extend(_format_issue(issue) for issue in errors[:5])
    if warnings:
        parts.append(f"warnings={len(warnings)}.")
        parts.extend(f"warning: {_format_issue(issue)}" for issue in warnings[:3])
    return "\n".join(parts)


def _format_issue(issue: WorkflowValidationIssue) -> str:
    location = ", ".join(
        value
        for value in (
            f"node_id={issue.node_id}" if issue.node_id else "",
            f"node_type={issue.node_type}" if issue.node_type else "",
            f"node_name={issue.node_name}" if issue.node_name else "",
            f"edge_id={issue.edge_id}" if issue.edge_id else "",
            f"field={issue.field}" if issue.field else "",
        )
        if value
    )
    source_line = issue.detail.get("source_node_line")
    target_line = issue.detail.get("target_node_line")
    line_location = _format_line_location(source_line, target_line)
    if line_location:
        location = f"{location}, {line_location}" if location else line_location
    code = issue.code or "UNKNOWN"
    message = issue.message or "no message"
    if location:
        summary = f"{code}: {message} ({location})"
    else:
        summary = f"{code}: {message}"
    dsl_context = _format_dsl_context(issue.detail)
    if dsl_context:
        return f"{summary}\n{dsl_context}"
    return summary


def _format_line_location(source_line: Any, target_line: Any) -> str:
    if source_line and target_line and source_line != target_line:
        return f"source_line={source_line}, target_line={target_line}"
    line = target_line or source_line
    if line:
        return f"line={line}"
    return ""


def _format_dsl_context(detail: dict[str, Any]) -> str:
    entries = []
    seen_code = set()
    for key, label in (
        ("node_code", "dsl_code"),
        ("target_node_code", "target_dsl_code"),
        ("source_node_code", "source_dsl_code"),
    ):
        value = detail.get(key)
        if not value:
            continue
        code = str(value)
        if code in seen_code:
            continue
        seen_code.add(code)
        entries.append(f"{label}: {code}")
    return "\n".join(entries)


def _parse_issues(value: Any) -> list[WorkflowValidationIssue]:
    if not isinstance(value, list):
        return []
    issues: list[WorkflowValidationIssue] = []
    for item in value:
        if isinstance(item, dict):
            issues.append(WorkflowValidationIssue.model_validate(item))
    return issues


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _truncate_response_text(text: str, limit: int = 1000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text) - limit} characters truncated]"


register_tool(ValidateWorkflowDslTool.name, ValidateWorkflowDslTool)
