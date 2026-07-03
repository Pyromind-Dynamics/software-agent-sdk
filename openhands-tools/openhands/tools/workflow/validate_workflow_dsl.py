from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Self, cast

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


DEFAULT_VALIDATE_URL = "https://pre-api-portal.pyromind.ai/std2/studio_api/api/agent/workflows/dsl/validate"


class WorkflowValidationIssue(BaseModel):
    code: str | None = Field(default=None, description="Validation issue code.")
    level: str | None = Field(default=None, description="Issue severity.")
    workflow_id: str | None = Field(default=None, description="Workflow id.")
    node_id: str | None = Field(default=None, description="Workflow node id.")
    node_type: str | None = Field(default=None, description="Workflow node type.")
    edge_id: str | None = Field(default=None, description="Workflow edge id.")
    field: str | None = Field(default=None, description="Location field.")
    message: str | None = Field(default=None, description="Issue message.")
    source: str | None = Field(default=None, description="Issue source.")
    detail: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific issue detail.",
    )


class ValidateWorkflowDslAction(Action):
    dsl: str = Field(description="Pyromind workflow DSL source to validate.")
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
        description="Whether the validation API request was processed.",
    )
    valid: bool | None = Field(
        default=None,
        description="Whether the workflow DSL passed business validation.",
    )
    workflow_id: str | None = Field(default=None, description="Workflow id.")
    errors: list[WorkflowValidationIssue] = Field(default_factory=list)
    warnings: list[WorkflowValidationIssue] = Field(default_factory=list)
    message: str | None = Field(default=None, description="API-level message.")
    error_code: str | None = Field(default=None, description="API-level error code.")
    raw_response: dict[str, Any] | None = Field(
        default=None,
        description="Raw JSON response from the validation API.",
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
WORKFLOW_SCHEMA_INVALID, NODE_TYPE_UNKNOWN, NODE_NOT_FOUND,
EDGE_SOURCE_NOT_FOUND, EDGE_TARGET_NOT_FOUND, EDGE_HANDLE_INVALID,
TYPE_INCOMPATIBLE, MISSING_REQUIRED_INPUT, EMPTY_REQUIRED_INPUT,
ENUM_VALUE_INVALID, PARAMETER_OUT_OF_RANGE, DAG_CYCLE_DETECTED,
PRIMITIVE_NODE_INVALID, and NODE_DEFINITION_UNRESOLVED.

Note: a successful tool call can still return valid=false. In that case, use
the returned node_id, node_type, edge_id, field, and detail line information to
fix the workflow DSL.
"""


class ValidateWorkflowDslExecutor(ToolExecutor):
    def __init__(
        self,
        endpoint_url: str = DEFAULT_VALIDATE_URL,
        headers: Mapping[str, str] | None = None,
        secret_headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._endpoint_url = endpoint_url
        self._headers = dict(headers or {})
        self._secret_headers = dict(secret_headers or {})
        self._timeout = timeout

    def __call__(
        self,
        action: ValidateWorkflowDslAction,
        conversation: BaseConversation | None = None,
    ) -> ValidateWorkflowDslObservation:
        headers = {
            "accept": "*/*",
            "content-type": "application/json",
            **self._headers,
        }
        try:
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
                json={"name": action.name, "dsl": action.dsl},
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            return ValidateWorkflowDslObservation.from_text(
                text=(
                    "Failed to call workflow DSL validation API: "
                    f"{type(exc).__name__}: {exc}"
                ),
                is_error=True,
            )

        if response.status_code >= 400:
            return ValidateWorkflowDslObservation.from_text(
                text=(
                    "Workflow DSL validation API returned HTTP "
                    f"{response.status_code}: {_truncate_response_text(response.text)}"
                ),
                is_error=True,
            )

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            return ValidateWorkflowDslObservation.from_text(
                text=(f"Workflow DSL validation API returned invalid JSON: {exc.msg}"),
                is_error=True,
            )

        if not isinstance(payload, dict):
            return ValidateWorkflowDslObservation.from_text(
                text="Workflow DSL validation API returned a non-object JSON payload.",
                is_error=True,
            )

        return _observation_from_payload(payload)

    def _resolve_secret_headers(
        self,
        conversation: BaseConversation | None,
    ) -> dict[str, str]:
        if not self._secret_headers:
            return {}
        if conversation is None:
            raise ValueError(
                "Cannot resolve validation API header secrets without an active "
                "conversation."
            )

        resolved: dict[str, str] = {}
        state = cast("ConversationState", conversation.state)
        secret_registry = state.secret_registry
        for header_name, secret_name in self._secret_headers.items():
            value = secret_registry.get_secret_value(secret_name)
            if not value:
                raise ValueError(
                    f"Secret '{secret_name}' required for validation API header "
                    f"'{header_name}' was not found."
                )
            resolved[header_name] = value
        return resolved


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
        endpoint_url = str(params.pop("endpoint_url", DEFAULT_VALIDATE_URL))
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
        )
    if not isinstance(data, dict):
        return ValidateWorkflowDslObservation.from_text(
            text="Workflow DSL validation API response is missing object field 'data'.",
            is_error=True,
            success=True,
            raw_response=payload,
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
    )


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
            f"edge_id={issue.edge_id}" if issue.edge_id else "",
            f"field={issue.field}" if issue.field else "",
        )
        if value
    )
    line = issue.detail.get("source_node_line") or issue.detail.get("target_node_line")
    if line:
        location = f"{location}, line={line}" if location else f"line={line}"
    code = issue.code or "UNKNOWN"
    message = issue.message or "no message"
    if location:
        return f"{code}: {message} ({location})"
    return f"{code}: {message}"


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
