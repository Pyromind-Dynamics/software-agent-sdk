from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from pyromind_common.tools.utils.platform_env import (
    platform_headers,
    platform_portal_url,
)
from pyromind_common.tools.workflow.task_submission import (
    WorkflowResourceValidationError,
)
from pyromind_common.tools.workflow.validate_workflow_dsl import (
    TOOL_DESCRIPTION as STRUCTURE_TOOL_DESCRIPTION,
    ValidateWorkflowDslAction,
    ValidateWorkflowDslExecutor as StructureValidationExecutor,
    ValidateWorkflowDslObservation,
    WorkflowValidationIssue,
    _format_issue,
    _format_validation_summary,
    _is_retryable_http_status,
    _parse_issues,
)


if TYPE_CHECKING:
    from pyromind_common.sdk.conversation.base import ToolConversation


class WorkflowAvailableResource(BaseModel):
    gpu_type: str | None = Field(
        default=None,
        description=(
            "GPU 型号；null 表示仅申请 CPU 和内存的资源选项，max_gpu_count 为 0。"
            "该选项可来自有 GPU 的节点，CPU 和内存取自同一节点。"
        ),
    )
    max_gpu_count: int = Field(ge=0)
    max_cpu_cores: float = Field(ge=0, allow_inf_nan=False)
    max_mem_gib: float = Field(ge=0, allow_inf_nan=False)


class WorkflowResourceValidationObservation(ValidateWorkflowDslObservation):
    available_resources: list[WorkflowAvailableResource] = Field(default_factory=list)
    # 仅供提交入口复用，不将完整工作流重复暴露到模型的工具结果中。
    validated_workflow: dict[str, Any] | None = Field(default=None, exclude=True)
    # 观测对象不可变；仅为新工作流扩展此只读字段的类型。
    failure_stage: (  # pyright: ignore[reportIncompatibleVariableOverride]
        Literal["transport", "dsl_parse", "sdk_schema", "platform_schema", "resources"]
        | None
    ) = None


RESOURCE_TOOL_DESCRIPTION = (
    STRUCTURE_TOOL_DESCRIPTION
    + """

After structure validation passes, check current single-machine resources.
Resource shortage returns valid=false, is_error=true and available_resources.
Report GPU types and one selected machine's GPU/CPU/memory resources per type.
gpu_type=null with max_gpu_count=0 denotes a CPU/memory-only resource option.
That option may come from a GPU-equipped machine. Explain this meaning when
reporting null; it does not indicate an unknown GPU model or missing data.
Use max GPU count per machine, never sum across machines. Each returned entry
contains resources from the same machine, with CPU and memory truncated down
to whole cores and GiB. Other single-machine options are not listed. Do not
silently change GPU type or reduce resources and submit. Revalidate edited DSL.
A resource snapshot does not reserve capacity or guarantee parallel scheduling.
"""
)


class ResourceValidatedWorkflowDslExecutor(StructureValidationExecutor):
    def __init__(
        self,
        endpoint_url: str | None = None,
        headers: Mapping[str, str] | None = None,
        secret_headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        *,
        env: str | None = None,
        resource_endpoint_url: str | None = None,
    ) -> None:
        super().__init__(endpoint_url, headers, secret_headers, timeout, env=env)
        self._resource_endpoint_url = resource_endpoint_url or (
            self._endpoint_url.removesuffix("/dsl/validate") + "/resources/validate"
        )

    def __call__(
        self,
        action: ValidateWorkflowDslAction,
        conversation: ToolConversation | None = None,
    ) -> ValidateWorkflowDslObservation:
        try:
            dsl = self._resolve_dsl(action, conversation)
        except ValueError as exc:
            return WorkflowResourceValidationObservation.from_text(
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
            return WorkflowResourceValidationObservation.from_text(
                text=str(exc),
                is_error=True,
            )
        structure = super().__call__(
            ValidateWorkflowDslAction(
                dsl=dsl, name=action.name, test_mode=action.test_mode
            ),
            conversation,
        )
        if structure.is_error or structure.valid is not True:
            return structure
        payload = structure.raw_response or {}
        data = payload.get("data")
        if not isinstance(data, dict):
            return _resource_check_failed(retryable=False)
        workflow = data.get("workflow")
        if (
            not isinstance(workflow, dict)
            or not isinstance(workflow.get("nodes"), list)
            or not isinstance(workflow.get("edges"), list)
        ):
            # 不回退到本地解析或重复 DSL 校验，避免使用未经校验的另一份定义。
            return _resource_check_failed(retryable=False)
        resources = self.validate_resources(workflow, headers=headers)
        warnings = [*structure.warnings, *resources.warnings]
        summary = resources.text
        if structure.warnings:
            summary += "\n" + "\n".join(
                f"warning: {_format_issue(issue)}" for issue in structure.warnings
            )
        return WorkflowResourceValidationObservation.from_text(
            text=summary,
            **resources.model_dump(
                exclude={"content", "workflow_id", "warnings", "raw_response"}
            ),
            raw_response={
                **payload,
                "data": {
                    **data,
                    "workflow": workflow,
                    "valid": resources.valid,
                    "errors": [
                        issue.model_dump(mode="json") for issue in resources.errors
                    ],
                    "warnings": [issue.model_dump(mode="json") for issue in warnings],
                    "available_resources": [
                        item.model_dump(mode="json")
                        for item in resources.available_resources
                    ],
                },
            },
            workflow_id=structure.workflow_id,
            warnings=warnings,
            validated_workflow=workflow,
        )

    def validate_resources(
        self, workflow: dict[str, Any], *, headers: Mapping[str, str] | None = None
    ) -> WorkflowResourceValidationObservation:
        return self._request(
            self._resource_endpoint_url,
            workflow,
            dict(headers) if headers is not None else self._headers,
        )

    def _request(
        self,
        endpoint: str,
        workflow: dict[str, Any],
        headers: Mapping[str, str],
    ) -> WorkflowResourceValidationObservation:
        try:
            response = httpx.post(
                platform_portal_url(endpoint, headers),
                headers=platform_headers(headers),
                json={"workflow": workflow},
                timeout=self._timeout,
            )
        except httpx.RequestError:
            return _resource_check_failed()
        if response.status_code >= 400:
            return _resource_check_failed(
                retryable=_is_retryable_http_status(response.status_code)
            )
        try:
            payload = response.json()
        except json.JSONDecodeError:
            return _resource_check_failed(retryable=False)
        if not isinstance(payload, dict):
            return _resource_check_failed(retryable=False)
        return _resource_observation(payload)


def check_workflow_resources(
    workflow: dict[str, Any], *, headers: Mapping[str, str]
) -> None:
    """只检查即将提交的实际工作流，不重新转换或校验 DSL。"""
    result = ResourceValidatedWorkflowDslExecutor(headers=headers).validate_resources(
        workflow
    )
    if result.is_error or result.valid is not True:
        raise WorkflowResourceValidationError(
            result.text or "workflow resource validation failed"
        )


def _resource_check_failed(
    *, retryable: bool = True
) -> WorkflowResourceValidationObservation:
    issue = WorkflowValidationIssue(
        code="RESOURCE_CHECK_FAILED",
        source="k8s",
        message="资源检查暂不可用，请稍后重新校验",
    )
    return WorkflowResourceValidationObservation.from_text(
        text=issue.message or "Resource check failed",
        valid=False,
        is_error=True,
        errors=[issue],
        failure_stage="resources",
        retryable=retryable,
    )


def _resource_observation(
    payload: dict[str, Any],
) -> WorkflowResourceValidationObservation:
    data = payload.get("data")
    if payload.get("success") is not True or not isinstance(data, dict):
        return _resource_check_failed()
    if not isinstance(data.get("valid"), bool):
        return _resource_check_failed(retryable=False)
    for key in ("errors", "warnings"):
        issues = data.get(key, [])
        if not isinstance(issues, list) or any(not isinstance(i, dict) for i in issues):
            return _resource_check_failed(retryable=False)
    try:
        errors = _parse_issues(data.get("errors"))
        warnings = _parse_issues(data.get("warnings"))
        available = [
            WorkflowAvailableResource.model_validate(item)
            for item in data.get("available_resources", [])
        ]
    except (ValidationError, TypeError):
        return _resource_check_failed(retryable=False)
    valid = data["valid"] and not errors
    if not valid and not errors:
        return _resource_check_failed(retryable=False)
    text = _format_validation_summary(valid, None, errors, warnings)
    if available:
        text += (
            "\n可用单机资源（GPU 按 max，不按 sum；每项资源来自同一节点，"
            "CPU 核数、内存 GiB 向下截断为整数，仅列出每类选中的一个节点）：\n"
            + json.dumps([item.model_dump() for item in available], ensure_ascii=False)
        )
        if any(item.gpu_type is None for item in available):
            text += (
                "\ngpu_type=null、max_gpu_count=0 表示仅申请 CPU 和内存的资源选项；"
                "该选项可来自有 GPU 的节点，CPU 和内存来自同一节点。"
            )
    return WorkflowResourceValidationObservation.from_text(
        text=text,
        success=True,
        valid=valid,
        is_error=not valid,
        errors=errors,
        warnings=warnings,
        available_resources=available,
        failure_stage="resources" if not valid else None,
        retryable=any(error.code == "RESOURCE_CHECK_FAILED" for error in errors),
    )
