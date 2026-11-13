from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError
from pyromind_runtime.application.pipeline_runs import PipelineRuns
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.storage_paths import resolve_conversation_storage_path
from pyromind_runtime.domain.task_correlation import TaskOrigin
from pyromind_runtime.domain.versions import VersionToolRequest

from harness_adapter.openhands_adapter.session_factory import current_user_from_context
from harness_adapter.pi_adapter.business_tools import (
    execute_validation_tool,
    validation_tool_spec,
)
from harness_adapter.pi_adapter.persistence import PiSessionFiles, SavedToolOutput
from harness_adapter.pi_adapter.sandbox_workspace import SandboxWorkspace
from harness_adapter.pi_adapter.tool_output import (
    ToolOutputPolicy,
    business_tool_output_id,
    normalize_utf8_text,
    resolve_tool_output_policy,
    truncate_tool_content,
)
from harness_adapter.version_tools import (
    ExecutionCall,
    ProductVersionsAction,
    ProductVersionsTool,
    VersionCall,
    submission_bridge,
)
from openhands.agent_server.pyromind_auth import parse_auth_token_from_cookie_header
from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.sdk.secret import StaticSecret
from openhands.sdk.tool import ToolDefinition
from openhands.sdk.workspace.base import BaseWorkspace
from openhands.sdk.workspace.local import LocalWorkspace
from openhands.tools.data_preparation import (
    DfCheckProgressTool,
    DfStopTaskTool,
    DfSubmitEvaluationTool,
    DfSubmitPipelineTool,
)
from openhands.tools.data_preparation.inference import InferencePipelineHandler
from openhands.tools.data_preparation.sandbox_execution import (
    ENV_SANDBOX_DATAFLOW_PYTHON,
)
from openhands.tools.environment_processing import (
    EdpAggregateTool,
    EdpRenderTool,
    EdpSubmitTool,
)
from openhands.tools.label_studio import (
    LABEL_STUDIO_TOKEN_SECRET,
    LabelStudioProjectTool,
)
from openhands.tools.pyromind_cleaning import RunDatasetCleaningTool
from openhands.tools.pyromind_dataset import (
    GetStorageUrlTool,
    UploadFileToPyromindTool,
)
from openhands.tools.sandbox import (
    SandboxCreateTool,
    SandboxDeleteFileTool,
    SandboxDeleteTool,
    SandboxDownloadTool,
    SandboxReadFileTool,
    SandboxTerminalTool,
    SandboxUploadTool,
    SandboxWriteFileTool,
)
from openhands.tools.training_analysis import TrainingAnalysisTool
from openhands.tools.update_plan import UpdatePlanTool
from openhands.tools.utils.platform_env import (
    platform_cluster_route,
    platform_cluster_target,
)
from openhands.tools.workflow.inspect_workflow_task import InspectWorkflowTaskTool
from openhands.tools.workflow.run_workflow import WORKFLOW_ATTEMPT_STATE_KEY
from openhands.tools.workflow.task_submission import (
    PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET,
    workflow_submission_scope,
)
from openhands.tools.workflow.validate_workflow_dsl import (
    PYROMIND_VALIDATE_AUTH_COOKIE_SECRET,
    PYROMIND_VALIDATE_HEADERS_STATE_KEY,
    ValidateWorkflowDslExecutor,
)
from openhands.tools.workflow_debug import WorkflowDebugTool


_STORAGE_COOKIE_SECRET = "PYROMIND_STORAGE_AUTH_COOKIE"
_VALIDATE_AUTHORIZATION_SECRET = "PYROMIND_VALIDATE_AUTHORIZATION"
_READ_ONLY_TOOLS = frozenset(
    {
        GetStorageUrlTool.name,
        "df_check_progress",
        SandboxReadFileTool.name,
        InspectWorkflowTaskTool.name,
    }
)
logger = logging.getLogger(__name__)


class _LabelStudioArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _LabelStudioCreateArgs(_LabelStudioArgs):
    dataset_path: str = Field(min_length=1)
    label_config_path: str = Field(min_length=1)
    adapter: Literal["avi_train", "aoi_export", "jsonl"] = "avi_train"
    field_map_path: str | None = None
    idempotency_key: str | None = None


class _LabelStudioProjectRefArgs(_LabelStudioArgs):
    project_ref: str = Field(min_length=1)


class _LabelStudioUpdateConfigArgs(_LabelStudioProjectRefArgs):
    label_config_path: str = Field(min_length=1)
    expected_config_version: int | None = Field(default=None, ge=1)


class _LabelStudioExportArgs(_LabelStudioProjectRefArgs):
    output_path: str | None = None


_LABEL_STUDIO_OPERATIONS: dict[str, tuple[str, type[_LabelStudioArgs], str]] = {
    "label_studio_create_project": (
        "create",
        _LabelStudioCreateArgs,
        "Create a Label Studio project from a Storage dataset. Inspect the data "
        "and write label_config.xml first. dataset_path is the source data; "
        "label_config_path is the XML. Use adapter='jsonl' for JSON Lines. "
        "field_map_path optionally binds custom fields. Authentication is managed "
        "by the server.",
    ),
    "label_studio_get_project": (
        "get",
        _LabelStudioProjectRefArgs,
        "Get an existing Label Studio project by project_ref.",
    ),
    "label_studio_project_status": (
        "status",
        _LabelStudioProjectRefArgs,
        "Inspect import progress and task counts for a Label Studio project.",
    ),
    "label_studio_update_config": (
        "update_config",
        _LabelStudioUpdateConfigArgs,
        "Update a Label Studio project's XML. Get its current config_version "
        "first and pass it as expected_config_version.",
    ),
    "label_studio_export_annotations": (
        "export",
        _LabelStudioExportArgs,
        "Export annotations from an existing Label Studio project to Storage. "
        "output_path is an optional destination directory.",
    ),
    "label_studio_refresh_media": (
        "refresh_media",
        _LabelStudioProjectRefArgs,
        "Refresh media URLs for an existing Label Studio project.",
    ),
}

_TRUNCATED_SUFFIX = "\n\n[output truncated: exceeded the runner frame budget]"
# Observation details (e.g. directory entries) are only machine-readable
# context; cap them well below the frame budget so the full response frame
# stays deliverable even when a storage listing holds thousands of entries.
_MAX_DETAILS_BYTES = 200_000
_MAX_DETAIL_ITEMS = 50
_MAX_DETAIL_TEXT_CHARS = 20_000


def _encoded_len(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


def _cap_details_size(details: dict[str, Any]) -> dict[str, Any]:
    if _encoded_len(details) <= _MAX_DETAILS_BYTES:
        return details
    capped: dict[str, Any] = {}
    for key, value in details.items():
        if isinstance(value, list) and len(value) > _MAX_DETAIL_ITEMS:
            capped[key] = value[:_MAX_DETAIL_ITEMS]
            capped[f"{key}_truncated"] = {
                "shown": _MAX_DETAIL_ITEMS,
                "total": len(value),
            }
        elif isinstance(value, str) and len(value) > _MAX_DETAIL_TEXT_CHARS:
            capped[key] = value[:_MAX_DETAIL_TEXT_CHARS] + _TRUNCATED_SUFFIX
        else:
            capped[key] = value
    if _encoded_len(capped) <= _MAX_DETAILS_BYTES:
        return capped
    return {"truncated": True, "keys": sorted(details)}


@dataclass(frozen=True, slots=True)
class BusinessToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    conversation_id: str
    workspace_root: Path
    request_context: RequestContext
    model_configuration: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)
    workspace: BaseWorkspace | None = None


def _workflow_storage_path(context: ToolExecutionContext, workflow_path: str) -> str:
    """Resolve a model-visible workflow path to the Storage object to read."""
    return resolve_conversation_storage_path(
        context.conversation_id, workflow_path, field="workflow_path"
    )


@dataclass(frozen=True, slots=True)
class BusinessToolResult:
    is_error: bool
    content: list[dict[str, Any]]
    details: dict[str, Any]
    signals: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_error": self.is_error,
            "content": self.content,
            "details": self.details,
            "signals": self.signals,
        }


def _llm_credential_secrets(context: ToolExecutionContext) -> dict[str, StaticSecret]:
    """Expose the session's own LLM config under the canonical edp names.

    ``resolve_llm_env`` checks the session secret registry first and only then
    falls back to process env (``LLM_AUTH_TOKEN`` -> ``DF_API_KEY`` ->
    ``ANTHROPIC_AUTH_TOKEN``). Without a registry entry the ``DF_API_KEY``
    fallback can win with a key that belongs to a different gateway, so seed
    the registry from the model configuration this conversation already runs
    on (mirroring the pi runner's own ``_api_key`` fallbacks).
    """
    model = context.model_configuration
    api_key = model.get("api_key")
    resolved = {
        "LLM_BASE_URL": model.get("base_url") or os.getenv("LLM_BASE_URL"),
        "LLM_AUTH_TOKEN": (api_key if isinstance(api_key, str) and api_key else None)
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY"),
        "LLM_MODEL": model.get("model") or os.getenv("LLM_MODEL"),
    }
    return {
        name: StaticSecret(value=SecretStr(value))
        for name, value in resolved.items()
        if isinstance(value, str) and value.strip()
    }


class _ToolConversationFacade:
    """The deliberately small OpenHands compatibility boundary used by Pi."""

    def __init__(
        self,
        context: ToolExecutionContext,
        persisted_state: dict[str, Any] | None = None,
    ) -> None:
        self.id = context.conversation_id
        self.workspace_root = context.workspace_root
        self.workspace = context.workspace or LocalWorkspace(
            working_dir=context.workspace_root
        )
        registry = SecretRegistry()
        secrets: dict[str, StaticSecret] = {}
        cookie = context.request_context.cookie
        if cookie:
            secrets[_STORAGE_COOKIE_SECRET] = StaticSecret(value=SecretStr(cookie))
            secrets[PYROMIND_VALIDATE_AUTH_COOKIE_SECRET] = StaticSecret(
                value=SecretStr(cookie)
            )
        if context.request_context.authorization:
            secrets[_VALIDATE_AUTHORIZATION_SECRET] = StaticSecret(
                value=SecretStr(context.request_context.authorization)
            )
        auth_token = _auth_token(context.request_context)
        if auth_token:
            secrets[PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET] = StaticSecret(
                value=SecretStr(auth_token)
            )
        label_studio_token = context.extra.get("label_studio_token")
        if not isinstance(label_studio_token, str) or not label_studio_token.strip():
            label_studio_token = os.getenv("LABEL_STUDIO_API_TOKEN", "")
        if label_studio_token.strip():
            secrets[LABEL_STUDIO_TOKEN_SECRET] = StaticSecret(
                value=SecretStr(label_studio_token)
            )
        secrets.update(_llm_credential_secrets(context))
        registry.secret_sources.update(secrets)
        model = context.model_configuration
        api_key = model.get("api_key")
        self.state = SimpleNamespace(
            agent=SimpleNamespace(
                llm=SimpleNamespace(
                    model=str(model.get("model") or ""),
                    base_url=model.get("base_url"),
                    api_key=SecretStr(api_key) if isinstance(api_key, str) else None,
                )
            ),
            agent_state={
                **(persisted_state or {}),
                PYROMIND_VALIDATE_HEADERS_STATE_KEY: _forward_headers(
                    context.request_context, include_cookie=False
                ),
            },
            secret_registry=registry,
        )
        self.signals: list[dict[str, Any]] = []

    def register_active_long_task(self, task: Any) -> None:
        value = task.model_dump(mode="json") if hasattr(task, "model_dump") else {}
        self.signals.append({"type": "external_task.submitted", "task": value})

    def send_agent_message(self, message: Any) -> None:
        self.signals.append({"type": "agent.message", "content": str(message)})


ToolFactory = Callable[[ToolExecutionContext], ToolDefinition[Any, Any]]


class PyromindBusinessToolHost:
    """Expose tested OpenHands business executors through Pi's JSONL bridge."""

    def __init__(
        self,
        skill_roots: Sequence[Path],
        *,
        skills_directory: Path | None = None,
        pipeline_runs: Callable[[str], PipelineRuns] | None = None,
    ) -> None:
        roots = (
            {
                name: skills_directory / name
                for name in ("data-processing", "training-analysis")
            }
            if skills_directory is not None
            else {}
        )
        roots.update({path.name: path.resolve() for path in skill_roots})
        missing = {"data-processing", "training-analysis"} - roots.keys()
        if missing:
            raise ValueError(
                f"Pi business tool resources not configured: {sorted(missing)}"
            )
        self._cleaning_runtime = roots["data-processing"] / "scripts" / "cleaning"
        self._preparation_runtime = roots["data-processing"] / "scripts" / "preparation"
        self._edp_runtime = roots["data-processing"] / "scripts" / "edp"
        self._training_runtime = roots["training-analysis"] / "scripts"
        self._inference_runtime = (
            skills_directory / "inference-evaluation" / "scripts"
            if skills_directory is not None
            else None
        )
        self._pipeline_runs = pipeline_runs
        self.version_call: VersionCall | None = None
        self.training_artifact_hook: Callable[..., Awaitable[int]] | None = None
        self.execution_call: ExecutionCall | None = None
        self._workflow_preparer: Callable[..., Awaitable[dict[str, Any]]] | None = None
        for path in (
            self._cleaning_runtime,
            self._preparation_runtime,
            self._edp_runtime,
            self._training_runtime,
        ):
            if not path.is_dir():
                raise ValueError(f"Pi business tool resource directory missing: {path}")
        self._locks: dict[str, asyncio.Lock] = {}
        self._active_executors: dict[tuple[str, str], Any] = {}
        self._factories: dict[str, ToolFactory] = {
            UploadFileToPyromindTool.name: lambda context: (
                UploadFileToPyromindTool.create(**self._storage_params(context))[0]
            ),
            GetStorageUrlTool.name: lambda context: GetStorageUrlTool.create(
                **self._storage_params(context)
            )[0],
            RunDatasetCleaningTool.name: lambda context: RunDatasetCleaningTool.create(
                **self._cleaning_params(context)
            )[0],
            DfSubmitPipelineTool.name: lambda context: DfSubmitPipelineTool.create(
                **self._preparation_params(context)
            )[0],
            DfSubmitEvaluationTool.name: lambda context: DfSubmitEvaluationTool.create(
                **self._evaluation_params(context)
            )[0],
            DfCheckProgressTool.name: lambda context: DfCheckProgressTool.create(
                **self._storage_params(context)
            )[0],
            DfStopTaskTool.name: lambda context: DfStopTaskTool.create(
                **self._stop_params(context)
            )[0],
            EdpRenderTool.name: lambda context: EdpRenderTool.create(
                **self._edp_params(context)
            )[0],
            EdpSubmitTool.name: lambda context: EdpSubmitTool.create(
                **self._edp_params(context)
            )[0],
            EdpAggregateTool.name: lambda context: EdpAggregateTool.create(
                **self._edp_params(context)
            )[0],
            SandboxCreateTool.name: lambda context: SandboxCreateTool.create(
                **self._sandbox_params(context, include_default_image=True)
            )[0],
            SandboxDeleteTool.name: lambda context: SandboxDeleteTool.create(
                **self._sandbox_params(context)
            )[0],
            SandboxReadFileTool.name: lambda context: SandboxReadFileTool.create(
                **self._sandbox_params(context)
            )[0],
            SandboxWriteFileTool.name: lambda context: SandboxWriteFileTool.create(
                **self._sandbox_params(context)
            )[0],
            SandboxDeleteFileTool.name: lambda context: SandboxDeleteFileTool.create(
                **self._sandbox_params(context)
            )[0],
            SandboxTerminalTool.name: lambda context: SandboxTerminalTool.create(
                **self._sandbox_params(context)
            )[0],
            SandboxUploadTool.name: lambda context: SandboxUploadTool.create(
                **self._sandbox_storage_params(context)
            )[0],
            SandboxDownloadTool.name: lambda context: SandboxDownloadTool.create(
                **self._sandbox_storage_params(context)
            )[0],
            WorkflowDebugTool.name: lambda context: WorkflowDebugTool.create(
                **self._workflow_debug_params(context)
            )[0],
            InspectWorkflowTaskTool.name: lambda context: (
                InspectWorkflowTaskTool.create(**self._analysis_params(context))[0]
            ),
            TrainingAnalysisTool.name: lambda context: TrainingAnalysisTool.create(
                runtime_dir=str(self._training_runtime),
                **self._training_analysis_params(context),
            )[0],
            LabelStudioProjectTool.name: lambda context: LabelStudioProjectTool.create(
                **self._label_studio_params(context)
            )[0],
            UpdatePlanTool.name: lambda context: UpdatePlanTool.create(
                cast(
                    Any,
                    SimpleNamespace(persistence_dir=str(context.workspace_root / "pi")),
                )
            )[0],
        }

    def terminal_environment(self, context: ToolExecutionContext) -> dict[str, str]:
        from openhands.tools.data_preparation.runner import terminal_model_environment

        environment = terminal_model_environment(
            _ToolConversationFacade(context).state.agent.llm
        )
        interpreter = os.environ.get(ENV_SANDBOX_DATAFLOW_PYTHON, "").strip()
        if interpreter:
            environment[ENV_SANDBOX_DATAFLOW_PYTHON] = interpreter
        return environment

    def specs(self, *, sandbox: bool = False) -> list[dict[str, Any]]:
        """Tool specs this session exposes.

        ``sandbox`` sessions mount the user's Storage, so they publish
        workspace files by writing under the mount and have no use for the
        host-side presigned upload tool.
        """
        specs = [validation_tool_spec()]
        if self.version_call is not None:
            specs.append(
                BusinessToolSpec(
                    ProductVersionsTool.name,
                    ProductVersionsTool.create()[0].description,
                    ProductVersionsAction.model_json_schema(),
                ).as_dict()
            )
        if self._workflow_preparer is not None:
            specs.append(
                BusinessToolSpec(
                    "submit_workflow",
                    "Submit a Storage workflow directly for formal execution. "
                    "The server validates the frozen definition and returns the "
                    "platform task ID without a confirmation dialog. "
                    "Call only when the user asks to run the workflow.",
                    {
                        "type": "object",
                        "properties": {
                            "workflow_path": {
                                "type": "string",
                                "description": (
                                    "Workflow Python DSL path. Use the conversation "
                                    "path such as "
                                    "public_data/workflow_canvas/workflow.py, or an "
                                    "existing /workspace/<path> Storage file."
                                ),
                            }
                        },
                        "required": ["workflow_path"],
                        "additionalProperties": False,
                    },
                ).as_dict()
            )
        tools: list[ToolDefinition[Any, Any]] = [
            UpdatePlanTool.create(cast(Any, SimpleNamespace(persistence_dir=None)))[0]
        ]
        for tool_type in (
            UploadFileToPyromindTool,
            GetStorageUrlTool,
            RunDatasetCleaningTool,
            DfSubmitPipelineTool,
            DfSubmitEvaluationTool,
            DfCheckProgressTool,
            DfStopTaskTool,
            EdpRenderTool,
            EdpSubmitTool,
            EdpAggregateTool,
            SandboxCreateTool,
            SandboxDeleteTool,
            SandboxReadFileTool,
            SandboxWriteFileTool,
            SandboxDeleteFileTool,
            SandboxTerminalTool,
            SandboxUploadTool,
            SandboxDownloadTool,
            WorkflowDebugTool,
            InspectWorkflowTaskTool,
            TrainingAnalysisTool,
        ):
            if sandbox and tool_type is UploadFileToPyromindTool:
                continue
            tools.append(tool_type.create()[0])
        for tool in tools:
            definition = tool.to_mcp_tool()
            schema = definition.get("inputSchema")
            if not isinstance(schema, dict):
                raise TypeError(f"{tool.name} schema must be an object")
            specs.append(
                BusinessToolSpec(tool.name, tool.description, schema).as_dict()
            )
        for name, (_, action_type, description) in _LABEL_STUDIO_OPERATIONS.items():
            specs.append(
                BusinessToolSpec(
                    name, description, action_type.model_json_schema()
                ).as_dict()
            )
        return specs

    def set_workflow_preparer(
        self, callback: Callable[..., Awaitable[dict[str, Any]]]
    ) -> None:
        self._workflow_preparer = callback

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
        tool_call_id: str | None = None,
        origin_run_id: str | None = None,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        origin = context.request_context.task_origin or TaskOrigin()
        if name == ProductVersionsTool.name:
            if self.version_call is None:
                raise ValueError("version management is unavailable")
            try:
                action = ProductVersionsAction.model_validate(arguments)
                result = await self.version_call(
                    context.conversation_id,
                    VersionToolRequest(
                        action=action.action, arguments=action.arguments
                    ),
                    action.command_id,
                    context.request_context,
                )
                return BusinessToolResult(
                    is_error=False,
                    content=[
                        {"type": "text", "text": json.dumps(result, ensure_ascii=False)}
                    ],
                    details=result,
                ).as_dict()
            except ValueError as exc:
                return BusinessToolResult(
                    is_error=True,
                    content=[{"type": "text", "text": str(exc)}],
                    details={},
                ).as_dict()
        if name == "submit_workflow":
            if self._workflow_preparer is None:
                raise ValueError("workflow submission is unavailable")
            workflow_path = arguments.get("workflow_path")
            if not isinstance(workflow_path, str) or not workflow_path.strip():
                raise ValueError("workflow_path is required")
            try:
                storage_path = _workflow_storage_path(context, workflow_path)
                prepared = await self._workflow_preparer(
                    context.conversation_id,
                    workflow_path,
                    context.request_context,
                    request_id=tool_call_id,
                    workflow_storage_path=storage_path,
                    **({"origin": origin} if origin.type == "subagent" else {}),
                )
                return BusinessToolResult(
                    is_error=False,
                    content=[
                        {
                            "type": "text",
                            "text": json.dumps(prepared, ensure_ascii=False),
                        }
                    ],
                    details=prepared,
                ).as_dict()
            except ValueError as exc:
                return BusinessToolResult(
                    is_error=True,
                    content=[{"type": "text", "text": str(exc)}],
                    details={"error": str(exc)},
                ).as_dict()
        if name == "validate_workflow_dsl":
            policy = ToolOutputPolicy()
            policy_source = "platform_default"
            raw = await execute_validation_tool(
                context.workspace or LocalWorkspace(working_dir=context.workspace_root),
                arguments,
                context.request_context,
            )
            response = await self._finalize_result(
                BusinessToolResult(
                    is_error=bool(raw["is_error"]),
                    content=list(raw["content"]),
                    details=dict(raw["details"]),
                ),
                policy=policy,
                policy_source=policy_source,
                context=context,
                tool_name=name,
                tool_call_id=tool_call_id,
            )
            self._log_execution(name, context, started_at, response)
            return response
        operation = _LABEL_STUDIO_OPERATIONS.get(name)
        if operation is not None:
            operation_name, action_type, _ = operation
            try:
                validated = action_type.model_validate(arguments)
            except ValidationError as exc:
                problems = "; ".join(
                    f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                    for error in exc.errors(include_input=False)
                )
                return BusinessToolResult(
                    is_error=True,
                    content=[
                        {
                            "type": "text",
                            "text": f"Validation failed for tool {name}: {problems}. "
                            "Correct the arguments before retrying.",
                        }
                    ],
                    details={"error_code": "invalid_tool_arguments"},
                ).as_dict()
            arguments = {
                "operation": operation_name,
                **validated.model_dump(exclude_none=True),
            }
            name = LabelStudioProjectTool.name
        factory = self._factories.get(name)
        if factory is None:
            raise ValueError(f"unsupported business tool: {name}")
        tool = factory(context)
        if (
            self.execution_call is not None
            and tool.executor is not None
            and hasattr(tool.executor, "submission_handler")
        ):
            inflight = PiSessionFiles(context.workspace_root).load_inflight() or {}
            identity = str(inflight.get("run_id", "")) + ":" + str(tool_call_id or "")
            if not tool_call_id:
                raise ValueError("submission requires a stable tool call ID")
            setattr(
                tool.executor,
                "submission_handler",
                submission_bridge(
                    self.execution_call,
                    context.conversation_id,
                    lambda: identity,
                    context.request_context,
                    asyncio.get_running_loop(),
                    graph_run_id=origin_run_id,
                ),
            )

        policy = resolve_tool_output_policy(tool.meta)
        policy_source = (
            "tool_meta"
            if tool.meta is not None and "output_policy" in tool.meta
            else "platform_default"
        )

        async def invoke() -> BusinessToolResult:
            action = tool.action_type.model_validate(arguments)
            files = PiSessionFiles(context.workspace_root)
            facade = _ToolConversationFacade(context, files.load_business_state())
            key = (context.conversation_id, tool_call_id or name)
            if tool.executor is not None:
                self._active_executors[key] = tool.executor
            try:
                with workflow_submission_scope(origin) as submissions:
                    observation = await asyncio.to_thread(
                        tool, action, cast(Any, facade)
                    )
            finally:
                self._active_executors.pop(key, None)
            if name == WorkflowDebugTool.name:
                debug_details = observation.model_dump(mode="json")
                debug_task_id = debug_details.get("task_id")
                if (
                    not observation.is_error
                    and isinstance(debug_task_id, str)
                    and debug_task_id
                ):
                    facade.signals.append(
                        {
                            "type": "external_task.submitted",
                            "task": {
                                "task_id": debug_task_id,
                                "kind": "workflow_debug",
                                "status": debug_details.get("status", "Pending"),
                            },
                        }
                    )
            observation_data = observation.model_dump(mode="json")
            recorded = {
                str(signal.get("task", {}).get("task_id"))
                for signal in facade.signals
                if signal.get("type") == "external_task.submitted"
            }
            for task_id in submissions.direct_tasks - recorded:
                facade.signals.append(
                    {
                        "type": "external_task.submitted",
                        "task": {
                            "task_id": task_id,
                            "kind": submissions.submissions[task_id].task_kind,
                            "status": "pending",
                        },
                    }
                )
            for signal in facade.signals:
                task = signal.get("task")
                if signal.get("type") == "external_task.submitted" and isinstance(
                    task, dict
                ):
                    correlation = submissions.submissions.get(
                        str(task.get("task_id") or observation_data.get("task_id", ""))
                    )
                    if correlation is not None:
                        task["correlation"] = correlation.model_dump(
                            mode="json", exclude_none=True
                        )
            attempts = facade.state.agent_state.get(WORKFLOW_ATTEMPT_STATE_KEY)
            files.save_business_state(
                {
                    WORKFLOW_ATTEMPT_STATE_KEY: attempts
                    if isinstance(attempts, int) and attempts >= 0
                    else 0
                }
            )
            content = [
                block.model_dump(mode="json") for block in observation.to_llm_content
            ]
            details = observation.model_dump(
                mode="json", exclude={"content", "is_error"}
            )
            if (
                name == TrainingAnalysisTool.name
                and arguments.get("operation") == "artifacts"
                and not observation.is_error
                and self.training_artifact_hook is not None
            ):
                artifacts = details.get("result") or {}
                nodes = artifacts.get("training_nodes")
                if isinstance(nodes, list) and artifacts.get(
                    "task_id"
                ) == arguments.get("task_id"):
                    try:
                        count = await self.training_artifact_hook(
                            context.conversation_id,
                            artifacts["task_id"],
                            nodes,
                            context.request_context,
                        )
                        receipt = {
                            "status": "registered" if count else "unavailable",
                            "checkpoints": count,
                        }
                    except Exception as exc:
                        logger.warning(
                            "Training artifact registration pending: %s",
                            type(exc).__name__,
                        )
                        receipt = {"status": "pending"}
                    details["lineage"] = receipt
                    content.insert(
                        0,
                        {
                            "type": "text",
                            "text": json.dumps({"lineage": receipt}) + "\n",
                        },
                    )
            if len(submissions.submissions) == 1:
                details["correlation"] = next(
                    iter(submissions.submissions.values())
                ).model_dump(mode="json", exclude_none=True)
            return BusinessToolResult(
                is_error=bool(observation.is_error),
                content=content,
                details=details,
                signals=facade.signals,
            )

        if name in _READ_ONLY_TOOLS:
            result = await invoke()
            response = await self._finalize_result(
                result,
                policy=policy,
                policy_source=policy_source,
                context=context,
                tool_name=name,
                tool_call_id=tool_call_id,
            )
            self._log_execution(name, context, started_at, response)
            return response
        lock = self._locks.setdefault(context.conversation_id, asyncio.Lock())
        async with lock:
            result = await invoke()
        response = await self._finalize_result(
            result,
            policy=policy,
            policy_source=policy_source,
            context=context,
            tool_name=name,
            tool_call_id=tool_call_id,
        )
        self._log_execution(name, context, started_at, response)
        return response

    @staticmethod
    async def _finalize_result(
        result: BusinessToolResult,
        *,
        policy: ToolOutputPolicy,
        policy_source: str,
        context: ToolExecutionContext,
        tool_name: str,
        tool_call_id: str | None,
    ) -> dict[str, Any]:
        details = dict(_cap_details_size(result.details))
        full_text = "".join(
            normalize_utf8_text(text)
            for block in result.content
            if block.get("type") == "text"
            and isinstance((text := block.get("text")), str)
        )
        output_id = business_tool_output_id(full_text)
        truncation = truncate_tool_content(
            result.content,
            tool_name=tool_name,
            output_id=output_id,
            policy=policy,
        )
        if not truncation.truncated:
            return BusinessToolResult(
                is_error=result.is_error,
                content=truncation.content,
                details=details,
                signals=result.signals,
            ).as_dict()

        saved = None
        persistence_failed = False
        try:
            saved = await asyncio.to_thread(
                PiSessionFiles(context.workspace_root).save_business_tool_output,
                full_text,
            )
            if saved.sha256 != output_id:
                raise RuntimeError("saved tool output hash does not match its content")
            await _publish_tool_output_to_execution_workspace(saved, context)
        except Exception:
            saved = None
            persistence_failed = True
            logger.warning(
                "Could not persist truncated business tool output "
                "conversation_id=%s tool_name=%s tool_call_id=%s sha256=%s",
                context.conversation_id,
                tool_name,
                tool_call_id,
                output_id,
                exc_info=True,
            )
            truncation = truncate_tool_content(
                result.content,
                tool_name=tool_name,
                output_id=output_id,
                policy=policy,
                full_output_available=False,
            )

        details["output_truncation"] = {
            "truncated": True,
            "original_chars": truncation.original_chars,
            "original_bytes": truncation.original_bytes,
            "retained_chars": truncation.retained_chars,
            "retained_bytes": truncation.retained_bytes,
            "omitted_chars": truncation.omitted_chars,
            "omitted_bytes": truncation.omitted_bytes,
            "original_lines": truncation.original_lines,
            "triggered_by": truncation.triggered_by,
            "invalid_unicode_replacements": (truncation.invalid_unicode_replacements),
            "sha256": output_id,
            "path": saved.relative_path if saved is not None else None,
            "persistence_failed": persistence_failed,
        }
        logger.info(
            "pi.business_tool_output_truncated conversation_id=%s tool_name=%s "
            "tool_call_id=%s policy_source=%s triggered_by=%s original_chars=%d "
            "original_bytes=%d retained_chars=%d retained_bytes=%d "
            "omitted_chars=%d omitted_bytes=%d persisted=%s "
            "persistence_failed=%s sha256=%s",
            context.conversation_id,
            tool_name,
            tool_call_id,
            policy_source,
            truncation.triggered_by,
            truncation.original_chars,
            truncation.original_bytes,
            truncation.retained_chars,
            truncation.retained_bytes,
            truncation.omitted_chars,
            truncation.omitted_bytes,
            saved is not None,
            persistence_failed,
            output_id,
        )
        return BusinessToolResult(
            is_error=result.is_error,
            content=truncation.content,
            details=details,
            signals=result.signals,
        ).as_dict()

    @staticmethod
    def _log_execution(
        name: str,
        context: ToolExecutionContext,
        started_at: float,
        result: dict[str, Any],
    ) -> None:
        logger.info(
            "pi.business_tool name=%s conversation_id=%s elapsed_ms=%.3f is_error=%s",
            name,
            context.conversation_id,
            (time.perf_counter() - started_at) * 1000,
            bool(result.get("is_error")),
        )

    def cancel(self, conversation_id: str) -> None:
        for (owner, _call_id), executor in tuple(self._active_executors.items()):
            if owner == conversation_id:
                executor.interrupt()

    def cancel_call(self, conversation_id: str, tool_call_id: str) -> dict[str, Any]:
        executor = self._active_executors.get((conversation_id, tool_call_id))
        if executor is None:
            return {
                "requested": False,
                "confirmed": False,
                "detail": "tool call is not active in this conversation",
            }
        interrupted = executor.interrupt()
        confirmed = interrupted is True
        return {
            "requested": True,
            "confirmed": confirmed,
            "detail": (
                "tool call interrupt confirmed"
                if confirmed
                else "tool call interrupt requested; termination is unconfirmed"
            ),
        }

    @staticmethod
    def reset_attempt_budget(workspace_root: Path) -> None:
        PiSessionFiles(workspace_root).save_business_state(
            {WORKFLOW_ATTEMPT_STATE_KEY: 0}
        )

    async def stop_platform_task(
        self, task_id: str, context: ToolExecutionContext
    ) -> dict[str, Any]:
        tool = DfStopTaskTool.create(**self._stop_params(context))[0]
        action = tool.action_type.model_validate({"task_id": task_id})
        facade = _ToolConversationFacade(context)
        observation = await asyncio.to_thread(
            tool, cast(Any, action), cast(Any, facade)
        )
        return {
            "is_error": bool(observation.is_error),
            "details": observation.model_dump(mode="json"),
        }

    def _storage_params(self, context: ToolExecutionContext) -> dict[str, Any]:
        params: dict[str, Any] = {}
        storage_url = context.extra.get(
            "storage_base_url", context.extra.get("storage_api_base_url")
        )
        if isinstance(storage_url, str) and storage_url:
            params["storage_base_url"] = storage_url
        headers = _forward_headers(context.request_context, include_cookie=False)
        if headers:
            params["headers"] = headers
        if context.request_context.cookie:
            params["secret_headers"] = {"cookie": _STORAGE_COOKIE_SECRET}
        return params

    def _label_studio_params(self, context: ToolExecutionContext) -> dict[str, Any]:
        params = self._storage_params(context)
        for key, legacy_key in (
            ("ls_base_url", None),
            ("portal_base_url", "sso_base_url"),
        ):
            value = context.extra.get(f"label_studio_{key}")
            if not (isinstance(value, str) and value.strip()) and legacy_key:
                value = context.extra.get(f"label_studio_{legacy_key}")
            if isinstance(value, str) and value.strip():
                params[key] = value.strip()
        params["cluster"] = platform_cluster_route(context.request_context.x_cluster)
        return params

    def _execution_params(self, context: ToolExecutionContext) -> dict[str, Any]:
        env, cluster = _execution_target(context)
        params: dict[str, Any] = {"env": env}
        params["cluster"] = cluster
        params["headers"] = _forward_headers(
            context.request_context, include_cookie=False
        )
        storage = self._storage_params(context)
        if "storage_base_url" in storage:
            params["storage_base_url"] = storage["storage_base_url"]
        if "headers" in storage:
            params["storage_headers"] = storage["headers"]
        if "secret_headers" in storage:
            params["storage_secret_headers"] = storage["secret_headers"]
        return params

    def _cleaning_params(self, context: ToolExecutionContext) -> dict[str, Any]:
        params = self._execution_params(context)
        params["runtime_dir"] = str(self._cleaning_runtime)
        output_root = context.extra.get("dataset_cleaning_output_root")
        if isinstance(output_root, str) and output_root:
            params["output_root"] = output_root
        return params

    def _preparation_params(self, context: ToolExecutionContext) -> dict[str, Any]:
        params = self._execution_params(context)
        params["runtime_dir"] = str(self._preparation_runtime)
        return params

    def _evaluation_params(self, context: ToolExecutionContext) -> dict[str, Any]:
        params = self._execution_params(context)
        if self._pipeline_runs is not None and self._inference_runtime is not None:
            runs = self._pipeline_runs(context.conversation_id).with_context(
                context.request_context
            )
            params["inference_handler"] = InferencePipelineHandler(
                self._inference_runtime,
                runs,
                ValidateWorkflowDslExecutor(
                    env=params["env"],
                    secret_headers={"authorization": _VALIDATE_AUTHORIZATION_SECRET}
                    if context.request_context.authorization
                    else None,
                ),
            )
        return params

    def _edp_params(self, context: ToolExecutionContext) -> dict[str, Any]:
        params = self._execution_params(context)
        params["runtime_dir"] = str(self._edp_runtime)
        return params

    def _sandbox_params(
        self,
        context: ToolExecutionContext,
        *,
        include_default_image: bool = False,
    ) -> dict[str, Any]:
        env, cluster = _execution_target(context)
        params: dict[str, Any] = {
            "current_user": current_user_from_context(context.request_context),
            "env": env,
            "headers": _forward_headers(
                context.request_context,
                include_cookie=False,
            ),
        }
        params["cluster"] = cluster
        default_image = context.extra.get("sandbox_default_image")
        if include_default_image and isinstance(default_image, str) and default_image:
            params["default_image"] = default_image
        return params

    def _sandbox_storage_params(
        self,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        params = self._sandbox_params(context)
        params.pop("default_image", None)
        storage = self._storage_params(context)
        if "storage_base_url" in storage:
            params["storage_base_url"] = storage["storage_base_url"]
        if "headers" in storage:
            params["storage_headers"] = storage["headers"]
        if "secret_headers" in storage:
            params["storage_secret_headers"] = storage["secret_headers"]
        return params

    def _stop_params(self, context: ToolExecutionContext) -> dict[str, Any]:
        params: dict[str, Any] = {
            "headers": _forward_headers(context.request_context, include_cookie=False)
        }
        if self._pipeline_runs is not None:
            params["task_resolver"] = self._pipeline_runs(
                context.conversation_id
            ).task_for
        if context.request_context.cookie:
            params["secret_headers"] = {"cookie": _STORAGE_COOKIE_SECRET}
        return params

    def _workflow_debug_params(self, context: ToolExecutionContext) -> dict[str, Any]:
        env, cluster = _execution_target(context)
        return {
            "cluster": cluster,
            "env": env,
            "current_user": current_user_from_context(context.request_context),
            "headers": _forward_headers(context.request_context, include_cookie=False),
        }

    @staticmethod
    def _analysis_params(context: ToolExecutionContext) -> dict[str, Any]:
        secret_headers: dict[str, str] = {}
        if context.request_context.cookie:
            secret_headers["cookie"] = PYROMIND_VALIDATE_AUTH_COOKIE_SECRET
        if context.request_context.authorization:
            secret_headers["authorization"] = _VALIDATE_AUTHORIZATION_SECRET
        params: dict[str, Any] = {
            "headers": _forward_headers(context.request_context, include_cookie=False)
        }
        if secret_headers:
            params["secret_headers"] = secret_headers
        api_base = context.extra.get("training_analysis_api_base")
        if isinstance(api_base, str) and api_base:
            params["api_base"] = api_base
        return params

    @classmethod
    def _training_analysis_params(cls, context: ToolExecutionContext) -> dict[str, Any]:
        params = cls._analysis_params(context)
        timeout = context.extra.get("training_analysis_timeout_seconds")
        if isinstance(timeout, (int, float)) and not isinstance(timeout, bool):
            params["timeout_seconds"] = float(timeout)
        return params


def _forward_headers(
    context: RequestContext, *, include_cookie: bool
) -> dict[str, str]:
    return {
        name: value
        for name, value in (
            ("cookie", context.cookie if include_cookie else None),
            ("authorization", context.authorization),
            ("x-cluster", platform_cluster_route(context.x_cluster)),
            ("accept-language", context.accept_language),
        )
        if value
    }


def _execution_target(context: ToolExecutionContext) -> tuple[str, str]:
    return platform_cluster_target(context.request_context.x_cluster)


async def _publish_tool_output_to_execution_workspace(
    saved: SavedToolOutput, context: ToolExecutionContext
) -> None:
    """Mirror a spilled tool output into the sandbox execution workspace.

    The truncation marker quotes a workspace-relative path, and with a remote
    sandbox the model's file tools resolve that path inside the container, so
    the control-plane spill file alone would be unreachable for the model.
    """
    workspace = context.workspace
    if not isinstance(workspace, SandboxWorkspace):
        return
    result = await asyncio.to_thread(
        workspace.file_upload, saved.path, saved.relative_path
    )
    if not result.success:
        raise RuntimeError(
            "could not publish tool output to the sandbox execution "
            f"workspace: {result.error}"
        )


def _auth_token(context: RequestContext) -> str | None:
    token = parse_auth_token_from_cookie_header(context.cookie)
    if token:
        return token
    authorization = context.authorization or ""
    scheme, _, value = authorization.partition(" ")
    return value.strip() if scheme.lower() == "bearer" and value.strip() else None
