from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from pydantic import SecretStr
from pyromind_runtime.domain.context import RequestContext

from harness_adapter.openhands_adapter.session_factory import current_user_from_context
from harness_adapter.pi_adapter.business_tools import (
    execute_validation_tool,
    validation_tool_spec,
)
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from openhands.agent_server.pyromind_auth import parse_auth_token_from_cookie_header
from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.sdk.secret import StaticSecret
from openhands.sdk.tool import ToolDefinition
from openhands.sdk.workspace.local import LocalWorkspace
from openhands.tools.data_preparation import (
    DfCheckProgressTool,
    DfRunPipelineTool,
    DfStopTaskTool,
    DfSubmitPipelineTool,
)
from openhands.tools.pyromind_cleaning import RunDatasetCleaningTool
from openhands.tools.pyromind_dataset import (
    PreviewDatasetTool,
    UploadFileToPyromindTool,
)
from openhands.tools.training_analysis import TrainingAnalysisTool
from openhands.tools.workflow.analyze_task_failure import AnalyzeTaskFailureTool
from openhands.tools.workflow.run_workflow import WORKFLOW_ATTEMPT_STATE_KEY
from openhands.tools.workflow.task_submission import (
    PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET,
)
from openhands.tools.workflow.validate_workflow_dsl import (
    PYROMIND_VALIDATE_AUTH_COOKIE_SECRET,
    PYROMIND_VALIDATE_HEADERS_STATE_KEY,
)
from openhands.tools.workflow_debug import WorkflowDebugTool


_STORAGE_COOKIE_SECRET = "PYROMIND_STORAGE_AUTH_COOKIE"
_VALIDATE_AUTHORIZATION_SECRET = "PYROMIND_VALIDATE_AUTHORIZATION"
_READ_ONLY_TOOLS = frozenset(
    {
        "preview_dataset",
        "df_check_progress",
        AnalyzeTaskFailureTool.name,
    }
)
logger = logging.getLogger(__name__)


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


class _ToolConversationFacade:
    """The deliberately small OpenHands compatibility boundary used by Pi."""

    def __init__(
        self,
        context: ToolExecutionContext,
        persisted_state: dict[str, Any] | None = None,
    ) -> None:
        self.id = context.conversation_id
        self.workspace = LocalWorkspace(working_dir=context.workspace_root)
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

    def __init__(self, skill_roots: Sequence[Path]) -> None:
        roots = {path.name: path.resolve() for path in skill_roots}
        self._cleaning_runtime = roots["data-processing"] / "scripts" / "cleaning"
        self._preparation_runtime = roots["data-processing"] / "scripts" / "preparation"
        self._training_runtime = roots["training-analysis"] / "scripts"
        self._locks: dict[str, asyncio.Lock] = {}
        self._active_executors: dict[tuple[str, str], Any] = {}
        self._factories: dict[str, ToolFactory] = {
            PreviewDatasetTool.name: lambda context: PreviewDatasetTool.create(
                **self._storage_params(context),
                extract_params=self._extraction_params(context),
            )[0],
            UploadFileToPyromindTool.name: lambda context: (
                UploadFileToPyromindTool.create(**self._storage_params(context))[0]
            ),
            RunDatasetCleaningTool.name: lambda context: RunDatasetCleaningTool.create(
                **self._cleaning_params(context)
            )[0],
            DfRunPipelineTool.name: lambda _context: DfRunPipelineTool.create(
                runtime_dir=str(self._preparation_runtime)
            )[0],
            DfSubmitPipelineTool.name: lambda context: DfSubmitPipelineTool.create(
                **self._preparation_params(context)
            )[0],
            DfCheckProgressTool.name: lambda context: DfCheckProgressTool.create(
                **self._storage_params(context)
            )[0],
            DfStopTaskTool.name: lambda context: DfStopTaskTool.create(
                **self._stop_params(context)
            )[0],
            WorkflowDebugTool.name: lambda context: WorkflowDebugTool.create(
                **self._workflow_debug_params(context)
            )[0],
            AnalyzeTaskFailureTool.name: lambda context: AnalyzeTaskFailureTool.create(
                **self._analysis_params(context)
            )[0],
            TrainingAnalysisTool.name: lambda context: TrainingAnalysisTool.create(
                runtime_dir=str(self._training_runtime),
                **self._training_analysis_params(context),
            )[0],
        }

    def specs(self) -> list[dict[str, Any]]:
        specs = [validation_tool_spec()]
        for tool_type in (
            PreviewDatasetTool,
            UploadFileToPyromindTool,
            RunDatasetCleaningTool,
            DfRunPipelineTool,
            DfSubmitPipelineTool,
            DfCheckProgressTool,
            DfStopTaskTool,
            WorkflowDebugTool,
            AnalyzeTaskFailureTool,
            TrainingAnalysisTool,
        ):
            tool = tool_type.create()[0]
            definition = tool.to_mcp_tool()
            schema = definition.get("inputSchema")
            if not isinstance(schema, dict):
                raise TypeError(f"{tool.name} schema must be an object")
            specs.append(
                BusinessToolSpec(tool.name, tool.description, schema).as_dict()
            )
        return specs

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
        tool_call_id: str | None = None,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        if name == "validate_workflow_dsl":
            result = await execute_validation_tool(
                context.workspace_root, arguments, context.request_context
            )
            response = {**result, "signals": []}
            self._log_execution(name, context, started_at, response)
            return response
        factory = self._factories.get(name)
        if factory is None:
            raise ValueError(f"unsupported business tool: {name}")

        async def invoke() -> dict[str, Any]:
            tool = factory(context)
            action = tool.action_type.model_validate(arguments)
            files = PiSessionFiles(context.workspace_root)
            facade = _ToolConversationFacade(context, files.load_business_state())
            key = (context.conversation_id, tool_call_id or name)
            if tool.executor is not None:
                self._active_executors[key] = tool.executor
            try:
                observation = await asyncio.to_thread(tool, action, cast(Any, facade))
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
            return BusinessToolResult(
                is_error=bool(observation.is_error),
                content=content,
                details=details,
                signals=facade.signals,
            ).as_dict()

        if name in _READ_ONLY_TOOLS:
            response = await invoke()
            self._log_execution(name, context, started_at, response)
            return response
        lock = self._locks.setdefault(context.conversation_id, asyncio.Lock())
        async with lock:
            response = await invoke()
        self._log_execution(name, context, started_at, response)
        return response

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

    def _execution_params(self, context: ToolExecutionContext) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if context.request_context.x_cluster:
            params["cluster"] = context.request_context.x_cluster
        env = context.extra.get("env")
        if isinstance(env, str) and env:
            params["env"] = env
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

    def _extraction_params(self, context: ToolExecutionContext) -> dict[str, Any]:
        params = self._execution_params(context)
        output_root = context.extra.get("dataset_extraction_output_root")
        if isinstance(output_root, str) and output_root:
            params["output_root"] = output_root
        return params

    def _stop_params(self, context: ToolExecutionContext) -> dict[str, Any]:
        params: dict[str, Any] = {
            "headers": _forward_headers(context.request_context, include_cookie=False)
        }
        if context.request_context.cookie:
            params["secret_headers"] = {"cookie": _STORAGE_COOKIE_SECRET}
        return params

    def _workflow_debug_params(self, context: ToolExecutionContext) -> dict[str, Any]:
        return {
            "cluster": context.request_context.x_cluster,
            "env": context.extra.get("env"),
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
            ("x-cluster", context.x_cluster),
            ("accept-language", context.accept_language),
        )
        if value
    }


def _auth_token(context: RequestContext) -> str | None:
    token = parse_auth_token_from_cookie_header(context.cookie)
    if token:
        return token
    authorization = context.authorization or ""
    scheme, _, value = authorization.partition(" ")
    return value.strip() if scheme.lower() == "bearer" and value.strip() else None
