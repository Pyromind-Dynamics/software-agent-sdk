from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyromind_agent_server.echomind_graph_runs import TaskRun
from pyromind_agent_server.echomind_projects import ProjectStore
from pyromind_agent_server.echomind_runtime_contracts import (
    ExternalTaskIntent,
    RuntimeArtifact,
    RuntimeCredentialUnavailableError,
    RuntimeStageContext,
    RuntimeSubmissionNotAttemptedError,
    StageCancellationResult,
    StagePollResult,
    StagePreparationResult,
    StageSubmissionResult,
)
from pyromind_agent_server.echomind_stage_executors import StageArtifactPayload
from pyromind_common.tools.utils.training_source import (
    WANDB_BUILDER_NODE_TYPE,
    WANDB_BUILDER_NODE_TYPES,
)
from pyromind_runtime.domain.context import RequestContext


TrainTaskStatus = Literal[
    "pending",
    "running",
    "succeeded",
    "failed",
    "canceled",
]

_JSON_PATH_TOKEN = re.compile(r"(?:^|\.)([^.[\]]+)|\[(\d+)\]")


class TrainRuntimeError(RuntimeError):
    pass


class TrainTaskState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: TrainTaskStatus
    message: str = Field(min_length=1, max_length=4000)
    output_uri: str | None = Field(default=None, max_length=4000)
    failure_category: Literal[
        "external", "validation", "infrastructure", "canceled"
    ] = "external"


class TrainWorkflowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_name: str = Field(min_length=1, max_length=256)
    conversation_id: str = Field(min_length=1, max_length=256)
    cluster: str = Field(min_length=1, max_length=256)
    owner_user_id: str | None = None
    workflow: dict[str, Any] | None = None
    workflow_path: str | None = Field(default=None, max_length=4000)
    wandb_credential_task_id: str | None = Field(
        default=None, pattern=r"^[0-9]+$", max_length=64
    )
    bindings: dict[str, str] = Field(default_factory=dict)
    workflow_templates: dict[str, str] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)
    output_uri: str | None = Field(default=None, max_length=4000)
    artifact_uri: str | None = Field(default=None, max_length=4000)
    artifact_selection: (
        Literal["latest_lora_checkpoint", "merged_model", "evaluation_report"] | None
    ) = None
    output_node_id: str | None = Field(default=None, max_length=128)
    output_parameter: str = Field(default="output_path", min_length=1, max_length=128)
    command_template: str | None = Field(default=None, max_length=20000)
    command_binding_path: str | None = Field(default=None, max_length=4000)
    template_values: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_workflow_source(self) -> TrainWorkflowSpec:
        if self.artifact_selection is not None and self.output_uri is None:
            raise ValueError("Artifact selection requires a configured output URI")
        if (self.workflow is None) == (self.workflow_path is None):
            raise ValueError("Workflow runtime requires exactly one source")
        if (self.command_template is None) != (self.command_binding_path is None):
            raise ValueError(
                "command_template and command_binding_path must be configured together"
            )
        return self


class TrainWorkflowBackend(Protocol):
    def find_task_by_name(
        self,
        *,
        name: str,
        spec: TrainWorkflowSpec,
        request_context: RequestContext | None,
    ) -> str | None: ...

    def submit_workflow(
        self,
        *,
        spec: TrainWorkflowSpec,
        request_context: RequestContext | None,
    ) -> str: ...

    def query_task(
        self,
        *,
        task_id: str,
        spec: TrainWorkflowSpec,
        request_context: RequestContext | None,
    ) -> TrainTaskState: ...

    def cancel_task(
        self,
        *,
        task_id: str,
        spec: TrainWorkflowSpec,
        request_context: RequestContext | None,
    ) -> None: ...


class TrainRuntimeExecutor:
    def __init__(
        self,
        *,
        projects_root: str,
        backend: TrainWorkflowBackend,
        request_context: RequestContext | None = None,
        provider: str = "pyromind.studio.training",
        output_port: str = "trained_model",
        config_key: str = "train_runtime",
        task_name_prefix: str = "echomind-train",
        pending_task_name: str = "echomind-train-pending",
        runtime_name: str = "Train",
        output_input_name: str = "trained_model_uri",
    ) -> None:
        self.projects_root = projects_root
        self.backend = backend
        self.request_context = request_context
        self.provider = provider
        self.output_port = output_port
        self.config_key = config_key
        self.task_name_prefix = task_name_prefix
        self.pending_task_name = pending_task_name
        self.runtime_name = runtime_name
        self.output_input_name = output_input_name

    def prepare(self, context: RuntimeStageContext) -> StagePreparationResult:
        try:
            spec = self._workflow_spec(context=context, task_run=None)
        except Exception as exc:
            return StagePreparationResult(
                status="failed",
                message=f"{self.runtime_name} workflow configuration is invalid.",
                failure_category="validation",
                error=f"{type(exc).__name__}: {exc}"[:4000],
            )
        if self.output_port not in context.node.outputs:
            return StagePreparationResult(
                status="failed",
                message=(
                    f"{self.runtime_name} node does not expose {self.output_port}."
                ),
                failure_category="validation",
                error=(f"Missing {self.runtime_name} output port: {self.output_port}"),
            )
        return StagePreparationResult(
            status="waiting_external",
            message=(f"{self.runtime_name} workflow is ready for platform submission."),
            external_task=ExternalTaskIntent(
                provider=self.provider,
                metadata={
                    "task_name": spec.task_name,
                    "workflow_path": spec.workflow_path,
                    "output_uri": spec.output_uri,
                    "artifact_uri": spec.artifact_uri,
                    "output_node_id": spec.output_node_id,
                    "output_parameter": spec.output_parameter,
                },
            ),
        )

    def submit(
        self,
        context: RuntimeStageContext,
        task_run: TaskRun,
    ) -> StageSubmissionResult:
        try:
            spec = self._workflow_spec(context=context, task_run=task_run)
            existing = self.backend.find_task_by_name(
                name=spec.task_name,
                spec=spec,
                request_context=self.request_context,
            )
        except RuntimeCredentialUnavailableError:
            raise
        except Exception as exc:
            raise RuntimeSubmissionNotAttemptedError(str(exc)) from exc
        if existing is not None:
            return StageSubmissionResult(
                status="submitted",
                message=(f"Recovered existing {self.runtime_name} task {existing}."),
                provider_run_id=existing,
            )
        if task_run.status == "unknown":
            raise TrainRuntimeError(
                f"Could not reconcile {self.runtime_name} task {spec.task_name}."
            )
        task_id = self.backend.submit_workflow(
            spec=spec,
            request_context=self.request_context,
        )
        return StageSubmissionResult(
            status="submitted",
            message=f"{self.runtime_name} task submitted: {task_id}.",
            provider_run_id=task_id,
        )

    def poll(
        self,
        context: RuntimeStageContext,
        task_run: TaskRun,
    ) -> StagePollResult:
        if task_run.provider_run_id is None:
            raise TrainRuntimeError(
                f"{self.runtime_name} TaskRun has no provider task ID"
            )
        spec = self._workflow_spec(context=context, task_run=task_run)
        state = self.backend.query_task(
            task_id=task_run.provider_run_id,
            spec=spec,
            request_context=self.request_context,
        )
        if state.status in {"pending", "running"}:
            return StagePollResult(status="running", message=state.message)
        if state.status in {"failed", "canceled"}:
            return StagePollResult(
                status="failed",
                message=state.message,
                failure_category=(
                    "canceled" if state.status == "canceled" else state.failure_category
                ),
            )
        output_uri = state.output_uri or spec.output_uri
        artifact_uri = (
            state.output_uri
            if spec.artifact_selection is not None
            else spec.artifact_uri or output_uri
        )
        if not artifact_uri:
            return StagePollResult(
                status="failed",
                message=(f"{self.runtime_name} succeeded without an output URI."),
                failure_category="validation",
            )
        output = context.node.outputs[self.output_port]
        return StagePollResult(
            status="completed",
            message=state.message,
            artifacts=[
                RuntimeArtifact(
                    port=self.output_port,
                    payload=StageArtifactPayload(
                        kind=output.artifact_type,
                        content={
                            "schema_version": 1,
                            "uri": artifact_uri,
                            "task_id": task_run.provider_run_id,
                            "task_name": spec.task_name,
                            "workflow_path": spec.workflow_path,
                        },
                    ),
                    metadata={
                        "source_uri": artifact_uri,
                        "provider_task_id": task_run.provider_run_id,
                    },
                )
            ],
        )

    def cancel(
        self,
        context: RuntimeStageContext,
        task_run: TaskRun,
    ) -> StageCancellationResult:
        task_id = task_run.provider_run_id
        if task_id is None and task_run.status != "unknown":
            return StageCancellationResult(
                status="canceled",
                message=(
                    f"{self.runtime_name} TaskRun has no submitted provider task."
                ),
            )
        spec = self._workflow_spec(context=context, task_run=task_run)
        if task_id is None:
            task_id = self.backend.find_task_by_name(
                name=spec.task_name,
                spec=spec,
                request_context=self.request_context,
            )
            if task_id is None:
                return StageCancellationResult(
                    status="failed",
                    message=f"Could not reconcile {self.runtime_name} task "
                    f"{spec.task_name} for cancellation.",
                )
        self.backend.cancel_task(
            task_id=task_id,
            spec=spec,
            request_context=self.request_context,
        )
        return StageCancellationResult(
            status="canceled",
            message=f"{self.runtime_name} task stopped: {task_id}.",
        )

    def _workflow_spec(
        self,
        *,
        context: RuntimeStageContext,
        task_run: TaskRun | None,
    ) -> TrainWorkflowSpec:
        project = ProjectStore(self.projects_root).load(
            context.project_id,
            owner_user_id=None,
        )
        config = _runtime_config(context.node.config, self.config_key)
        task_name = (
            _task_name(task_run.idempotency_key, prefix=self.task_name_prefix)
            if task_run is not None
            else str(config.get("task_name") or self.pending_task_name)
        )
        bindings = _bindings(config.get("workflow_bindings"))
        for port in bindings.values():
            if port not in context.inputs:
                raise TrainRuntimeError(
                    f"{self.runtime_name} workflow binding references "
                    f"missing input: {port}"
                )
        output_uri = _render_output_uri(
            _optional_string(config.get("output_uri"))
            or _optional_string(config.get("output_path"))
            or _optional_string(context.inputs.get(self.output_input_name)),
            context=context,
        )
        artifact_uri = _render_output_uri(
            _optional_string(config.get("artifact_uri")),
            context=context,
        )
        artifact_selection = config.get("artifact_selection")
        if artifact_selection not in (
            None,
            "latest_lora_checkpoint",
            "merged_model",
            "evaluation_report",
        ):
            raise TrainRuntimeError(f"Unknown artifact selection: {artifact_selection}")
        spec = TrainWorkflowSpec(
            task_name=task_name,
            conversation_id=project.project.conversation_id,
            cluster=project.project.cluster,
            owner_user_id=project.project.owner_user_id,
            workflow=_workflow_definition(config.get("workflow")),
            workflow_path=(_optional_string(config.get("workflow_path"))),
            wandb_credential_task_id=config.get("wandb_credential_task_id"),
            bindings=bindings,
            workflow_templates=_workflow_templates(config.get("workflow_templates")),
            inputs=dict(context.inputs),
            output_uri=output_uri,
            artifact_uri=artifact_uri,
            artifact_selection=artifact_selection,
            output_node_id=_optional_string(config.get("output_node_id")),
            output_parameter=(
                _optional_string(config.get("output_parameter")) or "output_path"
            ),
            command_template=_optional_string(config.get("command_template")),
            command_binding_path=_optional_string(config.get("command_binding_path")),
            template_values=_template_values(context=context, output_uri=output_uri),
        )
        if spec.workflow is not None:
            prepare_workflow_for_submission(spec.workflow, spec=spec)
        return spec


def bind_workflow_inputs(
    workflow: Mapping[str, Any],
    *,
    bindings: Mapping[str, str],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    effective = deepcopy(dict(workflow))
    for workflow_path, port in bindings.items():
        if port not in inputs:
            raise TrainRuntimeError(f"Workflow binding input is missing: {port}")
        _set_json_path(
            effective,
            workflow_path,
            _workflow_input_value(inputs[port]),
        )
    return effective


def bind_workflow_output(
    workflow: Mapping[str, Any],
    *,
    node_id: str,
    parameter: str,
    value: str,
) -> dict[str, Any]:
    effective = deepcopy(dict(workflow))
    nodes = effective.get("nodes")
    if not isinstance(nodes, list):
        raise TrainRuntimeError("Workflow definition has no nodes")
    matches = [
        node
        for node in nodes
        if isinstance(node, dict) and str(node.get("id")) == node_id
    ]
    if len(matches) != 1:
        raise TrainRuntimeError(
            f"Workflow output node must identify exactly one node: {node_id}"
        )
    node = matches[0]
    data = node.get("data")
    config = data.get("config") if isinstance(data, Mapping) else None
    if not isinstance(config, dict) or parameter not in config:
        raise TrainRuntimeError(
            f"Workflow output parameter does not exist: {node_id}.{parameter}"
        )
    config[parameter] = value
    return effective


def render_workflow_template(
    template: str,
    *,
    values: Mapping[str, str],
) -> str:
    class _Values(dict[str, str]):
        def __missing__(self, key: str) -> str:
            raise TrainRuntimeError(f"Unknown workflow template placeholder: {key}")

    try:
        return template.format_map(_Values(values))
    except (IndexError, KeyError, ValueError) as exc:
        raise TrainRuntimeError(f"Workflow template is invalid: {exc}") from exc


def normalize_wandb_builder_nodes(workflow: dict[str, Any]) -> None:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return
    for node in nodes:
        data = node.get("data") if isinstance(node, dict) else None
        if isinstance(data, dict) and data.get("nodeType") in WANDB_BUILDER_NODE_TYPES:
            data["nodeType"] = WANDB_BUILDER_NODE_TYPE


def prepare_workflow_for_submission(
    workflow: Mapping[str, Any],
    *,
    spec: TrainWorkflowSpec,
) -> dict[str, Any]:
    effective = bind_workflow_inputs(
        workflow,
        bindings=spec.bindings,
        inputs=spec.inputs,
    )
    template_values = {
        name: rendered
        for name, value in spec.inputs.items()
        if (rendered := _template_input_value(value)) is not None
    }
    template_values.update(spec.template_values)
    for workflow_path, template in spec.workflow_templates.items():
        _set_json_path(
            effective,
            workflow_path,
            render_workflow_template(template, values=template_values),
        )
    if spec.command_template and spec.command_binding_path:
        command = render_workflow_template(
            spec.command_template,
            values=template_values,
        )
        _set_json_path(effective, spec.command_binding_path, command)
    if spec.output_uri and spec.output_node_id:
        effective = bind_workflow_output(
            effective,
            node_id=spec.output_node_id,
            parameter=spec.output_parameter,
            value=spec.output_uri,
        )
    normalize_wandb_builder_nodes(effective)
    return effective


def _runtime_config(
    config: Mapping[str, Any],
    config_key: str,
) -> Mapping[str, Any]:
    nested = config.get(config_key)
    if isinstance(nested, Mapping):
        return nested
    return config


def _bindings(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("workflow_bindings must be an object")
    result: dict[str, str] = {}
    for path, port in value.items():
        if not isinstance(path, str) or not isinstance(port, str):
            raise ValueError("workflow_bindings keys and values must be strings")
        result[path] = port
    return result


def _workflow_templates(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("workflow_templates must be an object")
    result: dict[str, str] = {}
    for path, template in value.items():
        if not isinstance(path, str) or not isinstance(template, str):
            raise ValueError("workflow_templates keys and values must be strings")
        result[path] = template
    return result


def _workflow_definition(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("workflow must be an object")
    return deepcopy(dict(value))


def _task_name(idempotency_key: str, *, prefix: str) -> str:
    digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:24]}"


def _render_output_uri(
    template: str | None,
    *,
    context: RuntimeStageContext,
) -> str | None:
    if template is None:
        return None
    values = {
        "project_id": context.project_id,
        "release_id": context.release_id,
        "graph_run_id": context.graph_run_id,
        "stage_run_id": context.stage_run_id,
        "node_id": context.node.id,
    }
    try:
        return template.format_map(values)
    except KeyError as exc:
        raise TrainRuntimeError(
            f"Unknown output URI placeholder: {exc.args[0]}"
        ) from exc


def _template_values(
    *,
    context: RuntimeStageContext,
    output_uri: str | None,
) -> dict[str, str]:
    values = {
        name: rendered
        for name, value in context.inputs.items()
        if (rendered := _template_input_value(value)) is not None
    }
    values.update(
        {
            "project_id": context.project_id,
            "release_id": context.release_id,
            "graph_run_id": context.graph_run_id,
            "stage_run_id": context.stage_run_id,
            "node_id": context.node.id,
        }
    )
    if output_uri is not None:
        values["output_uri"] = output_uri
    return values


def _template_input_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Mapping):
        resolved = _workflow_input_value(value)
        if isinstance(resolved, str):
            return resolved
    return None


def _workflow_input_value(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        metadata = value.get("metadata")
        if isinstance(metadata, Mapping):
            source_uri = metadata.get("source_uri")
            if isinstance(source_uri, str):
                return source_uri
        uri = value.get("uri")
        if isinstance(uri, str):
            return uri
        content = value.get("content")
        if isinstance(content, Mapping):
            content_uri = content.get("uri")
            if isinstance(content_uri, str):
                return content_uri
    return value


def _set_json_path(target: Any, path: str, value: Any) -> None:
    tokens = _json_path_tokens(path)
    current = target
    for index, token in enumerate(tokens):
        next_token = tokens[index + 1] if index + 1 < len(tokens) else None
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                raise TrainRuntimeError(
                    f"Workflow binding path is out of range: {path}"
                )
            if next_token is None:
                current[token] = value
                return
            current = current[token]
            continue
        if not isinstance(current, dict):
            raise TrainRuntimeError(f"Workflow binding path is invalid: {path}")
        if next_token is None:
            if token not in current:
                raise TrainRuntimeError(f"Workflow binding path does not exist: {path}")
            current[token] = value
            return
        if token not in current:
            raise TrainRuntimeError(f"Workflow binding path does not exist: {path}")
        current = current[token]


def _json_path_tokens(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    position = 0
    for match in _JSON_PATH_TOKEN.finditer(path):
        if match.start() != position:
            raise TrainRuntimeError(f"Invalid workflow binding path: {path}")
        key, index = match.groups()
        tokens.append(int(index) if index is not None else str(key))
        position = match.end()
    if position != len(path) or not tokens:
        raise TrainRuntimeError(f"Invalid workflow binding path: {path}")
    return tokens


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
