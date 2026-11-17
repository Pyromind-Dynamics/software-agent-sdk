from __future__ import annotations

import json
import shlex
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from string import Formatter
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.storage_paths import normalize_workspace_path

from openhands.agent_server.echomind_graph_runs import TaskRun
from openhands.agent_server.echomind_projects import ProjectStore
from openhands.agent_server.echomind_runtime_contracts import (
    ExternalTaskIntent,
    RuntimeArtifact,
    RuntimeStageContext,
    StageCancellationResult,
    StagePollResult,
    StagePreparationResult,
    StageSubmissionResult,
)
from openhands.agent_server.echomind_stage_executors import StageArtifactPayload
from openhands.agent_server.echomind_train_runtime import (
    TrainWorkflowBackend,
    TrainWorkflowSpec,
    _task_name,
    _workflow_input_value,
    bind_workflow_inputs,
    bind_workflow_output,
    render_workflow_template,
)


class DatasetRuntimeError(RuntimeError):
    pass


_WORKFLOW_OUTPUTS = ("train_messages", "val_messages")
_WORKFLOW_RUNTIME_IDS = (
    "project_id",
    "release_id",
    "graph_run_id",
    "stage_run_id",
    "node_id",
)


class DatasetWorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow: dict[str, Any] = Field(min_length=1)
    workflow_bindings: dict[str, str] = Field(default_factory=dict)
    workflow_templates: dict[str, str] = Field(default_factory=dict)
    output_uri: str = Field(min_length=1, max_length=4000)
    output_node_id: str | None = Field(default=None, min_length=1, max_length=128)
    output_parameter: str = Field(default="output_path", min_length=1, max_length=128)
    train_filename: str = "train_messages.jsonl"
    val_filename: str = "val_messages.jsonl"
    command_template: str | None = Field(default=None, min_length=1, max_length=20000)
    command_binding_path: str | None = Field(
        default=None, min_length=1, max_length=4000
    )

    @model_validator(mode="after")
    def validate_config(self) -> DatasetWorkflowConfig:
        if (self.command_template is None) != (self.command_binding_path is None):
            raise ValueError(
                "command_template and command_binding_path must be configured together"
            )
        if "graph_run_id" not in _template_fields(self.output_uri):
            raise ValueError("Dataset output_uri must contain {graph_run_id}")
        if any(
            not name.endswith(".jsonl")
            or PurePosixPath(name).name != name
            or any(character in name for character in "\\:?#")
            for name in self.output_filenames
        ):
            raise ValueError("Dataset output filenames must be JSONL basenames")
        if self.train_filename == self.val_filename:
            raise ValueError("Dataset training and validation filenames must differ")
        return self

    @property
    def output_filenames(self) -> tuple[str, str]:
        return self.train_filename, self.val_filename

    def validate_workflow_inputs(self, *, input_ports: Iterable[str]) -> None:
        inputs = {port: "/workspace/dataset-input" for port in input_ports}
        if "source_batch" not in inputs:
            raise ValueError("Dataset workflow requires a source_batch input port")
        template_fields = {
            field
            for template in (
                *self.workflow_templates.values(),
                self.command_template or "",
            )
            for field in _template_fields(template)
        }
        if (
            "source_batch" not in self.workflow_bindings.values()
            and "source_batch" not in template_fields
        ):
            raise ValueError("Dataset workflow must bind the source_batch input")
        if self.output_node_id is None and "output_uri" not in template_fields:
            raise ValueError("Dataset workflow must bind the run-specific output_uri")
        self.render_workflow(
            inputs=inputs,
            runtime_values={name: "validation" for name in _WORKFLOW_RUNTIME_IDS},
        )

    def render_workflow(
        self,
        *,
        inputs: Mapping[str, Any],
        runtime_values: Mapping[str, str],
    ) -> tuple[dict[str, Any], str]:
        output_uri = normalize_workspace_path(
            render_workflow_template(self.output_uri, values=runtime_values),
            field="Dataset output_uri",
        )
        workflow = bind_workflow_inputs(
            self.workflow, bindings=self.workflow_bindings, inputs=inputs
        )
        template_values = {
            name: value
            for name, raw in inputs.items()
            if isinstance(value := _workflow_input_value(raw), str)
        }
        template_values.update(runtime_values)
        template_values["output_uri"] = output_uri
        rendered = {
            path: render_workflow_template(template, values=template_values)
            for path, template in self.workflow_templates.items()
        }
        if self.command_template and self.command_binding_path:
            rendered[self.command_binding_path] = render_workflow_template(
                self.command_template,
                values={
                    name: shlex.quote(value) for name, value in template_values.items()
                },
            )
        if rendered:
            workflow = bind_workflow_inputs(
                workflow,
                bindings={path: path for path in rendered},
                inputs=rendered,
            )
        if self.output_node_id is not None:
            workflow = bind_workflow_output(
                workflow,
                node_id=self.output_node_id,
                parameter=self.output_parameter,
                value=output_uri,
            )
        return workflow, output_uri


def _template_fields(template: str) -> set[str]:
    return {
        field for _, field, _, _ in Formatter().parse(template) if field is not None
    }


class DatasetWorkflowSpec(TrainWorkflowSpec):
    train_filename: str
    val_filename: str

    @property
    def output_filenames(self) -> tuple[str, str]:
        return self.train_filename, self.val_filename

    @property
    def output_directory(self) -> str:
        assert self.output_uri is not None
        return self.output_uri


class PlatformDatasetRuntimeExecutor:
    def __init__(
        self,
        *,
        projects_root: Path | str,
        backend: TrainWorkflowBackend,
        request_context: RequestContext | None = None,
    ) -> None:
        self.project_store = ProjectStore(projects_root)
        self.backend = backend
        self.request_context = request_context

    def prepare(self, context: RuntimeStageContext) -> StagePreparationResult:
        try:
            config = self._config(context)
            config.validate_workflow_inputs(input_ports=context.node.inputs)
            spec = self._workflow_spec(context, config=config)
        except (RuntimeError, ValueError) as exc:
            return StagePreparationResult(
                status="failed",
                message="Dataset workflow configuration is invalid.",
                failure_category="validation",
                error=str(exc)[:4000],
            )
        return StagePreparationResult(
            status="waiting_external",
            message="Dataset workflow is ready for platform submission.",
            external_task=ExternalTaskIntent(
                provider="pyromind.studio.dataset",
                metadata={"output_uri": spec.output_directory},
            ),
        )

    def submit(
        self, context: RuntimeStageContext, task_run: TaskRun
    ) -> StageSubmissionResult:
        spec = self._workflow_spec(context, task_run=task_run)
        existing = self._find_task(spec)
        if existing is not None:
            return StageSubmissionResult(
                status="submitted",
                message=f"Recovered existing Dataset task {existing}.",
                provider_run_id=existing,
            )
        if task_run.status == "unknown":
            raise DatasetRuntimeError(
                "Could not confirm whether the Dataset workflow was submitted."
            )
        task_id = self.backend.submit_workflow(
            spec=spec, request_context=self.request_context
        )
        return StageSubmissionResult(
            status="submitted",
            message=f"Dataset task submitted: {task_id}.",
            provider_run_id=task_id,
        )

    def poll(self, context: RuntimeStageContext, task_run: TaskRun) -> StagePollResult:
        task_id = task_run.provider_run_id
        if task_id is None:
            raise DatasetRuntimeError("Dataset TaskRun has no provider task ID")
        spec = self._workflow_spec(context, task_run=task_run)
        state = self.backend.query_task(
            task_id=task_id, spec=spec, request_context=self.request_context
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
        if state.output_uri != spec.output_directory:
            return StagePollResult(
                status="failed",
                message="Dataset succeeded without its expected Storage outputs.",
                failure_category="validation",
            )
        artifacts = []
        for port, filename in zip(
            _WORKFLOW_OUTPUTS, spec.output_filenames, strict=True
        ):
            uri = f"{spec.output_directory}/{filename}"
            artifacts.append(
                RuntimeArtifact(
                    port=port,
                    payload=StageArtifactPayload(
                        kind=context.node.outputs[port].artifact_type,
                        content={
                            "schema_version": 1,
                            "uri": uri,
                            "task_id": task_id,
                            "task_name": spec.task_name,
                        },
                    ),
                    metadata={"source_uri": uri, "provider_task_id": task_id},
                )
            )
        return StagePollResult(
            status="completed", message=state.message, artifacts=artifacts
        )

    def cancel(
        self, context: RuntimeStageContext, task_run: TaskRun
    ) -> StageCancellationResult:
        spec = self._workflow_spec(context, task_run=task_run)
        task_id = task_run.provider_run_id or self._find_task(spec)
        if task_id is None:
            return StageCancellationResult(
                status="failed",
                message="Could not locate the Dataset task to confirm cancellation.",
            )
        self.backend.cancel_task(
            task_id=task_id, spec=spec, request_context=self.request_context
        )
        return StageCancellationResult(
            status="canceled", message=f"Dataset task canceled: {task_id}."
        )

    def _find_task(self, spec: DatasetWorkflowSpec) -> str | None:
        return self.backend.find_task_by_name(
            name=spec.task_name, spec=spec, request_context=self.request_context
        )

    def _config(self, context: RuntimeStageContext) -> DatasetWorkflowConfig:
        return DatasetWorkflowConfig.model_validate(
            context.node.config.get("dataset_runtime")
        )

    def _workflow_spec(
        self,
        context: RuntimeStageContext,
        *,
        task_run: TaskRun | None = None,
        config: DatasetWorkflowConfig | None = None,
    ) -> DatasetWorkflowSpec:
        config = config or self._config(context)
        missing = set(_WORKFLOW_OUTPUTS).difference(context.node.outputs)
        if missing:
            raise DatasetRuntimeError(
                "Missing Dataset output ports: " + ", ".join(sorted(missing))
            )
        source = _workflow_input_value(context.inputs.get("source_batch"))
        if not isinstance(source, str):
            raise DatasetRuntimeError("Dataset source_batch must be a Storage path")
        inputs = {
            **context.inputs,
            "source_batch": normalize_workspace_path(source, field="source_batch"),
        }
        workflow, output_uri = config.render_workflow(
            inputs=inputs,
            runtime_values={
                "project_id": context.project_id,
                "release_id": context.release_id,
                "graph_run_id": context.graph_run_id,
                "stage_run_id": context.stage_run_id,
                "node_id": context.node.id,
            },
        )
        project = self.project_store.load(context.project_id, owner_user_id=None)
        return DatasetWorkflowSpec(
            task_name=(
                _task_name(task_run.idempotency_key, prefix="echomind-dataset")
                if task_run is not None
                else "echomind-dataset-pending"
            ),
            conversation_id=project.project.conversation_id,
            cluster=project.project.cluster,
            owner_user_id=project.project.owner_user_id,
            workflow=workflow,
            output_uri=output_uri,
            output_node_id=config.output_node_id,
            output_parameter=config.output_parameter,
            train_filename=config.train_filename,
            val_filename=config.val_filename,
        )


class TracerBatchReference(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    resource_id: str | None = Field(default=None, max_length=128)
    batch_id: str = Field(min_length=1, max_length=256)
    batch_index: int | None = Field(default=None, ge=0)
    uri: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="before")
    @classmethod
    def infer_batch_index(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        if payload.get("batch_index") is not None:
            return payload
        batch_id = payload.get("batch_id")
        if batch_id is None:
            return payload
        try:
            index = int(batch_id)
        except (TypeError, ValueError):
            return payload
        if index < 0:
            raise ValueError("Tracer batch index must not be negative")
        payload["batch_index"] = index
        return payload


class DatasetExport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    train_uri: str = Field(min_length=1, max_length=4000)
    val_uri: str = Field(min_length=1, max_length=4000)
    train_samples: int = Field(ge=0)
    val_samples: int = Field(ge=0)
    source_uri: str | None = Field(default=None, max_length=4000)
    source_files: list[str] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_samples(self) -> DatasetExport:
        if self.train_samples + self.val_samples == 0:
            raise ValueError("Dataset export must contain at least one sample")
        return self


class DatasetSourceProvider(Protocol):
    def materialize(
        self,
        *,
        source: Any,
        context: RuntimeStageContext,
    ) -> DatasetExport: ...

    def release(self, *, source: Any, context: RuntimeStageContext) -> None: ...


class StorageDatasetSourceProvider:
    def materialize(
        self,
        *,
        source: Any,
        context: RuntimeStageContext,
    ) -> DatasetExport:
        del context
        payload = _materialized_source_payload(source)
        if payload is None:
            raise DatasetRuntimeError(
                "Dataset source must provide train_uri and val_uri, or a "
                "materialization manifest."
            )

        train_samples = _sample_count(payload, "train_samples")
        val_samples = _sample_count(payload, "val_samples")
        stats = payload.get("stats")
        source_files = payload.get("source_files")
        return DatasetExport(
            train_uri=_required_string(payload, "train_uri"),
            val_uri=_required_string(payload, "val_uri"),
            train_samples=train_samples,
            val_samples=val_samples,
            source_uri=(
                payload.get("source_uri")
                if isinstance(payload.get("source_uri"), str)
                else None
            ),
            source_files=(
                [str(item) for item in source_files]
                if isinstance(source_files, list)
                else []
            ),
            stats=dict(stats) if isinstance(stats, Mapping) else {},
        )

    def release(self, *, source: Any, context: RuntimeStageContext) -> None:
        del source, context


class TracerBatchController(Protocol):
    def claim(self, batch: TracerBatchReference) -> str | None: ...

    def release(self, batch: TracerBatchReference) -> None: ...


class DatasetBatchExporter(Protocol):
    def export(
        self,
        *,
        batch: TracerBatchReference,
        context: RuntimeStageContext,
    ) -> DatasetExport: ...


class TracerBatchSourceProvider:
    def __init__(
        self,
        *,
        controller: TracerBatchController,
        exporter: DatasetBatchExporter,
        expected_tracer_resource_id: str | None = None,
    ) -> None:
        self.controller = controller
        self.exporter = exporter
        self.expected_tracer_resource_id = expected_tracer_resource_id

    def materialize(
        self,
        *,
        source: Any,
        context: RuntimeStageContext,
    ) -> DatasetExport:
        batch = _tracer_batch_reference(source)
        if batch is None:
            raise DatasetRuntimeError("Dataset input is not a Tracer batch.")
        if (
            self.expected_tracer_resource_id
            and batch.resource_id
            and batch.resource_id != self.expected_tracer_resource_id
        ):
            raise DatasetRuntimeError(
                f"Expected Tracer {self.expected_tracer_resource_id}, "
                f"received {batch.resource_id}"
            )
        self.controller.claim(batch)
        try:
            exported = self.exporter.export(batch=batch, context=context)
        except Exception:
            self.release(source=batch, context=context)
            raise
        return exported.model_copy(
            update={
                "source_uri": batch.uri,
                "stats": {
                    **exported.stats,
                    "tracer_resource_id": batch.resource_id,
                    "batch_id": batch.batch_id,
                    "batch_index": batch.batch_index,
                },
            }
        )

    def release(self, *, source: Any, context: RuntimeStageContext) -> None:
        del context
        batch = _tracer_batch_reference(source)
        if batch is not None:
            self.controller.release(batch)


class CompositeDatasetSourceProvider:
    def __init__(
        self,
        *,
        storage: StorageDatasetSourceProvider | None = None,
        tracer: TracerBatchSourceProvider | None = None,
    ) -> None:
        self.storage = storage or StorageDatasetSourceProvider()
        self.tracer = tracer
        self._active_sources: dict[str, DatasetSourceProvider] = {}

    def materialize(
        self,
        *,
        source: Any,
        context: RuntimeStageContext,
    ) -> DatasetExport:
        provider = self._provider_for(source)
        exported = provider.materialize(source=source, context=context)
        self._active_sources[context.stage_run_id] = provider
        return exported

    def release(self, *, source: Any, context: RuntimeStageContext) -> None:
        provider = self._active_sources.pop(context.stage_run_id, None)
        if provider is not None:
            provider.release(source=source, context=context)

    def _provider_for(self, source: Any) -> DatasetSourceProvider:
        if _tracer_batch_reference(source) is not None:
            if self.tracer is None:
                raise DatasetRuntimeError(
                    "Tracer batch input is not configured for this runtime."
                )
            return self.tracer
        return self.storage


class HttpTracerBatchController:
    def __init__(
        self,
        *,
        endpoint: str,
        admin_key: str,
        timeout_seconds: float = 30,
        client: httpx.Client | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.admin_key = admin_key
        self.timeout_seconds = timeout_seconds
        self._client = client

    def claim(self, batch: TracerBatchReference) -> str | None:
        payload = self._post_batch_action(batch, "claim")
        return str(payload["note"]) if payload.get("note") else None

    def release(self, batch: TracerBatchReference) -> None:
        self._post_batch_action(batch, "release")

    def _post_batch_action(
        self,
        batch: TracerBatchReference,
        action: str,
    ) -> Mapping[str, Any]:
        if batch.batch_index is None:
            raise DatasetRuntimeError(
                f"Tracer batch has no numeric index for {action}: {batch.batch_id}"
            )
        client = self._client or httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=True,
        )
        try:
            response = client.post(
                f"{self.endpoint}/admin/batches/{batch.batch_index}/{action}",
                headers={"Authorization": f"Bearer {self.admin_key}"},
                json={},
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise DatasetRuntimeError(
                f"Tracer batch {action} failed: {type(exc).__name__}"
            ) from exc
        finally:
            if self._client is None:
                client.close()
        if not isinstance(payload, dict) or payload.get("ok") is False:
            raise DatasetRuntimeError(
                f"Tracer batch {action} returned an invalid payload"
            )
        return payload


class DatasetRuntimeExecutor:
    def __init__(
        self,
        *,
        source_provider: DatasetSourceProvider | None = None,
        workflow_executor: PlatformDatasetRuntimeExecutor | None = None,
    ) -> None:
        self.source_provider = source_provider or CompositeDatasetSourceProvider()
        self.workflow_executor = workflow_executor

    def prepare(self, context: RuntimeStageContext) -> StagePreparationResult:
        if context.node.type != "dataset":
            return StagePreparationResult(
                status="failed",
                message="Dataset executor received a non-Dataset node.",
                failure_category="validation",
                error=f"Unexpected node type: {context.node.type}",
            )
        if "dataset_runtime" in context.node.config:
            if self.workflow_executor is None:
                return StagePreparationResult(
                    status="failed",
                    message="Dataset workflow execution is unavailable.",
                    failure_category="infrastructure",
                    error="No Dataset workflow executor is configured.",
                )
            return self.workflow_executor.prepare(context)
        source = _dataset_source(context.inputs)
        if source is None:
            return StagePreparationResult(
                status="failed",
                message="Dataset input source is missing.",
                failure_category="validation",
                error=(
                    "Dataset input must bind source_batch to a Storage dataset "
                    "source or Tracer batch."
                ),
            )
        required_ports = {"train_messages", "val_messages"}
        missing = required_ports.difference(context.node.outputs)
        if missing:
            return StagePreparationResult(
                status="failed",
                message="Dataset node does not expose standard train/validation ports.",
                failure_category="validation",
                error="Missing outputs: " + ", ".join(sorted(missing)),
            )
        try:
            exported = self.source_provider.materialize(
                source=source,
                context=context,
            )
        except Exception as exc:
            return StagePreparationResult(
                status="failed",
                message="Dataset source materialization failed.",
                failure_category=(
                    "validation"
                    if isinstance(exc, (ValueError, DatasetRuntimeError))
                    else "external"
                ),
                error=f"{type(exc).__name__}: {exc}"[:4000],
            )

        return StagePreparationResult(
            status="completed",
            message=(
                f"Dataset materialized {exported.train_samples} train and "
                f"{exported.val_samples} validation samples."
            ),
            artifacts=[
                _dataset_artifact(
                    context=context,
                    port="train_messages",
                    export=exported,
                    uri=exported.train_uri,
                    samples=exported.train_samples,
                    split="train",
                ),
                _dataset_artifact(
                    context=context,
                    port="val_messages",
                    export=exported,
                    uri=exported.val_uri,
                    samples=exported.val_samples,
                    split="validation",
                ),
            ],
        )

    def submit(
        self,
        context: RuntimeStageContext,
        task_run: TaskRun,
    ) -> StageSubmissionResult:
        return self._platform_executor().submit(context, task_run)

    def poll(
        self,
        context: RuntimeStageContext,
        task_run: TaskRun,
    ) -> StagePollResult:
        return self._platform_executor().poll(context, task_run)

    def cancel(
        self,
        context: RuntimeStageContext,
        task_run: TaskRun,
    ) -> StageCancellationResult:
        if "dataset_runtime" in context.node.config:
            return self._platform_executor().cancel(context, task_run)
        source = _dataset_source(context.inputs)
        if source is None:
            return StageCancellationResult(
                status="canceled",
                message="Dataset canceled without an active source.",
            )
        try:
            self.source_provider.release(source=source, context=context)
        except Exception as exc:
            return StageCancellationResult(
                status="failed",
                message=f"Dataset canceled, but source release failed: {exc}"[:4000],
            )
        return StageCancellationResult(
            status="canceled",
            message="Dataset canceled and source released.",
        )

    def _platform_executor(self) -> PlatformDatasetRuntimeExecutor:
        if self.workflow_executor is None:
            raise DatasetRuntimeError("No Dataset workflow executor is configured")
        return self.workflow_executor


def _dataset_artifact(
    *,
    context: RuntimeStageContext,
    port: str,
    export: DatasetExport,
    uri: str,
    samples: int,
    split: str,
) -> RuntimeArtifact:
    output = context.node.outputs[port]
    return RuntimeArtifact(
        port=port,
        payload=StageArtifactPayload(
            kind=output.artifact_type,
            content={
                "schema_version": 1,
                "split": split,
                "uri": uri,
                "sample_count": samples,
                "source_uri": export.source_uri,
                "train_uri": export.train_uri,
                "val_uri": export.val_uri,
                "train_samples": export.train_samples,
                "val_samples": export.val_samples,
                "source_files": export.source_files,
                "stats": export.stats,
            },
        ),
        metadata={"source_uri": uri},
    )


def _dataset_source(inputs: Mapping[str, Any]) -> Any | None:
    for name in ("source_batch", "dataset_source", "dataset", "tracer_batch"):
        if name in inputs and inputs[name] is not None:
            return inputs[name]
    return None


def _tracer_batch_reference(value: Any) -> TracerBatchReference | None:
    if isinstance(value, TracerBatchReference):
        return value
    if isinstance(value, str):
        if "batch" not in value.lower():
            return None
        try:
            return TracerBatchReference(batch_id=value)
        except ValueError:
            return None
    if not isinstance(value, Mapping):
        return None
    payload = dict(value)
    kind = payload.get("kind")
    if kind == "dataset.materialized" or (
        "train_uri" in payload and "val_uri" in payload
    ):
        return None
    if "batch_id" not in payload:
        return None
    try:
        return TracerBatchReference.model_validate(payload)
    except ValueError:
        return None


def _materialized_source_payload(source: Any) -> Mapping[str, Any] | None:
    if isinstance(source, Mapping):
        payload = dict(source)
    elif isinstance(source, str):
        manifest = _read_manifest(source)
        if manifest is None:
            return None
        payload = dict(manifest)
    else:
        return None
    nested = payload.get("materialized")
    if isinstance(nested, Mapping):
        payload = {**payload, **dict(nested)}
    if "train_uri" not in payload or "val_uri" not in payload:
        manifest_uri = payload.get("manifest_uri")
        if isinstance(manifest_uri, str):
            return _read_manifest(manifest_uri)
        return None
    return payload


def _read_manifest(uri: str) -> Mapping[str, Any] | None:
    path_value = uri.removeprefix("file://")
    if "://" in path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if "train_uri" not in payload or "val_uri" not in payload:
        raise DatasetRuntimeError(
            "Dataset materialization manifest must contain train_uri and val_uri."
        )
    return payload


def _required_string(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise DatasetRuntimeError(f"Dataset source is missing {name}.")
    return value


def _sample_count(payload: Mapping[str, Any], name: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DatasetRuntimeError(
            f"Dataset source {name} must be a non-negative integer."
        )
    return value
