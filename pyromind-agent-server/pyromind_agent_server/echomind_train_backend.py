from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, closing, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

from pydantic import ValidationError
from pyromind_sdk import PyroMindAPIClient
from pyromind_sdk.client.base import PyroMindAPIError
from pyromind_sdk.client.models import TrainingTaskResponse

from harness_adapter.workflow_submission import PlatformWorkflowSubmission
from pyromind_agent_server.echomind_runtime_access import platform_request_token
from pyromind_agent_server.echomind_runtime_contracts import (
    RuntimeCredentialUnavailableError,
    RuntimeSubmissionNotAttemptedError,
)
from pyromind_agent_server.echomind_runtime_credentials import (
    RuntimeCredentialStore,
    RuntimeCredentialStoreError,
)
from pyromind_agent_server.echomind_train_runtime import (
    TrainRuntimeError,
    TrainTaskState,
    TrainTaskStatus,
    TrainWorkflowSpec,
    prepare_workflow_for_submission,
)
from pyromind_agent_server.echomind_workflow_storage import (
    build_runtime_storage,
)
from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_common.tools.utils.platform_env import (
    platform_cluster_route,
    platform_cluster_target,
)
from pyromind_common.tools.utils.pyromind_api_client import get_pyromind_api_client
from pyromind_common.tools.utils.training_source import (
    WANDB_BUILDER_NODE_TYPES,
    wandb_environment,
)
from pyromind_common.tools.workflow.task_submission import (
    create_workflow_api_client,
    submit_workflow_task,
)
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.storage_paths import to_storage_api_path
from pyromind_runtime.domain.task_correlation import TaskCorrelation
from pyromind_runtime.domain.workflow_execution import WorkflowExecutionRequest
from pyromind_runtime.infrastructure.file_product_store import (
    FileProductStore,
    ProductStoreError,
)
from pyromind_runtime.ports.storage import (
    EvaluationStorageMissingError,
    WorkflowStorageReader,
    WorkflowStorageResolver,
)


_ACTIVE_TASK_STATUSES = frozenset({"new", "pending", "running"})
_SUCCEEDED_TASK_STATUSES = frozenset({"succeeded", "success", "completed"})
_FAILED_TASK_STATUSES = frozenset({"failed", "error"})
_CANCELED_TASK_STATUSES = frozenset({"canceled", "cancelled", "terminated"})
_WANDB_KEY_MARKER = "WANDB_API_KEY"
_WANDB_KEY_PATTERN = re.compile(r"wandb_[A-Za-z0-9_-]{20,}|[A-Za-z0-9]{40}")
_suppress_sdk_logs = ContextVar("suppress_wandb_sdk_logs", default=False)


def _filter_sdk_log(_record: logging.LogRecord) -> bool:
    return not _suppress_sdk_logs.get()


logging.getLogger("pyromind_sdk").addFilter(_filter_sdk_log)


@contextmanager
def _without_sensitive_sdk_logs(enabled: bool) -> Iterator[None]:
    # The SDK's request and response logger does not mask wandb_api_key.
    token = _suppress_sdk_logs.set(enabled)
    try:
        yield
    finally:
        _suppress_sdk_logs.reset(token)


@contextmanager
def _read_task_status(redact_wandb_key: bool) -> Iterator[None]:
    with _without_sensitive_sdk_logs(redact_wandb_key):
        try:
            yield
        except PyroMindAPIError as exc:
            if redact_wandb_key:
                raise PyroMindAPIError(
                    message=_WANDB_KEY_PATTERN.sub("[REDACTED]", str(exc)),
                    status_code=exc.status_code,
                ) from None
            raise
        except Exception as exc:
            if redact_wandb_key:
                raise TrainRuntimeError(
                    _WANDB_KEY_PATTERN.sub("[REDACTED]", str(exc))
                ) from None
            raise


class PlatformTrainWorkflowBackend:
    def __init__(
        self,
        *,
        task_kind: str = "echomind_train",
        recorded_storage: WorkflowStorageResolver | None = None,
        credential_store: RuntimeCredentialStore | None = None,
        conversations_dir: Path | str | None = None,
        projects_root: Path | str | None = None,
    ) -> None:
        self.task_kind = task_kind
        self.recorded_storage = recorded_storage
        self.credential_store = credential_store
        self.projects_root = Path(projects_root) if projects_root is not None else None
        self.conversations_dir = (
            Path(conversations_dir) if conversations_dir is not None else None
        )

    def _access_key(
        self, spec: TrainWorkflowSpec, request_context: RequestContext | None
    ) -> str | None:
        if platform_request_token(request_context) is not None:
            return None
        if self.credential_store is None or spec.owner_user_id is None:
            return None
        try:
            return self.credential_store.load(
                spec.owner_user_id, platform_cluster_route(spec.cluster)
            )
        except RuntimeCredentialStoreError as exc:
            raise RuntimeCredentialUnavailableError(
                "The project owner's Pyromind credential could not be loaded."
            ) from exc

    def _task_read_scope(self, spec: TrainWorkflowSpec) -> AbstractContextManager[None]:
        return _read_task_status(spec.wandb_credential_task_id is not None)

    def find_task_by_name(
        self,
        *,
        name: str,
        spec: TrainWorkflowSpec,
        request_context: RequestContext | None,
    ) -> str | None:
        client = _client(
            spec=spec,
            request_context=request_context,
            access_key=self._access_key(spec, request_context),
        )
        try:
            with self._task_read_scope(spec):
                matches = [task for task in client.studio.list() if task.name == name]
        finally:
            client.close()
        if not matches:
            return None
        if len(matches) != 1:
            raise TrainRuntimeError(
                f"Multiple platform tasks match the runtime task name {name}."
            )
        return str(matches[0].task_id)

    def submit_workflow(
        self,
        *,
        spec: TrainWorkflowSpec,
        request_context: RequestContext | None,
    ) -> str:
        try:
            workflow = _workflow(spec=spec, request_context=request_context)
            workflow["name"] = spec.task_name
            workflow = prepare_workflow_for_submission(
                workflow,
                spec=spec,
            )
            sensitive_values = self._bind_submission_credentials(
                workflow, spec=spec, request_context=request_context
            )
            client = _client(
                spec=spec,
                request_context=request_context,
                access_key=self._access_key(spec, request_context),
            )
        except RuntimeCredentialUnavailableError:
            raise
        except Exception as exc:
            raise RuntimeSubmissionNotAttemptedError(str(exc)) from exc
        try:
            with _without_sensitive_sdk_logs(bool(sensitive_values)):
                response = submit_workflow_task(
                    client=client,
                    workflow=workflow,
                    name=spec.task_name,
                    conversation_id=spec.conversation_id,
                    retry_create=False,
                    task_kind=self.task_kind,
                    origin=TaskCorrelation(
                        conversation_id=spec.conversation_id,
                        task_kind=self.task_kind,
                    ).origin,
                )
        except PyroMindAPIError as exc:
            if exc.status_code in {401, 403}:
                raise RuntimeCredentialUnavailableError(
                    "Pyromind rejected the project owner's workflow credential."
                ) from None
            if sensitive_values:
                raise PyroMindAPIError(
                    message=_redact_keys(str(exc), sensitive_values),
                    status_code=exc.status_code,
                ) from None
            raise
        except Exception as exc:
            if sensitive_values:
                raise TrainRuntimeError(
                    _redact_keys(str(exc), sensitive_values)
                ) from None
            raise
        finally:
            client.close()
        return str(response.task_id)

    def _bind_submission_credentials(
        self,
        workflow: dict[str, Any],
        *,
        spec: TrainWorkflowSpec,
        request_context: RequestContext | None,
    ) -> tuple[str, ...]:
        task_id = spec.wandb_credential_task_id
        if task_id is None:
            return ()
        archived = self._archived_credential_receipt(
            spec, request_context, task_id=task_id
        )
        source_nodes = _wandb_builder_configs(archived.workflow)
        targets = [
            (node_id, config)
            for node_id, config in _wandb_builder_configs(workflow).items()
            if config.get("wandb_api_key") == _WANDB_KEY_MARKER
        ]
        if not targets:
            raise TrainRuntimeError(
                "The Train workflow has no W&B credential reference."
            )
        keys = {
            node_id: source_nodes.get(node_id, {}).get("wandb_api_key")
            for node_id, _ in targets
        }
        remote = [
            node_id for node_id, value in keys.items() if value == _WANDB_KEY_MARKER
        ]
        if remote:
            with closing(
                _client(
                    spec=spec,
                    request_context=request_context,
                    access_key=self._access_key(spec, request_context),
                )
            ) as client:
                for node_id in remote:
                    keys[node_id] = _studio_wandb_key(client, task_id, node_id)
        bound: list[str] = []
        for node_id, config in targets:
            key = keys[node_id]
            if not isinstance(key, str) or not _WANDB_KEY_PATTERN.fullmatch(key):
                raise TrainRuntimeError(
                    "The referenced task has no W&B credential for the matching node."
                )
            config["wandb_api_key"] = key
            bound.append(key)
        return tuple(bound)

    def _archived_credential_receipt(
        self,
        spec: TrainWorkflowSpec,
        request_context: RequestContext | None,
        *,
        task_id: str,
        task_kind: str | None = None,
    ) -> WorkflowExecutionRequest:
        owner = spec.owner_user_id
        if self.conversations_dir is None or not owner:
            raise TrainRuntimeError("The credential source is unavailable.")
        if request_context is not None and (
            request_context.user_id != owner
            or (request_context.x_cluster or spec.cluster) != spec.cluster
        ):
            raise TrainRuntimeError("The credential source is outside this Project.")

        try:
            conversation_id = UUID(spec.conversation_id).hex
            store = FileProductStore(self.conversations_dir / conversation_id)
            store.authorize(owner, ECHOMIND_PRODUCT_PROFILE.profile_id)
            if store.load_x_cluster() != spec.cluster:
                raise PermissionError("conversation cluster mismatch")
            archived = [
                request
                for request in store.load_workflow_execution_requests().values()
                if request.task_id == task_id
                and request.conversation_id == conversation_id
                and request.status == "submitted"
                and (
                    task_kind is None
                    or (
                        request.correlation is not None
                        and request.correlation.task_kind == task_kind
                    )
                )
            ]
        except (OSError, ProductStoreError, ValidationError, ValueError):
            raise TrainRuntimeError(
                "The credential source is unavailable for this Project."
            ) from None
        if len(archived) != 1:
            raise TrainRuntimeError("The referenced credential task was not found.")
        receipt = archived[0]
        if task_kind is None and (
            receipt.request_id.startswith("sdk:")
            or any(
                config.get("wandb_api_key") == _WANDB_KEY_MARKER
                for config in _wandb_builder_configs(receipt.workflow).values()
            )
        ):
            _verify_train_credential_receipt(store, receipt, task_id, spec)
        if task_kind == "data_preparation" and receipt.request_id.startswith("sdk:"):
            _verify_dataset_credential_receipt(store, receipt, spec)
        return receipt

    def query_task(
        self,
        *,
        task_id: str,
        spec: TrainWorkflowSpec,
        request_context: RequestContext | None,
    ) -> TrainTaskState:
        access_key = self._access_key(spec, request_context)
        client = _client(
            spec=spec, request_context=request_context, access_key=access_key
        )
        redact_wandb_key = spec.wandb_credential_task_id is not None
        try:
            with self._task_read_scope(spec):
                task = client.studio.get_task(task_id)
                status = _status(task)
                output_uri = (
                    _node_output_uri(client=client, task_id=task_id, spec=spec)
                    if status == "succeeded" and spec.artifact_selection is None
                    else None
                )
        finally:
            client.close()
        if status == "succeeded" and spec.artifact_selection is not None:
            output_uri = _verified_artifact_uri(
                spec,
                request_context=request_context,
                access_key=access_key,
                projects_root=self.projects_root,
                credential_store=self.credential_store,
                recorded_storage=self.recorded_storage,
            )
            if output_uri is None:
                return TrainTaskState(
                    status="failed",
                    message=f"Platform task {task_id} finished without a valid "
                    f"{spec.artifact_selection} at {spec.output_uri}.",
                    failure_category="validation",
                )
        return TrainTaskState(
            status=status,
            message=_task_message(
                task=task,
                status=status,
                task_label="Eval"
                if self.task_kind == "echomind_evaluator"
                else "Train",
                redact_wandb_key=redact_wandb_key,
            ),
            output_uri=output_uri or spec.output_uri,
            failure_category="canceled" if status == "canceled" else "external",
        )

    def cancel_task(
        self,
        *,
        task_id: str,
        spec: TrainWorkflowSpec,
        request_context: RequestContext | None,
    ) -> None:
        client = _client(
            spec=spec,
            request_context=request_context,
            access_key=self._access_key(spec, request_context),
        )
        try:
            with self._task_read_scope(spec):
                if _status(client.studio.get_task(task_id)) in {"pending", "running"}:
                    stopped = client.studio.stop(task_id)
                    if _status(stopped) in {"pending", "running"}:
                        raise TrainRuntimeError(f"Task {task_id} is still stopping.")
        finally:
            client.close()


def _verified_artifact_uri(
    spec: TrainWorkflowSpec,
    *,
    request_context: RequestContext | None,
    access_key: str | None,
    projects_root: Path | str | None = None,
    credential_store: RuntimeCredentialStore | None = None,
    recorded_storage: WorkflowStorageResolver | None = None,
) -> str | None:
    assert spec.output_uri is not None
    storage = build_runtime_storage(
        spec=spec,
        request_context=request_context,
        access_key=access_key,
        projects_root=projects_root,
        credential_store=credential_store,
        recorded_storage=recorded_storage,
    )
    path = to_storage_api_path(spec.output_uri)
    if spec.artifact_selection == "evaluation_report":
        try:
            report = json.loads(storage.read(path))
        except (EvaluationStorageMissingError, ValueError):
            return None
        if (
            isinstance(report, dict)
            and report.get("status") == "succeeded"
            and isinstance(report.get("metrics"), dict)
            and report["metrics"]
        ):
            return spec.output_uri
        return None

    files = storage.list(path)
    if spec.artifact_selection == "merged_model":
        return spec.output_uri if _has_merged_model(storage, path, files) else None
    checkpoints = [
        (int(match[1]), name)
        for name in files
        if (match := re.fullmatch(r"checkpoint-(\d+)", name))
    ]
    for _, name in sorted(checkpoints, reverse=True):
        checkpoint = str(PurePosixPath(path) / name)
        if _is_lora_adapter(storage.list(checkpoint)):
            return f"{spec.output_uri.rstrip('/')}/{name}"
    return spec.output_uri if _is_lora_adapter(files) else None


def _has_merged_model(
    storage: WorkflowStorageReader, path: str, files: set[str]
) -> bool:
    if "config.json" not in files or "adapter_config.json" in files:
        return False
    directory = PurePosixPath(path)
    try:
        config = json.loads(storage.read(str(directory / "config.json")))
        if not isinstance(config, dict) or not config:
            return False
        for weights in ("model.safetensors", "pytorch_model.bin"):
            if weights in files:
                return True
            index_name = f"{weights}.index.json"
            if index_name in files:
                index = json.loads(storage.read(str(directory / index_name)))
                weight_map = (
                    index.get("weight_map") if isinstance(index, dict) else None
                )
                return bool(
                    isinstance(weight_map, dict)
                    and weight_map
                    and all(
                        isinstance(name, str)
                        and PurePosixPath(name).name == name
                        and name.endswith(PurePosixPath(weights).suffix)
                        and name in files
                        for name in weight_map.values()
                    )
                )
    except (EvaluationStorageMissingError, ValueError):
        return False
    return False


def _is_lora_adapter(files: set[str]) -> bool:
    return "adapter_config.json" in files and bool(
        files & {"adapter_model.safetensors", "adapter_model.bin"}
    )


def _verify_dataset_credential_receipt(
    store: FileProductStore, receipt: WorkflowExecutionRequest, spec: TrainWorkflowSpec
) -> None:
    snapshot = store.load_snapshot()

    def signed_success(item: WorkflowExecutionRequest) -> bool:
        c = item.correlation
        return bool(
            c
            and c.type == "main"
            and c.task_kind == "data_preparation"
            and c.cluster == spec.cluster
            and c.conversation_id == spec.conversation_id
            and item.request_id == "sdk:" + str(item.task_id)
            and item.status == "submitted"
            and store.verify_task_correlation(c)
            and any(
                t.task_id == item.task_id
                and t.kind == c.task_kind
                and t.correlation == c
                and t.status == "succeeded"
                and not t.graph_run_id
                and t.x_cluster == spec.cluster
                for t in snapshot.external_tasks
            )
        )

    if not signed_success(receipt):
        raise TrainRuntimeError(
            "The Dataset credential source has no signed success receipt."
        )
    source = receipt
    if receipt.dataset_recipe and receipt.dataset_recipe.replay_of_task_id:
        parent = receipt.dataset_recipe.replay_of_task_id
        original = store.load_workflow_execution_requests().get("sdk:" + parent)
        if (
            not original
            or not signed_success(original)
            or (original.dataset_recipe and original.dataset_recipe.replay_of_task_id)
            or original.correlation is None
            or receipt.correlation is None
            or receipt.correlation.execution
            not in (None, original.correlation.execution)
        ):
            raise TrainRuntimeError("The Dataset source has no signed replay lineage.")
        source = original
    ref = source.correlation.execution if source.correlation else None
    project = spec.template_values.get("project_id", "").replace("-", "")
    if (
        not ref
        or project != ref.project_id.replace("-", "")
        or not any(
            run.role_id == "main"
            and run.project_id == ref.project_id
            and run.task_id == ref.task_id
            and run.plan_sequence == ref.plan_sequence
            and run.plan_attempt == ref.attempt
            and (run.stage_snapshot or {}).get("kind") == "dataset"
            and source.task_id in (run.platform_task_id, *run.platform_task_ids)
            for run in snapshot.experiment_executions or ()
        )
    ):
        raise TrainRuntimeError(
            "The Dataset credential source is outside this Project execution."
        )


def _verify_train_credential_receipt(
    store: FileProductStore,
    receipt: WorkflowExecutionRequest,
    task_id: str,
    spec: TrainWorkflowSpec,
) -> None:
    correlation = receipt.correlation
    if not (
        correlation
        and correlation.type == "main"
        and correlation.execution
        and correlation.conversation_id == receipt.conversation_id
        and correlation.task_kind == "workflow_run"
        and store.verify_task_correlation(correlation)
    ):
        raise TrainRuntimeError(
            "The Train credential source has no signed Project receipt."
        )
    reference = correlation.execution
    snapshot = store.load_snapshot()
    if not any(
        task.task_id == task_id
        and task.correlation == correlation
        and task.kind == "workflow_run"
        and task.status == "succeeded"
        and task.x_cluster == spec.cluster
        for task in snapshot.external_tasks
    ) or not any(
        run.project_id == reference.project_id
        and run.task_id == reference.task_id
        and run.plan_sequence == reference.plan_sequence
        and run.plan_attempt == reference.attempt
        and run.role_id == "main"
        and (run.stage_snapshot or {}).get("kind") == "train"
        and task_id in (run.platform_task_ids or (run.platform_task_id,))
        for run in snapshot.experiment_executions or ()
    ):
        raise TrainRuntimeError(
            "The Train credential source does not match its Project execution."
        )


def _studio_wandb_key(
    client: PyroMindAPIClient, task_id: str, node_id: str
) -> str | None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", node_id):
        raise TrainRuntimeError("Invalid W&B builder node ID.")
    try:
        with _without_sensitive_sdk_logs(True):
            output = client.studio.get_node_output(task_id, node_id)
    except PyroMindAPIError as exc:
        if exc.status_code in {401, 403}:
            raise RuntimeCredentialUnavailableError(
                "Open this run while signed in to refresh Studio access."
            ) from None
        raise TrainRuntimeError(
            "Studio could not read the source W&B node output."
        ) from None
    except Exception:
        raise TrainRuntimeError(
            "Studio could not read the source W&B node output."
        ) from None
    parameters = (output or {}).get("parameters")
    if not isinstance(parameters, list):
        return None
    values = [
        row.get("value")
        for row in parameters
        if isinstance(row, dict) and row.get("name") == "wandb_config"
    ]
    if len(values) != 1:
        return None
    value = values[0]
    if isinstance(value, str) and value.lstrip().startswith("{"):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return wandb_environment({"wandb_config": value}).get("WANDB_API_KEY")


def _wandb_builder_configs(workflow: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return {}
    builders: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            continue
        data = node.get("data")
        if (
            not isinstance(data, dict)
            or data.get("nodeType") not in WANDB_BUILDER_NODE_TYPES
        ):
            continue
        config = data.get("config")
        if not isinstance(config, dict):
            continue
        node_id = node["id"]
        if node_id in builders:
            raise TrainRuntimeError("The workflow has duplicate W&B node IDs.")
        builders[node_id] = config
    return builders


def _redact_keys(message: str, keys: tuple[str, ...]) -> str:
    for key in keys:
        message = message.replace(key, "[REDACTED]")
    return message


def _workflow(
    *,
    spec: TrainWorkflowSpec,
    request_context: RequestContext | None,
) -> dict[str, Any]:
    if spec.workflow is not None:
        return deepcopy(spec.workflow)
    if spec.workflow_path is None:
        raise TrainRuntimeError("Train workflow source is missing")
    if request_context is None:
        raise RuntimeCredentialUnavailableError(
            "workflow_path requires the current request credentials; "
            "embed workflow to submit with a runtime service key"
        )
    prepared = PlatformWorkflowSubmission._prepare(
        spec.workflow_path,
        request_context,
    )
    workflow = prepared.get("workflow")
    if not isinstance(workflow, dict):
        raise TrainRuntimeError("Workflow preparation returned an invalid definition")
    return deepcopy(workflow)


def _client(
    *,
    spec: TrainWorkflowSpec,
    request_context: RequestContext | None,
    access_key: str | None = None,
) -> PyroMindAPIClient:
    route = platform_cluster_route(
        (request_context.x_cluster if request_context is not None else None)
        or spec.cluster
    )
    env, cluster = platform_cluster_target(route)
    auth_token = platform_request_token(request_context)
    if auth_token:
        assert request_context is not None
        return create_workflow_api_client(
            env=env,
            cluster=cluster,
            auth_token=auth_token,
            headers={
                name: value
                for name, value in {
                    "cookie": request_context.cookie,
                    "authorization": request_context.authorization,
                    "x-cluster": route,
                }.items()
                if value
            },
        )
    if not access_key:
        raise RuntimeCredentialUnavailableError(
            "The project owner's Pyromind access is unavailable. "
            "Open this run while signed in to refresh access."
        )
    return get_pyromind_api_client(
        env=env,
        cluster=cluster,
        api_key=access_key,
    )


def _status(task: TrainingTaskResponse) -> TrainTaskStatus:
    normalized = str(task.status).strip().lower()
    if normalized in _SUCCEEDED_TASK_STATUSES:
        return "succeeded"
    if normalized in _FAILED_TASK_STATUSES:
        return "failed"
    if normalized in _CANCELED_TASK_STATUSES:
        return "canceled"
    if normalized in _ACTIVE_TASK_STATUSES or not normalized:
        return "running" if normalized == "running" else "pending"
    return "running"


def _task_message(
    *,
    task: TrainingTaskResponse,
    status: str,
    task_label: str,
    redact_wandb_key: bool = False,
) -> str:
    if status == "failed" and task.error_message:
        message = task.error_message
        if redact_wandb_key:
            message = _WANDB_KEY_PATTERN.sub("[REDACTED]", message)
        return f"{task_label} task failed: {message}"[:4000]
    return f"{task_label} task {task.task_id} is {task.status}."


def _node_output_uri(
    *,
    client: PyroMindAPIClient,
    task_id: str,
    spec: TrainWorkflowSpec,
) -> str | None:
    if spec.output_node_id is None:
        return None
    output = client.studio.get_node_output(task_id, spec.output_node_id)
    if not isinstance(output, Mapping):
        return None
    parameters = output.get("parameters")
    if not isinstance(parameters, list):
        return None
    for item in parameters:
        if not isinstance(item, Mapping):
            continue
        if item.get("name") != spec.output_parameter:
            continue
        value = item.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
