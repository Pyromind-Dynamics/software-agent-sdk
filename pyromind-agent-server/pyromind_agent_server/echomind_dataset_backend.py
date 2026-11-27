from __future__ import annotations

import asyncio
import json
import re
import shlex
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from pyromind_sdk.client.base import PyroMindAPIError

from pyromind_agent_server.accepted_workflow import read_accepted_workflow
from pyromind_agent_server.echomind_dataset_runtime import (
    DatasetRuntimeError,
    DatasetWorkflowSpec,
)
from pyromind_agent_server.echomind_runtime_contracts import (
    RuntimeCredentialUnavailableError,
)
from pyromind_agent_server.echomind_runtime_credentials import RuntimeCredentialStore
from pyromind_agent_server.echomind_train_backend import (
    PlatformTrainWorkflowBackend,
    _client,
    _node_output_uri,
    _status,
    _without_sensitive_sdk_logs,
)
from pyromind_agent_server.echomind_train_runtime import (
    TrainTaskState,
    TrainWorkflowSpec,
)
from pyromind_agent_server.echomind_workflow_storage import build_runtime_storage
from pyromind_common.platform.credentials import materialize_runtime_environment
from pyromind_common.platform.recipe import dataset_public_profile
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.storage_paths import to_storage_api_path
from pyromind_runtime.ports.observation_source import ObservationForbiddenError
from pyromind_runtime.ports.storage import WorkflowStorageResolver


class PlatformDatasetWorkflowBackend(PlatformTrainWorkflowBackend):
    def __init__(
        self,
        *,
        recorded_storage: WorkflowStorageResolver | None = None,
        credential_store: RuntimeCredentialStore | None = None,
        projects_root: Path | str | None = None,
        conversations_dir: Path | str | None = None,
    ) -> None:
        super().__init__(
            task_kind="echomind_dataset",
            credential_store=credential_store,
            recorded_storage=recorded_storage,
            projects_root=projects_root,
            conversations_dir=conversations_dir,
        )

    def _bind_submission_credentials(
        self,
        workflow: dict[str, Any],
        *,
        spec: TrainWorkflowSpec,
        request_context: RequestContext | None,
    ) -> tuple[str, ...]:
        if not isinstance(spec, DatasetWorkflowSpec):
            raise DatasetRuntimeError("Dataset backend requires a DatasetWorkflowSpec")
        if spec.dataflow_credential_task_id is None:
            return ()
        source = self._archived_credential_receipt(
            spec,
            request_context,
            task_id=spec.dataflow_credential_task_id,
            task_kind="data_preparation",
        )
        target = _cpu_command_config(workflow)
        if target.get("param") != "DF_API_KEY":
            raise DatasetRuntimeError("Missing DataFlow credential reference")
        if source.request_id.startswith("sdk:"):
            key, profile = _sdk_dataflow_profile(
                source.workflow,
                spec.dataflow_credential_task_id,
                request_context,
                spec.cluster,
            )
        else:
            profile = _legacy_profile(_cpu_command_config(source.workflow)["command"])
            key = profile.pop("DF_API_KEY", "")
        if (
            not isinstance(key, str)
            or key == "DF_API_KEY"
            or not re.fullmatch(r"[A-Za-z0-9./_+=-]{8,512}", key)
        ):
            raise DatasetRuntimeError("Missing archived DataFlow credential")
        if not profile.get("DF_API_URL", "").startswith("https://") or not profile.get(
            "DF_MODEL_NAME"
        ):
            raise DatasetRuntimeError("Incomplete archived DataFlow model or endpoint")
        if _exported_profile(target["command"]) != profile:
            raise DatasetRuntimeError("DataFlow model or endpoint differs from source")
        target["param"] = key
        return (key,)

    @contextmanager
    def _task_read_scope(self, spec: TrainWorkflowSpec) -> Iterator[None]:
        if (
            not isinstance(spec, DatasetWorkflowSpec)
            or not spec.dataflow_credential_task_id
        ):
            with super()._task_read_scope(spec):
                yield
            return
        with _without_sensitive_sdk_logs(True):
            try:
                yield
            except PyroMindAPIError as exc:
                raise PyroMindAPIError(
                    message="Unable to read the DataFlow platform task.",
                    status_code=exc.status_code,
                ) from None

    def query_task(
        self,
        *,
        task_id: str,
        spec: TrainWorkflowSpec,
        request_context: RequestContext | None,
    ) -> TrainTaskState:
        if not isinstance(spec, DatasetWorkflowSpec):
            raise DatasetRuntimeError("Dataset backend requires a DatasetWorkflowSpec")
        access_key = self._access_key(spec, request_context)
        client = _client(
            spec=spec,
            request_context=request_context,
            access_key=access_key,
        )
        try:
            with self._task_read_scope(spec):
                task = client.studio.get_task(task_id)
            status = _status(task)
            if status != "succeeded":
                return TrainTaskState(
                    status=status,
                    message=(
                        (
                            task.error_message
                            if not spec.dataflow_credential_task_id
                            else None
                        )
                        or f"Dataset task {task_id} is {task.status}."
                    )[:4000],
                )
            output_uri = (
                _node_output_uri(client=client, task_id=task_id, spec=spec)
                if spec.output_node_id is not None
                else spec.output_directory
            )
        finally:
            client.close()

        prefix = to_storage_api_path(spec.output_directory)
        if output_uri is None or to_storage_api_path(output_uri) != prefix:
            return TrainTaskState(
                status="failed",
                message="Dataset task did not report the expected output directory.",
                failure_category="validation",
            )

        storage = build_runtime_storage(
            spec=spec,
            request_context=request_context,
            access_key=access_key,
            projects_root=self.projects_root,
            credential_store=self.credential_store,
            recorded_storage=self.recorded_storage,
        )
        missing = set(spec.output_filenames).difference(storage.list(prefix))
        if missing:
            return TrainTaskState(
                status="failed",
                message="Dataset output files are missing: "
                + ", ".join(sorted(missing)),
                failure_category="validation",
            )
        return TrainTaskState(
            status="succeeded",
            message=f"Dataset task {task_id} produced both splits.",
            output_uri=spec.output_directory,
        )


def _legacy_profile(command: str) -> dict[str, str]:
    profile: dict[str, str] = {}
    try:
        for token in shlex.split(command):
            name, sep, value = token.partition("=")
            if sep and name in (
                "DF_API_URL",
                "DF_API_BASE_URL",
                "DF_MODEL_NAME",
                "DF_API_KEY",
            ):
                if name in profile:
                    raise DatasetRuntimeError("Ambiguous archived DataFlow profile")
                profile[name] = value
    except ValueError:
        raise DatasetRuntimeError("Invalid archived DataFlow command") from None
    return profile


def _sdk_dataflow_profile(
    source: dict[str, Any], task_id: str, context: RequestContext | None, cluster: str
) -> tuple[str, dict[str, str]]:
    if context is None or not (context.cookie or context.authorization):
        raise RuntimeCredentialUnavailableError(
            "Reading the Dataset task requires its owner's Studio session."
        )
    try:
        accepted = asyncio.run(
            read_accepted_workflow(task_id, replace(context, x_cluster=cluster))
        )
    except ObservationForbiddenError:
        raise RuntimeCredentialUnavailableError(
            "Studio requires a renewed Project owner session."
        ) from None
    except Exception:
        raise DatasetRuntimeError(
            "The accepted Dataset workflow could not be read."
        ) from None
    remote_id, remote = _cpu_node(accepted)
    source_id, config = _cpu_node(source, historical=True)
    if source_id is not None and source_id != remote_id:
        raise DatasetRuntimeError("Studio returned a different Dataset node.")
    archived, actual = config["command"], remote["command"]
    if (
        archived == actual
        and source_id is not None
        and remote.get("param")
        and 'export DF_API_KEY="$param"' in actual.splitlines()
    ):
        return remote["param"], _exported_profile(actual)
    declarations = list(
        re.finditer(
            r"\{\{pyromind\.runtime:(vision|text):[A-Z_,]+(?::[0-9a-f]{64})?\}\}",
            archived,
        )
    )
    if len(declarations) != 1:
        raise DatasetRuntimeError(
            "The accepted Dataset task has no verified gateway binding."
        )
    (declaration,) = declarations
    prefix, suffix = archived[: declaration.start()], archived[declaration.end() :]
    if not actual.startswith(prefix) or not actual.endswith(suffix):
        raise DatasetRuntimeError(
            "The accepted Dataset command differs from the archive."
        )
    values = _legacy_profile(actual[len(prefix) : len(actual) - len(suffix)])
    try:
        rendered = materialize_runtime_environment(
            {"nodes": [{"data": {"config": {"command": archived}}}]},
            {"PYROMIND_DATAFLOW_PROFILES": json.dumps({declaration[1]: values})},
        )
    except ValueError:
        raise DatasetRuntimeError(
            "The platform Dataset model profile could not be verified."
        ) from None
    if rendered["nodes"][0]["data"]["config"]["command"] != actual:
        raise DatasetRuntimeError(
            "The platform Dataset environment differs from the signed source."
        )
    return values.get("DF_API_KEY", ""), dataset_public_profile(values)


def _exported_profile(command: str) -> dict[str, str]:
    profile: dict[str, str] = {}
    for line in command.splitlines():
        if line.startswith("#") or line in (
            "",
            "set -eu",
            "unset param",
            "unset DF_API_KEY DF_API_URL DF_API_BASE_URL DF_MODEL_NAME",
        ):
            continue
        if not line.startswith("export "):
            break
        for assignment in shlex.split(line, comments=True)[1:]:
            name, _, value = assignment.partition("=")
            if name in ("DF_API_URL", "DF_API_BASE_URL", "DF_MODEL_NAME"):
                if name in profile:
                    raise DatasetRuntimeError("Ambiguous DataFlow profile in command")
                profile[name] = value
    return profile


def _cpu_node(
    workflow: dict[str, Any], *, historical: bool = False
) -> tuple[str | None, dict[str, Any]]:
    nodes = workflow.get("nodes")
    if isinstance(nodes, list) and len(nodes) == 1 and isinstance(nodes[0], dict):
        node = nodes[0]
        data = node.get("data")
        if isinstance(data, dict):
            config = data.get("config")
            if isinstance(config, dict) and isinstance(config.get("command"), str):
                if data.get("nodeType") == "CustomCommandCPUNode":
                    identity = node.get("id")
                    if isinstance(identity, str) and identity:
                        return identity, config
                if historical and data.get("type") == "DataPreparationNode":
                    return None, config
    raise DatasetRuntimeError("DataFlow labeling requires a single CPU command node")


def _cpu_command_config(workflow: dict[str, Any]) -> dict[str, Any]:
    return _cpu_node(workflow)[1]
