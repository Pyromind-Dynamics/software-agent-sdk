from __future__ import annotations

import re
import shlex
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pyromind_sdk.client.base import PyroMindAPIError

from pyromind_agent_server.echomind_dataset_runtime import (
    DatasetRuntimeError,
    DatasetWorkflowSpec,
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
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.storage_paths import to_storage_api_path


class PlatformDatasetWorkflowBackend(PlatformTrainWorkflowBackend):
    def __init__(
        self,
        *,
        credential_store: RuntimeCredentialStore | None = None,
        projects_root: Path | str | None = None,
        conversations_dir: Path | str | None = None,
    ) -> None:
        super().__init__(
            task_kind="echomind_dataset",
            credential_store=credential_store,
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
        source = self._archived_credential_workflow(
            spec,
            request_context,
            task_id=spec.dataflow_credential_task_id,
            task_kind="data_preparation",
        )
        target = _cpu_command_config(workflow)
        if target.get("param") != "DF_API_KEY":
            raise DatasetRuntimeError("Missing DataFlow credential reference")
        try:
            tokens = shlex.split(_cpu_command_config(source)["command"])
        except ValueError:
            raise DatasetRuntimeError("Invalid archived DataFlow command") from None
        profile: dict[str, str] = {}
        for token in tokens:
            name, sep, value = token.partition("=")
            if sep and name in ("DF_API_URL", "DF_MODEL_NAME", "DF_API_KEY"):
                if name in profile:
                    raise DatasetRuntimeError("Ambiguous archived DataFlow profile")
                profile[name] = value
        key = profile.get("DF_API_KEY", "")
        endpoint = profile.get("DF_API_URL", "")
        model = profile.get("DF_MODEL_NAME", "")
        if not re.fullmatch(r"[A-Za-z0-9./_+=-]{8,512}", key):
            raise DatasetRuntimeError("Missing archived DataFlow credential")
        if not endpoint.startswith("https://") or not model:
            raise DatasetRuntimeError("Incomplete archived DataFlow model or endpoint")
        if _exported_profile(target["command"]) != {
            "DF_API_URL": endpoint,
            "DF_MODEL_NAME": model,
        }:
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


def _exported_profile(command: str) -> dict[str, str]:
    profile: dict[str, str] = {}
    for line in command.splitlines():
        if line.startswith("#") or line in ("", "set -eu", "unset param"):
            continue
        if not line.startswith("export "):
            break
        for assignment in shlex.split(line, comments=True)[1:]:
            name, _, value = assignment.partition("=")
            if name in ("DF_API_URL", "DF_MODEL_NAME"):
                if name in profile:
                    raise DatasetRuntimeError("Ambiguous DataFlow profile in command")
                profile[name] = value
    return profile


def _cpu_command_config(workflow: dict[str, Any]) -> dict[str, Any]:
    nodes = workflow.get("nodes")
    if isinstance(nodes, list) and len(nodes) == 1 and isinstance(nodes[0], dict):
        data = nodes[0].get("data")
        if isinstance(data, dict) and data.get("nodeType") == "CustomCommandCPUNode":
            config = data.get("config")
            if isinstance(config, dict) and isinstance(config.get("command"), str):
                return config
    raise DatasetRuntimeError("DataFlow labeling requires a single CPU command node")
