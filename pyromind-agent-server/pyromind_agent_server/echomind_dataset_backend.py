from __future__ import annotations

from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.storage_paths import to_storage_api_path

from openhands.agent_server.echomind_dataset_runtime import (
    DatasetRuntimeError,
    DatasetWorkflowSpec,
)
from openhands.agent_server.echomind_train_runtime import (
    TrainTaskState,
    TrainWorkflowSpec,
)
from pyromind_agent_server.echomind_runtime_credentials import RuntimeCredentialStore
from pyromind_agent_server.echomind_train_backend import (
    PlatformTrainWorkflowBackend,
    _client,
    _node_output_uri,
    _status,
)
from pyromind_agent_server.echomind_workflow_storage import build_runtime_storage


class PlatformDatasetWorkflowBackend(PlatformTrainWorkflowBackend):
    def __init__(
        self, *, credential_store: RuntimeCredentialStore | None = None
    ) -> None:
        super().__init__(
            task_kind="echomind_dataset", credential_store=credential_store
        )

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
            spec=spec, request_context=request_context, access_key=access_key
        )
        try:
            task = client.studio.get_task(task_id)
            status = _status(task)
            if status != "succeeded":
                return TrainTaskState(
                    status=status,
                    message=(
                        task.error_message
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
            spec=spec, request_context=request_context, access_key=access_key
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
