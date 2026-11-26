"""Submit agent-authored dataset cleaning scripts to Pyromind Studio."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Self, cast

from pydantic import Field
from pyromind_sdk import PyroMindAPIError
from rich.text import Text

from pyromind_common.platform.scene_core import (
    _build_cleaning_command as _build_cleaning_command,
    _build_cleaning_workflow as _build_cleaning_workflow,
    _runtime_public_names as _runtime_public_names,
    validate_cleaning_script as validate_cleaning_script,
)
from pyromind_common.platform.scene_options import CleaningOptions
from pyromind_common.sdk.conversation.state import ActiveLongTask
from pyromind_common.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from pyromind_common.tools.pyromind_cleaning.task_store import (
    TASK_ASSOCIATION_DIRNAME,
    DatasetCleaningTaskAssociation,
    DatasetCleaningTaskStore,
)
from pyromind_common.tools.pyromind_dataset.definition import (
    PYROMIND_AGENT_STORAGE_ROOT,
    _default_storage_base_url,
    _resolve_conversation_headers,
    _resolve_secret_headers,
    download_file_from_pyromind,
    upload_local_file_to_pyromind,
)
from pyromind_common.tools.utils.conversation_dirs import conversation_state_dir
from pyromind_common.tools.utils.storage_paths import (
    StoragePathError,
    to_agent_storage_path,
    to_tool_storage_path,
)
from pyromind_common.tools.workflow.task_submission import (
    PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET,
    create_workflow_api_client,
    submit_workflow_task,
)


if TYPE_CHECKING:
    from pyromind_sdk.client.models import TrainingTaskCreateResponse

    from pyromind_common.sdk.conversation.base import ToolConversation
    from pyromind_common.sdk.conversation.state import ToolState


RUNTIME_FILENAMES = ("cleaning_utils.py", "validate_format.py")
MAX_CLEANING_SCRIPT_BYTES = 1024 * 1024
MAX_SUBMIT_ERROR_CHARS = 500


def _submit_error_text(exc: Exception) -> str:
    if not isinstance(exc, PyroMindAPIError) or not isinstance(exc.response, dict):
        return str(exc)
    response = exc.response
    error = response.get("error")
    detail = error.get("message") if isinstance(error, dict) else None
    message = detail or response.get("message") or response.get("detail")
    if not isinstance(message, str) or not message.strip():
        return str(exc)
    message = message.strip()
    if isinstance(error, dict):
        code = error.get("code")
        if isinstance(code, str) and code.strip():
            message = f"{code.strip()}: {message}"
    if len(message) > MAX_SUBMIT_ERROR_CHARS:
        message = message[:MAX_SUBMIT_ERROR_CHARS] + "..."
    return f"{exc}\nPlatform response: {message}"


class RunDatasetCleaningAction(CleaningOptions, Action):
    continue_after_completion: bool = Field(
        default=False,
        description="Enable only when the user asks for work after cleaning completes.",
    )

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Run dataset cleaning: ", style="bold blue")
        content.append(self.input_path)
        return content


class RunDatasetCleaningObservation(Observation):
    """Initial result of a submitted dataset cleaning task."""

    status: str = Field(description="Initial Studio task status.")
    task_id: str | None = Field(default=None)
    run_id: str | None = Field(default=None)
    output_dir: str | None = Field(default=None)
    resumed: bool = Field(default=False)

    @property
    def visualize(self) -> Text:
        content = Text()
        if self.is_error:
            content.append("Dataset cleaning submission failed", style="bold red")
            return content
        content.append("Dataset cleaning submitted", style="bold green")
        if self.task_id:
            content.append(f"\ntask_id={self.task_id}")
        if self.output_dir:
            content.append(f"\noutput_dir={self.output_dir}")
        return content


TOOL_DESCRIPTION = """Run an uploaded dataset cleaner entirely on Pyromind.

Call this after previewing the source dataset, generating a cleaning script that
implements the CLI contract below, and uploading that script to Pyromind storage.
The tool creates a one-node CustomCommandNode workflow; do not construct or run
shell commands yourself.

The script must accept `--input`, `--output`, `--state-dir`, optional `--resume`,
and optional `--limit N`. `--output` is the concrete output.jsonl path, while
`--state-dir` is the run directory. A limit applies to source records considered
by that invocation. Output is homogeneous messages or text DPO preference JSONL.

The submission is asynchronous. A new run gets a unique result directory under
`/workspace/.pyromind-agent/<conversation_id>/data_cleaning/<run_id>`. The tool uploads
`cleaning_utils.py` and
`validate_format.py` beside the frozen `clean_script.py`, runs the cleaner, then
validates output.jsonl in the same Pod and merges validation into report.json.
The only generated artifacts are output.jsonl and report.json. Before creating a
new run it statically checks script syntax and explicit cleaning_utils imports; it
does not execute the script or read dataset content. Use `limit=3` for a sample
run. To continue an interrupted run, call this tool again with `resume_run_id`;
the platform reuses all frozen runtime files and passes `--resume`.

Set continue_after_completion=true only when the user explicitly asks to check
the results or do further work after cleaning. Otherwise the backend reports
completion. End the current turn after submission, without sleeping or polling.
When the terminal workflow callback resumes the conversation, use the
`output_dir` returned by this tool: read report.json for validation,
counters, errors, and checkpoint state, then output.jsonl for cleaned rows.
Do not re-run the cleaning pipeline locally to validate platform output.
"""


class DatasetCleaningRunner:
    """Build and submit a fixed CustomCommandNode cleaning workflow."""

    def __init__(
        self,
        *,
        env: str | None = None,
        cluster: str | None = None,
        output_root: str | None = None,
        headers: dict[str, str] | None = None,
        runtime_dir: str | None = None,
        storage_base_url: str | None = None,
        storage_headers: dict[str, str] | None = None,
        storage_secret_headers: dict[str, str] | None = None,
        task_store_dir: str | None = None,
        timeout: int = 30,
    ) -> None:
        self._env = env
        self._cluster = cluster
        self._output_root = (
            _normalize_storage_path(output_root, "output_root")
            if output_root is not None
            else None
        )
        self._headers = dict(headers or {})
        self._runtime_dir = Path(runtime_dir) if runtime_dir else None
        self._storage_base_url = (
            storage_base_url or _default_storage_base_url()
        ).rstrip("/")
        self._storage_headers = dict(storage_headers or {})
        self._storage_secret_headers = dict(storage_secret_headers or {})
        self._task_store_dir = Path(task_store_dir) if task_store_dir else None
        self._timeout = timeout
        self.submission_handler: (
            Callable[[dict, str, str | None], TrainingTaskCreateResponse] | None
        ) = None

    @classmethod
    def from_options(cls, **params: Any) -> Self:
        env_value = params.pop("env", None)
        env = str(env_value) if env_value is not None else None
        cluster_value = params.pop("cluster", None)
        cluster = str(cluster_value) if cluster_value is not None else None
        params.pop("current_user", None)
        output_root_value = params.pop("output_root", None)
        output_root = str(output_root_value) if output_root_value is not None else None
        headers = _normalize_headers(params.pop("headers", None))
        runtime_dir_value = params.pop("runtime_dir", None)
        runtime_dir = str(runtime_dir_value) if runtime_dir_value is not None else None
        storage_base_url_value = params.pop("storage_base_url", None)
        storage_base_url = (
            str(storage_base_url_value) if storage_base_url_value is not None else None
        )
        storage_headers = _normalize_headers(params.pop("storage_headers", None))
        legacy_secret_headers = params.pop("secret_headers", None)
        storage_secret_headers = _normalize_headers(
            params.pop("storage_secret_headers", legacy_secret_headers)
        )
        env, cluster = _resolve_execution_target(env, cluster, headers)
        params.pop("endpoint_url", None)
        task_store_dir_value = params.pop("task_store_dir", None)
        task_store_dir = (
            str(task_store_dir_value) if task_store_dir_value is not None else None
        )
        timeout = int(params.pop("timeout", 30))
        if params:
            names = ", ".join(sorted(params))
            raise ValueError(f"RunDatasetCleaningTool got unknown params: {names}")
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if output_root is not None:
            _normalize_storage_path(output_root, "output_root")
        return cls(
            env=env,
            cluster=cluster,
            output_root=output_root,
            headers=headers,
            runtime_dir=runtime_dir,
            storage_base_url=storage_base_url,
            storage_headers=storage_headers,
            storage_secret_headers=storage_secret_headers,
            task_store_dir=task_store_dir,
            timeout=timeout,
        )

    def run(
        self,
        action: RunDatasetCleaningAction,
        conversation: ToolConversation | None = None,
    ) -> RunDatasetCleaningObservation:
        try:
            if conversation is None:
                raise ValueError(
                    "run_dataset_cleaning requires an active conversation."
                )
            input_path = _normalize_storage_path(action.input_path, "input_path")
            output_root = (
                self._output_root
                or f"{PYROMIND_AGENT_STORAGE_ROOT}/{conversation.id}/data_cleaning"
            )
            uploaded_script_path = None
            if action.script_path is not None:
                uploaded_script_path = _normalize_storage_path(
                    action.script_path, "script_path"
                )
                if PurePosixPath(uploaded_script_path).suffix.lower() != ".py":
                    raise ValueError("script_path must point to a Python .py file.")
            task_store = self._task_store(conversation)
            run_id = action.resume_run_id or uuid.uuid4()
            resumed = action.resume_run_id is not None
            if resumed:
                prior_run = task_store.get_by_run_id(str(run_id))
                if prior_run is None:
                    raise ValueError(
                        f"Cannot resume unknown dataset cleaning run {run_id}."
                    )
                if input_path != prior_run.input_path:
                    raise ValueError(
                        "input_path must match the original dataset cleaning run."
                    )
                effective_script_path = prior_run.script_path
                output_dir = _normalize_storage_path(
                    prior_run.output_dir, "persisted output_dir"
                )
                if PurePosixPath(output_dir).name != str(run_id):
                    raise ValueError(
                        "Persisted dataset cleaning output directory does not "
                        "match resume_run_id."
                    )
            else:
                if uploaded_script_path is None:
                    raise ValueError("script_path is required for a new run.")
                effective_script_path = uploaded_script_path
                output_dir = str(PurePosixPath(output_root) / str(run_id))
                self._preflight_script(effective_script_path, conversation)
                self._stage_runtime_files(output_dir, conversation)
            command = _build_cleaning_command(
                input_path=input_path,
                script_path=effective_script_path,
                output_root=output_root,
                output_dir=output_dir,
                limit=action.limit,
                resumed=resumed,
            )
            workflow = _build_cleaning_workflow(action, run_id, command)
        except ValueError as exc:
            return RunDatasetCleaningObservation.from_text(
                text=str(exc),
                status="Failed",
                is_error=True,
            )

        try:
            state = cast("ToolState", conversation.state)
            auth_token = state.secret_registry.get_secret_value(
                PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET
            )
            client = create_workflow_api_client(
                env=self._env,
                cluster=self._cluster,
                auth_token=auth_token,
                headers=self._headers,
                timeout=self._timeout,
            )
            response = submit_workflow_task(
                client=client,
                workflow=workflow,
                name=str(workflow["name"]),
                conversation_id=str(conversation.id),
                task_kind="data_cleaning",
                retry_create=False,
                submission_handler=self.submission_handler,
            )
            task_id = response.task_id
        except Exception as exc:
            return RunDatasetCleaningObservation.from_text(
                text=(
                    "Failed to submit dataset cleaning workflow: "
                    f"{_submit_error_text(exc)}"
                ),
                status="Failed",
                is_error=True,
            )

        association = DatasetCleaningTaskAssociation(
            task_id=task_id,
            conversation_id=str(conversation.id),
            run_id=str(run_id),
            output_dir=output_dir,
            input_path=input_path,
            script_path=effective_script_path,
            limit=action.limit,
            resumed=resumed,
            status=response.status,
        )
        agent_output_dir = to_agent_storage_path(output_dir, field="output_dir")
        try:
            task_store.save(association)
        except OSError as exc:
            return RunDatasetCleaningObservation.from_text(
                text=(
                    f"Studio accepted dataset cleaning task {task_id}, but task "
                    f"association persistence failed: {exc}"
                ),
                status=response.status,
                task_id=task_id,
                run_id=str(run_id),
                output_dir=agent_output_dir,
                resumed=resumed,
                is_error=True,
            )

        # Register the task on the conversation so it stays presented as
        # running while the task is in flight, and can be stopped together
        # with the conversation.
        conversation.register_active_long_task(
            ActiveLongTask(
                task_id=task_id,
                kind="data_cleaning",
                status=response.status,
            )
        )
        conversation.send_agent_message(
            f"已提交数据清洗任务（task_id={task_id}, run_id={run_id}），"
            "平台正在后台异步执行。任务完成或失败后我会在此通知你，"
            "也可以随时让我查询进度。"
        )

        return RunDatasetCleaningObservation.from_text(
            text=(
                "Dataset cleaning workflow submitted. "
                f"task_id={task_id}, run_id={run_id}, output_dir={agent_output_dir}. "
                "A submission confirmation message has already been sent "
                "to the user, so do not repeat the submission details in "
                "your reply. "
                "After the terminal callback, preview "
                f"{agent_output_dir}/report.json, then output.jsonl."
            ),
            status=response.status,
            task_id=task_id,
            run_id=str(run_id),
            output_dir=agent_output_dir,
            resumed=resumed,
        )

    def _task_store(self, conversation: ToolConversation) -> DatasetCleaningTaskStore:
        if self._task_store_dir is not None:
            return DatasetCleaningTaskStore(self._task_store_dir)
        state_dir = conversation_state_dir(conversation, TASK_ASSOCIATION_DIRNAME)
        if state_dir is None:
            raise ValueError(
                "Cannot resolve the host conversation directory for task associations."
            )
        return DatasetCleaningTaskStore(state_dir)

    def _stage_runtime_files(
        self,
        output_dir: str,
        conversation: ToolConversation,
    ) -> None:
        if self._runtime_dir is None:
            raise ValueError("Dataset cleaning runtime_dir is not configured.")
        headers = self._resolved_storage_headers(conversation)
        for filename in RUNTIME_FILENAMES:
            local_path = self._runtime_dir / filename
            if not local_path.is_file():
                raise ValueError(
                    f"Dataset cleaning runtime file is missing: {local_path}"
                )
            try:
                upload_local_file_to_pyromind(
                    local_path=local_path,
                    target_dir=output_dir,
                    storage_base_url=self._storage_base_url,
                    headers=headers,
                    timeout=float(self._timeout),
                )
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"Failed to stage dataset cleaning runtime {filename}: {exc}"
                ) from exc

    def _resolved_storage_headers(
        self,
        conversation: ToolConversation,
    ) -> dict[str, str]:
        headers = {"accept": "*/*", **self._storage_headers}
        headers.update(_resolve_conversation_headers(conversation))
        headers.update(
            _resolve_secret_headers(conversation, self._storage_secret_headers)
        )
        return headers

    def _preflight_script(
        self,
        script_path: str,
        conversation: ToolConversation,
    ) -> None:
        if self._runtime_dir is None:
            raise ValueError("Dataset cleaning runtime_dir is not configured.")
        utils_path = self._runtime_dir / "cleaning_utils.py"
        if not utils_path.is_file():
            raise ValueError(f"Dataset cleaning runtime file is missing: {utils_path}")
        try:
            source_bytes = download_file_from_pyromind(
                storage_path=script_path,
                storage_base_url=self._storage_base_url,
                headers=self._resolved_storage_headers(conversation),
                timeout=float(self._timeout),
                max_bytes=MAX_CLEANING_SCRIPT_BYTES,
            )
            source = source_bytes.decode("utf-8-sig")
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(
                f"Failed to read dataset cleaning script for preflight: {exc}"
            ) from exc
        validate_cleaning_script(source, utils_path, filename=script_path)


class RunDatasetCleaningExecutor(
    DatasetCleaningRunner,
    ToolExecutor[RunDatasetCleaningAction, RunDatasetCleaningObservation],
):
    def __call__(
        self,
        action: RunDatasetCleaningAction,
        conversation: ToolConversation | None = None,
    ) -> RunDatasetCleaningObservation:
        return self.run(action, conversation)


def create_dataset_cleaning_runner(**params: Any) -> DatasetCleaningRunner:
    return DatasetCleaningRunner.from_options(**params)


class RunDatasetCleaningTool(
    ToolDefinition[RunDatasetCleaningAction, RunDatasetCleaningObservation]
):
    """Tool definition for asynchronous dataset cleaning submissions."""

    @classmethod
    def create(
        cls,
        conv_state: ToolState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=RunDatasetCleaningAction,
                observation_type=RunDatasetCleaningObservation,
                executor=RunDatasetCleaningExecutor.from_options(**params),
                annotations=ToolAnnotations(
                    title="run_dataset_cleaning",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


def _normalize_headers(value: Any) -> dict[str, str] | None:
    if not value:
        return None
    if not isinstance(value, dict):
        raise ValueError("headers must be a dictionary when provided")
    return {str(name): str(header_value) for name, header_value in value.items()}


def _resolve_execution_target(
    env: str | None,
    cluster: str | None,
    headers: dict[str, str] | None,
) -> tuple[str | None, str | None]:
    routed_cluster = next(
        (
            value
            for name, value in (headers or {}).items()
            if name.lower() == "x-cluster"
        ),
        None,
    )
    if not routed_cluster:
        return env, cluster

    cluster_part, separator, env_part = routed_cluster.partition("#")
    resolved_cluster = cluster or cluster_part.strip() or None
    resolved_env = env or (env_part.strip().lower() if separator else "prod")
    return resolved_env, resolved_cluster


def _normalize_storage_path(value: str, field_name: str) -> str:
    try:
        return to_tool_storage_path(value, field=field_name)
    except StoragePathError as exc:
        raise ValueError(str(exc)) from exc


def _pod_path(storage_path: str) -> str:
    return f"/target-workspace{storage_path}"


register_tool(RunDatasetCleaningTool.name, RunDatasetCleaningTool)
