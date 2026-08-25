"""Submit fixed embodied-data cleaning jobs to Pyromind sandbox compute."""

from __future__ import annotations

import hashlib
import os
import shlex
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from pydantic import BaseModel, Field, model_validator
from rich.text import Text

from openhands.sdk.conversation.state import ActiveLongTask
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.pyromind_dataset.definition import PYROMIND_AGENT_STORAGE_ROOT
from openhands.tools.workflow.task_submission import (
    PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET,
    create_workflow_api_client,
    submit_workflow_task,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


SANDBOX_NODE_TYPE = "CustomCommandCPUNode"
TASK_ASSOCIATION_DIRNAME = ".pyromind_embodied_cleaning_tasks"
RUNTIME_PACKAGE_ENV = "PYROMIND_EMBODIED_RUNTIME_PACKAGE"
RUNTIME_WHEEL_PATH_ENV = "PYROMIND_EMBODIED_RUNTIME_WHEEL_PATH"


TOOL_DESCRIPTION = """\
Inspect and clean self-collected robot data entirely in Pyromind sandbox compute.

Use mode='plan' first. It reads the mounted Storage source, inspects the dataset,
and writes one representative EpisodePlan plus report.json under the returned
output_dir. After the user confirms the shared task, source-derived subtask ranges,
and derived next-state action convention, call mode='full' with the returned run_id.
Use mode='resume' with the same arguments only after a runtime failure.

The source videos, depth files, Parquet output, and checkpoints never enter the
conversation workspace. The sandbox uses a server-configured immutable runtime,
runs the deterministic batch once, validates the merged LeRobot v2.1 dataset, and
publishes it to target_path only after validation. A terminal platform callback
resumes this conversation. After each callback inspect output_dir/report.json with
preview_dataset. Completion requires report.complete=true and a successful preview
of target_path. Do not call local embodied-data tools for this workflow.
"""


class RunEmbodiedSandboxAction(Action):
    source_path: str = Field(description="Source dataset path in Pyromind Storage")
    mode: Literal["plan", "full", "resume"] = "plan"
    run_id: uuid.UUID | None = Field(
        default=None,
        description="Run id returned by plan; required for full and resume",
    )
    target_path: str | None = Field(
        default=None,
        description="Final LeRobot v2.1 Storage directory for full and resume",
    )
    task_text: str | None = Field(
        default=None,
        description="Confirmed task shared by every episode",
    )
    confirm_subtasks: bool = False
    confirm_derived_action: bool = False
    representative_episode_id: str | None = None
    robot_type: str = "s2"
    motion_speed_threshold: float = Field(default=0.02, gt=0)
    idle_min_duration_s: float = Field(default=1.5, gt=0)
    context_s: float = Field(default=0.5, ge=0)
    cpu: int = Field(default=8, ge=1, le=64)
    memory: int = Field(default=32, ge=1, le=256)

    @model_validator(mode="after")
    def validate_phase_arguments(self) -> RunEmbodiedSandboxAction:
        if self.mode == "plan":
            if self.run_id is not None:
                raise ValueError("run_id must be omitted for a new plan")
            return self
        if self.run_id is None:
            raise ValueError("run_id is required for full and resume")
        if self.target_path is None or not self.target_path.strip():
            raise ValueError("target_path is required for full and resume")
        if self.task_text is None or not self.task_text.strip():
            raise ValueError("task_text is required for full and resume")
        if not self.confirm_subtasks:
            raise ValueError("full and resume require subtask confirmation")
        if not self.confirm_derived_action:
            raise ValueError("full and resume require derived action confirmation")
        return self

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Run embodied sandbox: ", style="bold blue")
        content.append(f"{self.mode} {self.source_path}")
        return content


class RunEmbodiedSandboxObservation(Observation):
    status: str = "Failed"
    task_id: str | None = None
    run_id: str | None = None
    phase: str = ""
    output_dir: str | None = None
    target_path: str | None = None


class EmbodiedTaskAssociation(BaseModel):
    schema_version: int = 1
    task_id: str
    conversation_id: str
    run_id: str
    phase: Literal["plan", "full", "resume"]
    output_dir: str
    source_path: str
    target_path: str | None = None
    task_text: str | None = None
    robot_type: str = "s2"
    motion_speed_threshold: float = 0.02
    idle_min_duration_s: float = 1.5
    context_s: float = 0.5
    status: str = "Pending"
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EmbodiedTaskStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, association: EmbodiedTaskAssociation) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        updated = association.model_copy(update={"updated_at": datetime.now(UTC)})
        target = self._path(updated.task_id)
        temporary = target.with_name(f".{target.name}-{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(updated.model_dump_json(indent=2) + "\n")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def get_by_run_id(self, run_id: str) -> EmbodiedTaskAssociation | None:
        matches: list[EmbodiedTaskAssociation] = []
        try:
            for path in self.root.glob("*.json"):
                association = self._read(path)
                if association is not None and association.run_id == run_id:
                    matches.append(association)
        except OSError:
            return None
        return max(matches, key=lambda item: item.updated_at, default=None)

    def get_by_task_id(self, task_id: str) -> EmbodiedTaskAssociation | None:
        return self._read(self._path(task_id))

    def update_status(
        self,
        task_id: str,
        status: str,
    ) -> EmbodiedTaskAssociation | None:
        association = self.get_by_task_id(task_id)
        if association is None:
            return None
        updated = association.model_copy(update={"status": status})
        self.save(updated)
        return updated

    @staticmethod
    def _read(path: Path) -> EmbodiedTaskAssociation | None:
        try:
            return EmbodiedTaskAssociation.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None

    def _path(self, task_id: str) -> Path:
        digest = hashlib.sha256(task_id.encode()).hexdigest()
        return self.root / f"{digest}.json"


class RunEmbodiedSandboxExecutor(
    ToolExecutor[RunEmbodiedSandboxAction, RunEmbodiedSandboxObservation]
):
    def __init__(
        self,
        *,
        env: str | None = None,
        cluster: str | None = None,
        output_root: str | None = None,
        headers: dict[str, str] | None = None,
        runtime_package: str | None = None,
        runtime_wheel_storage_path: str | None = None,
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
        package_spec = runtime_package or os.environ.get(RUNTIME_PACKAGE_ENV)
        self._runtime_package = (
            _normalize_runtime_package(package_spec) if package_spec else None
        )
        wheel_path = runtime_wheel_storage_path or os.environ.get(
            RUNTIME_WHEEL_PATH_ENV
        )
        self._runtime_wheel_storage_path = (
            _normalize_runtime_wheel_storage_path(wheel_path) if wheel_path else None
        )
        self._task_store_dir = Path(task_store_dir) if task_store_dir else None
        self._timeout = timeout

    def __call__(
        self,
        action: RunEmbodiedSandboxAction,
        conversation: BaseConversation | None = None,
    ) -> RunEmbodiedSandboxObservation:
        try:
            if conversation is None:
                raise ValueError(
                    "run_embodied_cleaning_sandbox requires a conversation"
                )
            if not self._runtime_package and not self._runtime_wheel_storage_path:
                raise ValueError(
                    "embodied sandbox runtime is not configured; set "
                    f"{RUNTIME_PACKAGE_ENV} or {RUNTIME_WHEEL_PATH_ENV}"
                )
            source_path = _normalize_storage_path(action.source_path, "source_path")
            store = self._task_store(conversation)
            run_id, output_dir, target_path = self._resolve_run(
                action,
                source_path=source_path,
                store=store,
                conversation=conversation,
            )
            command = _build_sandbox_command(
                action,
                source_path=source_path,
                output_dir=output_dir,
                target_path=target_path,
                runtime_package=self._runtime_package,
                runtime_wheel_storage_path=self._runtime_wheel_storage_path,
            )
            workflow = _build_sandbox_workflow(action, run_id, command)
        except ValueError as exc:
            return RunEmbodiedSandboxObservation.from_text(
                text=str(exc),
                is_error=True,
                status="Failed",
                phase=action.mode,
            )

        try:
            state = cast("ConversationState", conversation.state)
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
            )
            task_id = response.task_id
        except Exception as exc:
            return RunEmbodiedSandboxObservation.from_text(
                text=f"Failed to submit embodied sandbox job: {exc}",
                is_error=True,
                status="Failed",
                run_id=str(run_id),
                phase=action.mode,
                output_dir=output_dir,
                target_path=target_path,
            )

        association = EmbodiedTaskAssociation(
            task_id=task_id,
            conversation_id=str(conversation.id),
            run_id=str(run_id),
            phase=action.mode,
            output_dir=output_dir,
            source_path=source_path,
            target_path=target_path,
            task_text=action.task_text,
            robot_type=action.robot_type,
            motion_speed_threshold=action.motion_speed_threshold,
            idle_min_duration_s=action.idle_min_duration_s,
            context_s=action.context_s,
            status=response.status,
        )
        try:
            store.save(association)
        except OSError as exc:
            return RunEmbodiedSandboxObservation.from_text(
                text=(
                    f"Sandbox accepted task {task_id}, but task association "
                    f"persistence failed: {exc}"
                ),
                is_error=True,
                status=response.status,
                task_id=task_id,
                run_id=str(run_id),
                phase=action.mode,
                output_dir=output_dir,
                target_path=target_path,
            )

        conversation.register_active_long_task(
            ActiveLongTask(
                task_id=task_id,
                kind="embodied_data_cleaning",
                status=response.status,
            )
        )
        return RunEmbodiedSandboxObservation.from_text(
            text=(
                f"Embodied sandbox {action.mode} job submitted. "
                f"task_id={task_id}, run_id={run_id}, output_dir={output_dir}. "
                "After the terminal callback, inspect "
                f"{output_dir}/report.json with preview_dataset."
            ),
            status=response.status,
            task_id=task_id,
            run_id=str(run_id),
            phase=action.mode,
            output_dir=output_dir,
            target_path=target_path,
        )

    def _resolve_run(
        self,
        action: RunEmbodiedSandboxAction,
        *,
        source_path: str,
        store: EmbodiedTaskStore,
        conversation: BaseConversation,
    ) -> tuple[uuid.UUID, str, str | None]:
        if action.mode == "plan":
            run_id = uuid.uuid4()
            root = self._output_root or (
                f"{PYROMIND_AGENT_STORAGE_ROOT}/{conversation.id}/embodied_cleaning"
            )
            output_dir = str(PurePosixPath(root) / str(run_id))
            if _storage_paths_overlap(source_path, output_dir):
                raise ValueError("source_path and output_dir must not overlap")
            return run_id, output_dir, None

        assert action.run_id is not None
        prior = store.get_by_run_id(str(action.run_id))
        if prior is None:
            raise ValueError(f"Cannot continue unknown embodied run {action.run_id}")
        if source_path != prior.source_path:
            raise ValueError("source_path must match the plan phase")
        if action.mode == "full" and prior.phase != "plan":
            raise ValueError("full may run once after plan; use resume after failures")
        if action.mode == "resume" and prior.phase not in {"full", "resume"}:
            raise ValueError("resume requires a prior full or resume submission")

        target_path = _normalize_storage_path(action.target_path or "", "target_path")
        if _storage_paths_overlap(source_path, target_path):
            raise ValueError("source_path and target_path must not overlap")
        if _storage_paths_overlap(prior.output_dir, target_path):
            raise ValueError("output_dir and target_path must not overlap")
        if prior.target_path is not None and target_path != prior.target_path:
            raise ValueError("target_path must match the prior full submission")
        if prior.task_text is not None and action.task_text != prior.task_text:
            raise ValueError("task_text must match the prior full submission")
        for name in (
            "robot_type",
            "motion_speed_threshold",
            "idle_min_duration_s",
            "context_s",
        ):
            if getattr(action, name) != getattr(prior, name):
                raise ValueError(f"{name} must match the prior full submission")
        return action.run_id, prior.output_dir, target_path

    def _task_store(self, conversation: BaseConversation) -> EmbodiedTaskStore:
        if self._task_store_dir is not None:
            return EmbodiedTaskStore(self._task_store_dir)
        workspace = cast(Any, conversation).workspace
        conversations_dir = Path(workspace.working_dir).resolve().parent
        return EmbodiedTaskStore(conversations_dir / TASK_ASSOCIATION_DIRNAME)


class RunEmbodiedSandboxTool(
    ToolDefinition[RunEmbodiedSandboxAction, RunEmbodiedSandboxObservation]
):
    name = "run_embodied_cleaning_sandbox"

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        env_value = params.pop("env", None)
        env = str(env_value) if env_value is not None else None
        cluster_value = params.pop("cluster", None)
        cluster = str(cluster_value) if cluster_value is not None else None
        params.pop("current_user", None)
        output_root_value = params.pop("output_root", None)
        output_root = str(output_root_value) if output_root_value is not None else None
        headers = _normalize_headers(params.pop("headers", None))
        runtime_package_value = params.pop("runtime_package", None)
        runtime_package = (
            str(runtime_package_value) if runtime_package_value is not None else None
        )
        wheel_value = params.pop("runtime_wheel_storage_path", None)
        runtime_wheel_storage_path = (
            str(wheel_value) if wheel_value is not None else None
        )
        task_store_value = params.pop("task_store_dir", None)
        task_store_dir = str(task_store_value) if task_store_value is not None else None
        timeout = int(params.pop("timeout", 30))
        env, cluster = _resolve_execution_target(env, cluster, headers)
        if params:
            names = ", ".join(sorted(params))
            raise ValueError(f"RunEmbodiedSandboxTool got unknown params: {names}")
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=RunEmbodiedSandboxAction,
                observation_type=RunEmbodiedSandboxObservation,
                executor=RunEmbodiedSandboxExecutor(
                    env=env,
                    cluster=cluster,
                    output_root=output_root,
                    headers=headers,
                    runtime_package=runtime_package,
                    runtime_wheel_storage_path=runtime_wheel_storage_path,
                    task_store_dir=task_store_dir,
                    timeout=timeout,
                ),
                annotations=ToolAnnotations(
                    title="run embodied cleaning in sandbox",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


def _build_sandbox_command(
    action: RunEmbodiedSandboxAction,
    *,
    source_path: str,
    output_dir: str,
    target_path: str | None,
    runtime_package: str | None,
    runtime_wheel_storage_path: str | None,
) -> str:
    venv_python = "/tmp/embodied-cleaning-venv/bin/python"
    install_target = runtime_package
    checks = [f"test -e {shlex.quote(_pod_path(source_path))}"]
    if runtime_wheel_storage_path is not None:
        install_target = _pod_path(runtime_wheel_storage_path)
        checks.append(f"test -f {shlex.quote(install_target)}")
    if install_target is None:
        raise ValueError("embodied sandbox runtime is not configured")

    command = [
        venv_python,
        "-m",
        "openhands_embodied_runtime.sandbox_runner",
        "--mode",
        action.mode,
        "--source",
        _pod_path(source_path),
        "--run-dir",
        _pod_path(output_dir),
        "--robot-type",
        action.robot_type,
        "--motion-speed-threshold",
        str(action.motion_speed_threshold),
        "--idle-min-duration-s",
        str(action.idle_min_duration_s),
        "--context-s",
        str(action.context_s),
        "--runtime-revision",
        runtime_wheel_storage_path or runtime_package or "",
    ]
    if action.representative_episode_id:
        command.extend(
            ["--representative-episode-id", action.representative_episode_id]
        )
    if action.mode != "plan":
        assert target_path is not None
        assert action.task_text is not None
        command.extend(
            [
                "--target",
                _pod_path(target_path),
                "--task-text",
                action.task_text,
                "--confirm-subtasks",
                "--confirm-derived-action",
            ]
        )

    setup = [
        *checks,
        "python3 -m venv /tmp/embodied-cleaning-venv",
        (
            f"{venv_python} -m pip install --disable-pip-version-check "
            f"{shlex.quote(install_target)}"
        ),
        f"mkdir -p {shlex.quote(_pod_path(output_dir))}",
    ]
    return " && ".join([*setup, shlex.join(command)])


def _build_sandbox_workflow(
    action: RunEmbodiedSandboxAction,
    run_id: uuid.UUID,
    command: str,
) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "name": f"embodied-{action.mode}-{str(run_id)[:8]}",
        "nodes": [
            {
                "id": "1",
                "type": "default",
                "position": {"x": 0, "y": 0},
                "data": {
                    "display_name": "Embodied Data Cleaning",
                    "nodeType": SANDBOX_NODE_TYPE,
                    "config": {
                        "command": command,
                        "cpu": action.cpu,
                        "memory": action.memory,
                    },
                },
            }
        ],
        "edges": [],
        "viewport": {"x": 0, "y": 0, "zoom": 1},
        "timestamp": datetime.now(UTC).isoformat(),
    }


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
    raw = value.strip().replace("\\", "/")
    if raw.startswith("/workspace/"):
        raw = raw.removeprefix("/workspace")
    if not raw or any(ord(character) < 32 for character in raw):
        raise ValueError(f"{field_name} must be a non-empty Storage path")
    parts = [part for part in raw.split("/") if part not in {"", "."}]
    if not parts or ".." in parts:
        raise ValueError(f"{field_name} must not be the root or contain '..'")
    return "/" + "/".join(parts)


def _normalize_runtime_package(value: str) -> str:
    package, separator, version = value.strip().partition("==")
    if (
        not separator
        or not package
        or not version
        or "*" in version
        or any(character.isspace() for character in value)
    ):
        raise ValueError(
            "runtime_package must be an exact package pin such as "
            "openhands-embodied-runtime==1.29.3"
        )
    return f"{package}=={version}"


def _normalize_runtime_wheel_storage_path(value: str) -> str:
    path = _normalize_storage_path(value, "runtime_wheel_storage_path")
    filename = PurePosixPath(path).name
    if not filename.startswith("openhands_embodied_runtime-") or not filename.endswith(
        ".whl"
    ):
        raise ValueError(
            "runtime_wheel_storage_path must point to an "
            "openhands_embodied_runtime-*.whl file"
        )
    return path


def _pod_path(storage_path: str) -> str:
    return f"/target-workspace{storage_path}"


def _storage_paths_overlap(left: str, right: str) -> bool:
    left_path = PurePosixPath(left)
    right_path = PurePosixPath(right)
    return (
        left_path == right_path
        or left_path in right_path.parents
        or right_path in left_path.parents
    )


def _normalize_headers(value: Any) -> dict[str, str] | None:
    if not value:
        return None
    if not isinstance(value, dict):
        raise ValueError("headers must be a dictionary when provided")
    return {str(name): str(header_value) for name, header_value in value.items()}


register_tool(RunEmbodiedSandboxTool.name, RunEmbodiedSandboxTool)
