"""Workspace staging and transport shared by script and evaluation submissions.

Request credentials are resolved in memory and never become frozen run inputs.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import Field
from rich.text import Text

from openhands.sdk.conversation.state import ActiveLongTask
from openhands.sdk.llm import ImageContent, TextContent
from openhands.sdk.tool import Observation
from openhands.tools.data_preparation.workspace_paths import (
    resolve_workspace_file,
    workspace_relative_path,
)
from openhands.tools.pyromind_dataset.definition import (
    _default_storage_base_url,
    _resolve_conversation_headers,
    _resolve_secret_headers,
    upload_local_file_to_pyromind,
)
from openhands.tools.utils.workspace_staging import (
    WorkspaceStagingError,
    is_remote_workspace,
    resolve_workspace_path,
    stage_workspace_path,
    staged_remote_dir,
)
from openhands.tools.workflow.task_submission import (
    PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET,
    create_workflow_api_client,
)
from openhands.tools.workflow.validate_workflow_dsl import WorkflowValidationIssue


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


@dataclass(frozen=True, kw_only=True)
class SubmissionConfig:
    env: str | None = None
    cluster: str | None = None
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    storage_base_url: str = field(default_factory=_default_storage_base_url)
    storage_headers: dict[str, str] = field(default_factory=dict, repr=False)
    storage_secret_headers: dict[str, str] = field(default_factory=dict, repr=False)
    timeout: int = 30


def submission_config(params: dict[str, Any]) -> SubmissionConfig:
    """Consume transport options; the caller validates its remaining tool options."""
    headers = dict(params.pop("headers", None) or {})
    env = params.pop("env", None)
    cluster = params.pop("cluster", None)
    route = next((v for k, v in headers.items() if k.lower() == "x-cluster"), None)
    if route:
        cluster_part, separator, env_part = route.partition("#")
        cluster = cluster or cluster_part.strip() or None
        env = env or (env_part.strip().lower() if separator else "prod")
    params.pop("current_user", None)
    params.pop("endpoint_url", None)
    legacy_secrets = params.pop("secret_headers", None)
    config = SubmissionConfig(
        env=env,
        cluster=cluster,
        headers=headers,
        storage_base_url=str(
            params.pop("storage_base_url", None) or _default_storage_base_url()
        ).rstrip("/"),
        storage_headers=dict(params.pop("storage_headers", None) or {}),
        storage_secret_headers=dict(
            params.pop("storage_secret_headers", legacy_secrets) or {}
        ),
        timeout=int(params.pop("timeout", 30)),
    )
    if config.timeout <= 0:
        raise ValueError("timeout must be greater than 0")
    return config


def submission_client(config: SubmissionConfig, conversation: BaseConversation) -> Any:
    return create_workflow_api_client(
        env=config.env,
        cluster=config.cluster,
        headers=config.headers,
        timeout=config.timeout,
        auth_token=cast(
            "ConversationState", conversation.state
        ).secret_registry.get_secret_value(PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET),
    )


def resolved_storage_headers(
    config: SubmissionConfig, conversation: BaseConversation
) -> dict[str, str]:
    headers = {"accept": "*/*", **config.storage_headers}
    headers.update(_resolve_conversation_headers(conversation))
    headers.update(_resolve_secret_headers(conversation, config.storage_secret_headers))
    return headers


def stage_submission_file(
    config: SubmissionConfig,
    script_path: str,
    output_dir: str,
    conversation: BaseConversation,
    *,
    frozen_script_name: str,
) -> None:
    """Publish a local file under its immutable run/revision name."""
    headers = resolved_storage_headers(config, conversation)
    try:
        with tempfile.TemporaryDirectory(prefix="submission-file-") as directory:
            target = Path(directory) / frozen_script_name
            target.write_bytes(Path(script_path).read_bytes())
            upload_local_file_to_pyromind(
                local_path=target,
                target_dir=output_dir,
                storage_base_url=config.storage_base_url,
                headers=headers,
                timeout=float(config.timeout),
            )
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Failed to stage submission file into {output_dir}: {exc}"
        ) from exc


def register_submitted_task(
    conversation: BaseConversation, task_id: str, status: str
) -> None:
    conversation.register_active_long_task(
        ActiveLongTask(task_id=task_id, kind="data_preparation", status=status)
    )


class SubmissionObservation(Observation):
    """Result of a platform pipeline submission."""

    status: str = Field(default="Unknown")
    task_id: str | None = Field(default=None)
    run_id: str | None = Field(default=None)
    output_dir: str | None = Field(default=None)
    resumed: bool = Field(default=False)
    execution_revision: int | None = Field(default=None)
    failure_stage: str | None = Field(default=None)
    retryable: bool | None = Field(default=None)
    validation_errors: list[WorkflowValidationIssue] = Field(default_factory=list)
    validation_warnings: list[WorkflowValidationIssue] = Field(default_factory=list)

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        content = list(super().to_llm_content)
        if self.failure_stage or self.validation_errors or self.validation_warnings:
            content.append(
                TextContent(
                    text=self.model_dump_json(
                        include={
                            "failure_stage",
                            "retryable",
                            "validation_errors",
                            "validation_warnings",
                        },
                        exclude_none=True,
                    )
                )
            )
        return content

    @property
    def visualize(self) -> Text:
        text = Text()
        style = "green" if self.status != "Failed" else "red"
        text.append(f"Submission: {self.status}\n", style=style)
        if self.run_id:
            text.append(f"run_id={self.run_id}\n")
        if self.output_dir:
            text.append(f"output_dir={self.output_dir}\n")
        text.append(self.text)
        return text


class SubmissionFiles:
    """Resolve submission inputs to host paths.

    Local workspaces resolve in place. A remote (sandbox) workspace has no host
    filesystem, so inputs are staged into a private host directory that lives
    for the duration of the submission.
    """

    def __init__(
        self,
        conversation: BaseConversation,
        staging: Path | None,
    ) -> None:
        self._conversation = conversation
        self._workspace = cast(Any, conversation).workspace
        self._staging = staging

    def resolve(self, path: str) -> tuple[Path, str]:
        """Return the host path holding ``path`` and its workspace-relative form."""
        if self._staging is None:
            resolved = resolve_workspace_file(
                self._conversation,
                path,
                allow_virtual_conversation_path=True,
            )
            return resolved, workspace_relative_path(self._conversation, resolved)
        try:
            staged = stage_workspace_path(
                self._workspace, path, destination=self._staging
            )
        except WorkspaceStagingError as exc:
            raise ValueError(str(exc)) from exc
        relative, from_storage = resolve_workspace_path(self._workspace, path)
        mount = getattr(self._workspace, "storage_path", None)
        display = (
            f"{str(mount).rstrip('/')}/{relative.as_posix()}"
            if from_storage and mount
            else relative.as_posix()
        )
        return staged, display

    def read_text(self, path: str) -> str:
        """Return the text of a workspace file that the submission reads."""
        return self.resolve(path)[0].read_text()


@contextmanager
def submission_files(conversation: BaseConversation) -> Iterator[SubmissionFiles]:
    workspace = cast(Any, conversation).workspace
    if not is_remote_workspace(workspace):
        yield SubmissionFiles(conversation, staging=None)
        return
    with staged_remote_dir() as staging:
        yield SubmissionFiles(conversation, staging=staging)


MODEL_PREPARATION_NODES = frozenset(
    {"CloneAndCacheModel", "DownloadAndCacheModel", "ModelMergeLoraNode"}
)


def validate_pipeline_topology(
    workflow: dict[str, Any], *, require_inference: bool = False
) -> None:
    nodes, edges = workflow.get("nodes", []), workflow.get("edges", [])
    kinds = [node["data"]["nodeType"] for node in nodes]
    if not require_inference and kinds == ["CustomCommandCPUNode"] and not edges:
        return
    preparation = [
        node for node in nodes if node["data"]["nodeType"] in MODEL_PREPARATION_NODES
    ]
    expected = ["CustomCommandCPUNode", "VLLMInference"]
    if len(preparation) == 1:
        expected.append(preparation[0]["data"]["nodeType"])
    if sorted(kinds) != sorted(expected) or len(edges) != len(nodes) - 1:
        raise ValueError(
            "evaluation requires inference -> CPU with at most one "
            "clone or download node"
        )
    if len({str(node["id"]) for node in nodes}) != len(nodes):
        raise ValueError("evaluation node IDs must be unique")
    source = next(node for node in nodes if node["data"]["nodeType"] == "VLLMInference")
    target = next(
        node for node in nodes if node["data"]["nodeType"] == "CustomCommandCPUNode"
    )
    expected_edges = {(str(source["id"]), str(target["id"]), "endpoint", "param")}
    if preparation:
        expected_edges.add(
            (
                str(preparation[0]["id"]),
                str(source["id"]),
                "merged_model_path"
                if preparation[0]["data"]["nodeType"] == "ModelMergeLoraNode"
                else "model_path",
                "model_path",
            )
        )
    actual_edges = {
        (
            str(edge.get("source")),
            str(edge.get("target")),
            edge.get("sourceHandle"),
            edge.get("targetHandle"),
        )
        for edge in edges
    }
    if actual_edges != expected_edges:
        raise ValueError(
            "pipeline must connect preparation.model_path to inference.model_path "
            "and inference.endpoint to CPU.param"
        )
