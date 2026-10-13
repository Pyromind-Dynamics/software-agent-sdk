"""Harness-neutral version records. Credentials are never part of these models."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Literal
from uuid import uuid4

from pydantic import Field, JsonValue, model_validator

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.content import JsonObject


def new_id() -> str:
    return str(uuid4())


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def check_persistable(value: JsonValue) -> None:
    """Reject credential containers and temporary URLs, including nested configs."""
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.lower().replace("-", "_")
            if normalized in {
                "cookie",
                "authorization",
                "token",
                "access_token",
                "refresh_token",
                "api_key",
                "password",
                "secret",
                "headers",
                "env",
                "environment_variables",
            }:
                raise ValueError(f"credential-bearing field cannot be versioned: {key}")
            check_persistable(item)
    elif isinstance(value, list):
        for item in value:
            check_persistable(item)
    elif isinstance(value, str) and value.startswith(("https://", "http://")):
        if "?" in value or "@" in value.split("/", 3)[2]:
            raise ValueError("persist a stable storage reference, not a signed URL")


type ArtifactKind = Literal[
    "dataset",
    "annotation",
    "code",
    "workflow",
    "config",
    "model",
    "evaluation",
    "trajectory",
    "report",
]
type VersionState = Literal["pending", "ready", "reference_only", "failed"]


class Iteration(ContractModel):
    id: str = Field(default_factory=new_id)
    parent_id: str | None = None
    goal: str = "建立基线"


class Selection(ContractModel):
    command_id: str
    slot: str
    previous: str | None = Field(default=None, alias="from")
    to: str
    reason: str = ""
    at: int = Field(default_factory=now_ms)


class ProjectContext(ContractModel):
    schema_version: Literal[1] = 1
    current_iteration_id: str
    iterations: list[Iteration]
    selected_versions: dict[str, str] = Field(default_factory=dict)
    selection_history: list[Selection] = Field(default_factory=list)


class VersionProject(ContractModel):
    id: str = Field(default_factory=new_id)
    conversation_id: str
    owner_user_id: str
    name: str = "模型项目"
    context: ProjectContext
    revision: int = 0
    created_at: int = Field(default_factory=now_ms)


class RunSpec(ContractModel):
    schema_version: Literal[1] = 1
    kind: str
    iteration_id: str
    inputs: dict[str, str] = Field(default_factory=dict)
    parameters: JsonObject = Field(default_factory=dict)
    environment: JsonObject = Field(default_factory=dict)


class RunSource(ContractModel):
    kind: str
    source_id: str
    execution_revision: int = 1
    backend: str = "pyromind"
    task_id: str | None = None
    retry_of: str | None = None


class VersionRun(ContractModel):
    id: str = Field(default_factory=new_id)
    project_id: str
    request_key: str
    spec: RunSpec
    source: RunSource
    created_at: int = Field(default_factory=now_ms)


class VersionParent(ContractModel):
    version_id: str
    relation: str = "derived_from"


class SnapshotSource(ContractModel):
    storage_ref: str | None = None
    text: str | None = None
    filename: str = "content.txt"
    # Dataset manifests may reference media outside their directory. Such inputs
    # remain references until a backend can prove a complete dependency snapshot.
    reference_only: bool = False

    @model_validator(mode="after")
    def one_source(self) -> SnapshotSource:
        if (self.storage_ref is None) == (self.text is None):
            raise ValueError("provide exactly one of storage_ref or text")
        if (
            not self.filename
            or "/" in self.filename
            or "\\" in self.filename
            or self.filename in {".", ".."}
        ):
            raise ValueError("filename must be a basename")
        if self.storage_ref is not None:
            check_persistable(self.storage_ref)
        return self


class VersionManifest(ContractModel):
    schema_version: Literal[1] = 1
    storage_ref: str | None = None
    manifest_ref: str | None = None
    digest: str | None = None
    parents: list[VersionParent] = Field(default_factory=list)
    output_role: str | None = None
    summary: JsonObject = Field(default_factory=dict)
    source: SnapshotSource
    target: str
    request_digest: str
    error: str | None = None


class ArtifactVersion(ContractModel):
    id: str = Field(default_factory=new_id)
    project_id: str
    producer_run_id: str | None = None
    kind: ArtifactKind
    logical_key: str = Field(min_length=1)
    version_no: int = Field(ge=1)
    registration_key: str = Field(min_length=1)
    state: VersionState = "pending"
    manifest: VersionManifest
    created_at: int = Field(default_factory=now_ms)


class SaveVersion(ContractModel):
    kind: ArtifactKind
    logical_key: str = Field(min_length=1)
    source: SnapshotSource
    parents: list[VersionParent] = Field(default_factory=list)
    summary: JsonObject = Field(default_factory=dict)
    producer_run_id: str | None = None
    output_role: str | None = None


class SnapshotResult(ContractModel):
    storage_ref: str
    manifest_ref: str | None = None
    digest: str | None = None
    reference_only: bool = False


class VersionToolRequest(ContractModel):
    action: Literal["save", "list", "compare", "select", "new_iteration", "recover"]
    arguments: JsonObject = Field(default_factory=dict)
