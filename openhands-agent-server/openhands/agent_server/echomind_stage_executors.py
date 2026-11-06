from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from openhands.agent_server.echomind_projects import (
    ProjectInfo,
    ProjectNode,
)


class StageArtifactPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    format: Literal["json"] = "json"
    schema_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    content: dict[str, Any]


class StageExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message: str
    input_refs: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)
    artifacts: list[StageArtifactPayload] = Field(default_factory=list)


class NodeExecutionContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    run_id: str
    project: ProjectInfo
    node: ProjectNode


class ProjectNodeExecutor(Protocol):
    kind: str

    def run(
        self,
        *,
        context: NodeExecutionContext,
    ) -> StageExecutionResult: ...


class ContractProbeExecutor:
    kind = "contract_probe"

    def run(
        self,
        *,
        context: NodeExecutionContext,
    ) -> StageExecutionResult:
        input_refs = [
            resource["uri"]
            for resource in context.node.config.get("resources", [])
            if isinstance(resource, dict)
            and isinstance(resource.get("uri"), str)
            and resource["uri"]
        ]
        return StageExecutionResult(
            message=(
                f"{context.node.title} node validation completed. "
                "No business stage was executed."
            ),
            input_refs=input_refs,
            artifacts=[
                StageArtifactPayload(
                    kind="node_validation_report",
                    content={
                        "schema_version": 1,
                        "validation_mode": "contract_probe",
                        "project_id": context.project.project.project_id,
                        "graph_id": context.project.graph.graph_id,
                        "run_id": context.run_id,
                        "node_id": context.node.id,
                        "node_type": context.node.type,
                        "task_type": context.node.task_type,
                        "executor_kind": context.node.executor.kind,
                        "input_refs": input_refs,
                    },
                )
            ],
        )


_CONTRACT_PROBE_EXECUTOR = ContractProbeExecutor()
_EXECUTORS: dict[str, ProjectNodeExecutor] = {
    "contract_probe": _CONTRACT_PROBE_EXECUTOR,
    "shell": _CONTRACT_PROBE_EXECUTOR,
}


def get_project_node_executor(executor_name: str) -> ProjectNodeExecutor:
    executor = _EXECUTORS.get(executor_name)
    if executor is None:
        raise ValueError(f"Unknown project node executor: {executor_name}")
    return executor
