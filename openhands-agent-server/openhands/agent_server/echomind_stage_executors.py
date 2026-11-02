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
    content: dict[str, Any]


class StageExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message: str
    input_refs: list[str] = Field(default_factory=list)
    artifacts: list[StageArtifactPayload] = Field(default_factory=list)


class ProjectNodeExecutor(Protocol):
    def run(
        self,
        *,
        project: ProjectInfo,
        node: ProjectNode,
    ) -> StageExecutionResult: ...


class ContractProbeExecutor:
    def run(
        self,
        *,
        project: ProjectInfo,
        node: ProjectNode,
    ) -> StageExecutionResult:
        input_refs = [
            resource["uri"]
            for resource in node.config.get("resources", [])
            if isinstance(resource, dict)
            and isinstance(resource.get("uri"), str)
            and resource["uri"]
        ]
        return StageExecutionResult(
            message=(
                f"{node.title} node validation completed. "
                "No business stage was executed."
            ),
            input_refs=input_refs,
            artifacts=[
                StageArtifactPayload(
                    kind="node_validation_report",
                    content={
                        "schema_version": 1,
                        "validation_mode": "contract_probe",
                        "project_id": project.project.project_id,
                        "graph_id": project.graph.graph_id,
                        "node_id": node.id,
                        "node_type": node.type,
                        "executor": node.executor,
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
