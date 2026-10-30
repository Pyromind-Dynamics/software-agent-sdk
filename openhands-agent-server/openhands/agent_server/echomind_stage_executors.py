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
        return StageExecutionResult(
            message=(
                f"{node.title} contract probe completed. "
                "No business stage was executed."
            ),
            artifacts=[
                StageArtifactPayload(
                    kind="stage_contract_probe",
                    content={
                        "probe_version": 1,
                        "project_id": project.project.project_id,
                        "graph_id": project.graph.graph_id,
                        "node_id": node.id,
                        "node_type": node.type,
                        "executor": node.executor,
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
