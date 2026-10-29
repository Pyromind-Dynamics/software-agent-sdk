from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from openhands.agent_server.echomind_projects import (
    ProjectInfo,
    ProjectNode,
    ProjectNodeType,
)


class ProjectNodeExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    node_id: str
    node_type: ProjectNodeType
    executor: str
    status: Literal["not_implemented"]
    message: str


class ProjectNodeExecutor(Protocol):
    def run(
        self,
        *,
        project: ProjectInfo,
        node: ProjectNode,
    ) -> ProjectNodeExecutionResult: ...


class ShellProjectNodeExecutor:
    def run(
        self,
        *,
        project: ProjectInfo,
        node: ProjectNode,
    ) -> ProjectNodeExecutionResult:
        return ProjectNodeExecutionResult(
            project_id=project.project.project_id,
            node_id=node.id,
            node_type=node.type,
            executor=node.executor,
            status="not_implemented",
            message=(
                f"{node.title} is registered but its executor is still a shell. "
                "No external task or artifact was created."
            ),
        )


_SHELL_EXECUTOR = ShellProjectNodeExecutor()


def execute_project_node(
    *,
    project: ProjectInfo,
    node: ProjectNode,
) -> ProjectNodeExecutionResult:
    if node.executor != "shell":
        raise ValueError(f"Unknown project node executor: {node.executor}")
    return _SHELL_EXECUTOR.run(project=project, node=node)
