from __future__ import annotations

from pathlib import Path

from pyromind_runtime.domain.experiment_plans import PlanOwner
from pyromind_runtime.domain.project_graphs import GraphPatch, ProjectGraphView

from openhands.agent_server.echomind_projects import ProjectInfo, ProjectStore


class EchoMindProjectGraphStore:
    def __init__(self, conversations_dir: Path | str) -> None:
        self._conversations_dir = Path(conversations_dir)

    def get(self, owner: PlanOwner, project_id: str) -> ProjectGraphView:
        project = self._load(owner, project_id)
        return self._store().get_graph(
            project.project.project_id,
            owner_user_id=owner.user_id,
        )

    def apply(
        self,
        owner: PlanOwner,
        project_id: str,
        patch: GraphPatch,
    ) -> ProjectGraphView:
        self._load(owner, project_id)
        return self._store().apply_graph_patch(
            project_id,
            owner_user_id=owner.user_id,
            patch=patch,
        )

    def _load(self, owner: PlanOwner, project_id: str) -> ProjectInfo:
        project = self._store().load(project_id, owner_user_id=owner.user_id)
        if project.project.echomind_conversation_id != owner.conversation_id:
            raise PermissionError("Project belongs to another conversation")
        return project

    def _store(self) -> ProjectStore:
        return ProjectStore.for_conversations_dir(self._conversations_dir)
