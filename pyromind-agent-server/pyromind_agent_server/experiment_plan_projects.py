from __future__ import annotations

from pathlib import Path

from pyromind_runtime.domain.experiment_plans import PlanOwner, PlanRevision
from pyromind_runtime.domain.preparation import ProjectView

from openhands.agent_server.echomind_projects import ProjectStore


class EchoMindExperimentPlanProjectBinder:
    def __init__(self, conversations_dir: Path | str) -> None:
        self._conversations_dir = Path(conversations_dir)

    def bind(self, revision: PlanRevision, owner: PlanOwner) -> None:
        ProjectStore.for_conversations_dir(self._conversations_dir).bind_plan(
            revision,
            echomind_conversation_id=owner.conversation_id,
            owner_user_id=owner.user_id,
        )


class EchoMindPreparationProjectMaterializer:
    def __init__(self, conversations_dir: Path | str) -> None:
        self._conversations_dir = Path(conversations_dir)

    def materialize(
        self, project: ProjectView, owner: PlanOwner
    ) -> tuple[str, str] | None:
        info = ProjectStore.for_conversations_dir(self._conversations_dir).bind_project(
            project, owner_user_id=owner.user_id
        )
        targets = {edge.target for edge in info.graph.edges}
        roots = [node for node in info.graph.nodes if node.id not in targets]
        if (
            len(roots) != 1
            or roots[0].launch_mode != "agent"
            or not roots[0].execution_role
        ):
            return None
        return roots[0].id, roots[0].execution_role
