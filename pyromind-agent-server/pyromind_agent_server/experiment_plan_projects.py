from __future__ import annotations

from pathlib import Path

from pyromind_runtime.domain.experiment_plans import PlanOwner, PlanRevision

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
