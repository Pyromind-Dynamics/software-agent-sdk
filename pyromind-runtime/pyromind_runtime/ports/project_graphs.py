from __future__ import annotations

from typing import Protocol

from pyromind_runtime.domain.experiment_plans import PlanOwner
from pyromind_runtime.domain.project_graphs import GraphPatch, ProjectGraphView


class ProjectGraphStore(Protocol):
    def get(self, owner: PlanOwner, project_id: str) -> ProjectGraphView: ...

    def apply(
        self,
        owner: PlanOwner,
        project_id: str,
        patch: GraphPatch,
    ) -> ProjectGraphView: ...
