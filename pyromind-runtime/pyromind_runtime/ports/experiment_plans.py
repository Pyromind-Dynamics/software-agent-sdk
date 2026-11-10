from __future__ import annotations

from typing import Protocol

from pyromind_runtime.domain.experiment_plans import (
    PlanOwner,
)
from pyromind_runtime.domain.preparation import ProjectView


class ExperimentPlanProjectBinder(Protocol):
    def bind_preparation(
        self, project: ProjectView, owner: PlanOwner, *, cluster: str
    ) -> None: ...
