from __future__ import annotations

from typing import Protocol

from pyromind_runtime.domain.experiment_plans import (
    ExperimentPlanView,
    PlanOwner,
    PlanProposal,
    PlanRevision,
)


class ExperimentPlanStore(Protocol):
    def close(self) -> None: ...

    def view(self, owner: PlanOwner) -> ExperimentPlanView: ...

    def revision(self, owner: PlanOwner, revision: int) -> PlanRevision: ...

    def propose(
        self,
        owner: PlanOwner,
        proposal: PlanProposal,
        *,
        source_proposal_id: str | None,
        initial: bool,
    ) -> PlanProposal: ...

    def confirm(
        self,
        owner: PlanOwner,
        proposal_id: str,
        expected_revision: int,
        command_id: str,
    ) -> PlanRevision: ...

    def reject(self, owner: PlanOwner, proposal_id: str) -> None: ...
