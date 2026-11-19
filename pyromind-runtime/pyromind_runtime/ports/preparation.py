from __future__ import annotations

from typing import Protocol, runtime_checkable

from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_plans import PlanOwner
from pyromind_runtime.domain.preparation import (
    PreparationState,
    ProjectView,
)
from pyromind_runtime.domain.reward_bundle import RewardBundle


class PreparationStore(Protocol):
    def initialize(self, owner: PlanOwner) -> None: ...

    def get(self, owner: PlanOwner) -> PreparationState: ...

    def save(
        self, owner: PlanOwner, view: PreparationState, expected_revision: int
    ) -> PreparationState: ...

    def approve_reward(
        self, owner: PlanOwner, request_id: str, bundle: RewardBundle
    ) -> bool: ...

    def invalidate(self, owner: PlanOwner, command_id: str) -> int | None: ...

    def project(self, owner: PlanOwner) -> ProjectView | None: ...

    def activate(
        self,
        owner: PlanOwner,
        expected_revision: int,
        command_id: str,
        *,
        cluster: str | None = None,
    ) -> ProjectView: ...


@runtime_checkable
class ResourceMediaReader(Protocol):
    def read_media(self, path: str, context: RequestContext) -> tuple[bytes, str]:
        """Read a bounded raster preview with current request credentials."""
        ...
