from __future__ import annotations

from typing import Protocol, runtime_checkable

from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_plans import PlanOwner
from pyromind_runtime.domain.preparation import (
    InspectResource,
    PreparationPolicy,
    PreparationView,
    ProjectView,
)


class PreparationStore(Protocol):
    def initialize(self, owner: PlanOwner, policy: PreparationPolicy) -> None: ...

    def get(self, owner: PlanOwner) -> PreparationView: ...

    def save(
        self, owner: PlanOwner, view: PreparationView, expected_revision: int
    ) -> PreparationView: ...

    def invalidate(self, owner: PlanOwner, command_id: str) -> int | None: ...

    def add_evidence(
        self, owner: PlanOwner, reference: str, description: str
    ) -> int: ...

    def project(self, owner: PlanOwner) -> ProjectView | None: ...

    def activate(
        self,
        owner: PlanOwner,
        expected_revision: int,
        command_id: str,
        *,
        cluster: str | None = None,
    ) -> ProjectView: ...


class ResourceInspector(Protocol):
    def inspect(
        self, request: InspectResource, context: RequestContext
    ) -> JsonObject: ...


@runtime_checkable
class ResourceMediaReader(Protocol):
    def read_media(self, path: str, context: RequestContext) -> tuple[bytes, str]:
        """Read a bounded raster preview with current request credentials."""
        ...
