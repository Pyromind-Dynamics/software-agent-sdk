"""Derived SQL projection of an archived experiment."""

from typing import Protocol

from pyromind_runtime.domain.experiment_plans import PlanOwner
from pyromind_runtime.domain.experiments import ExperimentVersion


class ExperimentIndex(Protocol):
    def save(
        self,
        owner: PlanOwner,
        experiment: ExperimentVersion,
        *,
        archive_ref: str,
        manifest_digest: str,
        source_sequence: int,
        observed_at: int,
    ) -> None: ...
