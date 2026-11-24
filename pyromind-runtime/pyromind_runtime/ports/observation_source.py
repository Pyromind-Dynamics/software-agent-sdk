from typing import Protocol

from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.observations import (
    MetricConfig,
    RunBinding,
    SourcePage,
)


class ObservationUnavailableError(RuntimeError):
    pass


class ObservationForbiddenError(PermissionError):
    pass


class TrainingMetricSource(Protocol):
    async def read(
        self,
        binding: RunBinding,
        metrics: tuple[MetricConfig, ...],
        start_step: int,
        step_count: int,
        context: RequestContext,
        /,
        *,
        previous_source: str = "",
    ) -> SourcePage: ...
