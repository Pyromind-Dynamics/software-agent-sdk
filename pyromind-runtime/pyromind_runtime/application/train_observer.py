"""Bounded metric reads with cursors tied to authorized query identity."""

import asyncio
import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime

from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.observations import MetricConfig, RunBinding, SeriesPage
from pyromind_runtime.ports.observation_source import (
    ObservationUnavailableError,
    TrainingMetricSource,
)


def record_version(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()[:24]


def observed_at() -> str:
    return datetime.now(UTC).isoformat()


class ObservationCursor:
    def __init__(self, secret: bytes) -> None:
        self.secret = secret

    def encode(self, position: int, scope: str, source: str = "") -> str:
        payload = json.dumps([position, scope, source], separators=(",", ":")).encode()
        signature = hmac.digest(self.secret, payload, "sha256")
        return base64.urlsafe_b64encode(signature + payload).decode()

    def decode(self, cursor: str | None, scope: str) -> tuple[int, str]:
        if cursor is None:
            return 0, ""
        try:
            if len(cursor) > 2048:
                raise ValueError("cursor too long")
            raw = base64.b64decode(cursor, altchars=b"-_", validate=True)
            signature, payload = raw[:32], raw[32:]
            if not hmac.compare_digest(
                signature, hmac.digest(self.secret, payload, "sha256")
            ):
                raise ValueError("invalid signature")
            position, stored_scope, source = json.loads(payload)
            if (
                type(position) is not int
                or not 0 <= position <= 2**53
                or stored_scope != scope
                or not isinstance(source, str)
            ):
                raise ValueError("cursor scope mismatch")
            return position, source
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid or expired observations cursor") from exc


class TrainObserver:
    def __init__(
        self, source: TrainingMetricSource | None, cursors: ObservationCursor
    ) -> None:
        self.source = source
        self.cursors = cursors
        self._inflight: set[str] = set()

    async def series(
        self,
        *,
        project_id: str,
        config_version: str,
        binding: RunBinding,
        metrics: tuple[MetricConfig, ...],
        context: RequestContext,
        cursor: str | None,
        limit: int,
    ) -> SeriesPage:
        if not metrics or len(metrics) > 20 or not len(metrics) <= limit <= 1000:
            raise ValueError("Select 1–20 configured metrics and limit 1–1000")
        binding_identity = [
            binding.project_node_id,
            binding.run_id,
            binding.task_id,
            binding.platform_node_id,
        ]
        scope = record_version(
            [
                context.user_id,
                context.profile_id,
                context.x_cluster,
                project_id,
                config_version,
                binding_identity,
                [metric.model_dump() for metric in metrics],
            ]
        )
        start, previous_source = self.cursors.decode(cursor, scope)
        response = SeriesPage(
            observation_status="not_configured",
            project_id=project_id,
            project_node_id=binding.project_node_id,
            run_id=binding.run_id,
            config_version=config_version,
            observed_at=observed_at(),
        )
        if self.source is None:
            return response
        target = record_version(
            [context.user_id, context.x_cluster, project_id, binding_identity]
        )
        unavailable = response.model_copy(
            update={
                "observation_status": "unavailable",
                "message": "Training source unavailable; retry later",
            }
        )
        if target in self._inflight:
            return unavailable
        self._inflight.add(target)
        try:
            page = await asyncio.wait_for(
                self.source.read(
                    binding,
                    metrics,
                    start,
                    max(1, min(200, limit // len(metrics))),
                    context,
                    previous_source=previous_source,
                ),
                timeout=12,
            )
        except (TimeoutError, ObservationUnavailableError):
            return unavailable
        finally:
            self._inflight.discard(target)
        if previous_source and page.source_run_id != previous_source:
            raise ValueError("Source run changed; restart this metric query")
        points = {point.source_point_id: point for point in page.points}
        keys = {metric.key for metric in metrics}
        if (
            len(points) > limit
            or len(page.model_dump_json().encode()) > 1024 * 1024
            or page.next_step < start
            or any(
                point.key not in keys or point.source_run_id != page.source_run_id
                for point in points.values()
            )
        ):
            return unavailable
        return response.model_copy(
            update={
                "points": tuple(points.values()),
                "next_cursor": self.cursors.encode(
                    page.next_step, scope, page.source_run_id
                ),
                "truncated": page.has_more,
                "gap": page.gap,
                "observation_status": "available",
            }
        )
