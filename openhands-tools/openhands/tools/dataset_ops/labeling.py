"""Strict model-response parsing and append-only failure accounting."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from openhands.tools.dataset_ops.contracts import (
    AnalyzedSample,
    LabelAssignment,
    Taxonomy,
)


_assignment_adapter = TypeAdapter(tuple[LabelAssignment, ...])


def parse_label_response(
    payload: str | dict[str, Any], *, sample_id: str, taxonomy: Taxonomy
) -> AnalyzedSample:
    """Parse one strict JSON label response and enforce the taxonomy."""

    value = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(value, dict) or set(value) != {"assignments"}:
        raise ValueError("label response must contain only 'assignments'")
    try:
        assignments = _assignment_adapter.validate_python(value["assignments"])
    except ValidationError as exc:
        raise ValueError(f"invalid label response schema: {exc}") from exc

    dimensions = {item.id: item for item in taxonomy.dimensions}
    if {item.dimension for item in assignments} != set(dimensions):
        raise ValueError("label response must assign every taxonomy dimension once")
    if len(assignments) != len(dimensions):
        raise ValueError("label response contains a duplicate dimension")
    for assignment in assignments:
        dimension = dimensions[assignment.dimension]
        known = {label.id for label in dimension.labels}
        selected = set(assignment.labels)
        if not selected.issubset(known):
            raise ValueError(
                f"unknown labels for {dimension.id!r}: {sorted(selected - known)}"
            )
        if dimension.assignment == "single" and len(selected) != 1:
            raise ValueError(f"dimension {dimension.id!r} requires one label")
        if len(selected) != len(assignment.labels):
            raise ValueError(f"dimension {dimension.id!r} contains duplicate labels")
    return AnalyzedSample(sample_id=sample_id, assignments=assignments, source="model")


class FailureLedger:
    """Thread-safe JSONL ledger; failed records are never silently dropped."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def record(
        self,
        *,
        sample_id: str,
        stage: str,
        error_code: str,
        message: str,
        raw_output: str | None = None,
    ) -> None:
        value = {
            "schema_version": 1,
            "sample_id": sample_id,
            "stage": stage,
            "error_code": error_code,
            "message": message,
            "raw_output_preview": raw_output[:500] if raw_output else None,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
