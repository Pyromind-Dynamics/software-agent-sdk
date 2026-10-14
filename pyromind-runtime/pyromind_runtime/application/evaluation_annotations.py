"""Evaluation annotation save protocol (single-writer, Product Store commit).

Protocol invariants:

- Storage holds immutable ``evaluation_annotations.r{N}.jsonl`` /
  ``labeled_evaluation_results.r{N}.jsonl`` objects; names are only ever
  created, and committed objects are never overwritten.
- The conversation Product Store holds the commit state (current revision,
  object references, bounded save history). Reads and downloads resolve
  through it.
- ``save_id`` replays return the stored result when the request content hash
  matches; a mismatching replay is rejected (422). A stale ``base_revision``
  conflicts (409).
- Canonical-name files (``evaluation_annotations.json(l)``,
  ``labeled_evaluation_results.jsonl``) are best-effort convenience views,
  refreshed after every commit and lazily repaired on access. They never
  participate in the commit protocol and are not a recovery source: a
  missing commit state with ``r{N>0}`` objects on Storage is a fault that
  stops writes until explicit recovery.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import UUID

from pydantic import ValidationError

from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.errors import (
    EvaluationRunNotFoundError,
    EvaluationRunUnavailableError,
)
from pyromind_runtime.domain.evaluation import (
    AnnotationLoadResult,
    AnnotationSaveResult,
    AnnotationSummaryResult,
    EvaluationCommitState,
    SaveHistoryEntry,
)
from pyromind_runtime.infrastructure.file_product_store import (
    FileProductStore,
    ProductStoreError,
)
from pyromind_runtime.ports.product_store import EvaluationCommitStore
from pyromind_runtime.ports.storage import (
    DeletableEvaluationStorage,
    EvaluationStorage,
    EvaluationStorageMissingError,
)


logger = logging.getLogger(__name__)

OverallStatus = Literal["completed", "completed_with_open_items", "in_progress"]

SCHEMA_VERSION = 1
REVIEW_SCOPE = "machine_failed"

DECISIONS = {"confirmed_badcase", "evaluation_false_alarm", "needs_discussion"}
FAILURE_MODES = {
    "model_miss",
    "model_false_positive",
    "classification_error",
    "severity_misclassification",
    "gt_issue",
    "rubric_issue",
    "inference_input_issue",
    "report_issue",
    "other",
}
SEVERITIES = {"P0", "P1", "P2", "P3"}
BASE_DISPOSITIONS = {
    "fix_ground_truth",
    "data_augmentation",
    "training_workflow_adjustment",
    "evaluation_rule_fix",
    "needs_analysis",
    "no_action",
}
RESOLUTION_CODES = {
    "gt_error",
    "poor_data",
    "model_gap",
    "coverage_gap",
    "acceptable",
    "uncertain",
}
RESOLUTION_LEGACY_FIELDS = {
    "gt_error": ("evaluation_false_alarm", "fix_ground_truth"),
    "poor_data": ("evaluation_false_alarm", "needs_analysis"),
    "model_gap": ("confirmed_badcase", "training_workflow_adjustment"),
    "coverage_gap": ("confirmed_badcase", "data_augmentation"),
    "acceptable": ("evaluation_false_alarm", "no_action"),
    "uncertain": ("needs_discussion", "needs_analysis"),
}
DATA_QUALITY_REASONS = {
    "needs_cleaning",
    "exclude",
    "missing",
    "format_error",
    "corrupt",
    "other",
}
MAX_COMMENT_CHARS = 2000
MAX_CORRECTED_CHARS = 2000
MAX_LABELS = 50
MAX_PREDICTIONS_BYTES = 256 * 1024 * 1024
MAX_REPORTED_ERRORS = 20
HISTORY_LIMIT = 5
ANNOTATION_KEEP = 3
LABELED_KEEP = 2

RUN_MANIFEST_NAME = "run_manifest.json"
PREDICTIONS_NAME = "predictions.jsonl"
RESOLVED_REPORT_SPEC_NAME = "resolved_report_spec.json"
ANNOTATION_OBJECT_TEMPLATE = "evaluation_annotations.r{}.jsonl"
ANNOTATION_OBJECT_PREFIX = "evaluation_annotations.r"
LABELED_OBJECT_TEMPLATE = "labeled_evaluation_results.r{}.jsonl"
LABELED_OBJECT_PREFIX = "labeled_evaluation_results.r"
VIEW_ANNOTATIONS_JSONL = "evaluation_annotations.jsonl"
VIEW_ANNOTATIONS_JSON = "evaluation_annotations.json"
VIEW_LABELED_JSONL = "labeled_evaluation_results.jsonl"

RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "evaluation_config_sha256",
        "case_id",
        "review_status",
        "machine_result",
        "human_annotation",
    }
)
ANNOTATION_FIELDS = frozenset(
    {
        "decision",
        "resolution_code",
        "resolution_details_version",
        "failure_modes",
        "taxonomy_labels",
        "disposition",
        "severity",
        "comment",
        "corrected_answer",
        "data_quality_reasons",
        "training_suggestion",
        "missing_scenario",
        "target_sample_count",
        "acceptance_reason",
        "verification_items",
        "confirmation_owner",
        "reviewer_id",
        "updated_at",
    }
)


class EvaluationAnnotationError(RuntimeError):
    """Base error; message is safe to surface to the report page."""


class AnnotationNotInitializedError(EvaluationAnnotationError):
    """No annotation products exist for this run (never a validated eval)."""


class AnnotationStateFault(EvaluationAnnotationError):
    """Commit state is missing while later-revision objects exist on Storage.

    Writes stop until explicit recovery; automated rebuild from the lagging
    views could roll back committed annotations.
    """


class AnnotationConflictError(EvaluationAnnotationError):
    """base_revision is behind the committed revision (HTTP 409)."""


class AnnotationReplayMismatchError(EvaluationAnnotationError):
    """A save_id replay arrived with different content (HTTP 422)."""


class AnnotationValidationError(EvaluationAnnotationError):
    """Record-level contract violations (HTTP 422)."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__(_summarize(errors))
        self.errors = errors


def _summarize(errors: list[str]) -> str:
    summary = "annotation records rejected; " + "; ".join(errors[:MAX_REPORTED_ERRORS])
    if len(errors) > MAX_REPORTED_ERRORS:
        summary += f" … and {len(errors) - MAX_REPORTED_ERRORS} more"
    return summary


def canonical_jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    if not records:
        return b""
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    return ("\n".join(lines) + "\n").encode("utf-8")


def request_content_hash(
    save_id: str,
    base_revision: int,
    records: list[dict[str, Any]],
    reviewer_id: str,
) -> str:
    """Pure function of the save request body + authenticated reviewer.

    Including reviewer_id prevents cross-user replay: the same save_id and
    records submitted by different users must produce distinct hashes, so
    the idempotent replay cache cannot return user A's result to user B.
    """
    payload = json.dumps(
        {
            "save_id": save_id,
            "base_revision": base_revision,
            "records": records,
            "reviewer_id": reviewer_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _annotation_without_reviewer(
    annotation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Strip reviewer_id for content comparison.

    Two annotations are considered 'unchanged' if all fields except
    reviewer_id match. This preserves the original reviewer when a
    full-snapshot save re-submits unchanged annotations.
    """
    if not isinstance(annotation, dict):
        return None
    return {key: value for key, value in annotation.items() if key != "reviewer_id"}


def _parse_jsonl(data: bytes, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationAnnotationError(
                f"{source} line {index}: invalid JSON ({exc.msg})"
            ) from exc
        if not isinstance(row, dict):
            raise EvaluationAnnotationError(f"{source} line {index}: not an object")
        rows.append(row)
    return rows


def _prediction_rows(data: bytes) -> list[dict[str, Any]]:
    rows = _parse_jsonl(data, PREDICTIONS_NAME)
    for row in rows:
        if not isinstance(row.get("id"), str):
            raise EvaluationAnnotationError(
                f"{PREDICTIONS_NAME}: row without a string id"
            )
    return rows


def _annotation_complete(annotation: dict[str, Any]) -> bool:
    if not annotation.get("decision") or not annotation.get("disposition"):
        return False
    if annotation.get("decision") == "confirmed_badcase":
        if not annotation.get("failure_modes"):
            return False
        if not annotation.get("severity"):
            return False
    if annotation.get("resolution_details_version") != 1:
        return True
    resolution_code = annotation.get("resolution_code")
    required_fields = {
        "gt_error": "corrected_answer",
        "coverage_gap": "missing_scenario",
        "acceptable": "acceptance_reason",
        "uncertain": "verification_items",
    }
    required_field = (
        required_fields.get(resolution_code)
        if isinstance(resolution_code, str)
        else None
    )
    if required_field and not annotation.get(required_field):
        return False
    if resolution_code == "poor_data" and not annotation.get("data_quality_reasons"):
        return False
    return True


def _string_labels(value: Any) -> set[str] | None:
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(item, str) and item.strip() for item in value):
        return None
    return set(value)


def _annotation_value_errors(
    annotation: dict[str, Any],
    locate: str,
    taxonomy: set[str],
    dispositions: set[str],
) -> list[str]:
    errors: list[str] = []
    for field in sorted(set(annotation) - ANNOTATION_FIELDS):
        errors.append(f"{locate}: unexpected human_annotation field {field}")
    decision = annotation.get("decision")
    if decision and decision not in DECISIONS:
        errors.append(f"{locate}: invalid decision {decision!r}")
    resolution_code = annotation.get("resolution_code")
    if resolution_code is not None and resolution_code not in RESOLUTION_CODES:
        errors.append(f"{locate}: invalid resolution_code {resolution_code!r}")
    details_version = annotation.get("resolution_details_version")
    if details_version is not None and details_version != 1:
        errors.append(f"{locate}: resolution_details_version must be 1")
    disposition = annotation.get("disposition")
    if disposition and disposition not in dispositions:
        errors.append(f"{locate}: invalid disposition {disposition!r}")
    if resolution_code in RESOLUTION_LEGACY_FIELDS:
        expected_decision, expected_disposition = RESOLUTION_LEGACY_FIELDS[
            resolution_code
        ]
        if decision != expected_decision or disposition != expected_disposition:
            errors.append(
                f"{locate}: resolution_code {resolution_code!r} requires decision"
                f" {expected_decision!r} and disposition {expected_disposition!r}"
            )
    for field_name, allowed in (
        ("failure_modes", FAILURE_MODES),
        ("taxonomy_labels", taxonomy),
        ("data_quality_reasons", DATA_QUALITY_REASONS),
    ):
        values = annotation.get(field_name) or []
        if not isinstance(values, list):
            errors.append(f"{locate}: {field_name} is not a list")
            continue
        if len(values) > MAX_LABELS:
            errors.append(f"{locate}: {field_name} exceeds {MAX_LABELS} labels")
        for label in values:
            if not isinstance(label, str) or label not in allowed:
                errors.append(f"{locate}: invalid {field_name[:-1]} {label!r}")
    severity = annotation.get("severity")
    if severity is not None and (
        not isinstance(severity, str) or severity not in SEVERITIES
    ):
        errors.append(f"{locate}: invalid severity {severity!r}")
    for field_name, limit in (
        ("comment", MAX_COMMENT_CHARS),
        ("corrected_answer", MAX_CORRECTED_CHARS),
        ("training_suggestion", MAX_COMMENT_CHARS),
        ("missing_scenario", MAX_COMMENT_CHARS),
        ("acceptance_reason", MAX_COMMENT_CHARS),
        ("verification_items", MAX_COMMENT_CHARS),
        ("confirmation_owner", 200),
    ):
        value = annotation.get(field_name)
        if value is not None and (not isinstance(value, str) or len(value) > limit):
            errors.append(
                f"{locate}: {field_name} must be a string of at most {limit} characters"
            )
    target_sample_count = annotation.get("target_sample_count")
    if target_sample_count is not None and (
        not isinstance(target_sample_count, int)
        or isinstance(target_sample_count, bool)
        or not 1 <= target_sample_count <= 1_000_000
    ):
        errors.append(
            f"{locate}: target_sample_count must be an integer from 1 to 1000000"
        )
    if annotation.get("reviewer_id") is not None:
        errors.append(f"{locate}: reviewer_id must be null")
    updated_at = annotation.get("updated_at")
    if not isinstance(updated_at, str):
        errors.append(f"{locate}: updated_at must be a timestamp string")
    else:
        try:
            parsed = datetime.fromisoformat(updated_at)
        except ValueError:
            parsed = None
        if parsed is None or parsed.tzinfo is None:
            errors.append(f"{locate}: updated_at must be timezone-aware")
    return errors


def _validate_records(
    records: list[dict[str, Any]],
    *,
    run_id: str,
    evaluation_config_sha256: str,
    failed_ids: list[str],
    annotation_schema: dict[str, Any],
) -> None:
    errors: list[str] = []
    taxonomy = _string_labels(annotation_schema.get("taxonomy_labels")) or set()
    dispositions = BASE_DISPOSITIONS | (
        _string_labels(annotation_schema.get("dispositions")) or set()
    )
    record_ids = [record.get("case_id") for record in records]
    if record_ids != failed_ids:
        errors.append(
            f"records must cover exactly the {len(failed_ids)} machine-failed"
            f" cases in predictions order (got {len(records)} records)"
        )
        failed = set(failed_ids)
        seen: set[str] = set()
        for index, case_id in enumerate(record_ids, 1):
            locate = f"record {index} ({case_id})"
            if not isinstance(case_id, str):
                errors.append(f"{locate}: case_id is not a failed case of this run")
                continue
            if case_id not in failed:
                errors.append(f"{locate}: case_id is not a failed case of this run")
                continue
            if case_id in seen:
                errors.append(f"{locate}: duplicate case_id")
            seen.add(case_id)
    for index, record in enumerate(records, 1):
        case_id = record.get("case_id")
        locate = f"case {case_id}" if isinstance(case_id, str) else f"record {index}"
        for field in sorted(set(record) - RECORD_FIELDS):
            errors.append(f"{locate}: unexpected field {field}")
        if record.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"{locate}: schema_version mismatch")
        if record.get("run_id") != run_id:
            errors.append(f"{locate}: run_id mismatch")
        if record.get("evaluation_config_sha256") != evaluation_config_sha256:
            errors.append(f"{locate}: evaluation_config_sha256 mismatch")
        status = record.get("review_status")
        annotation = record.get("human_annotation")
        if status == "reviewed":
            if not isinstance(annotation, dict):
                errors.append(f"{locate}: reviewed but human_annotation is missing")
                continue
            if not _annotation_complete(annotation):
                errors.append(
                    f"{locate}: reviewed record is incomplete (decision and"
                    " disposition are always required; confirmed badcase also"
                    " requires failure_modes and severity)"
                )
        elif status == "in_progress":
            if not isinstance(annotation, dict):
                errors.append(
                    f"{locate}: in_progress record must carry its partial"
                    " human_annotation"
                )
                continue
        elif status == "unreviewed":
            if annotation is not None:
                errors.append(f"{locate}: unreviewed but human_annotation is set")
            continue
        else:
            errors.append(f"{locate}: invalid review_status {status!r}")
            continue
        errors.extend(
            _annotation_value_errors(annotation, locate, taxonomy, dispositions)
        )
    if errors:
        raise AnnotationValidationError(errors)


def _machine_result(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(row.get("passed")),
        "reference": row.get("reference"),
        "prediction": row.get("prediction"),
        "overall_score": row.get("overall_score"),
        "failure_tags": [
            rubric["name"]
            for rubric in row.get("rubric_results") or []
            if isinstance(rubric, dict)
            and rubric.get("name")
            and not rubric.get("passed")
        ],
    }


def merge_labeled_results(
    prediction_rows: list[dict[str, Any]],
    annotation_records: list[dict[str, Any]],
    *,
    run_id: str,
    evaluation_config_sha256: str,
) -> list[dict[str, Any]]:
    """Full prediction rows plus the human-review slot, in predictions order."""
    by_case = {record["case_id"]: record for record in annotation_records}
    merged: list[dict[str, Any]] = []
    for row in prediction_rows:
        record = by_case.get(row["id"])
        review_status = "not_in_scope"
        annotation: dict[str, Any] | None = None
        if record is not None:
            review_status = record["review_status"]
            annotation = record.get("human_annotation")
        merged.append(
            {
                **row,
                "run_id": run_id,
                "evaluation_config_sha256": evaluation_config_sha256,
                "human_review": {
                    "review_status": review_status,
                    "annotation": annotation,
                },
            }
        )
    return merged


def _counts(records: list[dict[str, Any]]) -> tuple[int, int, int, OverallStatus]:
    reviewed = sum(
        1
        for record in records
        if record.get("review_status") == "reviewed"
        and isinstance(record.get("human_annotation"), dict)
        and _annotation_complete(record["human_annotation"])
    )
    in_progress = sum(
        1 for record in records if record.get("review_status") == "in_progress"
    )
    unreviewed = len(records) - reviewed - in_progress
    overall: OverallStatus
    # reviewed == total includes the empty review set (an all-pass run has
    # nothing to review and is completed by definition).
    if reviewed == len(records):
        overall = (
            "completed_with_open_items" if _has_open_items(records) else "completed"
        )
    else:
        overall = "in_progress"
    return reviewed, in_progress, unreviewed, overall


def _has_open_items(records: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(record.get("human_annotation"), dict)
        and record["human_annotation"].get("decision") == "needs_discussion"
        for record in records
    )


def _object_revision(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix) or not name.endswith(".jsonl"):
        return None
    body = name[len(prefix) : -len(".jsonl")]
    try:
        return int(body)
    except ValueError:
        return None


class EvaluationLockRegistry:
    """Shared per-run lock registry for one server process.

    Services for the same run must share this registry to serialize reads
    and writes. Separate registries do not provide mutual exclusion.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def for_run(self, run_id: str) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(run_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[run_id] = lock
            return lock


class EvaluationAnnotationService:
    """Single-writer annotation save/load for one conversation's runs.

    Writes are serialized per run through the (shared) lock registry;
    cross-instance mutual exclusion requires conversation affinity or an
    external single-writer guarantee.
    """

    def __init__(
        self,
        *,
        storage: EvaluationStorage,
        commits: EvaluationCommitStore,
        locks: EvaluationLockRegistry | None = None,
    ):
        self._storage = storage
        self._commits = commits
        self._locks = locks if locks is not None else EvaluationLockRegistry()

    def load(self, *, run_id: str, output_dir: str) -> AnnotationLoadResult:
        prefix = output_dir.rstrip("/")
        # The per-run lock is shared with save: concurrent saves advance the
        # revision and garbage-collect superseded objects, so an unlocked
        # load could resolve a revision whose objects were just deleted.
        with self._lock_for(run_id):
            state = self._commit_state(run_id, prefix)
            annotations_bytes = self._read_committed(
                run_id, prefix, state.annotations_object
            )
            annotations = _parse_jsonl(annotations_bytes, state.annotations_object)
            reviewed, in_progress, unreviewed, overall = _counts(annotations)
            self._repair_views(prefix, state, annotations_bytes)
            return AnnotationLoadResult(
                revision=state.revision,
                records=annotations,
                reviewed=reviewed,
                in_progress=in_progress,
                unreviewed=unreviewed,
                overall_status=overall,
            )

    def load_summary(self, *, run_id: str, output_dir: str) -> AnnotationSummaryResult:
        prefix = output_dir.rstrip("/")
        with self._lock_for(run_id):
            state = self._commit_state(run_id, prefix)
            for entry in reversed(state.history):
                if entry.revision == state.revision:
                    result = entry.result
                    return AnnotationSummaryResult(
                        revision=state.revision,
                        reviewed=result.reviewed,
                        in_progress=result.in_progress,
                        unreviewed=result.unreviewed,
                        overall_status=result.overall_status,
                    )
            annotations = _parse_jsonl(
                self._read_committed(run_id, prefix, state.annotations_object),
                state.annotations_object,
            )
            reviewed, in_progress, unreviewed, overall = _counts(annotations)
            return AnnotationSummaryResult(
                revision=state.revision,
                reviewed=reviewed,
                in_progress=in_progress,
                unreviewed=unreviewed,
                overall_status=overall,
            )

    def export(self, *, run_id: str, output_dir: str, filename: str) -> bytes:
        """Assemble one canonical export from the committed revision.

        Downloads never touch the best-effort canonical-name views: they are
        assembled from the objects the commit state references, so a view
        that failed to refresh cannot make an export serve stale data.
        """
        prefix = output_dir.rstrip("/")
        with self._lock_for(run_id):
            state = self._commit_state(run_id, prefix)
            annotations_bytes = self._read_committed(
                run_id, prefix, state.annotations_object
            )
            if filename == VIEW_ANNOTATIONS_JSONL:
                return annotations_bytes
            labeled_bytes = self._read_committed(run_id, prefix, state.labeled_object)
            if filename == VIEW_LABELED_JSONL:
                return labeled_bytes
            records = _parse_jsonl(annotations_bytes, state.annotations_object)
            reviewed, in_progress, unreviewed, _ = _counts(records)
            snapshot = {
                "schema_version": SCHEMA_VERSION,
                "run_id": state.run_id,
                "evaluation_config_sha256": self._manifest_config_sha(prefix),
                "revision": state.revision,
                "exported_at": datetime.now(UTC).isoformat(),
                "review_scope": REVIEW_SCOPE,
                "total": len(records),
                "reviewed": reviewed,
                "in_progress": in_progress,
                "unreviewed": unreviewed,
                "annotations": records,
            }
            return (json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n").encode(
                "utf-8"
            )

    def save(
        self,
        *,
        run_id: str,
        output_dir: str,
        save_id: str,
        base_revision: int,
        records: list[dict[str, Any]],
        reviewer_id: str,
    ) -> AnnotationSaveResult:
        if not isinstance(save_id, str) or not save_id.strip():
            raise AnnotationValidationError(["save_id must be a non-empty string"])
        if not isinstance(records, list) or not all(
            isinstance(record, dict) for record in records
        ):
            raise AnnotationValidationError(["records must be a list of objects"])
        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            raise AnnotationValidationError(["reviewer_id must be a non-empty string"])
        prefix = output_dir.rstrip("/")
        with self._lock_for(run_id):
            return self._save_locked(
                run_id=run_id,
                prefix=prefix,
                save_id=save_id,
                base_revision=base_revision,
                records=records,
                reviewer_id=reviewer_id,
            )

    def save_changes(
        self,
        *,
        run_id: str,
        output_dir: str,
        save_id: str,
        base_revision: int,
        changes: list[dict[str, Any]],
        reviewer_id: str,
    ) -> AnnotationSaveResult:
        if not isinstance(save_id, str) or not save_id.strip():
            raise AnnotationValidationError(["save_id must be a non-empty string"])
        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            raise AnnotationValidationError(["reviewer_id must be a non-empty string"])
        errors = []
        case_ids = []
        for index, change in enumerate(changes):
            if not isinstance(change, dict):
                errors.append(f"changes[{index}] must be an object")
                continue
            if set(change) - {"case_id", "human_annotation"}:
                errors.append(f"changes[{index}] has unexpected fields")
            case_id = change.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                errors.append(f"changes[{index}].case_id must be a non-empty string")
            else:
                case_ids.append(case_id)
            annotation = change.get("human_annotation")
            if annotation is not None and not isinstance(annotation, dict):
                errors.append(
                    f"changes[{index}].human_annotation must be an object or null"
                )
        if len(case_ids) != len(set(case_ids)):
            errors.append("changes contains duplicate case_id values")
        if not changes:
            errors.append("changes must not be empty")
        if errors:
            raise AnnotationValidationError(errors)

        prefix = output_dir.rstrip("/")
        content_hash = request_content_hash(
            save_id, base_revision, changes, reviewer_id
        )
        with self._lock_for(run_id):
            state = self._commit_state(run_id, prefix)
            replay = self._replayed_result(state, save_id, content_hash)
            if replay is not None:
                return replay
            if base_revision != state.revision:
                raise AnnotationConflictError(
                    f"your save is based on revision {base_revision}, but the"
                    f" current committed revision is {state.revision}; reload the"
                    " latest annotations and re-apply your changes"
                )
            previous_bytes = self._read_committed(
                run_id, prefix, state.annotations_object
            )
            records = _parse_jsonl(previous_bytes, state.annotations_object)
            by_case = {record["case_id"]: record for record in records}
            for change in changes:
                case_id = change["case_id"]
                if case_id not in by_case:
                    raise AnnotationValidationError(
                        [f"changes: unknown case_id {case_id!r}"]
                    )
                annotation = change.get("human_annotation")
                by_case[case_id] = {
                    **by_case[case_id],
                    "review_status": (
                        "unreviewed"
                        if annotation is None
                        else (
                            "reviewed"
                            if _annotation_complete(annotation)
                            else "in_progress"
                        )
                    ),
                    "human_annotation": annotation,
                }
            merged_records = [by_case[record["case_id"]] for record in records]
            return self._save_locked(
                run_id=run_id,
                prefix=prefix,
                save_id=save_id,
                base_revision=base_revision,
                records=merged_records,
                reviewer_id=reviewer_id,
                content_hash=content_hash,
            )

    @staticmethod
    def _replayed_result(
        state: EvaluationCommitState, save_id: str, content_hash: str
    ) -> AnnotationSaveResult | None:
        for entry in state.history:
            if entry.save_id != save_id:
                continue
            if entry.content_hash != content_hash:
                raise AnnotationReplayMismatchError(
                    f"save_id {save_id!r} was already used with different"
                    " content; use a new save_id for new edits"
                )
            return entry.result.model_copy(
                update={
                    "superseded_by": (
                        state.revision
                        if entry.result.revision < state.revision
                        else None
                    )
                }
            )
        return None

    def _save_locked(
        self,
        *,
        run_id: str,
        prefix: str,
        save_id: str,
        base_revision: int,
        records: list[dict[str, Any]],
        reviewer_id: str,
        content_hash: str | None = None,
    ) -> AnnotationSaveResult:
        state = self._commit_state(run_id, prefix)
        content_hash = content_hash or request_content_hash(
            save_id, base_revision, records, reviewer_id
        )
        replay = self._replayed_result(state, save_id, content_hash)
        if replay is not None:
            return replay
        if base_revision != state.revision:
            raise AnnotationConflictError(
                f"your save is based on revision {base_revision}, but the"
                f" current committed revision is {state.revision}; reload the"
                " latest annotations and re-apply your changes"
            )

        evaluation_config_sha256 = self._run_binding(prefix, run_id)
        prediction_rows = _prediction_rows(
            self._read_artifact(f"{prefix}/{PREDICTIONS_NAME}", MAX_PREDICTIONS_BYTES)
        )
        failed_ids = [row["id"] for row in prediction_rows if not row.get("passed")]
        spec = self._read_artifact_json(f"{prefix}/{RESOLVED_REPORT_SPEC_NAME}")
        annotation_schema = spec.get("annotation_schema") or {}
        if not isinstance(annotation_schema, dict):
            annotation_schema = {}
        _validate_records(
            records,
            run_id=run_id,
            evaluation_config_sha256=evaluation_config_sha256,
            failed_ids=failed_ids,
            annotation_schema=annotation_schema,
        )

        # Read previous annotations to preserve reviewer_id for unchanged
        # annotations. A full-snapshot save re-submits all records; without
        # this diff, user B modifying only case 2 would overwrite user A's
        # reviewer attribution on case 1.
        #
        # If the committed object is unreadable (storage fault / corruption),
        # the save must abort rather than silently overwrite all reviewers.
        previous_by_case: dict[str, dict[str, Any]] = {}
        if state.annotations_object:
            previous_bytes = self._read_committed(
                run_id, prefix, state.annotations_object
            )
            previous_records = _parse_jsonl(previous_bytes, state.annotations_object)
            previous_by_case = {
                record["case_id"]: record for record in previous_records
            }

        by_id = {row["id"]: row for row in prediction_rows}
        rebuilt = []
        for record in records:
            entry = {
                **record,
                "machine_result": _machine_result(by_id[record["case_id"]]),
            }
            annotation = entry.get("human_annotation")
            if isinstance(annotation, dict):
                previous = previous_by_case.get(record["case_id"], {})
                previous_annotation = previous.get("human_annotation")
                # Compare content excluding reviewer_id: if unchanged, keep
                # the previous reviewer (even if it was None from legacy
                # data); otherwise attribute to current user.
                unchanged = isinstance(
                    previous_annotation, dict
                ) and _annotation_without_reviewer(
                    annotation
                ) == _annotation_without_reviewer(previous_annotation)
                effective_reviewer_id = (
                    previous_annotation.get("reviewer_id")
                    if isinstance(previous_annotation, dict) and unchanged
                    else reviewer_id
                )
                entry["human_annotation"] = {
                    **annotation,
                    "reviewer_id": effective_reviewer_id,
                }
            rebuilt.append(entry)
        merged = merge_labeled_results(
            prediction_rows,
            rebuilt,
            run_id=run_id,
            evaluation_config_sha256=evaluation_config_sha256,
        )

        revision = state.revision + 1
        annotations_object = ANNOTATION_OBJECT_TEMPLATE.format(revision)
        labeled_object = LABELED_OBJECT_TEMPLATE.format(revision)
        annotations_bytes = canonical_jsonl_bytes(rebuilt)
        labeled_bytes = canonical_jsonl_bytes(merged)
        self._write_object(f"{prefix}/{annotations_object}", annotations_bytes)
        self._write_object(f"{prefix}/{labeled_object}", labeled_bytes)

        reviewed, in_progress, unreviewed, overall = _counts(rebuilt)
        result = AnnotationSaveResult(
            revision=revision,
            reviewed=reviewed,
            in_progress=in_progress,
            unreviewed=unreviewed,
            overall_status=overall,
        )
        new_state = EvaluationCommitState(
            run_id=run_id,
            revision=revision,
            annotations_object=annotations_object,
            labeled_object=labeled_object,
            history=(
                state.history
                + [
                    SaveHistoryEntry(
                        save_id=save_id,
                        revision=revision,
                        content_hash=content_hash,
                        result=result,
                    )
                ]
            )[-HISTORY_LIMIT:],
            updated_at=datetime.now(UTC),
        )
        self._commits.save_evaluation_commit(new_state)
        self._refresh_views(
            prefix,
            new_state,
            annotations_bytes,
            labeled_bytes,
            rebuilt,
            evaluation_config_sha256,
        )
        self._collect_garbage(prefix, revision)
        return result

    # -- internals ----------------------------------------------------------

    def _lock_for(self, run_id: str) -> threading.Lock:
        return self._locks.for_run(run_id)

    def _commit_state(self, run_id: str, prefix: str) -> EvaluationCommitState:
        try:
            commit = self._commits.load_evaluation_commit(run_id)
        except Exception as exc:
            raise EvaluationAnnotationError(
                f"annotation commit state for run {run_id} is unreadable: {exc}"
            ) from exc
        if commit is not None:
            return commit
        names = self._storage.list(prefix)
        revisions = [
            revision
            for name in names
            if (revision := _object_revision(name, ANNOTATION_OBJECT_PREFIX))
            is not None
        ]
        if revisions and max(revisions) > 0:
            raise AnnotationStateFault(
                f"annotation commit state for run {run_id} is missing while"
                f" revision objects {sorted(revisions)} exist on Storage;"
                " writes are stopped until the state is recovered explicitly"
            )
        if 0 not in revisions:
            raise AnnotationNotInitializedError(
                f"run {run_id} has no annotation products; a validated"
                " evaluation initializes them on completion"
            )
        try:
            # Both authoritative objects must be readable before the commit
            # state may reference them; eval interrupted mid-init leaves a
            # repairable half-set, never a committable one.
            self._storage.read(f"{prefix}/{ANNOTATION_OBJECT_TEMPLATE.format(0)}")
            self._storage.read(f"{prefix}/{LABELED_OBJECT_TEMPLATE.format(0)}")
        except EvaluationStorageMissingError as exc:
            raise AnnotationNotInitializedError(
                f"evaluation annotation products for run {run_id} are"
                f" incomplete ({exc}); re-run the evaluation to regenerate them"
            ) from exc
        except Exception as exc:
            # A storage outage must not be classified as "not initialized":
            # the products exist, the server just cannot reach them.
            raise EvaluationAnnotationError(
                f"annotation products for run {run_id} could not be verified: {exc}"
            ) from exc
        state = EvaluationCommitState(
            run_id=run_id,
            revision=0,
            annotations_object=ANNOTATION_OBJECT_TEMPLATE.format(0),
            labeled_object=LABELED_OBJECT_TEMPLATE.format(0),
            history=[],
            updated_at=datetime.now(UTC),
        )
        self._commits.save_evaluation_commit(state)
        return state

    def _run_binding(self, prefix: str, run_id: str) -> str:
        try:
            manifest = self._read_artifact_json(f"{prefix}/{RUN_MANIFEST_NAME}")
        except EvaluationAnnotationError:
            raise
        except Exception as exc:
            raise EvaluationAnnotationError(
                f"run artifacts under {prefix} are unreadable: {exc}"
            ) from exc
        if manifest.get("run_id") != run_id:
            raise EvaluationAnnotationError(
                "run_manifest.json does not match the requested run"
            )
        evaluation_config_sha256 = manifest.get("evaluation_config_sha256")
        if not isinstance(evaluation_config_sha256, str):
            raise EvaluationAnnotationError(
                "run_manifest.json is missing evaluation_config_sha256"
            )
        return evaluation_config_sha256

    def _write_object(self, path: str, payload: bytes) -> None:
        if self._read_maybe(path) == payload:
            return
        self._storage.write(path, payload)
        if self._storage.read(path) != payload:
            raise EvaluationAnnotationError(
                f"storage re-read of {path} does not match the written bytes;"
                " the save was not committed"
            )

    def _read_committed(self, run_id: str, prefix: str, name: str) -> bytes:
        try:
            return self._storage.read(f"{prefix}/{name}")
        except Exception as exc:
            raise EvaluationAnnotationError(
                f"committed annotation objects for run {run_id} are unreadable: {exc}"
            ) from exc

    def _read_maybe(self, path: str) -> bytes | None:
        try:
            return self._storage.read(path)
        except Exception:
            return None

    def _read_artifact(self, path: str, max_bytes: int) -> bytes:
        try:
            data = self._storage.read(path)
        except EvaluationAnnotationError:
            raise
        except Exception as exc:
            raise EvaluationAnnotationError(
                f"run artifacts under the output directory are unreadable: {exc}"
            ) from exc
        if len(data) > max_bytes:
            raise EvaluationAnnotationError(f"{path} exceeds the size limit")
        return data

    def _read_artifact_json(self, path: str) -> dict[str, Any]:
        value = json.loads(self._read_artifact(path, 8 * 1024 * 1024).decode("utf-8"))
        if not isinstance(value, dict):
            raise EvaluationAnnotationError(f"{path} is not a JSON object")
        return value

    def _manifest_config_sha(self, prefix: str) -> str:
        manifest = self._read_maybe(f"{prefix}/{RUN_MANIFEST_NAME}")
        if manifest is None:
            return ""
        try:
            return str(
                json.loads(manifest.decode("utf-8")).get("evaluation_config_sha256", "")
            )
        except Exception:
            return ""

    def _refresh_views(
        self,
        prefix: str,
        state: EvaluationCommitState,
        annotations_bytes: bytes,
        labeled_bytes: bytes,
        records: list[dict[str, Any]],
        evaluation_config_sha256: str,
    ) -> None:
        try:
            reviewed, in_progress, unreviewed, _ = _counts(records)
            snapshot = {
                "schema_version": SCHEMA_VERSION,
                "run_id": state.run_id,
                "evaluation_config_sha256": evaluation_config_sha256,
                "revision": state.revision,
                "exported_at": datetime.now(UTC).isoformat(),
                "review_scope": REVIEW_SCOPE,
                "total": len(records),
                "reviewed": reviewed,
                "in_progress": in_progress,
                "unreviewed": unreviewed,
                "annotations": records,
            }
            self._storage.write(f"{prefix}/{VIEW_ANNOTATIONS_JSONL}", annotations_bytes)
            self._storage.write(f"{prefix}/{VIEW_LABELED_JSONL}", labeled_bytes)
            self._storage.write(
                f"{prefix}/{VIEW_ANNOTATIONS_JSON}",
                (json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n").encode(
                    "utf-8"
                ),
            )
        except Exception as exc:
            logger.warning(
                "evaluation annotation view refresh failed for %s: %s",
                state.run_id,
                exc,
            )

    def _repair_views(
        self,
        prefix: str,
        state: EvaluationCommitState,
        annotations_bytes: bytes,
    ) -> None:
        labeled_bytes = self._read_committed(state.run_id, prefix, state.labeled_object)
        stale = (
            self._read_maybe(f"{prefix}/{VIEW_ANNOTATIONS_JSONL}") != annotations_bytes
            or self._read_maybe(f"{prefix}/{VIEW_LABELED_JSONL}") != labeled_bytes
        )
        snapshot = self._read_maybe(f"{prefix}/{VIEW_ANNOTATIONS_JSON}")
        if snapshot is None:
            stale = True
        else:
            try:
                stale = (
                    stale
                    or json.loads(snapshot.decode("utf-8")).get("revision")
                    != state.revision
                )
            except Exception:
                stale = True
        if not stale:
            return
        records = _parse_jsonl(annotations_bytes, state.annotations_object)
        self._refresh_views(
            prefix,
            state,
            annotations_bytes,
            labeled_bytes,
            records,
            self._manifest_config_sha(prefix),
        )

    def _collect_garbage(self, prefix: str, revision: int) -> None:
        # runtime_checkable isinstance confirms the capability at runtime;
        # the cast recovers the static type pyright cannot narrow.
        deletable = (
            cast(DeletableEvaluationStorage, self._storage)
            if isinstance(self._storage, DeletableEvaluationStorage)
            else None
        )
        if deletable is None:
            return
        try:
            names = self._storage.list(prefix)
        except Exception as exc:
            logger.warning("annotation GC listing failed for %s: %s", prefix, exc)
            return
        keep_annotations = set(
            range(max(0, revision - ANNOTATION_KEEP + 1), revision + 1)
        )
        keep_labeled = set(range(max(0, revision - LABELED_KEEP + 1), revision + 1))
        for name in names:
            annotation_revision = _object_revision(name, ANNOTATION_OBJECT_PREFIX)
            if annotation_revision is not None and (
                annotation_revision not in keep_annotations
            ):
                self._delete_maybe(deletable, f"{prefix}/{name}")
                continue
            labeled_revision = _object_revision(name, LABELED_OBJECT_PREFIX)
            if labeled_revision is not None and labeled_revision not in keep_labeled:
                self._delete_maybe(deletable, f"{prefix}/{name}")

    def _delete_maybe(self, deletable: DeletableEvaluationStorage, path: str) -> None:
        try:
            deletable.delete(path)
        except Exception as exc:
            logger.warning("annotation GC delete failed for %s: %s", path, exc)


def resolve_evaluation_run(
    conversation_id: str,
    store: FileProductStore,
    run_id: str,
    context: RequestContext,
    storage: EvaluationStorage,
    locks: EvaluationLockRegistry,
) -> tuple[str, str, EvaluationAnnotationService]:
    """Resolve the canonical run id, output dir, and annotation service.

    The URL-supplied ``run_id`` is parsed exactly once; the canonical form is
    what addresses the commit state, so an alternate spelling of the same UUID
    can never fork a second revision history. Confirmed absence/ownership
    becomes :class:`EvaluationRunNotFoundError`; an unreadable Product Store
    becomes :class:`EvaluationRunUnavailableError`.
    """
    if not store.metadata_path.is_file():
        raise EvaluationRunNotFoundError(f"conversation {conversation_id} not found")
    try:
        store.authorize(context.user_id)
    except PermissionError as exc:
        raise EvaluationRunNotFoundError(
            "conversation does not belong to current user"
        ) from exc
    except ProductStoreError as exc:
        raise EvaluationRunUnavailableError(
            f"conversation {conversation_id} state is unreadable"
        ) from exc
    try:
        canonical_run_id = str(UUID(run_id))
    except ValueError as exc:
        raise EvaluationRunNotFoundError(
            f"evaluation run id {run_id!r} is not a valid UUID"
        ) from exc
    try:
        runs = store.load_pipeline_runs()
    except (ProductStoreError, OSError, ValidationError) as exc:
        raise EvaluationRunUnavailableError(
            f"conversation {conversation_id} pipeline run registry is unreadable"
        ) from exc
    run = runs.get(canonical_run_id)
    if run is None or run.conversation_id != conversation_id or not run.output_dir:
        raise EvaluationRunNotFoundError(
            f"evaluation run {canonical_run_id} not found in this conversation"
        )
    service = EvaluationAnnotationService(storage=storage, commits=store, locks=locks)
    return canonical_run_id, run.output_dir.rstrip("/"), service
