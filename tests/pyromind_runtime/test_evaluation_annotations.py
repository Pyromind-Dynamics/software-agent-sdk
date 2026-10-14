"""Tests for the evaluation annotation save protocol (Phase 2)."""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pyromind_runtime.application.evaluation_annotations import (
    AnnotationConflictError,
    AnnotationNotInitializedError,
    AnnotationReplayMismatchError,
    AnnotationStateFault,
    AnnotationValidationError,
    EvaluationAnnotationError,
    EvaluationAnnotationService,
    canonical_jsonl_bytes,
    merge_labeled_results,
)
from pyromind_runtime.domain.evaluation import EvaluationCommitState
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.product_store import EvaluationCommitStore
from pyromind_runtime.ports.storage import (
    EvaluationStorageMissingError,
    EvaluationStorageUnavailableError,
)


RUN_ID = "11111111-1111-4111-8111-111111111111"
CONFIG_SHA = "cfgsha"
OUTPUT = "/runs/out"
TIMESTAMP = "2026-10-09T12:00:00+00:00"
REVIEWER_ID = "test-reviewer-42"

PREDICTIONS = [
    {
        "id": "a",
        "passed": True,
        "reference": "hello",
        "prediction": "hello",
        "overall_score": 1.0,
        "rubric_results": [{"name": "correct", "passed": True}],
        "latency": 0.1,
    },
    {
        "id": "b",
        "passed": False,
        "reference": "hello",
        "prediction": "wrong",
        "overall_score": 0.0,
        "rubric_results": [{"name": "correct", "passed": False}],
        "latency": 0.2,
    },
    {
        "id": "c",
        "passed": False,
        "reference": "hello",
        "prediction": "no answer",
        "overall_score": 0.0,
        "error": "evaluation mismatch",
        "rubric_results": [],
    },
]
FAILED_IDS = ["b", "c"]


class FakeStorage:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.fail_list = False
        self.fail_writes: set[str] = set()
        self.fail_reads: set[str] = set()
        self.list_calls = 0

    def read(self, path: str) -> bytes:
        if path in self.fail_reads:
            raise EvaluationStorageUnavailableError(f"injected read failure for {path}")
        if path not in self.files:
            raise EvaluationStorageMissingError(f"missing {path}")
        return self.files[path]

    def write(self, path: str, payload: bytes) -> None:
        if path in self.fail_writes:
            raise EvaluationStorageUnavailableError(
                f"injected write failure for {path}"
            )
        self.files[path] = payload

    def list(self, prefix: str) -> set[str]:
        self.list_calls += 1
        if self.fail_list:
            raise EvaluationStorageUnavailableError("storage unreachable")
        marker = prefix.rstrip("/") + "/"
        return {path[len(marker) :] for path in self.files if path.startswith(marker)}


class DeletableFakeStorage(FakeStorage):
    def delete(self, path: str) -> None:
        self.files.pop(path, None)


class FakeCommitStore(EvaluationCommitStore):
    def __init__(self) -> None:
        self.states: dict[str, EvaluationCommitState] = {}

    def load_evaluation_commit(self, run_id: str) -> EvaluationCommitState | None:
        return self.states.get(run_id)

    def save_evaluation_commit(self, state: EvaluationCommitState) -> None:
        self.states[state.run_id] = state


def make_service(
    storage: FakeStorage | None = None,
    commits: FakeCommitStore | None = None,
) -> tuple[EvaluationAnnotationService, FakeStorage, FakeCommitStore]:
    storage = storage if storage is not None else FakeStorage()
    commits = commits if commits is not None else FakeCommitStore()
    return (
        EvaluationAnnotationService(storage=storage, commits=commits),
        storage,
        commits,
    )


def seed_eval_artifacts(storage: FakeStorage, *, with_r0: bool = True) -> None:
    def put(name: str, payload: Any) -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        storage.files[f"{OUTPUT}/{name}"] = body

    put("run_manifest.json", {"run_id": RUN_ID, "evaluation_config_sha256": CONFIG_SHA})
    put(
        "predictions.jsonl",
        b"".join(
            (json.dumps(row, ensure_ascii=False) + "\n").encode() for row in PREDICTIONS
        ),
    )
    put(
        "resolved_report_spec.json",
        {
            "annotation_schema": {
                "taxonomy_labels": ["small_object"],
                "dispositions": ["fix_ground_truth"],
            }
        },
    )
    if with_r0:
        annotations = [
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "evaluation_config_sha256": CONFIG_SHA,
                "case_id": case_id,
                "review_status": "unreviewed",
                "human_annotation": None,
            }
            for case_id in FAILED_IDS
        ]
        put("evaluation_annotations.r0.jsonl", canonical_jsonl_bytes(annotations))
        put(
            "labeled_evaluation_results.r0.jsonl",
            b"{}\n",  # shape is irrelevant for the protocol tests
        )


def record(
    case_id: str,
    status: str = "unreviewed",
    annotation: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "evaluation_config_sha256": CONFIG_SHA,
        "case_id": case_id,
        "review_status": status,
        "human_annotation": annotation,
    }
    base.update(overrides)
    return base


def complete_badcase(case_id: str, **extra: Any) -> dict[str, Any]:
    return record(
        case_id,
        "reviewed",
        {
            "decision": "confirmed_badcase",
            "failure_modes": ["model_miss"],
            "taxonomy_labels": [],
            "disposition": "fix_ground_truth",
            "severity": "P1",
            "comment": "",
            "corrected_answer": None,
            "reviewer_id": None,
            "updated_at": TIMESTAMP,
            **extra,
        },
    )


def test_load_lazy_initializes_revision_zero_from_eval_products() -> None:
    service, storage, commits = make_service()
    seed_eval_artifacts(storage)
    result = service.load(run_id=RUN_ID, output_dir=OUTPUT)
    assert result.revision == 0
    assert [item["case_id"] for item in result.records] == FAILED_IDS
    assert result.reviewed == 0
    assert result.overall_status == "in_progress"
    assert commits.states[RUN_ID].revision == 0


def test_save_writes_immutable_objects_commit_and_views() -> None:
    service, storage, commits = make_service()
    seed_eval_artifacts(storage)
    tampered = complete_badcase("b")
    tampered["machine_result"] = {"passed": "fabricated"}
    untouched = record("c")
    result = service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s1",
        base_revision=0,
        records=[tampered, untouched],
        reviewer_id=REVIEWER_ID,
    )
    assert result.revision == 1
    assert result.reviewed == 1
    assert result.unreviewed == 1
    assert result.overall_status == "in_progress"
    # The committed annotation object never trusts request machine results.
    r1 = [
        json.loads(line)
        for line in storage.files[f"{OUTPUT}/evaluation_annotations.r1.jsonl"]
        .decode()
        .splitlines()
        if line
    ]
    assert r1[0]["machine_result"]["passed"] is False
    assert r1[0]["machine_result"]["prediction"] == "wrong"
    # The merged object is the full prediction row plus the review slot.
    labeled = [
        json.loads(line)
        for line in storage.files[f"{OUTPUT}/labeled_evaluation_results.r1.jsonl"]
        .decode()
        .splitlines()
        if line
    ]
    by_id = {row["id"]: row for row in labeled}
    assert set(by_id) == {"a", "b", "c"}
    assert by_id["a"]["latency"] == 0.1
    assert by_id["a"]["human_review"] == {
        "review_status": "not_in_scope",
        "annotation": None,
    }
    assert by_id["b"]["human_review"]["review_status"] == "reviewed"
    assert by_id["c"]["human_review"]["review_status"] == "unreviewed"
    # Views mirror revision 1 and the snapshot carries the counts.
    assert (
        storage.files[f"{OUTPUT}/evaluation_annotations.jsonl"]
        == storage.files[f"{OUTPUT}/evaluation_annotations.r1.jsonl"]
    )
    snapshot = json.loads(storage.files[f"{OUTPUT}/evaluation_annotations.json"])
    assert snapshot["revision"] == 1
    assert snapshot["reviewed"] == 1
    assert snapshot["in_progress"] == 0
    assert snapshot["unreviewed"] == 1
    # The commit state is the single pointer to revision 1.
    assert commits.states[RUN_ID].revision == 1
    assert commits.states[RUN_ID].annotations_object.endswith("r1.jsonl")


def test_stale_base_revision_conflicts() -> None:
    service, storage, _ = make_service()
    seed_eval_artifacts(storage)
    service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s1",
        base_revision=0,
        records=[complete_badcase("b"), record("c")],
        reviewer_id=REVIEWER_ID,
    )
    with pytest.raises(AnnotationConflictError) as exc_info:
        service.save(
            run_id=RUN_ID,
            output_dir=OUTPUT,
            save_id="s2",
            base_revision=0,
            records=[record("b"), record("c")],
            reviewer_id=REVIEWER_ID,
        )
    message = str(exc_info.value)
    assert "revision 0" in message and "1" in message


def test_save_id_replay_replays_and_marks_superseded() -> None:
    service, storage, _ = make_service()
    seed_eval_artifacts(storage)
    first = service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s1",
        base_revision=0,
        records=[record("b"), record("c")],
        reviewer_id=REVIEWER_ID,
    )
    replayed = service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s1",
        base_revision=0,
        records=[record("b"), record("c")],
        reviewer_id=REVIEWER_ID,
    )
    assert replayed.revision == first.revision == 1
    assert replayed.superseded_by is None
    service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s2",
        base_revision=1,
        records=[complete_badcase("b"), record("c")],
        reviewer_id=REVIEWER_ID,
    )
    late_replay = service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s1",
        base_revision=0,
        records=[record("b"), record("c")],
        reviewer_id=REVIEWER_ID,
    )
    assert late_replay.revision == 1
    assert late_replay.superseded_by == 2
    with pytest.raises(AnnotationReplayMismatchError):
        service.save(
            run_id=RUN_ID,
            output_dir=OUTPUT,
            save_id="s1",
            base_revision=0,
            records=[complete_badcase("b"), record("c")],
            reviewer_id=REVIEWER_ID,
        )


def test_in_progress_draft_saves_partial_annotation() -> None:
    service, storage, _ = make_service()
    seed_eval_artifacts(storage)
    draft = record(
        "b",
        "in_progress",
        {"decision": "confirmed_badcase", "updated_at": TIMESTAMP},
    )
    result = service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s1",
        base_revision=0,
        records=[draft, record("c")],
        reviewer_id=REVIEWER_ID,
    )
    assert result.in_progress == 1
    assert result.reviewed == 0
    assert result.overall_status == "in_progress"
    labeled = [
        json.loads(line)
        for line in storage.files[f"{OUTPUT}/labeled_evaluation_results.r1.jsonl"]
        .decode()
        .splitlines()
        if line
    ]
    review = {row["id"]: row["human_review"] for row in labeled}
    assert review["b"]["review_status"] == "in_progress"
    assert review["b"]["annotation"]["decision"] == "confirmed_badcase"


def test_reviewed_incomplete_and_bad_coverage_are_rejected() -> None:
    service, storage, _ = make_service()
    seed_eval_artifacts(storage)
    incomplete = record(
        "b",
        "reviewed",
        {
            "decision": "confirmed_badcase",
            "disposition": "fix_ground_truth",
            "updated_at": TIMESTAMP,
        },
    )
    with pytest.raises(AnnotationValidationError) as exc_info:
        service.save(
            run_id=RUN_ID,
            output_dir=OUTPUT,
            save_id="s1",
            base_revision=0,
            records=[incomplete],
            reviewer_id=REVIEWER_ID,
        )
    assert "incomplete" in str(exc_info.value)
    with pytest.raises(AnnotationValidationError):
        service.save(
            run_id=RUN_ID,
            output_dir=OUTPUT,
            save_id="s1",
            base_revision=0,
            records=[record("a"), record("b")],  # "a" is not a failed case
            reviewer_id=REVIEWER_ID,
        )
    with pytest.raises(AnnotationValidationError):
        service.save(
            run_id=RUN_ID,
            output_dir=OUTPUT,
            save_id="s1",
            base_revision=0,
            records=[record("c"), record("b")],  # wrong order
            reviewer_id=REVIEWER_ID,
        )
    assert not any(
        name.startswith("evaluation_annotations.r1") for name in storage.files
    )


def test_missing_state_with_later_objects_is_a_fault() -> None:
    service, storage, _ = make_service()
    seed_eval_artifacts(storage)
    storage.files[f"{OUTPUT}/evaluation_annotations.r2.jsonl"] = b"{}\n"
    with pytest.raises(AnnotationStateFault):
        service.load(run_id=RUN_ID, output_dir=OUTPUT)
    with pytest.raises(AnnotationStateFault):
        service.save(
            run_id=RUN_ID,
            output_dir=OUTPUT,
            save_id="s1",
            base_revision=0,
            records=[record("b"), record("c")],
            reviewer_id=REVIEWER_ID,
        )


def test_unreachable_list_is_not_treated_as_empty() -> None:
    service, storage, _ = make_service()
    seed_eval_artifacts(storage, with_r0=False)
    storage.fail_list = True
    with pytest.raises(RuntimeError, match="unreachable"):
        service.load(run_id=RUN_ID, output_dir=OUTPUT)


def test_run_without_products_is_not_initialized() -> None:
    service, storage, _ = make_service()
    seed_eval_artifacts(storage, with_r0=False)
    with pytest.raises(AnnotationNotInitializedError):
        service.load(run_id=RUN_ID, output_dir=OUTPUT)


def test_orphaned_uncommitted_object_is_recovered_by_retry() -> None:
    """Reviewer scenario: failed save leaves rN, retry recovers.

    Commit at r1, then a crashed save wrote diverging r2 bytes without
    committing. The retry overwrites the uncommitted name, commits, and the
    protocol continues; a byte-identical orphan is reused without a write.
    """
    service, storage, commits = make_service()
    seed_eval_artifacts(storage)
    service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s1",
        base_revision=0,
        records=[record("b"), record("c")],
        reviewer_id=REVIEWER_ID,
    )
    orphan = b'{"schema_version": 1, "case_id": "b", "torn": true}\n'
    storage.files[f"{OUTPUT}/evaluation_annotations.r2.jsonl"] = orphan
    result = service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s2",
        base_revision=1,
        records=[complete_badcase("b"), record("c")],
        reviewer_id=REVIEWER_ID,
    )
    assert result.revision == 2
    assert b"torn" not in storage.files[f"{OUTPUT}/evaluation_annotations.r2.jsonl"]
    assert commits.states[RUN_ID].revision == 2
    # A fresh service instance (e.g. after a server restart) continues from
    # the committed revision instead of re-initializing.
    resumed, _, _ = make_service(storage, commits)
    state = resumed.load(run_id=RUN_ID, output_dir=OUTPUT)
    assert state.revision == 2
    follow_up = resumed.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s3",
        base_revision=2,
        records=[complete_badcase("b"), complete_badcase("c")],
        reviewer_id=REVIEWER_ID,
    )
    assert follow_up.revision == 3
    assert follow_up.reviewed == 2
    assert follow_up.overall_status == "completed"


def test_gc_keeps_only_recent_revisions_when_deletable() -> None:
    storage = DeletableFakeStorage()
    service, _, _ = make_service(storage)
    seed_eval_artifacts(storage)
    service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s1",
        base_revision=0,
        records=[record("b"), record("c")],
        reviewer_id=REVIEWER_ID,
    )
    service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s2",
        base_revision=1,
        records=[complete_badcase("b"), record("c")],
        reviewer_id=REVIEWER_ID,
    )
    service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s3",
        base_revision=2,
        records=[complete_badcase("b"), complete_badcase("c")],
        reviewer_id=REVIEWER_ID,
    )
    service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s4",
        base_revision=3,
        records=[complete_badcase("b"), record("c", "unreviewed")],
        reviewer_id=REVIEWER_ID,
    )
    annotation_revisions = {
        name.split(".r")[1].removesuffix(".jsonl")
        for name in storage.files
        if name.startswith(f"{OUTPUT}/evaluation_annotations.r")
    }
    labeled_revisions = {
        name.split(".r")[1].removesuffix(".jsonl")
        for name in storage.files
        if name.startswith(f"{OUTPUT}/labeled_evaluation_results.r")
    }
    assert annotation_revisions == {"2", "3", "4"}
    assert labeled_revisions == {"3", "4"}
    assert (
        storage.files[f"{OUTPUT}/evaluation_annotations.jsonl"]
        == (storage.files[f"{OUTPUT}/evaluation_annotations.r4.jsonl"])
    )


def test_non_deletable_storage_skips_gc_quietly() -> None:
    storage = FakeStorage()
    service, _, _ = make_service(storage)
    seed_eval_artifacts(storage)
    for index, save_id in enumerate(["s1", "s2", "s3", "s4"]):
        service.save(
            run_id=RUN_ID,
            output_dir=OUTPUT,
            save_id=save_id,
            base_revision=index,
            records=[record("b"), record("c")],
            reviewer_id=REVIEWER_ID,
        )
    revisions = {
        name
        for name in storage.files
        if name.startswith(f"{OUTPUT}/evaluation_annotations.r")
    }
    assert len(revisions) == 5  # r0..r4 retained, no deletion capability


def test_history_window_bounds_replay() -> None:
    storage = FakeStorage()
    service, _, _ = make_service(storage)
    seed_eval_artifacts(storage)
    records = [record("b"), record("c")]
    for index in range(6):
        service.save(
            run_id=RUN_ID,
            output_dir=OUTPUT,
            save_id=f"s{index}",
            base_revision=index,
            records=records,
            reviewer_id=REVIEWER_ID,
        )
    # s0 fell out of the 5-entry history: replay is no longer possible and
    # the save falls through to the base_revision conflict path.
    with pytest.raises(AnnotationConflictError):
        service.save(
            run_id=RUN_ID,
            output_dir=OUTPUT,
            save_id="s0",
            base_revision=0,
            records=records,
            reviewer_id=REVIEWER_ID,
        )


def test_needs_discussion_completes_with_open_items() -> None:
    storage = FakeStorage()
    service, _, _ = make_service(storage)
    seed_eval_artifacts(storage)
    result = service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s1",
        base_revision=0,
        records=[
            complete_badcase("b"),
            complete_badcase(
                "c", decision="needs_discussion", failure_modes=["gt_issue"]
            ),
        ],
        reviewer_id=REVIEWER_ID,
    )
    assert result.reviewed == 2
    assert result.overall_status == "completed_with_open_items"


def test_merge_labeled_results_is_pure_and_complete() -> None:
    annotations = [complete_badcase("b"), record("c")]
    merged = merge_labeled_results(
        PREDICTIONS,
        annotations,
        run_id=RUN_ID,
        evaluation_config_sha256=CONFIG_SHA,
    )
    by_id = {row["id"]: row for row in merged}
    for source in PREDICTIONS:
        for key, value in source.items():
            assert by_id[source["id"]][key] == value
    assert by_id["a"]["human_review"]["review_status"] == "not_in_scope"
    assert by_id["b"]["human_review"]["annotation"]["decision"] == ("confirmed_badcase")
    assert by_id["c"]["human_review"]["review_status"] == "unreviewed"


def test_load_shares_the_save_lock() -> None:
    service, storage, _ = make_service()
    seed_eval_artifacts(storage)
    lock = service._lock_for(RUN_ID)
    with lock:
        done = threading.Event()

        def run_load() -> None:
            service.load(run_id=RUN_ID, output_dir=OUTPUT)
            done.set()

        thread = threading.Thread(target=run_load)
        thread.start()
        # While the save lock is held elsewhere, load must wait: a concurrent
        # save could otherwise garbage-collect the objects it resolves.
        assert not done.wait(0.2)
    thread.join(timeout=5)
    assert done.is_set()


def test_concurrent_saves_and_loads_stay_consistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyromind_runtime.application import (
        evaluation_annotations as evaluation_annotations_module,
    )

    # Aggressive GC widens the failure window for an unlocked load: the
    # revision it resolved can be deleted before its object read completes.
    monkeypatch.setattr(evaluation_annotations_module, "ANNOTATION_KEEP", 1)
    monkeypatch.setattr(evaluation_annotations_module, "LABELED_KEEP", 1)
    storage = DeletableFakeStorage()
    service, _, _ = make_service(storage)
    seed_eval_artifacts(storage)
    original_read = storage.read

    def slow_read(path: str) -> bytes:
        if "evaluation_annotations.r" in path:
            time.sleep(0.003)
        return original_read(path)

    storage.read = slow_read  # type: ignore[method-assign]
    errors: list[Exception] = []

    def saver() -> None:
        try:
            for index in range(6):
                records = (
                    [complete_badcase("b"), record("c")]
                    if index % 2
                    else [record("b"), record("c")]
                )
                service.save(
                    run_id=RUN_ID,
                    output_dir=OUTPUT,
                    save_id=f"s{index}",
                    base_revision=index,
                    records=records,
                    reviewer_id=REVIEWER_ID,
                )
        except Exception as exc:  # pragma: no cover - surfaced via errors
            errors.append(exc)

    def loader() -> None:
        try:
            for _ in range(40):
                service.load(run_id=RUN_ID, output_dir=OUTPUT)
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=saver),
        threading.Thread(target=loader),
        threading.Thread(target=loader),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


def test_missing_view_files_are_lazily_repaired() -> None:
    service, storage, _ = make_service()
    seed_eval_artifacts(storage)
    service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s1",
        base_revision=0,
        records=[complete_badcase("b"), record("c")],
        reviewer_id=REVIEWER_ID,
    )
    revision = "1"
    for missing in (
        "evaluation_annotations.jsonl",
        "labeled_evaluation_results.jsonl",
        "evaluation_annotations.json",
    ):
        del storage.files[f"{OUTPUT}/{missing}"]
        result = service.load(run_id=RUN_ID, output_dir=OUTPUT)
        assert result.revision == 1
        assert f"{OUTPUT}/{missing}" in storage.files, missing
    # Every restored view again matches revision {revision}.
    snapshot = json.loads(storage.files[f"{OUTPUT}/evaluation_annotations.json"])
    assert snapshot["revision"] == int(revision)


def test_incomplete_r0_products_do_not_initialize() -> None:
    service, storage, commits = make_service()
    seed_eval_artifacts(storage, with_r0=True)
    del storage.files[f"{OUTPUT}/labeled_evaluation_results.r0.jsonl"]
    with pytest.raises(AnnotationNotInitializedError, match="incomplete"):
        service.load(run_id=RUN_ID, output_dir=OUTPUT)
    # The half-set must not be committed as a usable revision-0 state.
    assert RUN_ID not in commits.states


def test_zero_badcase_run_is_completed() -> None:
    storage = FakeStorage()
    service, _, _ = make_service(storage)
    # An all-pass evaluation: the review scope is empty from the start.
    storage.files[f"{OUTPUT}/run_manifest.json"] = json.dumps(
        {"run_id": RUN_ID, "evaluation_config_sha256": CONFIG_SHA}
    ).encode()
    storage.files[f"{OUTPUT}/predictions.jsonl"] = (
        json.dumps(PREDICTIONS[0], ensure_ascii=False) + "\n"
    ).encode()
    storage.files[f"{OUTPUT}/resolved_report_spec.json"] = json.dumps({}).encode()
    storage.files[f"{OUTPUT}/evaluation_annotations.r0.jsonl"] = b""
    storage.files[f"{OUTPUT}/labeled_evaluation_results.r0.jsonl"] = b"{}\n"
    result = service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s1",
        base_revision=0,
        records=[],
        reviewer_id=REVIEWER_ID,
    )
    assert result.revision == 1
    assert result.reviewed == 0
    assert result.overall_status == "completed"


def test_second_object_write_failure_keeps_state_and_retry_recovers() -> None:
    """Reviewer scenario: first object written, second fails mid-save."""
    service, storage, commits = make_service()
    seed_eval_artifacts(storage)
    storage.fail_writes = {f"{OUTPUT}/labeled_evaluation_results.r1.jsonl"}
    with pytest.raises(EvaluationStorageUnavailableError):
        service.save(
            run_id=RUN_ID,
            output_dir=OUTPUT,
            save_id="s1",
            base_revision=0,
            records=[record("b"), record("c")],
            reviewer_id=REVIEWER_ID,
        )
    # The write failure happened before the commit: state stays at revision 0.
    assert commits.states[RUN_ID].revision == 0
    assert f"{OUTPUT}/evaluation_annotations.jsonl" not in storage.files
    # The same save_id retries with identical content and now succeeds.
    storage.fail_writes.clear()
    result = service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s1",
        base_revision=0,
        records=[record("b"), record("c")],
        reviewer_id=REVIEWER_ID,
    )
    assert result.revision == 1
    assert commits.states[RUN_ID].revision == 1
    assert f"{OUTPUT}/evaluation_annotations.jsonl" in storage.files


def test_file_product_store_commit_roundtrip(tmp_path: Path) -> None:
    conversation = tmp_path / "conversation"
    conversation.mkdir()
    store = FileProductStore(conversation)
    from pyromind_runtime.domain.capabilities import HarnessCapabilities
    from pyromind_runtime.domain.snapshot import ConversationSnapshot

    store.create(
        ConversationSnapshot(
            conversation_id="conv-1", capabilities=HarnessCapabilities()
        ),
        user_id="test",
    )
    state = EvaluationCommitState(
        run_id=RUN_ID,
        revision=2,
        annotations_object="evaluation_annotations.r2.jsonl",
        labeled_object="labeled_evaluation_results.r2.jsonl",
        updated_at=datetime(2026, 10, 9, 12, 0, tzinfo=UTC),
    )
    store.save_evaluation_commit(state)
    # A fresh store instance (server restart) reads the same commit state.
    reloaded = FileProductStore(conversation).load_evaluation_commit(RUN_ID)
    assert reloaded is not None
    assert reloaded.revision == 2
    assert reloaded.annotations_object.endswith("r2.jsonl")
    assert FileProductStore(conversation).load_evaluation_commit("other") is None


def test_reviewer_id_is_written_for_new_annotation() -> None:
    """A new annotation from user '42' must land with reviewer_id='42'."""
    service, storage, _ = make_service()
    seed_eval_artifacts(storage)
    service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s1",
        base_revision=0,
        records=[complete_badcase("b"), record("c")],
        reviewer_id="42",
    )
    ann_bytes = storage.files[f"{OUTPUT}/evaluation_annotations.r1.jsonl"]
    rows = [
        json.loads(line) for line in ann_bytes.decode().splitlines() if line.strip()
    ]
    by_case = {row["case_id"]: row for row in rows}
    assert by_case["b"]["human_annotation"]["reviewer_id"] == "42"
    # Unreviewed record keeps annotation=null (no reviewer injected).
    assert by_case["c"]["human_annotation"] is None


def test_unchanged_annotation_preserves_previous_reviewer() -> None:
    """User A annotates case b; user C modifies only case c.

    After C's save, case b must still carry A's reviewer_id, not C's.
    """
    service, storage, _ = make_service()
    seed_eval_artifacts(storage)
    # User A annotates case b (and leaves c unreviewed).
    service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s1",
        base_revision=0,
        records=[complete_badcase("b"), record("c")],
        reviewer_id="user-A",
    )
    # User C loads the latest state, modifies only case c, and re-submits
    # the full snapshot (including case b unchanged from A's save).
    service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s2",
        base_revision=1,
        records=[complete_badcase("b"), complete_badcase("c")],
        reviewer_id="user-C",
    )
    ann_bytes = storage.files[f"{OUTPUT}/evaluation_annotations.r2.jsonl"]
    rows = [
        json.loads(line) for line in ann_bytes.decode().splitlines() if line.strip()
    ]
    by_case = {row["case_id"]: row for row in rows}
    # Case b: unchanged content -> keep A's reviewer.
    assert by_case["b"]["human_annotation"]["reviewer_id"] == "user-A"
    # Case c: newly annotated by C -> reviewer is C.
    assert by_case["c"]["human_annotation"]["reviewer_id"] == "user-C"


def test_replay_mismatch_when_reviewer_differs() -> None:
    """Same save_id + records but different reviewer must not replay."""
    service, storage, _ = make_service()
    seed_eval_artifacts(storage)
    service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s1",
        base_revision=0,
        records=[complete_badcase("b"), record("c")],
        reviewer_id="user-A",
    )
    # Same save_id and records, but a different reviewer: the content hash
    # now includes reviewer_id, so this must raise replay mismatch.
    with pytest.raises(AnnotationReplayMismatchError):
        service.save(
            run_id=RUN_ID,
            output_dir=OUTPUT,
            save_id="s1",
            base_revision=0,
            records=[complete_badcase("b"), record("c")],
            reviewer_id="user-B",
        )


def test_save_aborts_when_previous_annotations_unreadable() -> None:
    """A storage fault reading the committed object must abort the save.

    If we silently fell back to 'all annotations are new', user B's save
    would overwrite user A's reviewer attribution and advance the commit
    revision, masking the outage. The save must fail and leave the commit
    revision unchanged.
    """
    service, storage, commits = make_service()
    seed_eval_artifacts(storage)
    service.save(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="s1",
        base_revision=0,
        records=[complete_badcase("b"), record("c")],
        reviewer_id="user-A",
    )
    assert commits.states[RUN_ID].revision == 1
    # Simulate a storage outage on the committed annotation object.
    storage.fail_reads.add(f"{OUTPUT}/evaluation_annotations.r1.jsonl")
    with pytest.raises(EvaluationAnnotationError) as exc_info:
        service.save(
            run_id=RUN_ID,
            output_dir=OUTPUT,
            save_id="s2",
            base_revision=1,
            records=[complete_badcase("b"), complete_badcase("c")],
            reviewer_id="user-B",
        )
    assert "unreadable" in str(exc_info.value)
    # Commit revision must not advance: no new objects were written.
    assert commits.states[RUN_ID].revision == 1
    assert f"{OUTPUT}/evaluation_annotations.r2.jsonl" not in storage.files


def test_save_changes_merges_only_dirty_cases_and_attributes_reviewer() -> None:
    service, storage, _ = make_service()
    seed_eval_artifacts(storage)
    annotation = {
        **complete_badcase("b")["human_annotation"],
        "resolution_code": "model_gap",
        "disposition": "training_workflow_adjustment",
    }

    result = service.save_changes(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="patch-1",
        base_revision=0,
        changes=[{"case_id": "b", "human_annotation": annotation}],
        reviewer_id="user-A",
    )

    assert result.revision == 1
    rows = [
        json.loads(line)
        for line in storage.files[f"{OUTPUT}/evaluation_annotations.r1.jsonl"]
        .decode()
        .splitlines()
    ]
    by_case = {row["case_id"]: row for row in rows}
    assert by_case["b"]["review_status"] == "reviewed"
    assert by_case["b"]["human_annotation"]["resolution_code"] == "model_gap"
    assert by_case["b"]["human_annotation"]["reviewer_id"] == "user-A"
    assert by_case["c"]["review_status"] == "unreviewed"
    assert by_case["c"]["human_annotation"] is None

    replay = service.save_changes(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="patch-1",
        base_revision=0,
        changes=[{"case_id": "b", "human_annotation": annotation}],
        reviewer_id="user-A",
    )
    assert replay.revision == 1
    assert replay.superseded_by is None


def test_resolution_specific_fields_control_completion() -> None:
    service, storage, _ = make_service()
    seed_eval_artifacts(storage)
    annotation = {
        "resolution_code": "poor_data",
        "resolution_details_version": 1,
        "decision": "evaluation_false_alarm",
        "failure_modes": [],
        "taxonomy_labels": [],
        "disposition": "needs_analysis",
        "severity": None,
        "comment": "",
        "corrected_answer": None,
        "data_quality_reasons": [],
        "reviewer_id": None,
        "updated_at": TIMESTAMP,
    }
    partial = service.save_changes(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="poor-data-partial",
        base_revision=0,
        changes=[{"case_id": "b", "human_annotation": annotation}],
        reviewer_id="user-A",
    )
    assert partial.in_progress == 1

    complete = service.save_changes(
        run_id=RUN_ID,
        output_dir=OUTPUT,
        save_id="poor-data-complete",
        base_revision=1,
        changes=[
            {
                "case_id": "b",
                "human_annotation": {
                    **annotation,
                    "data_quality_reasons": ["needs_cleaning"],
                },
            }
        ],
        reviewer_id="user-A",
    )
    assert complete.reviewed == 1
    assert complete.in_progress == 0


def test_save_changes_rejects_duplicate_or_unknown_cases() -> None:
    service, storage, _ = make_service()
    seed_eval_artifacts(storage)
    annotation = complete_badcase("b")["human_annotation"]
    with pytest.raises(AnnotationValidationError):
        service.save_changes(
            run_id=RUN_ID,
            output_dir=OUTPUT,
            save_id="duplicates",
            base_revision=0,
            changes=[
                {"case_id": "b", "human_annotation": annotation},
                {"case_id": "b", "human_annotation": annotation},
            ],
            reviewer_id="user-A",
        )
    with pytest.raises(AnnotationValidationError):
        service.save_changes(
            run_id=RUN_ID,
            output_dir=OUTPUT,
            save_id="invalid-target-count",
            base_revision=0,
            changes=[
                {
                    "case_id": "b",
                    "human_annotation": {
                        **annotation,
                        "target_sample_count": 0,
                    },
                }
            ],
            reviewer_id="user-A",
        )
    with pytest.raises(AnnotationValidationError):
        service.save_changes(
            run_id=RUN_ID,
            output_dir=OUTPUT,
            save_id="mismatched-resolution",
            base_revision=0,
            changes=[
                {
                    "case_id": "b",
                    "human_annotation": {
                        **annotation,
                        "resolution_code": "acceptable",
                    },
                }
            ],
            reviewer_id="user-A",
        )
    with pytest.raises(AnnotationValidationError):
        service.save_changes(
            run_id=RUN_ID,
            output_dir=OUTPUT,
            save_id="unknown",
            base_revision=0,
            changes=[{"case_id": "unknown", "human_annotation": annotation}],
            reviewer_id="user-A",
        )
    with pytest.raises(AnnotationValidationError):
        service.save_changes(
            run_id=RUN_ID,
            output_dir=OUTPUT,
            save_id="invalid-resolution",
            base_revision=0,
            changes=[
                {
                    "case_id": "b",
                    "human_annotation": {
                        **annotation,
                        "resolution_code": "not-a-resolution",
                    },
                }
            ],
            reviewer_id="user-A",
        )
