from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from pyromind_runtime.application.batch_schedules import BatchSchedules
from pyromind_runtime.domain.batch_schedule import (
    BatchReceipt,
    BatchRunState,
    BatchScheduleConfig,
    ReadyProjectBatch,
)
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.infrastructure.file_batch_schedule_store import (
    FileBatchScheduleStore,
)
from pyromind_runtime.ports.batch_schedule import BatchScheduleBlocked


@dataclass
class BatchRuntime:
    tracer_resource_id: str = "tracer-1"
    release_id: str = "release-1"
    ready: list[ReadyProjectBatch] = field(default_factory=list)
    runs: dict[str, BatchRunState] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)
    release_error: bool = False
    submission_disconnect: bool = False
    claim_disconnect: bool = False
    unavailable: str | None = None

    def preflight(self, config: BatchScheduleConfig) -> None:
        if self.unavailable:
            raise BatchScheduleBlocked(self.unavailable)

    def list_ready(self) -> list[ReadyProjectBatch]:
        return self.ready

    def claim(self, receipt: BatchReceipt) -> None:
        self.calls.append(("claim", receipt.batch_id))
        if self.claim_disconnect:
            self.ready = [
                item for item in self.ready if item.batch_id != receipt.batch_id
            ]
            raise RuntimeError("claim response was lost")

    def materialize(
        self, receipt: BatchReceipt, config: BatchScheduleConfig
    ) -> JsonObject:
        return {"source_batch": f"/workspace/batches/{receipt.batch_id}"}

    def find_run(self, receipt: BatchReceipt) -> BatchRunState | None:
        return self.runs.get(receipt.batch_id)

    def start_run(self, receipt: BatchReceipt, inputs: JsonObject) -> BatchRunState:
        self.calls.append(("start", receipt.batch_id))
        run = BatchRunState(run_id="run-" + receipt.batch_id, status="running")
        self.runs[receipt.batch_id] = run
        if self.submission_disconnect:
            raise RuntimeError("secret must not be persisted")
        return run

    def release(self, receipt: BatchReceipt) -> None:
        self.calls.append(("release", receipt.batch_id))
        if self.release_error:
            raise BatchScheduleBlocked("Release is temporarily unavailable")


@pytest.fixture
def batch_setup(tmp_path):
    store = FileBatchScheduleStore(tmp_path)
    runtime = BatchRuntime(
        ready=[
            ReadyProjectBatch(batch_id="10", index=10),
            ReadyProjectBatch(batch_id="2", index=2),
        ]
    )
    service = BatchSchedules(store, lambda _project, _owner: runtime)
    config = BatchScheduleConfig(
        enabled=True,
        source_manifest_uri="/workspace/confirmed/train.jsonl",
        validation_dataset_uri="/workspace/holdout/val.jsonl",
    )
    return str(uuid4()), service, runtime, config


def test_two_batches_continue_after_report_archive_and_survive_restart(batch_setup):
    project, service, runtime, config = batch_setup
    assert service.get(project, "owner").status == "disabled"
    service.configure(project, "owner", config)
    state = service.tick(project, "owner", force=True)
    assert state.receipts[0].batch_id == "2"
    assert runtime.calls == [("claim", "2"), ("start", "2")]
    runtime.runs["2"] = BatchRunState(run_id="run-2", status="succeeded")
    assert service.tick(project, "owner", force=True).status == "blocked"
    assert ("release", "2") not in runtime.calls
    runtime.runs["2"] = runtime.runs["2"].model_copy(update={"report_archived": True})
    restarted = BatchSchedules(service.store, lambda _project, _owner: runtime)
    assert restarted.tick(project, "owner", force=True).receipts[0].status == "released"
    assert restarted.tick(project, "owner", force=True).receipts[1].batch_id == "10"
    assert runtime.calls.count(("start", "2")) == 1
    assert runtime.calls.count(("start", "10")) == 1


def test_submission_receipt_recovers_without_duplicate_start(batch_setup):
    project, service, runtime, config = batch_setup
    runtime.submission_disconnect = True
    service.configure(project, "owner", config)
    state = service.tick(project, "owner", force=True)
    assert state.status == "blocked" and state.receipts[0].run_id is None
    assert "secret" not in state.model_dump_json()
    runtime.submission_disconnect = False
    state = service.tick(project, "owner", force=True)
    assert state.receipts[0].run_id == "run-2"
    assert runtime.calls.count(("start", "2")) == 1


def test_lost_claim_response_retains_explicit_uncertainty_until_ready(batch_setup):
    project, service, runtime, config = batch_setup
    runtime.claim_disconnect = True
    service.configure(project, "owner", config)
    state = service.tick(project, "owner", force=True)
    assert state.receipts[0].status == "claiming"
    restarted = BatchSchedules(service.store, lambda _project, _owner: runtime)
    state = restarted.tick(project, "owner", force=True)
    assert state.status == "blocked"
    assert state.receipts[0].status == "claim_uncertain"
    assert state.last_error and "uncertain" in state.last_error
    assert runtime.calls == [("claim", "2")]
    runtime.claim_disconnect = False
    runtime.ready.append(ReadyProjectBatch(batch_id="2", index=2))
    state = restarted.tick(project, "owner", force=True)
    assert state.receipts[0].status == "running"
    assert runtime.calls == [("claim", "2"), ("claim", "2"), ("start", "2")]


def test_release_failure_retries_and_disabled_schedule_finishes_active_batch(
    batch_setup,
):
    project, service, runtime, config = batch_setup
    service.configure(project, "owner", config)
    service.tick(project, "owner", force=True)
    runtime.runs["2"] = BatchRunState(
        run_id="run-2", status="succeeded", report_archived=True
    )
    runtime.release_error = True
    assert service.tick(project, "owner", force=True).status == "blocked"
    service.configure(project, "owner", config.model_copy(update={"enabled": False}))
    runtime.release_error = False
    assert service.tick(project, "owner", force=True).status == "disabled"
    service.tick(project, "owner", force=True)
    assert ("start", "10") not in runtime.calls
    assert service.store.list_projects() == []


@pytest.mark.parametrize("status", ["failed", "canceled"])
def test_failed_batch_is_retained_and_stops_next_batch(batch_setup, status):
    project, service, runtime, config = batch_setup
    service.configure(project, "owner", config)
    service.tick(project, "owner", force=True)
    runtime.runs["2"] = BatchRunState(run_id="run-2", status=status)
    state = service.tick(project, "owner", force=True)
    assert state.status == "blocked" and state.receipts[0].status == "failed"
    assert not any(call[0] == "release" for call in runtime.calls)
    assert ("start", "10") not in runtime.calls


def test_preflight_is_fail_closed_and_owner_isolation_precedes_side_effects(
    batch_setup,
):
    project, service, runtime, config = batch_setup
    runtime.unavailable = "Tracer batch control requires admin authorization"
    state = service.configure(project, "owner", config)
    assert state.config.enabled and state.status == "blocked"
    assert service.tick(project, "owner", force=True).status == "blocked"
    assert runtime.calls == []
    with pytest.raises(PermissionError):
        service.tick(project, "another-owner", force=True)
    assert runtime.calls == []


def test_missing_holdout_or_same_source_is_blocked(batch_setup):
    project, service, runtime, config = batch_setup
    for uri in [None, config.source_manifest_uri]:
        state = service.configure(
            project, "owner", config.model_copy(update={"validation_dataset_uri": uri})
        )
        assert state.status == "blocked"
    assert runtime.calls == []


def test_poll_interval_and_project_lease_prevent_overlapping_ticks(batch_setup):
    project, service, runtime, config = batch_setup
    service.configure(project, "owner", config)
    first = service.tick(project, "owner")
    assert service.tick(project, "owner") == first
    with service.store.lease(project):
        with pytest.raises(BlockingIOError):
            service.tick(project, "owner", force=True)
    assert runtime.calls.count(("start", "2")) == 1
