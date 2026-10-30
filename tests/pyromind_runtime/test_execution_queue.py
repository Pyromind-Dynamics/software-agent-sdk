from datetime import timedelta

import pytest
import sqlalchemy as sa
from pyromind_runtime.domain.experiment_plans import PlanOwner
from pyromind_runtime.infrastructure.sql_experiment_execution import (
    SQLExperimentExecutionStore,
    metadata,
)
from pyromind_runtime.infrastructure.sql_work_queue import (
    StaleLeaseError,
    now,
    work,
)


@pytest.fixture
def ledger(tmp_path):
    engine = sa.create_engine("sqlite:///" + str(tmp_path / "execution.db"))
    metadata.create_all(engine)
    from pyromind_runtime.infrastructure.sql_experiment_execution import owner_key
    from pyromind_runtime.infrastructure.sql_experiment_plan_store import (
        metadata as plan_metadata,
        projects,
    )

    plan_metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            projects.insert().values(
                project_id="project",
                owner_key=owner_key(
                    PlanOwner(
                        user_id="user",
                        conversation_id="conversation",
                        profile_id="test",
                    )
                ),
                user_id="user",
                conversation_id="conversation",
                profile_id="test",
                current_revision=1,
            )
        )
    yield SQLExperimentExecutionStore(engine)
    engine.dispose()


@pytest.fixture
def owner():
    return PlanOwner(user_id="user", conversation_id="conversation", profile_id="test")


def test_role_serialization_and_deduplication(ledger):
    q = ledger.queue
    first = q.enqueue("owner", "role", "a", "prepare", {"x": 1})
    assert q.enqueue("owner", "role", "a", "prepare", {"x": 1}) == first
    q.enqueue("owner", "role", "b", "prepare", {})
    claimed = q.claim()
    assert claimed["id"] == first
    assert q.claim() is None
    q.finish(first, claimed["generation"], {"prepared": True})
    assert q.claim()["id"] != first
    with pytest.raises(ValueError, match="different work"):
        q.enqueue("owner", "role", "a", "prepare", {"x": 2})


def test_expired_work_requires_explicit_reconciliation(ledger):
    q = ledger.queue
    identity = q.enqueue("owner", "role", "a", "submit", {})
    claimed = q.claim()
    with ledger.engine.begin() as conn:
        conn.execute(
            work.update()
            .where(work.c.id == identity)
            .values(lease_until=now() - timedelta(seconds=1))
        )
    assert q.claim() is None
    with pytest.raises(StaleLeaseError):
        q.finish(identity, claimed["generation"], {})
    assert q.expired()[0]["id"] == identity
    q.reconcile_expired(identity, safe_to_retry=False)
    assert q.claim() is None
    assert q.list("owner")[0]["state"] == "reconciliation_required"


def test_execution_confirm_and_submission_response_loss(ledger, owner):
    run = ledger.request(owner, "project", 1, "task", "evaluator", "request")
    assert (
        ledger.request(owner, "project", 1, "task", "evaluator", "request")["id"]
        == run["id"]
    )
    proposal = ledger.propose(owner, run["id"], {"configuration": {"model": "v1"}})
    with pytest.raises(ValueError):
        ledger.decide(owner, run["id"], "confirm", 2, proposal["fingerprint"], "allow")
    ledger.decide(owner, run["id"], "confirm", 1, proposal["fingerprint"], "allow")
    ledger.decide(owner, run["id"], "confirm", 1, proposal["fingerprint"], "allow")
    prepare = ledger.queue.claim()
    ledger.queue.finish(prepare["id"], prepare["generation"], {})
    submit = ledger.queue.claim()
    ledger.begin_submission(owner, run["id"], submit["id"], submit["generation"])
    # Crash between platform acceptance and storing its returned task ID.
    assert ledger.recover_submissions() == 1
    assert ledger.list(owner)[0]["state"] == "submission_uncertain"
    with pytest.raises(ValueError, match="replayed"):
        ledger.begin_submission(owner, run["id"], submit["id"], submit["generation"])


def test_owner_isolation_and_active_task_uniqueness(ledger, owner):
    run = ledger.request(owner, "project", 1, "task", "evaluator", "a")
    foreign = owner.model_copy(update={"user_id": "other"})
    assert ledger.list(foreign) == []
    with pytest.raises(PermissionError):
        ledger.propose(foreign, run["id"], {})
    with pytest.raises(ValueError, match="conflicts with an active task"):
        ledger.request(owner, "project", 2, "task", "evaluator", "b")
    assert len(ledger.list(owner)) == 1


def test_inbox_persists_early_and_duplicate_events(ledger):
    ledger.ingest("message", "unbound-task", "kafka", {"status": "succeeded"})
    ledger.ingest("message", "unbound-task", "kafka", {"status": "succeeded"})
    from pyromind_runtime.infrastructure.sql_experiment_execution import inbox

    with ledger.engine.connect() as conn:
        rows = list(conn.execute(sa.select(inbox)).mappings())
    assert len(rows) == 1
    assert rows[0]["state"] == "pending"


def test_completed_result_does_not_allow_expired_runner_to_overwrite(ledger):
    identity = ledger.queue.enqueue("owner", "lane", "item", "prepare", {})
    item = ledger.queue.claim()
    ledger.queue.finish(identity, item["generation"], {"answer": "first"})
    with pytest.raises(StaleLeaseError):
        ledger.queue.finish(identity, item["generation"], {"answer": "stale"})
    assert ledger.queue.list("owner")[0]["checkpoint"] == {"answer": "first"}


def test_uncertain_process_keeps_role_blocked(ledger):
    identity = ledger.queue.enqueue("owner", "lane", "first", "prepare", {})
    ledger.queue.enqueue("owner", "lane", "second", "prepare", {})
    item = ledger.queue.claim()
    ledger.queue.finish(identity, item["generation"], {}, "reconciliation_required")
    assert ledger.queue.claim() is None


async def test_worker_completes_and_stops_cleanly(ledger):
    from pyromind_runtime.application.work_queue import DurableWorker

    identity = ledger.queue.enqueue("owner", "lane", "first", "prepare", {})

    async def handle(item):
        return {"prepared": item["id"]}

    async def reconcile(item):
        return False

    worker = DurableWorker(ledger.queue, {"prepare": handle}, reconcile)
    assert await worker.run_once()
    assert ledger.queue.list("owner")[0]["checkpoint"] == {"prepared": identity}
    assert not await worker.run_once()
    await worker.close()


async def test_worker_does_not_replay_external_side_effect_on_handler_failure(ledger):
    from pyromind_runtime.application.work_queue import DurableWorker

    ledger.queue.enqueue("owner", "lane", "submit", "submit", {})
    calls = []

    async def handle(item):
        calls.append(item["id"])
        raise OSError("response lost")

    async def reconcile(item):
        return False

    worker = DurableWorker(ledger.queue, {"submit": handle}, reconcile)
    assert await worker.run_once()
    assert not await worker.run_once()
    assert len(calls) == 1
    assert ledger.queue.list("owner")[0]["state"] == "reconciliation_required"


def test_early_observation_applied_after_binding_without_regression(ledger, owner):
    from pyromind_runtime.infrastructure.sql_experiment_execution import submissions

    run = ledger.request(owner, "project", 1, "task", "evaluator", "request")
    proposal = ledger.propose(owner, run["id"], {"expected_artifacts": ["metrics"]})
    ledger.decide(owner, run["id"], "confirm", 1, proposal["fingerprint"], "allow")
    ledger.ingest("complete", "platform-1", "kafka", {"status": "succeeded"})
    assert ledger.apply_observations() == 0
    ledger.submission_result(owner, run["id"], "platform-1")
    assert ledger.apply_observations() == 1
    ledger.ingest("old-progress", "platform-1", "kafka", {"status": "running"})
    ledger.apply_observations()
    with ledger.engine.connect() as conn:
        assert conn.execute(sa.select(submissions.c.state)).scalar_one() == "succeeded"
    assert ledger.list(owner)[0]["state"] == "waiting_artifacts"
    with pytest.raises(ValueError, match="frozen manifest"):
        ledger.register_artifacts(owner, run["id"], [], {})
    assert ledger.list(owner)[0]["state"] == "waiting_artifacts"
    ledger.register_artifacts(
        owner,
        run["id"],
        [
            {"name": "metrics", "uri": "storage://metrics", "sha256": "a" * 64},
        ],
        {"accuracy": 0.9},
    )
    row = ledger.list(owner)[0]
    assert row["state"] == "succeeded" and row["report_state"] == "queued"
    assert len(row["artifacts"]) == 1


def test_plan_changed_after_proposal_cannot_be_confirmed(ledger, owner):
    from pyromind_runtime.infrastructure.sql_experiment_plan_store import projects

    run = ledger.request(owner, "project", 1, "task", "evaluator", "request")
    proposal = ledger.propose(owner, run["id"], {"configuration": {}})
    with ledger.engine.begin() as conn:
        conn.execute(projects.update().values(current_revision=2))
    with pytest.raises(ValueError, match="plan changed"):
        ledger.decide(owner, run["id"], "confirm", 1, proposal["fingerprint"], "allow")
    assert ledger.list(owner)[0]["submission"] is None


def test_mariadb_concurrent_enqueue_and_claim():
    import os
    from concurrent.futures import ThreadPoolExecutor
    from uuid import uuid4

    from pyromind_runtime.infrastructure.sql_work_queue import SQLWorkQueue

    url = os.getenv("EXPERIMENT_EXECUTION_TEST_DATABASE_URL")
    if not url:
        pytest.skip("dedicated MariaDB test database is not configured")
    first_engine = sa.create_engine(url, pool_pre_ping=True)
    second_engine = sa.create_engine(url, pool_pre_ping=True)
    try:
        metadata.create_all(first_engine)
        first = SQLWorkQueue(first_engine)
        second = SQLWorkQueue(second_engine)
        scope = uuid4().hex
        with ThreadPoolExecutor(max_workers=2) as pool:
            ids = list(
                pool.map(
                    lambda q: q.enqueue(scope, "role", "same", "prepare", {}),
                    [first, second],
                )
            )
        assert ids[0] == ids[1]
        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = list(pool.map(lambda q: q.claim(), [first, second]))
        claimed = [item for item in claims if item]
        assert len(claimed) == 1
        item = claimed[0]
        first.finish(item["id"], item["generation"], {"ok": True})
        assert second.list(scope)[0]["checkpoint"] == {"ok": True}
    finally:
        first_engine.dispose()
        second_engine.dispose()


@pytest.mark.parametrize(
    "payload",
    [
        {"headers": {"Authorization": "Bearer example"}},
        {"model": {"api_key": "example"}},
        {"uri": "https://example.test/file?X-Amz-Signature=example"},
    ],
)
def test_execution_queue_does_not_persist_request_credentials(ledger, payload):
    with pytest.raises(ValueError, match="persisted"):
        ledger.queue.enqueue("owner", "lane", "secret", "prepare", payload)
    assert ledger.queue.list("owner") == []


def test_retry_requires_proven_termination_and_never_replays_submission(ledger):
    identity = ledger.queue.enqueue("owner", "lane", "first", "prepare", {})
    item = ledger.queue.claim()
    ledger.queue.finish(identity, item["generation"], {}, "reconciliation_required")
    with pytest.raises(ValueError, match="proved stopped"):
        ledger.queue.retry("owner", identity, previous_execution_stopped=False)
    ledger.queue.retry("owner", identity, previous_execution_stopped=True)
    assert ledger.queue.list("owner")[0]["state"] == "queued"
    assert ledger.queue.claim() is None  # backoff is persisted
    second = ledger.queue.enqueue(
        "owner", "submit-lane", "submit", "submit_execution", {}
    )
    item = ledger.queue.claim()
    ledger.queue.finish(second, item["generation"], {}, "reconciliation_required")
    with pytest.raises(ValueError, match="cannot be retried"):
        ledger.queue.retry("owner", second, previous_execution_stopped=True)


def test_cancel_before_submission_prevents_platform_creation(ledger, owner):
    run = ledger.request(owner, "project", 1, "task", "evaluator", "request")
    proposal = ledger.propose(owner, run["id"], {"configuration": {}})
    ledger.decide(owner, run["id"], "confirm", 1, proposal["fingerprint"], "allow")
    preparing = ledger.queue.claim()
    ledger.queue.finish(preparing["id"], preparing["generation"], {})
    submitting = ledger.queue.claim()
    assert ledger.cancel(owner, run["id"])["state"] == "cancel_requested"
    with pytest.raises(ValueError, match="no longer eligible"):
        ledger.begin_submission(
            owner, run["id"], submitting["id"], submitting["generation"]
        )


def test_uncertain_submission_requires_explicit_resolution(ledger, owner):
    run = ledger.request(owner, "project", 1, "task", "evaluator", "request")
    proposal = ledger.propose(owner, run["id"], {"configuration": {}})
    ledger.decide(owner, run["id"], "confirm", 1, proposal["fingerprint"], "allow")
    ledger.submission_result(owner, run["id"], None)
    with pytest.raises(ValueError, match="acceptance is uncertain"):
        ledger.cancel(owner, run["id"])
    result = ledger.resolve(
        owner, run["id"], "platform-1", "resolve", {"workflow_verified": True}
    )
    assert result["state"] == "waiting_platform"
    assert (
        ledger.resolve(owner, run["id"], "platform-1", "resolve", {})["state"]
        == "waiting_platform"
    )
    with pytest.raises(ValueError, match="different platform task"):
        ledger.resolve(owner, run["id"], "platform-2", "resolve", {})


def test_abandon_records_remote_task_may_still_be_running(ledger, owner):
    run = ledger.request(owner, "project", 1, "task", "evaluator", "request")
    proposal = ledger.propose(owner, run["id"], {"configuration": {}})
    ledger.decide(owner, run["id"], "confirm", 1, proposal["fingerprint"], "allow")
    ledger.submission_result(owner, run["id"], None)
    result = ledger.abandon(owner, run["id"], "abandon")
    assert result["state"] == "abandoned" and result["active_key"] is None
    resolution = ledger.list(owner)[0]["submission"]["evidence"]["resolution"]
    assert resolution["remote_may_be_running"] is True


def test_recovery_retains_role_ownership_until_original_work_is_reclaimed(ledger):
    q = ledger.queue
    original = q.enqueue("owner", "role", "original", "prepare", {})
    item = q.claim()
    q.enqueue("owner", "role", "urgent", "prepare", {}, priority=0)
    with ledger.engine.begin() as conn:
        conn.execute(
            work.update()
            .where(work.c.id == original)
            .values(lease_until=now() - timedelta(seconds=1))
        )
    q.reconcile_expired(original, safe_to_retry=True)
    resumed = q.claim()
    assert resumed["id"] == original
    assert resumed["generation"] > item["generation"]
    with pytest.raises(StaleLeaseError):
        q.heartbeat(original, item["generation"])


def test_role_proposal_rejects_wrong_execution_and_expired_work(ledger, owner):
    run = ledger.request(owner, "project", 1, "task", "evaluator", "prepare")
    item = ledger.queue.claim()
    scope = {"work_id": item["id"], "generation": item["generation"]}
    with pytest.raises(PermissionError):
        ledger.propose_from_role(owner, run["id"], "different-role", scope, {})
    with ledger.engine.begin() as conn:
        conn.execute(
            work.update()
            .where(work.c.id == item["id"])
            .values(lease_until=now() - timedelta(seconds=1))
        )
    with pytest.raises(StaleLeaseError):
        ledger.propose_from_role(owner, run["id"], "evaluator", scope, {})


def test_failed_worker_receipt_keeps_native_reference_for_recovery(ledger):
    identity = ledger.queue.enqueue("owner", "role", "prepare", "prepare", {})
    item = ledger.queue.claim()
    ledger.save_work_checkpoint(
        identity, item["generation"], {"execution_ref": "native-run"}
    )
    ledger.queue.finish(
        identity, item["generation"], {"reason": "reauthorize"}, "waiting_authorization"
    )
    assert ledger.queue.list("owner")[0]["checkpoint"]["execution_ref"] == "native-run"


async def test_pending_launch_recovery_queries_correlation_without_respawning(
    ledger, owner
):
    from types import SimpleNamespace
    from typing import cast

    from pyromind_runtime.application.delegated_execution import DelegatedExecution
    from pyromind_runtime.ports.experiment_plans import ExperimentPlanStore

    run = ledger.request(owner, "project", 1, "task", "evaluator", "prepare")
    ledger.propose(owner, run["id"], {"configuration": {}})
    item = ledger.queue.claim()
    ledger.save_work_checkpoint(
        item["id"], item["generation"], {"launch_pending": True}
    )
    with ledger.engine.begin() as conn:
        conn.execute(
            work.update()
            .where(work.c.id == item["id"])
            .values(lease_until=now() - timedelta(seconds=1))
        )
    ledger.queue.reconcile_expired(item["id"], True)
    recovered = ledger.queue.claim()
    task = SimpleNamespace(id="task")
    plans = cast(
        ExperimentPlanStore,
        SimpleNamespace(
            revision=lambda owner, revision: SimpleNamespace(
                plan=SimpleNamespace(tasks=[task])
            )
        ),
    )
    calls = []

    async def control(owner, role, request):
        calls.append(request)
        return {"execution_ref": "opaque-ref", "status": "succeeded"}

    result = await DelegatedExecution(plans, ledger, control).prepare(recovered)
    assert result["proposal_ready"] is True
    assert [entry["action"] for entry in calls] == ["authorize", "status"]
    assert calls[1]["launch_request_id"] == recovered["id"]
    assert "execution_ref" not in calls[1]
