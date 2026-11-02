"""Independent task admission and real-ledger result handling without a Graph."""

import hashlib
import json
from datetime import timedelta

import pytest
import sqlalchemy as sa
from pyromind_runtime.application.experiment_execution import ExperimentExecutions
from pyromind_runtime.domain.experiment_execution import RequestExperimentTask
from pyromind_runtime.domain.experiment_plans import PlanOwner, UpdateExperimentContext
from pyromind_runtime.infrastructure.sql_experiment_execution import (
    SQLExperimentExecutionStore,
    metadata,
)
from pyromind_runtime.infrastructure.sql_experiment_plan_store import (
    SQLExperimentPlanStore,
    metadata as plan_metadata,
)
from pyromind_runtime.infrastructure.sql_work_queue import StaleLeaseError, now, work


@pytest.fixture
def experiment(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'ledger.db'}")
    plan_metadata.create_all(engine)
    metadata.create_all(engine)
    plans, store = SQLExperimentPlanStore(engine), SQLExperimentExecutionStore(engine)
    owner = PlanOwner(
        user_id="42", conversation_id="conversation", profile_id="profile"
    )
    yield ExperimentExecutions(plans, store), owner, engine
    engine.dispose()


def request(identity="request", intent="execute", **changes):
    return RequestExperimentTask.model_validate(
        {
            "role_id": "eval",
            "request_id": identity,
            "intent": intent,
            "task": {
                "name": "先看看这个模型的效果",
                "instruction": "Run the baseline with existing evaluation tools",
                "inputs": {
                    "model_path": "/models/base",
                    "input_path": "/dataset.jsonl",
                },
            },
            **changes,
        }
    )


def claim(service, owner, run):
    item = service.store.queue.claim(kinds=("execute_task",))
    assert item and item["payload"]["run_id"] == run["id"]
    return {
        "run_id": run["id"],
        "work_id": item["id"],
        "generation": item["generation"],
    }


def test_task_does_not_require_graph_or_plan_and_is_idempotent(experiment):
    service, owner, _ = experiment
    run = service.request(owner, request(), ("eval",))
    assert run["revision"] is None and run["state"] == "queued"
    assert service.plans.view(owner).current is None
    assert service.request(owner, request(), ("eval",))["id"] == run["id"]
    with pytest.raises(ValueError, match="different task"):
        service.request(
            owner,
            request(task={"name": "Changed", "instruction": "Different"}),
            ("eval",),
        )
    with pytest.raises(PermissionError, match="not enabled"):
        service.request(owner, request(role_id="train"), ("eval",))


def test_advisory_graph_can_loop_and_change_without_changing_task(experiment):
    service, owner, _ = experiment
    run = service.request(owner, request(), ("eval",))
    guide = UpdateExperimentContext.model_validate(
        {
            "context": {
                "goal": "Explore",
                "stages": [
                    {"id": "init", "name": "Init"},
                    {"id": "data", "name": "Data"},
                ],
                "edges": [
                    {"source": "init", "target": "data"},
                    {"source": "data", "target": "init"},
                ],
            }
        }
    )
    view = service.plans.update_context(owner, guide)
    assert view.sequence == 1
    assert service.store.list(owner)[0]["task_spec"] == run["task_spec"]
    assert service.list(owner)[0].plan_revision is None
    with pytest.raises(ValueError, match="context changed"):
        service.plans.update_context(owner, guide)


async def test_eval_submission_uses_shared_executor_once_and_prepare_cannot_submit(
    experiment,
):
    service, owner, _ = experiment
    run = service.request(owner, request(), ("eval",))
    scope = claim(service, owner, run)
    calls = []

    async def execute(*args):
        calls.append(args)
        return {
            "content": [{"type": "text", "text": "Submitted"}],
            "is_error": False,
            "details": {
                "task_id": "platform-1",
                "run_id": "pipeline-1",
                "output_dir": "/outputs/one",
                "status": "Pending",
            },
        }

    async def read(*_):
        raise AssertionError("no input fingerprint approval is required")

    arguments = {
        "operation": "inference",
        "input_path": "/dataset.jsonl",
        "model_path": "/models/base",
    }
    await service.role_call(
        owner, "eval", "df_submit_evaluation", arguments, scope, execute, read
    )
    await service.role_call(
        owner, "eval", "df_submit_evaluation", arguments, scope, execute, read
    )
    assert len(calls) == 1 and calls[0][3] == "df_submit_evaluation"
    assert calls[0][2]["operation_id"]
    assert service.list(owner)[0].platform_task_id == "platform-1"
    assert service.list(owner)[0].state == "waiting_platform"
    service.store.queue.finish(scope["work_id"], scope["generation"], {})
    prepared = service.request(owner, request("prepare", "prepare"), ("eval",))
    prepare_scope = claim(service, owner, prepared)
    with pytest.raises(PermissionError, match="only authorizes preparation"):
        await service.role_call(
            owner,
            "eval",
            "df_submit_evaluation",
            arguments,
            prepare_scope,
            execute,
            read,
        )
    assert len(calls) == 1


async def test_verified_result_and_report_retry_do_not_resubmit_evaluation(experiment):
    service, owner, _ = experiment
    run = service.request(owner, request(), ("eval",))
    scope = claim(service, owner, run)
    metrics = {"total": 1, "evaluated_cases": 1, "api_or_evaluation_errors": 0}
    report = {
        "status": "succeeded",
        "evaluation_status": "validated",
        "metrics": metrics,
    }
    payloads = {
        "/outputs/one/metrics.json": json.dumps(metrics).encode(),
        "/outputs/one/report.json": json.dumps(report).encode(),
        "/outputs/one/predictions.jsonl": b'{"id":"1","prediction":"ok"}\n',
    }
    report_path = f"public_data/experiment-runs/{run['id']}/public_data/report.md"
    payloads[report_path] = b"One real sample was evaluated."

    async def execute(*_):
        return {
            "is_error": False,
            "content": [],
            "details": {
                "task_id": "platform-1",
                "run_id": "pipeline-1",
                "output_dir": "/outputs/one",
            },
        }

    async def read(_owner, path):
        return payloads[path]

    async def query(_owner, task_id):
        assert task_id == "platform-1"
        return {"status": "Succeeded"}

    await service.role_call(
        owner,
        "eval",
        "df_submit_evaluation",
        {"operation": "inference"},
        scope,
        execute,
        read,
    )
    service.store.queue.finish(
        scope["work_id"], scope["generation"], {"execution_ref": "native-eval"}
    )
    await service.poll(query, read)
    assert service.list(owner)[0].state == "succeeded"
    assert service.list(owner)[0].metrics == metrics
    item = service.store.queue.claim(kinds=("generate_report",))
    assert item
    scope = {
        "run_id": run["id"],
        "work_id": item["id"],
        "generation": item["generation"],
    }
    with pytest.raises(PermissionError, match="cannot start GPU"):
        await service.role_call(
            owner,
            "eval",
            "df_submit_evaluation",
            {"operation": "inference"},
            scope,
            execute,
            read,
        )
    await service.role_call(
        owner,
        "eval",
        "record_experiment_result",
        {
            "run_id": run["id"],
            "summary": "One sample, no execution errors",
            "report_uri": report_path,
        },
        scope,
        execute,
        read,
    )
    view = service.list(owner)[0]
    assert view.report_state == "ready" and view.summary
    assert (
        next(a for a in view.artifacts if a["name"] == "agent_report")["sha256"]
        == hashlib.sha256(payloads[report_path]).hexdigest()
    )
    service.store.ingest("duplicate", "platform-1", "kafka", {"status": "succeeded"})
    service.store.apply_observations()
    assert service.list(owner)[0].state == "succeeded"
    assert (
        len(
            [
                item
                for item in service.store.queue.list(
                    service.store.list(owner)[0]["owner"]
                )
                if item["kind"] == "generate_report"
            ]
        )
        == 1
    )


async def test_missing_artifacts_do_not_create_fake_success(experiment):
    service, owner, _ = experiment
    run = service.request(owner, request(), ("eval",))
    scope = claim(service, owner, run)
    service.store.managed_submission(
        owner, run["id"], scope, {"operation": "inference"}
    )
    service.store.managed_receipt(
        owner,
        run["id"],
        scope,
        "inference",
        {"task_id": "task", "run_id": "pipeline", "output_dir": "/missing"},
    )

    async def query(*_):
        return {"status": "Succeeded"}

    async def read(*_):
        raise FileNotFoundError("not uploaded yet")

    await service.poll(query, read)
    view = service.list(owner)[0]
    assert view.state == "waiting_artifacts" and view.metrics is None
    assert view.report_state == "not_started"


def test_other_owners_and_expired_leases_cannot_submit(experiment):
    service, owner, engine = experiment
    run = service.request(owner, request(), ("eval",))
    scope = claim(service, owner, run)
    other = owner.model_copy(update={"profile_id": "other"})
    with pytest.raises(PermissionError):
        service.store.role_scope(other, run["id"], "eval", scope)
    with engine.begin() as conn:
        conn.execute(
            work.update()
            .where(work.c.id == scope["work_id"])
            .values(lease_until=now() - timedelta(seconds=1))
        )
    with pytest.raises(StaleLeaseError):
        service.store.managed_submission(
            owner, run["id"], scope, {"operation": "inference"}
        )


def test_native_end_with_missing_submit_receipt_is_uncertain_and_cannot_repeat(experiment):
    service, owner, _ = experiment
    run = service.request(owner, request(), ("eval",))
    scope = claim(service, owner, run)
    service.store.managed_submission(owner, run["id"], scope, {"operation": "inference"})
    service.store.complete_agent_turn(owner, run["id"], scope, "Native process exited")
    assert service.list(owner)[0].state == "submission_uncertain"
    with pytest.raises(ValueError, match="cannot submit"):
        service.store.managed_submission(owner, run["id"], scope, {"operation": "inference"})
    service.store.queue.finish(scope["work_id"], scope["generation"], {})
    sub = service.store.list(owner)[0]["submission"]
    service.store.recover_managed_receipt(owner, run["id"], sub["id"], {"task_id": "original", "run_id": sub["evidence"]["pipeline_run_id"], "output_dir": "/output"})
    assert service.list(owner)[0].platform_task_id == "original"
    assert service.list(owner)[0].state == "waiting_platform"


async def test_conflicting_terminal_notification_uses_authoritative_query(experiment):
    service, owner, _ = experiment
    run = service.request(owner, request(), ("eval",))
    scope = claim(service, owner, run)
    service.store.managed_submission(owner, run["id"], scope, {"operation": "inference"})
    service.store.managed_receipt(owner, run["id"], scope, "inference", {"task_id": "task", "output_dir": "/output"})
    service.store.ingest("success", "task", "kafka", {"status": "succeeded"})
    service.store.apply_observations()
    service.store.ingest("failure", "task", "kafka", {"status": "failed"})
    service.store.apply_observations()
    assert service.store.conflicting_tasks() == {"task"}
    calls = []

    async def query(*_):
        calls.append("query")
        return {"status": "failed"}

    async def read(*_):
        raise AssertionError("failed task has no successful artifacts")

    await service.poll(query, read)
    assert calls == ["query"] and service.list(owner)[0].state == "failed"
    assert not service.store.conflicting_tasks()


def test_completed_platform_result_wins_cancel_race(experiment):
    service, owner, _ = experiment
    run = service.request(owner, request(), ("eval",))
    scope = claim(service, owner, run)
    service.store.managed_submission(owner, run["id"], scope, {"operation": "inference"})
    service.store.managed_receipt(owner, run["id"], scope, "inference", {"task_id": "task", "output_dir": "/output"})
    service.store.queue.finish(scope["work_id"], scope["generation"], {})
    assert service.store.cancel(owner, run["id"])["state"] == "cancel_requested"
    control = service.store.queue.claim(kinds=("cancel_execution",))
    scope = {"work_id": control["id"], "generation": control["generation"]}
    service.store.ingest("won-race", "task", "query", {"status": "succeeded"})
    service.store.apply_observations()
    assert service.store.cancellation_result(owner, run["id"], scope, True, True)["state"] == "waiting_artifacts"
    assert service.list(owner)[0].metrics is None

@pytest.mark.parametrize("action", ["bind", "abandon"])
def test_reconcile_targets_uncertain_operation_and_preserves_previous_task(
    experiment, action
):
    service, owner, _ = experiment
    run = service.request(owner, request(), ("eval",))
    scope = claim(service, owner, run)
    store = service.store
    store.managed_submission(owner, run["id"], scope, {"operation": "inference"})
    store.managed_receipt(
        owner,
        run["id"],
        scope,
        "inference",
        {"task_id": "gpu-original", "output_dir": "/original"},
    )
    store.managed_submission(owner, run["id"], scope, {"operation": "report"})
    store.managed_receipt(owner, run["id"], scope, "report", {"uncertain": True})
    with pytest.raises(ValueError, match="acceptance is uncertain"):
        store.cancel(owner, run["id"])
    if action == "bind":
        result = store.resolve(
            owner, run["id"], "cpu-report", "resolve", {"verified": True}
        )
        assert result["state"] == "waiting_platform"
        assert (
            store.resolve(owner, run["id"], "cpu-report", "resolve", {})["id"]
            == run["id"]
        )
        assert store.cancel(owner, run["id"])["state"] == "cancel_requested"
    else:
        assert store.abandon(owner, run["id"], "abandon")["state"] == "abandoned"
    entries = {
        entry["operation_key"]: entry for entry in store.list(owner)[0]["submissions"]
    }
    assert entries["inference"]["task_id"] == "gpu-original"
    assert "resolution" not in entries["inference"]["evidence"]
    assert entries["report"]["evidence"]["resolution"]["action"] == action
