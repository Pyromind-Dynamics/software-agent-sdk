"""Workflow failures do not seal delegated tasks or replace prior receipts."""

import json
from datetime import timedelta

import pytest
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.delegated_execution import DelegatedExecution
from pyromind_runtime.domain.commands import PermissionResponseCommand
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.infrastructure.sql_experiment_execution import (
    SQLExperimentExecutionStore,
    submissions,
)
from pyromind_runtime.infrastructure.sql_work_queue import StaleLeaseError, now, work
from pyromind_runtime.ports.harness import SessionSpec

from .fake_adapter import FakeAdapter
from .test_standalone_experiments import claim, experiment as experiment, request
from .test_workflow_execution import FakeSubmission


@pytest.fixture
async def training_execution(experiment, tmp_path, monkeypatch):
    service, owner, engine = experiment
    run = service.request(
        owner,
        request(role_id="trainer"),
        ("trainer",),
        execution_policy={
            "submission_tool": "submit_workflow",
            "artifact_contract": "model",
        },
    )
    scope = claim(service, owner, run)
    submission = FakeSubmission()
    original_submit = submission.submit

    async def submit(*args, **kwargs):
        await original_submit(*args, **kwargs)
        return f"task-{len(submission.submissions)}"

    monkeypatch.setattr(submission, "submit", submit)
    runtime = ConversationRuntime(
        tmp_path / "product",
        {"pi": FakeAdapter("pi")},
        default_harness_id="pi",
        experiment_plan_store=service.plans,
        experiment_execution_store=service.store,
        execution_roles=("trainer",),
        workflow_submission=submission,
    )
    context = RequestContext(user_id=owner.user_id, profile_id=owner.profile_id)
    await runtime.create_conversation(
        SessionSpec(
            conversation_id=owner.conversation_id,
            user_id=owner.user_id,
            profile_id=owner.profile_id,
            workspace_root=str(tmp_path / "product" / owner.conversation_id),
        ),
        context,
    )
    yield runtime, service, owner, run, scope, submission, engine
    await runtime.close()


async def stage(runtime, owner, scope):
    return await runtime._execute_role_tool(
        owner, "trainer", scope, "submit_workflow", {"workflow_path": "/train.py"}
    )


async def approve(runtime, owner, request_id, command_id):
    return await runtime.submit_command(
        owner.conversation_id,
        PermissionResponseCommand(
            command_id=command_id,
            permission_id=f"workflow-run:{request_id}",
            decision="allow_once",
        ),
        RequestContext(user_id=owner.user_id, profile_id=owner.profile_id),
    )


async def test_subagent_workflow_path_resolves_under_delegated_workspace(
    training_execution,
):
    runtime, _, owner, run, scope, submission, _ = training_execution
    workflow_path = "public_data/workflow_canvas/workflow.py"

    prepared = await runtime._execute_role_tool(
        owner,
        "trainer",
        scope,
        "submit_workflow",
        {"workflow_path": workflow_path},
    )

    delegated_path = (
        f"public_data/experiment-runs/{run['id']}/"
        "public_data/workflow_canvas/workflow.py"
    )
    assert prepared["summary"]["workflow_path"] == delegated_path
    assert submission.prepared_paths == [
        f"/workspace/.pyromind-agent/{owner.conversation_id}/{delegated_path}"
    ]


@pytest.mark.parametrize(
    ("prior_status", "launch", "interrupted"),
    [
        ("stopped", "spawn", False),
        ("succeeded", "resume", False),
        ("unknown", None, False),
        ("succeeded", "resume", True),
    ],
)
async def test_artifact_report_recovers_only_known_context(
    training_execution, prior_status, launch, interrupted
):
    runtime, service, owner, run, scope, submission, _ = training_execution
    prepared = await stage(runtime, owner, scope)
    await approve(runtime, owner, prepared["request_id"], "approval")
    service.store.queue.finish(
        scope["work_id"], scope["generation"], {"execution_ref": "previous-child"}
    )
    service.store.ingest("done", "task-1", "callback", {"status": "succeeded"})
    service.store.apply_observations()
    service.store.queue_artifact_report(owner, run["id"])
    item = service.store.queue.claim(kinds=("execute_task",))
    assert item is not None
    calls = []

    async def control(_owner, _role, request):
        calls.append(request)
        if request["action"] == "authorize":
            return {"status": "authorized"}
        if request["action"] == "status":
            return {"status": prior_status, "execution_ref": "previous-child"}
        if interrupted:
            return {
                "status": "failed",
                "execution_ref": "retry-child"
                if request["request_id"].endswith(":interruption")
                else "report-child",
                "interrupted": True,
                "result_text": "Execution transport lost",
            }
        return {"status": "succeeded", "execution_ref": "report-child"}

    delegated = DelegatedExecution(service.plans, service.store, control)
    if launch is None:
        with pytest.raises(ValueError, match="report context requires reconciliation"):
            await delegated.prepare(item)
        assert [call["action"] for call in calls] == ["authorize", "status"]
    else:
        result = await delegated.prepare(item)
        expected = ["authorize", "status", launch]
        if interrupted:
            expected.append("resume")
            assert calls[-1]["execution_ref"] == "report-child"
            assert calls[-1]["request_id"] == item["id"] + ":interruption"
            assert result["execution_ref"] == "retry-child"
            assert result["report_state"] == "failed"
        assert [call["action"] for call in calls] == expected
        assert calls[2].get("execution_ref") == (
            "previous-child" if launch == "resume" else None
        )
        task = json.loads(calls[-1]["task"])
        assert task["phase"] == "report"
        assert task["submission"]["task_id"] == "task-1"
    assert len(submission.submissions) == 1


@pytest.mark.parametrize(
    ("child_status", "reply", "expected_state", "report_state"),
    [
        (
            "succeeded",
            "Training finished. See `{report_path}`.",
            "succeeded",
            "ready",
        ),
        ("succeeded", " ", "waiting_artifacts", "failed"),
        ("failed", "Host disconnected", "waiting_artifacts", "failed"),
    ],
)
async def test_model_report_accepts_the_native_completion(
    training_execution, child_status, reply, expected_state, report_state
):
    runtime, service, owner, run, scope, submission, _ = training_execution
    prepared = await stage(runtime, owner, scope)
    await approve(runtime, owner, prepared["request_id"], "approval")
    service.store.queue.finish(scope["work_id"], scope["generation"], {})
    service.store.ingest("done", "task-1", "callback", {"status": "succeeded"})
    service.store.apply_observations()
    service.store.queue_artifact_report(owner, run["id"])
    item = service.store.queue.claim(kinds=("execute_task",))
    assert item is not None
    reply = reply.format(
        report_path=f"public_data/experiment-runs/{run['id']}/public_data/report.md"
    )

    async def control(_owner, _role, request):
        if request["action"] == "authorize":
            return {"status": "authorized"}
        assert request["action"] == "spawn"
        assert "record_experiment_result" not in request["task"]
        return {
            "execution_ref": "report-child",
            "status": child_status,
            "result_text": reply,
        }

    delegated = DelegatedExecution(service.plans, service.store, control)
    await delegated.prepare(item)

    view = service.list(owner)[0]
    assert (view.state, view.report_state) == (expected_state, report_state)
    assert view.artifacts == ()
    assert len(submission.submissions) == 1
    if report_state == "ready":
        assert view.summary == reply
        assert service.store.list(owner)[0]["active_key"] is None
        eval_request = request(
            "evaluate-output",
            task={
                "name": "Evaluate training output",
                "instruction": "Use the registered model.",
                "artifact_refs": [
                    {
                        "input_name": "model_path",
                        "task_run_id": run["id"],
                        "artifact_name": "model",
                    }
                ],
            },
        )
        with pytest.raises(ValueError, match="unknown artifact"):
            service.request(owner, eval_request, ("eval",))


@pytest.mark.parametrize("reported", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
async def test_failed_workflow_can_be_revised_and_resubmitted(
    training_execution, reported, legacy
):
    runtime, service, owner, run, scope, submission, engine = training_execution
    first = await stage(runtime, owner, scope)
    await approve(runtime, owner, first["request_id"], "first-approval")
    if legacy:
        with engine.begin() as conn:
            conn.execute(submissions.update().values(operation_key="workflow"))
    service.store.queue.finish(scope["work_id"], scope["generation"], {})

    async def query(*_):
        return {"status": "Failed"}

    async def read(*_):
        raise AssertionError("failed tasks have no model artifacts")

    await service.poll(query, read)
    item = service.store.queue.claim(kinds=("execute_task",))
    assert item is not None
    assert "without submitting" not in item["payload"]["message"]
    scope = {
        "run_id": run["id"],
        "work_id": item["id"],
        "generation": item["generation"],
    }
    if reported:
        service.store.complete_agent_turn(
            owner,
            run["id"],
            scope,
            "Configuration failed. See `failure-report.md`.",
            report=True,
            completed=True,
        )
        service.store.queue.finish(scope["work_id"], scope["generation"], {})
        service.store.continue_task(owner, run["id"], "continue", "Fix the input")
        scope = claim(service, owner, run)

    submission.workflow["name"] = "repaired"
    second = await stage(runtime, owner, scope)
    assert second["request_id"] != first["request_id"]
    assert second["status"] == "awaiting_confirmation"
    assert (await stage(runtime, owner, scope)) == second
    assert len(submission.submissions) == 1
    service.store.complete_agent_turn(
        owner, run["id"], scope, "Revised workflow awaits confirmation", report=True
    )
    row = service.store.list(owner)[0]
    assert row["state"] == "waiting_confirmation"
    assert row["report_state"] == "not_started"
    assert row["result"] is None and row["artifacts"] == []
    if reported:
        assert (
            row["submissions"][0]["evidence"]["result"]["summary"]
            == "Configuration failed. See `failure-report.md`."
        )
    service.store.queue.finish(scope["work_id"], scope["generation"], {})

    for command_id in ("second-approval", "duplicate-approval"):
        await approve(runtime, owner, second["request_id"], command_id)
    assert len(submission.submissions) == 2
    assert submission.submissions[-1][0]["name"] == "repaired"
    row = service.store.list(owner)[0]
    assert row["state"] == "waiting_platform"
    assert row["submission"]["task_id"] == "task-2"
    assert {sub["task_id"] for sub in row["submissions"]} == {"task-1", "task-2"}

    # A corrected observation for the older attempt stays historical.
    service.store.ingest("late-query", "task-1", "query", {"status": "succeeded"})
    service.store.apply_observations()
    recovered = SQLExperimentExecutionStore(engine).list(owner)[0]
    assert recovered["state"] == "waiting_platform"
    assert recovered["submission"]["task_id"] == "task-2"
    service.store.ingest("second-failed", "task-2", "callback", {"status": "failed"})
    service.store.apply_observations()
    service.store.continue_task(owner, run["id"], "continue-again", "Revise again")
    scope = claim(service, owner, run)
    third = await stage(runtime, owner, scope)
    await approve(runtime, owner, third["request_id"], "third-approval")
    assert len(submission.submissions) == 3
    service.store.ingest(
        "third-complete", "task-3", "callback", {"status": "succeeded"}
    )
    service.store.apply_observations()
    assert service.store.list(owner)[0]["state"] == "waiting_artifacts"


async def test_restart_during_resubmission_preserves_failed_receipt(
    training_execution,
):
    runtime, service, owner, run, scope, _, _ = training_execution
    first = await stage(runtime, owner, scope)
    await approve(runtime, owner, first["request_id"], "first-approval")
    service.store.ingest("failed", "task-1", "callback", {"status": "failed"})
    service.store.apply_observations()
    service.store.queue.finish(scope["work_id"], scope["generation"], {})
    service.store.continue_task(owner, run["id"], "continue", "Fix the input")
    scope = claim(service, owner, run)
    second = await stage(runtime, owner, scope)
    second_id = service.store.begin_workflow(
        owner, run["id"], second["request_id"], "sha256", True
    )
    assert service.store.recover_submissions() == 1
    row = service.store.list(owner)[0]
    assert row["state"] == "submission_uncertain"
    assert {sub["state"] for sub in row["submissions"]} == {
        "failed",
        "submission_uncertain",
    }
    service.store.recover_managed_receipt(
        owner, run["id"], second_id, {"task_id": "task-2"}
    )
    assert service.store.list(owner)[0]["submission"]["task_id"] == "task-2"


@pytest.mark.parametrize("status", ["pending", "running", "succeeded"])
async def test_new_workflow_call_can_submit_after_platform_acceptance(
    training_execution, status
):
    runtime, service, owner, run, scope, submission, _ = training_execution
    first = await stage(runtime, owner, {**scope, "tool_call_id": "first"})
    await approve(runtime, owner, first["request_id"], "first-approval")
    service.store.ingest("status", "task-1", "callback", {"status": status})
    service.store.apply_observations()
    next_call = {**scope, "tool_call_id": "second"}
    second = await stage(runtime, owner, next_call)
    assert second["request_id"] != first["request_id"]
    assert await stage(runtime, owner, next_call) == second
    await approve(runtime, owner, second["request_id"], "second-approval")
    assert len(submission.submissions) == 2
    assert service.store.list(owner)[0]["submission"]["task_id"] == "task-2"


@pytest.mark.parametrize("case", ["profile", "conversation", "lease", "task"])
def test_workflow_submission_preserves_execution_isolation(experiment, case):
    service, owner, engine = experiment
    run = service.request(
        owner,
        request(role_id="trainer"),
        ("trainer",),
        execution_policy={"submission_tool": "submit_workflow"},
    )
    scope = claim(service, owner, run)
    if case in {"profile", "conversation"}:
        field = "profile_id" if case == "profile" else "conversation_id"
        owner = owner.model_copy(update={field: "other"})
    elif case == "lease":
        with engine.begin() as conn:
            conn.execute(
                work.update()
                .where(work.c.id == scope["work_id"])
                .values(lease_until=now() - timedelta(seconds=1))
            )
    else:
        other = service.request(
            owner, request(identity="other", role_id="trainer"), ("trainer",)
        )
        run = other
    with pytest.raises((PermissionError, StaleLeaseError)):
        service.store.stage_workflow(owner, run["id"], scope, "request")
