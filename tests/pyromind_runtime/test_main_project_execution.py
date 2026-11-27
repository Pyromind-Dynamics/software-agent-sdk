"""Exercise a single real Pi session with deterministic model and Studio responses."""

import asyncio
import json
import re
from dataclasses import replace
from uuid import uuid4

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from harness_adapter.pi_adapter import PiAdapter
from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.plan_execution import with_plan_report
from pyromind_runtime.application.train_observer import ObservationCursor, TrainObserver
from pyromind_runtime.application.versions import Versions
from pyromind_runtime.domain.commands import (
    CreateProjectCommand,
    EnqueueTaskCommand,
    UserMessageCommand,
)
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.observations import MetricPoint, SourcePage
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.preparation import PreparationPolicy, SetPreparationPlan
from pyromind_runtime.domain.session_policy import SessionPolicy
from pyromind_runtime.domain.versions import SaveVersion, VersionToolRequest
from pyromind_runtime.infrastructure.file_artifact_snapshots import (
    FileArtifactSnapshots,
)
from pyromind_runtime.infrastructure.sqlite_version_store import SQLiteVersionStore
from pyromind_runtime.ports.harness import DelegatedRole, SessionSpec

from .fake_adapter import FakeAdapter
from .test_native_preparation import stream
from .test_preparation import (
    preparation_setup as preparation_setup,
    ready_request as ready_request,
)
from .test_workflow_execution import FakeSubmission


_GATEWAY_MODEL = "deepseek/deepseek-v4.1-flash-1017"


@pytest.mark.parametrize(
    ("queued_preparation", "model_id", "legacy_control"),
    [
        (False, "test", False),
        (True, "test", False),
        (False, _GATEWAY_MODEL, True),
        (False, _GATEWAY_MODEL, False),
    ],
    ids=["user", "queue", "deepseek-control", "deepseek"],
)
@pytest.mark.parametrize("state_format", ["task_records", "tasks"])
async def test_main_runs_plans_and_evaluation_starts_only_its_report_runner(
    preparation_setup,
    tmp_path,
    monkeypatch,
    queued_preparation,
    state_format,
    model_id,
    legacy_control,
):
    preparation, owner, _ = preparation_setup
    scope = owner.conversation_id
    context = RequestContext(
        user_id=owner.user_id, profile_id=owner.profile_id, x_cluster="test"
    )
    profile = ECHOMIND_PRODUCT_PROFILE
    if not legacy_control:
        profile = profile.for_model(model_id)
    assert profile.skill_allowlist and profile.development_skill_allowlist
    assert profile.development_product_instructions
    root = tmp_path / "conversations"
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    adapter = PiAdapter(root, terminal_backend="os-sandbox")
    versions = Versions(
        SQLiteVersionStore(tmp_path / "versions.db"),
        FileArtifactSnapshots(tmp_path / "storage"),
    )
    studio = FakeSubmission()

    class Metrics:
        async def read(
            self, binding, _metrics, start, _count, _context, *, previous_source=""
        ):
            assert binding.task_id == "task-123"
            return SourcePage(
                points=(
                    MetricPoint(
                        key="loss",
                        value=0.25,
                        unit="number",
                        source_point_id=f"point:{start}",
                        source_run_id="source",
                        log_step=start,
                        training_step=start,
                    ),
                ),
                source_run_id="source",
                next_step=start + 1,
            )

    runtime = ConversationRuntime(
        root,
        adapter,
        preparation_store=preparation,
        versions=versions,
        workflow_submission=studio,
        train_observer=TrainObserver(Metrics(), ObservationCursor(b"test-only")),
    )
    for _ in range(4):
        await runtime._stage_analysis_capacity.acquire()
    adapter.set_version_handler(runtime.version_call)
    adapter.set_preparation_handler(
        runtime.preparation_call, runtime.preparation_revision
    )
    adapter.set_workflow_preparer(runtime.prepare_workflow_execution)
    adapter.set_plan_submission_handler(runtime.bind_plan_submission)

    jobs = []
    execute = adapter._business_tools.execute

    async def platform_tool(name, arguments, tool_context, **kwargs):
        if name != "df_submit_evaluation":
            return await execute(name, arguments, tool_context, **kwargs)
        assert "plan_task_id" not in arguments
        task_id = f"evaluation-{len(jobs) + 1}"
        assert arguments["operation"] == "inference"
        assert tool_context.request_context.plan_task_id in {"baseline", "candidate"}
        runtime._store(scope).save_pipeline_run(
            PipelineRun(
                run_id=uuid4(),
                conversation_id=scope,
                task_id=task_id,
                definition={
                    "inference": {"model_path": arguments["model_path"]},
                    "input_path": arguments["input_path"],
                },
                fingerprint="test",
                output_dir="/test",
                submission_status="submitted",
            ),
            expected_revision=None,
        )
        jobs.append(task_id)
        return {
            "is_error": False,
            "details": {"task_id": task_id},
            "content": [{"type": "text", "text": task_id}],
            "signals": [
                {
                    "type": "external_task.submitted",
                    "task": {"kind": "data_preparation", "task_id": task_id},
                }
            ],
        }

    monkeypatch.setattr(adapter._business_tools, "execute", platform_tool)
    stages = [
        ("baseline", "init", "bench"),
        ("training", "train", "train_run"),
        ("candidate", "eval", "model_eval"),
    ]
    plan = {
        "goal": "Optimize and evaluate",
        "resources": [
            {
                "id": "data",
                "name": "Dataset",
                "storage_path": "/workspace/samples.jsonl",
            },
            {
                "id": "merged",
                "name": "Merged model",
                "storage_path": "/workspace/model",
            },
        ],
        "stages": [{"id": key, "kind": node, "name": key} for key, node, _ in stages],
        "tasks": [
            {
                "id": key,
                "kind": kind,
                "stage_id": key,
                "name": key,
                "outputs": [{"name": "primary_results", "kind": "report"}],
                "inputs": [
                    {"name": "model", "resource_id": "merged"},
                    {"name": "dataset", "resource_id": "data"},
                ]
                if key == "candidate"
                else [{"name": "data", "resource_id": "data"}],
            }
            for key, _, kind in stages
        ],
    }
    request = SetPreparationPlan.model_validate(
        {"plan": plan, "entry_task_id": "baseline"}
    )
    evaluations = {
        "input_path": "/workspace/samples.jsonl",
        "model_path": "/workspace/model",
        "operation": "inference",
    }
    evidence = json.dumps(
        {
            "summary": "Final evidence",
            "metrics": [{"label": "通过率", "value": 0.75, "unit": "ratio"}],
        }
    )
    report = {
        "action": "save",
        "command_id": "report",
        "arguments": {
            "kind": "report",
            "logical_key": "candidate/evidence",
            "plan_task_id": "candidate",
            "plan_attempt": 1,
            "source": {"text": evidence, "filename": "report.json"},
        },
    }
    actions = [
        ("df_submit_evaluation", {**evaluations, "plan_task_id": "baseline"}),
        ("df_submit_evaluation", {**evaluations, "plan_task_id": "baseline"}),
        None,
        (
            "submit_workflow",
            {
                "plan_task_id": "training",
                "plan_attempt": 3,
                "workflow_path": "public_data/workflow_canvas/workflow.py",
            },
        ),
        None,
        (
            "df_submit_evaluation",
            {
                "operation": "score",
                "input_path": "/workspace/predictions",
                "plan_task_id": "candidate",
            },
        ),
        (
            "df_submit_evaluation",
            {
                **evaluations,
                "model_path": "/workspace/wrong-model",
                "plan_task_id": "candidate",
            },
        ),
        ("df_submit_evaluation", {**evaluations, "plan_task_id": "candidate"}),
        None,
        ("product_versions", report),
        ("write", {}),
        ("edit", {}),
        None,
    ]
    requests = []
    analysis_requests = []
    reviewed = asyncio.Event()
    review_started = asyncio.Event()
    finish_review = asyncio.Event()
    finish_plan_turn = asyncio.Event()

    async def respond(http_request):
        body = await http_request.json()
        system = body["messages"][0]["content"]
        # Report handoffs are separate Main Agent turns in this same session.
        handoff = "A stage report was validated" in str(body["messages"][-1]["content"])
        analyzing = "Summarize the candidate evaluation" in system
        content = None
        if handoff:
            content = "[[echomind:reply]]阶段报告已完成。"
        elif analyzing:
            assert reviewed.is_set()
            assert not body.get("tools")
            analysis_requests.append(body)
            content = json.dumps({"summary": "评估已完成", "findings": []})
        if content is not None:
            chunks = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }
                    ]
                },
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            return web.Response(
                text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
                + "data: [DONE]\n\n",
                content_type="text/event-stream",
            )
        requests.append(body)
        if len(requests) == 1:
            return stream(
                "set_preparation_plan", request.model_dump(mode="json"), "plan"
            )
        if len(requests) == 2:
            return stream()
        if not actions and not review_started.is_set():
            review_started.set()
            await asyncio.wait_for(finish_review.wait(), timeout=10)
        action = actions.pop(0) if actions else None
        if action and action[0] == "submit_workflow":
            store = runtime._store(scope)
            saved = store.load_snapshot()
            assert any(
                task.task_id == "evaluation-1" and task.status == "succeeded"
                for task in saved.external_tasks
            )
            baseline = next(
                run
                for run in saved.experiment_executions or ()
                if run.task_id == "baseline"
            )
            version_project = versions.project(scope, context)
            assert version_project
            baseline_report = versions.save(
                version_project,
                "test:baseline-report",
                SaveVersion.model_validate(
                    {
                        "kind": "report",
                        "logical_key": "baseline/evidence",
                        "source": {"text": evidence, "filename": "baseline.json"},
                    }
                ),
                context,
            )
            runtime._record_execution(
                store,
                with_plan_report(
                    baseline.model_copy(update={"state": "succeeded"}), baseline_report
                ),
            )
        elif action and action[0] == "write":
            (artifact,) = [
                item
                for item in runtime._artifact_versions(scope, context)
                if item.kind == "report" and item.logical_key == "candidate/evidence"
            ]
            rows: list[dict] = [
                {"id": key, "stage_id": key, "business_status": "completed"}
                for key, _, _ in stages
            ]
            rows[1]["plan_attempt"] = 3
            rows[-1]["report_version_id"] = artifact.id
            rows[-1]["verified_outputs"] = {"primary_results": artifact.id}
            rows[-1]["business_status"] = "in_progress"
            plan_file: dict[str, object] = {"tasks": rows}
            if state_format == "task_records":
                plan_file = {
                    "plan": plan,
                    "task_records": {
                        row["id"]: {
                            "plan_attempt": 1,
                            **{
                                name: value
                                for name, value in row.items()
                                if name not in {"id", "stage_id"}
                            },
                        }
                        for row in rows
                    },
                }
            action = (
                "write",
                {
                    "path": "public_data/workflowPLAN.json",
                    "content": json.dumps(plan_file),
                },
            )
        elif action and action[0] == "edit":
            await wait_for(
                lambda s: any(
                    r.task_id == "baseline" and r.state == "succeeded"
                    for r in s.experiment_executions or ()
                )
            )
            assert not analysis_requests
            reviewed.set()
            action = (
                "edit",
                {
                    "path": "public_data/workflowPLAN.json",
                    "edits": [
                        {
                            "oldText": '"business_status": "in_progress"',
                            "newText": '"business_status": "completed"',
                        }
                    ],
                },
            )
        elif not actions and not review_started.is_set():
            await asyncio.wait_for(finish_plan_turn.wait(), timeout=12)
        return stream(*action, call=f"tool-{len(requests)}") if action else stream()

    app = web.Application()
    app.router.add_post("/v1/chat/completions", respond)

    async def wait_for(predicate):
        async with asyncio.timeout(15):
            while True:
                snapshot = await runtime.get_snapshot(scope, context)
                if predicate(snapshot):
                    return snapshot
                await asyncio.sleep(0.02)

    try:
        async with TestServer(app) as server:
            await runtime.create_conversation(
                SessionSpec(
                    conversation_id=scope,
                    user_id=owner.user_id,
                    profile_id=owner.profile_id,
                    workspace_root=str(root / scope),
                    preparation_policy=profile.preparation_policy,
                    tool_allowlist=profile.tool_allowlist,
                    skill_allowlist=profile.skill_allowlist,
                    development_tool_allowlist=profile.development_tool_allowlist,
                    development_skill_allowlist=profile.development_skill_allowlist,
                    development_product_instructions=profile.development_product_instructions,
                    delegated_roles=(),
                    command_allowlist=profile.command_allowlist,
                    product_instructions=profile.instructions,
                    model_configuration={
                        "model": model_id,
                        "base_url": str(server.make_url("/v1")),
                        "api_key": "local-test-key",
                    },
                ),
                context,
            )
            if queued_preparation:
                await runtime.enqueue_chat_requirement(
                    scope,
                    EnqueueTaskCommand(
                        command_id="prepare", instruction="Review the dataset."
                    ),
                    context,
                )
            else:
                await runtime.submit_command(
                    scope,
                    UserMessageCommand.model_validate(
                        {
                            "command_id": "prepare",
                            "content": [
                                {"type": "text", "text": "Review the dataset."}
                            ],
                        }
                    ),
                    context,
                )
            prepared = await wait_for(
                lambda s: (
                    s.preparation
                    and s.preparation.plan
                    and not s.experiment_queue
                    and s.status != "running"
                )
            )
            assert prepared.preparation
            preparation_prompt = requests[0]["messages"][0]["content"]
            assert profile.instructions in preparation_prompt
            assert profile.development_product_instructions not in preparation_prompt
            if model_id == _GATEWAY_MODEL and not legacy_control:
                assert profile is not ECHOMIND_PRODUCT_PROFILE
                assert ECHOMIND_PRODUCT_PROFILE.instructions not in preparation_prompt
                assert preparation_prompt.count("[[echomind:reply]]") == 1
                assert "update_plan" not in preparation_prompt
            assert set(re.findall(r"<name>([^<]+)</name>", preparation_prompt)) == set(
                profile.skill_allowlist
            )
            assert "frozen-reward-evaluation" not in preparation_prompt
            create = CreateProjectCommand(
                command_id="approve",
                expected_preparation_revision=prepared.preparation.revision,
            )
            receipt = await runtime.submit_command(scope, create, context)
            assert receipt == await runtime.submit_command(scope, create, context)
            missing_binding = {
                "kind": "report",
                "logical_key": "candidate/evidence",
                "source": {"text": evidence, "filename": "report.json"},
            }
            with pytest.raises(ValueError, match="arguments.plan_task_id"):
                await runtime.version_call(
                    scope,
                    VersionToolRequest(action="save", arguments=missing_binding),
                    "missing-binding",
                    context,
                )
            with pytest.raises(ValueError, match="conflicts"):
                await runtime.version_call(
                    scope,
                    VersionToolRequest(
                        action="save",
                        arguments={**missing_binding, "plan_task_id": "baseline"},
                    ),
                    "wrong-binding",
                    context,
                )
            for i, platform_id in enumerate(
                ["evaluation-1", "task-123", "evaluation-2"]
            ):
                snapshot = await wait_for(
                    lambda s: any(
                        run.platform_task_id == platform_id
                        and run.state == "waiting_platform"
                        for run in s.experiment_executions or ()
                    )
                )
                assert snapshot.experiment_executions
                assert (
                    next(
                        r.stage_id
                        for r in snapshot.experiment_executions or ()
                        if r.platform_task_id == platform_id
                    )
                    == stages[i][0]
                )
                if i == 1:
                    assert snapshot.project is not None
                    project_id = snapshot.project.project_id
                    overview = await runtime.get_project_observations(
                        scope, project_id, context
                    )
                    (binding,) = overview.bindings
                    assert binding.project_node_id == "training"
                    for wrong_node in ("baseline", "candidate"):
                        with pytest.raises(ValueError, match="not bound"):
                            await runtime.get_project_training_series(
                                scope,
                                project_id,
                                wrong_node,
                                binding.run_id,
                                (),
                                None,
                                100,
                                context,
                            )
                    first = await runtime.get_project_training_series(
                        scope,
                        project_id,
                        "training",
                        binding.run_id,
                        ("loss", "unconfigured_metric"),
                        None,
                        100,
                        context,
                    )
                    second = await runtime.get_project_training_series(
                        scope,
                        project_id,
                        "training",
                        binding.run_id,
                        ("loss",),
                        first.next_cursor,
                        100,
                        context,
                    )
                    assert [
                        point.log_step for point in first.points + second.points
                    ] == [0, 1]
                    assert first.ignored_keys == ("unconfigured_metric",)
                    assert not second.ignored_keys
                for _ in range(2):
                    await runtime.deliver_external_task_status(
                        scope, task_id=platform_id, status="succeeded"
                    )
            snapshot = await wait_for(
                lambda s: (
                    not actions
                    and any(
                        r.task_id == "candidate"
                        and r.state == "succeeded"
                        and r.report_state == "ready"
                        for r in s.experiment_executions or ()
                    )
                )
            )
            finish_plan_turn.set()
            await wait_for(
                lambda s: (
                    s.status != "running"
                    and any(job.status == "queued" for job in s.stage_analyses)
                )
            )
            assert runtime._active[scope].inflight > 0
            assert not await runtime._release(scope, reason="test-capacity")
            for _ in range(4):
                runtime._stage_analysis_capacity.release()
            snapshot = await wait_for(
                lambda s: (
                    s.status != "running"
                    and any(job.status == "ready" for job in s.stage_analyses)
                )
            )
            (job,) = snapshot.stage_analyses
            assert job.stage_id == "candidate" and job.role == "evaluation_analysis"
            assert job.result is not None
            assert job.result.metrics[0].value == 0.75
            assert len(analysis_requests) == 1
            assert snapshot.project is not None
            overview = await runtime.get_project_observations(
                scope, snapshot.project.project_id, context
            )
            assert overview.stage_analyses == (job,)
            await runtime.submit_command(
                scope,
                create.model_copy(update={"command_id": "approve-again"}),
                context,
            )
            snapshot = await runtime.get_snapshot(scope, context)
            assert snapshot.experiment_executions
            assert all(r.state == "succeeded" for r in snapshot.experiment_executions)
            execution_prompt = requests[2]["messages"][0]["content"]
            assert profile.development_product_instructions in execution_prompt
            assert profile.instructions not in execution_prompt
            if model_id == _GATEWAY_MODEL and not legacy_control:
                assert execution_prompt.count("[[echomind:reply]]") == 1
                assert "update_plan" not in execution_prompt
                assert (
                    profile.development_product_instructions
                    in requests[-1]["messages"][0]["content"]
                )
            assert set(re.findall(r"<name>([^<]+)</name>", execution_prompt)) == set(
                profile.development_skill_allowlist
            )
            (training,) = (
                r for r in snapshot.experiment_executions if r.stage_id == "training"
            )
            assert training.run_id.endswith(":training:3")
            assert training.platform_task_id == "task-123"
            events = runtime._store(scope).replay()
            (candidate_evidence,) = [
                item
                for item in runtime._artifact_versions(scope, context)
                if item.logical_key == "candidate/evidence"
            ]
            artifact_id = candidate_evidence.id
            assert any(
                event.type == "artifact.registered"
                and event.payload["artifact_id"] == artifact_id
                for event in events
            )
            read = VersionToolRequest(
                action="read", arguments={"artifact_id": artifact_id}
            )
            assert (await runtime.version_call(scope, read, "lookup", context))[
                "text"
            ] == evidence
            with pytest.raises(PermissionError):
                await runtime.version_call(
                    scope, read, "foreign", replace(context, profile_id="other")
                )
            old_request = VersionToolRequest(action="save", arguments=missing_binding)
            legacy = versions.call(scope, old_request, "legacy-save", context)
            replay = await runtime.version_call(
                scope, old_request, "legacy-save", context
            )
            assert replay["id"] == legacy["id"]
            assert len(studio.submissions) == 1 and len(jobs) == 2
            assert not (root / scope / "pi" / "subagents").exists()
            for model_request in requests:
                names = {tool["function"]["name"] for tool in model_request["tools"]}
                assert {"find", "grep", "terminal"} <= names
                assert {"subagent", "ls", "bash"}.isdisjoint(names)
            assert all(e.payload.get("category") != "subagent" for e in events)
            queued = EnqueueTaskCommand(
                command_id="additional", instruction="Review the reports"
            )
            with pytest.raises(PermissionError):
                await runtime.submit_command(
                    scope, queued, replace(context, profile_id="other")
                )
            await runtime.submit_command(scope, queued, context)
            await runtime.submit_command(scope, queued, context)
            queued_requirements = [
                e.payload["task_id"]
                for e in runtime._store(scope).replay()
                if e.type == "task.queued" and e.payload["kind"] == "requirement"
            ]
            expected = ["requirement:additional"]
            if queued_preparation:
                expected.insert(0, "requirement:prepare")
            assert queued_requirements == expected
            await asyncio.wait_for(review_started.wait(), timeout=10)
            follow_up = UserMessageCommand.model_validate(
                {
                    "command_id": "while-reviewing",
                    "content": [{"type": "text", "text": "Check sample coverage"}],
                }
            )
            await runtime.submit_command(scope, follow_up, context)
            snapshot = await runtime.get_snapshot(scope, context)
            assert snapshot.experiment_queue
            assert [item["task_id"] for item in snapshot.experiment_queue] == [
                "requirement:while-reviewing"
            ]
            assert "Check sample coverage" not in json.dumps(requests[-1])
            finish_review.set()
            await wait_for(
                lambda s: (
                    not s.experiment_queue
                    and "Check sample coverage" in json.dumps(requests[-1])
                )
            )
    finally:
        finish_plan_turn.set()
        finish_review.set()
        await runtime.close()


async def test_retire_roles_after_running_stage_and_recover_queued_requirement(
    preparation_setup, ready_request, tmp_path, monkeypatch
):
    preparation, owner, old_policy = preparation_setup
    scope = owner.conversation_id
    root = tmp_path / "conversations"
    context = RequestContext(
        user_id=owner.user_id, profile_id=owner.profile_id, x_cluster="test"
    )
    adapter = FakeAdapter()
    runtime = ConversationRuntime(root, adapter, preparation_store=preparation)
    policy = SessionPolicy(
        configuration_version=ECHOMIND_PRODUCT_PROFILE.configuration_version,
        product_instructions="Review the plan.",
        development_product_instructions="Execute the approved plan directly.",
        preparation_policy=PreparationPolicy(
            workflow_nodes=tuple(
                node.model_copy(update={"roles": ()})
                for node in old_policy.workflow_nodes
            )
        ),
        development_tool_allowlist=("read", "write"),
        command_allowlist=("user_message", "create_project", "enqueue_task"),
        delegated_roles=(),
    )
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=scope,
                user_id=owner.user_id,
                profile_id=owner.profile_id,
                workspace_root=str(root),
                preparation_policy=old_policy,
                tool_allowlist=("read", "set_preparation_plan"),
                development_tool_allowlist=("read", "subagent"),
                delegated_roles=(DelegatedRole(name="eval"),),
                command_allowlist=("create_project",),
            ),
            context,
        )
        previous_policy = SessionPolicy(
            configuration_version=28,
            preparation_policy=old_policy,
            tool_allowlist=("read", "set_preparation_plan"),
            development_tool_allowlist=("read", "subagent"),
            command_allowlist=("create_project",),
            delegated_roles=({"name": "eval"},),
        )
        assert await runtime.apply_session_configuration(
            scope, previous_policy, context
        )
        assert runtime._store(scope).configuration_version() == 28
        await runtime.preparation_call(
            scope,
            "set_preparation_plan",
            ready_request.model_dump(mode="json"),
            replace(context, preparation_revision=0),
        )
        await runtime.submit_command(
            scope,
            CreateProjectCommand(command_id="start", expected_preparation_revision=1),
            context,
        )
        store = runtime._store(scope)
        store.append(
            ProductEvent(
                conversation_id=scope,
                type="operation.started",
                payload={
                    "operation_id": "legacy-run",
                    "name": "eval",
                    "category": "subagent",
                },
            )
        )
        assert not await runtime.apply_session_configuration(scope, policy, context)
        assert store.access_policy()["delegated_roles"] == ("eval",)
        store.append(
            ProductEvent(
                conversation_id=scope,
                type="operation.completed",
                payload={"operation_id": "legacy-run", "name": "eval"},
            )
        )
        assert await runtime.apply_session_configuration(scope, policy, context)
        assert adapter.session_policies[-1][1].product_instructions == (
            policy.development_product_instructions
        )
        assert store.configuration_version() == policy.configuration_version
        assert store.access_policy()["delegated_roles"] == ()
        snapshot = await runtime.get_snapshot(scope, context)
        assert snapshot.project is not None
        assert snapshot.project.plan == ready_request.plan
        assert snapshot.experiment_executions
        assert any(
            r.run_id == "legacy-run" and r.stage_id == "evaluation"
            for r in snapshot.experiment_executions or ()
        )

        # The receiver has not acknowledged this requirement before shutdown.
        async def unavailable(*args):
            return {"accepted": False}

        monkeypatch.setattr(adapter, "notify_external_task", unavailable)
        queued = EnqueueTaskCommand(
            command_id="durable", instruction="Review the accepted report"
        )
        await runtime.submit_command(scope, queued, context)
        queue = store.load_snapshot().experiment_queue
        assert queue and queue[0]["task_id"] == "requirement:durable"
    finally:
        await runtime.close()

    restored = FakeAdapter()
    runtime = ConversationRuntime(root, restored, preparation_store=preparation)
    try:
        await runtime.get_snapshot(scope, context)
        async with asyncio.timeout(5):
            while not restored.external_task_notifications:
                await asyncio.sleep(0.02)
        await runtime.submit_command(scope, queued, context)
        assert [
            notice.task_id for _, notice, _ in restored.external_task_notifications
        ] == ["requirement:durable"]
        assert not (await runtime.get_snapshot(scope, context)).experiment_queue
    finally:
        await runtime.close()
