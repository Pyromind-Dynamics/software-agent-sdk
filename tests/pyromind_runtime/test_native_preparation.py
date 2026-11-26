"""Real Pi calls with model responses supplied by a local streaming server."""

import asyncio
import json
from dataclasses import replace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from harness_adapter.pi_adapter import PiAdapter
from harness_adapter.pi_adapter.business_tools import execute_validation_tool
from pyromind_common.sdk.workspace.local import LocalWorkspace
from pyromind_common.tools.workflow.dsl_to_xyflow import convert_dsl_to_xyflow
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.versions import Versions
from pyromind_runtime.domain.commands import (
    CreateProjectCommand,
    InputResponseCommand,
    UserMessageCommand,
)
from pyromind_runtime.domain.content import JsonObject, TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.reward_bundle import START_REWARD_REVIEW
from pyromind_runtime.infrastructure.file_artifact_snapshots import (
    FileArtifactSnapshots,
)
from pyromind_runtime.infrastructure.sqlite_version_store import SQLiteVersionStore
from pyromind_runtime.ports.harness import DelegatedRole, SessionSpec

from .test_preparation import (
    preparation_setup as preparation_setup,
    ready_request as ready_request,
)
from .test_workflow_execution import FakeSubmission


def stream(tool=None, arguments=None, call="call"):
    delta = (
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": call,
                    "type": "function",
                    "function": {"name": tool, "arguments": json.dumps(arguments)},
                }
            ]
        }
        if tool
        else {"content": "Plan submitted for review."}
    )
    chunks = [
        {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls" if tool else "stop",
                }
            ]
        },
    ]
    return web.Response(
        text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        + "data: [DONE]\n\n",
        content_type="text/event-stream",
    )


@pytest.mark.parametrize("concurrent_edit", [False, True])
async def test_single_plan_tool_after_question_and_stale_model_request(
    preparation_setup, ready_request, tmp_path, monkeypatch, concurrent_edit
):
    store, owner, policy = preparation_setup
    context = RequestContext(user_id=owner.user_id, profile_id=owner.profile_id)
    captured = []
    request_started, release = asyncio.Event(), asyncio.Event()

    async def respond(request):
        payload = await request.json()
        captured.append(payload)
        if len(captured) == 1:
            return stream(
                "ask_user",
                {
                    "questions": [
                        {"id": "scope", "prompt": "Which data?", "options": []}
                    ]
                },
                "question",
            )
        if len(captured) == 2:
            request_started.set()
            await release.wait()
            return stream(
                "set_preparation_plan",
                {
                    **ready_request.model_dump(mode="json"),
                    "plan": ready_request.plan.model_dump_json(),
                },
                "plan-first",
            )
        if concurrent_edit and len(captured) == 3:
            return stream(
                "set_preparation_plan",
                ready_request.model_dump(mode="json"),
                "plan-latest",
            )
        return stream()

    app = web.Application()
    app.router.add_post("/v1/chat/completions", respond)
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    root = tmp_path / "conversations"
    adapter = PiAdapter(root, terminal_backend="os-sandbox")
    runtime = ConversationRuntime(
        root, adapter, default_harness_id="pi", preparation_store=store
    )
    adapter.set_preparation_handler(
        runtime.preparation_call, runtime.preparation_revision
    )
    try:
        async with TestServer(app) as server:
            await runtime.create_conversation(
                SessionSpec(
                    conversation_id=owner.conversation_id,
                    user_id=owner.user_id,
                    profile_id=owner.profile_id,
                    workspace_root=str(root / owner.conversation_id),
                    user_input_enabled=True,
                    tool_allowlist=("ask_user", "set_preparation_plan"),
                    delegated_roles=(DelegatedRole(name="eval"),),
                    preparation_policy=policy,
                    model_configuration={
                        "model": "eval-test-model",
                        "base_url": str(server.make_url("/v1")),
                        "api_key": "local-test-key",
                    },
                ),
                context,
            )
            await runtime.submit_command(
                owner.conversation_id,
                UserMessageCommand(
                    command_id="start", content=(TextContent(text="Evaluate data"),)
                ),
                context,
            )
            async with asyncio.timeout(20):
                while not (
                    snapshot := await runtime.get_snapshot(
                        owner.conversation_id, context
                    )
                ).pending_inputs:
                    await asyncio.sleep(0.01)
                await runtime.submit_command(
                    owner.conversation_id,
                    InputResponseCommand.model_validate(
                        {
                            "command_id": "answer",
                            "request_id": snapshot.pending_inputs[0].request_id,
                            "status": "answered",
                            "answers": [{"question_id": "scope", "text": "All"}],
                        }
                    ),
                    context,
                )
                await request_started.wait()
                if concurrent_edit:
                    store.invalidate(owner, "newer-input")
                release.set()
                while (
                    await runtime.get_preparation(owner.conversation_id, context)
                ).phase != "ready":
                    await asyncio.sleep(0.01)
            ready = store.get(owner)
            assert ready.revision == (4 if concurrent_edit else 3)
            assert store.project(owner) is None
            names = {tool["function"]["name"] for tool in captured[0]["tools"]}
            assert names == {"ask_user", "set_preparation_plan"}
            answer = json.loads(captured[1]["messages"][-1]["content"])
            assert set(answer) == {"status", "answers"}
            properties = next(
                tool["function"]["parameters"]["properties"]
                for tool in captured[0]["tools"]
                if tool["function"]["name"] == "set_preparation_plan"
            )
            assert set(properties) == {
                "plan",
                "entry_task_id",
                "training_method",
                "reward_approval_id",
            }
            if concurrent_edit:
                assert "Review context changed" in json.dumps(captured[2]["messages"])
    finally:
        release.set()
        await runtime.close()


async def test_reward_review_freezes_the_approved_bytes_and_registers_once(
    preparation_setup, ready_request, tmp_path, monkeypatch
):
    store, owner, policy = preparation_setup
    scope = owner.conversation_id
    context = RequestContext(
        user_id=owner.user_id, profile_id=owner.profile_id, x_cluster="test-cluster"
    )
    calls = []

    async def respond(request):
        calls.append(await request.json())
        if len(calls) == 1:
            return stream(START_REWARD_REVIEW, review_arguments, "review")
        if len(calls) == 2:
            answer = json.loads(calls[-1]["messages"][-1]["content"])
            assert answer["status"] == "approved"
            return stream(
                "set_preparation_plan",
                {
                    **ready_request.model_dump(mode="json"),
                    "training_method": "rubric_rl",
                    "reward_approval_id": answer["approval_id"],
                },
                "plan",
            )
        return stream()

    review_arguments: JsonObject = {
        "candidate_directory": "candidate",
        "judge_kind": "code",
        "requirements": "Python 3.12",
    }
    app = web.Application()
    app.router.add_post("/v1/chat/completions", respond)
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    root = tmp_path / "conversations"
    snapshots = FileArtifactSnapshots(tmp_path / "storage")
    version_store = SQLiteVersionStore(tmp_path / "versions.sqlite")
    versions = Versions(version_store, snapshots)
    submission = FakeSubmission()
    adapter = PiAdapter(root, terminal_backend="os-sandbox")
    runtime = ConversationRuntime(
        root,
        adapter,
        default_harness_id="pi",
        preparation_store=store,
        versions=versions,
        workflow_submission=submission,
    )
    adapter.set_preparation_handler(
        runtime.preparation_call,
        runtime.preparation_revision,
        runtime.preparation_review,
    )
    try:
        async with TestServer(app) as server:
            await runtime.create_conversation(
                SessionSpec(
                    conversation_id=scope,
                    user_id=owner.user_id,
                    profile_id=owner.profile_id,
                    workspace_root=str(root / scope),
                    user_input_enabled=True,
                    tool_allowlist=(
                        "ask_user",
                        "set_preparation_plan",
                        START_REWARD_REVIEW,
                    ),
                    development_tool_allowlist=("read",),
                    delegated_roles=(DelegatedRole(name="eval"),),
                    preparation_policy=policy,
                    model_configuration={
                        "model": "eval-test-model",
                        "base_url": str(server.make_url("/v1")),
                        "api_key": "local-test-key",
                    },
                ),
                context,
            )
            candidate = root / scope / "public_data/rubric_work/candidate"
            candidate.mkdir(parents=True)
            original_task = "Return the classification and a supporting reason.\n"
            for name, content in {
                "task_prompt.md": original_task,
                "rubric.json": json.dumps(
                    {
                        "items": [{"criterion": "Reason matches evidence"}],
                        "aggregation": "Mean",
                    }
                ),
                "reward.py": (
                    "def studio_reward(completions, **kwargs):\n"
                    "    return [1.0] * len(completions)\n"
                    "def parse_record(record):\n    return record\n"
                    "def score_case(case, response):\n    return 1.0\n"
                ),
                "calibration_review.json": (
                    '[{"case_id":"case-1","score":1,'
                    '"summary":"Supported reason, user agreed."}]'
                ),
            }.items():
                (candidate / name).write_text(content, encoding="utf-8")

            with pytest.raises(PermissionError):
                await runtime.preparation_review(
                    scope,
                    START_REWARD_REVIEW,
                    review_arguments,
                    "foreign",
                    replace(context, profile_id="foreign"),
                )

            await runtime.submit_command(
                scope,
                UserMessageCommand(
                    command_id="start-reward", content=(TextContent(text="Review"),)
                ),
                context,
            )
            async with asyncio.timeout(20):
                while not (
                    snapshot := await runtime.get_snapshot(scope, context)
                ).pending_inputs:
                    await asyncio.sleep(0.01)
                pending = snapshot.pending_inputs[0]
                assert pending.presentation is not None
                task_summary, criteria, _ = pending.presentation.content
                assert isinstance(task_summary, TextContent)
                assert isinstance(criteria, TextContent)
                assert original_task.strip() in task_summary.text
                assert "Reason matches evidence" in criteria.text
                state = store.get(owner)
                assert state.reward_review is not None and state.approved_reward is None
                unapproved_id = state.reward_review.bundle.approval_id
                with pytest.raises(ValueError, match="user-approved"):
                    await runtime.preparation_call(
                        scope,
                        "set_preparation_plan",
                        {
                            **ready_request.model_dump(mode="json"),
                            "training_method": "rubric_rl",
                            "reward_approval_id": unapproved_id,
                        },
                        replace(context, preparation_revision=state.revision),
                    )

                (candidate / "task_prompt.md").write_text(
                    "Edited draft", encoding="utf-8"
                )
                await runtime.submit_command(
                    scope,
                    InputResponseCommand.model_validate(
                        {
                            "command_id": "approve-current-review",
                            "request_id": pending.request_id,
                            "status": "answered",
                            "answers": [
                                {"question_id": "reward_bundle", "option_id": "approve"}
                            ],
                        }
                    ),
                    context,
                )
                while True:
                    state = store.get(owner)
                    snapshot = await runtime.get_snapshot(scope, context)
                    if state.phase == "ready" and snapshot.status != "running":
                        break
                    await asyncio.sleep(0.01)
            command = CreateProjectCommand(
                command_id="create-reward-project",
                expected_preparation_revision=state.revision,
            )

            def unavailable(*args, **kwargs):
                raise OSError("version store temporarily unavailable")

            with monkeypatch.context() as patch:
                patch.setattr(version_store, "reserve_version", unavailable)
                with pytest.raises(OSError, match="temporarily unavailable"):
                    await runtime.submit_command(scope, command, context)
                assert runtime._store(scope).permits_tool(START_REWARD_REVIEW)
            receipt = await runtime.submit_command(scope, command, context)
            assert receipt.status == "completed"
            assert not runtime._store(scope).permits_tool(START_REWARD_REVIEW)
            assert await runtime.submit_command(scope, command, context) == receipt
            project = await runtime.get_project(scope, context)
            assert project is not None and project.reward_bundle is not None
            snapshot_ref = project.reward_bundle.snapshot
            assert (
                snapshots.path(
                    snapshot_ref.storage_ref + "/task_prompt.md", context
                ).read_text()
                == original_task
            )
            ledger = versions.project(scope, context, create=False)
            assert ledger is not None
            (version,) = version_store.list_versions(ledger.id)
            (run,) = version_store.list_runs(ledger.id)
            assert version.logical_key == "reward_bundle"
            assert version.manifest.storage_ref == snapshot_ref.storage_ref
            assert version.producer_run_id == run.id
            assert run.spec.kind == "rubric_calibration"
            assert "reward_bundle" not in project.plan.model_dump_json()
            assert await runtime.project_reward(scope, context) == project.reward_bundle
            dsl = (
                f'reward = RewardItemBuilderCustomNode(name="rubric", '
                f'entry="{project.reward_bundle.studio_entry}", weight=1.0)\n'
                "config = RewardConfigBuilderNode(reward_item_1=reward.reward_item)\n"
                "train = ModelTrainGRPONode(reward_config=config.reward_config)\n"
            )
            workflow = convert_dsl_to_xyflow(dsl)
            assert await runtime.validate_training_reward(scope, workflow, context) == {
                "reward_bundle": version.id
            }
            wrong_weight = convert_dsl_to_xyflow(
                dsl.replace("weight=1.0", "weight=0.5")
            )
            with pytest.raises(ValueError, match="weight 1"):
                await runtime.submit_versioned_workflow(
                    scope, "bad-training", wrong_weight, None, context
                )
            validation = await execute_validation_tool(
                LocalWorkspace(working_dir=str(candidate)),
                {"dsl": dsl.replace("weight=1.0", "weight=0.5")},
                context,
                project.reward_bundle,
            )
            assert validation["is_error"] and validation["details"]["valid"] is False
            wrong_entry = convert_dsl_to_xyflow(
                dsl.replace(
                    project.reward_bundle.studio_entry, "/workspace/other.py:score"
                )
            )
            with pytest.raises(ValueError, match="Reward Custom Entry"):
                await runtime.validate_training_reward(scope, wrong_entry, context)
            await runtime.submit_versioned_workflow(
                scope, "bound-training", workflow, dsl, context
            )
            training = next(
                run
                for run in version_store.list_runs(ledger.id)
                if run.request_key == "tool:bound-training"
            )
            assert training.spec.inputs["reward_bundle"] == version.id
    finally:
        await runtime.close()
