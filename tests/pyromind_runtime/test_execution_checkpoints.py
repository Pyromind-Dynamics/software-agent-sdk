"""Sample review and multi-capability execution on the existing SQL ledger."""

import hashlib
import json
import subprocess
import sys

import pytest
import sqlalchemy as sa
from pyromind_runtime.application.experiment_execution import ExperimentExecutions
from pyromind_runtime.domain.experiment_execution import RequestExperimentTask
from pyromind_runtime.domain.experiment_plans import PlanOwner
from pyromind_runtime.infrastructure.sql_experiment_execution import (
    SQLExperimentExecutionStore,
    metadata,
)
from pyromind_runtime.infrastructure.sql_experiment_plan_store import (
    SQLExperimentPlanStore,
    metadata as plan_metadata,
)


@pytest.fixture
def sample_execution(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'runs.db'}")
    plan_metadata.create_all(engine)
    metadata.create_all(engine)
    service = ExperimentExecutions(
        SQLExperimentPlanStore(engine), SQLExperimentExecutionStore(engine)
    )
    owner = PlanOwner(user_id="42", conversation_id="sample-chat", profile_id="example")
    request = RequestExperimentTask.model_validate(
        {
            "role_id": "cleaner",
            "request_id": "clean",
            "task": {"name": "Clean", "instruction": "Prepare and review a sample"},
        }
    )
    policy = {
        "submission_tool": "df_submit_pipeline",
        "artifact_contract": "data",
        "require_review": True,
        "sample_limit": 3,
    }
    run = service.request(owner, request, ("cleaner",), execution_policy=policy)
    root = f"public_data/experiment-runs/{run['id']}/public_data"
    source = b'{"text":" a "}\n{"text":" b "}\n{"text":" c "}\n{"text":" d "}\n'
    files = {"/source.jsonl": source}
    script = (
        "import json,sys\nwith open(sys.argv[2],'w') as out:\n"
        " for line in open(sys.argv[1]):\n"
        "  row=json.loads(line);row['text']=row['text'].strip();"
        "out.write(json.dumps(row)+'\\n')\n"
    )
    for name, body in {
        "pipeline.py": script.encode(),
        "sample-in.jsonl": b"\n".join(source.splitlines()[:3]) + b"\n",
    }.items():
        target = tmp_path / root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        files[f"{root}/{name}"] = body
    output = tmp_path / root / "sample-out.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(tmp_path / root / "pipeline.py"),
            str(tmp_path / root / "sample-in.jsonl"),
            str(output),
        ],
        check=True,
    )
    files[f"{root}/sample-out.jsonl"] = output.read_bytes()
    config = {
        "mode": "full",
        "input_path": "/source.jsonl",
        "script_path": f"{root}/pipeline.py",
        "model_profile": "none",
        "output_schema": "text",
    }
    document = {
        "run_id": run["id"],
        "summary": "Three real rows cleaned",
        "sample_count": 3,
        "source_count": 4,
        "retained_count": 3,
        "source_path": "/source.jsonl",
        "sample_input": f"{root}/sample-in.jsonl",
        "sample_output": f"{root}/sample-out.jsonl",
        "script_path": config["script_path"],
        "configuration": config,
        "preview": [{"invented": "ignored"}],
        "validation": {"status": "passed"},
    }

    async def read(_owner, path):
        assert _owner == owner
        return files[path]

    async def inspect(_owner, path, kind):
        assert _owner == owner and kind == "file"
        return {
            "sha256": hashlib.sha256(files[path]).hexdigest(),
            "scope": "complete file",
            "nonempty_lines": sum(
                bool(line.strip()) for line in files[path].splitlines()
            ),
        }

    yield service, owner, run, files, document, read, inspect
    engine.dispose()


def claim(service, run_id, kind="execute_task"):
    item = service.store.queue.claim(kinds=(kind,))
    assert item is not None and item["payload"]["run_id"] == run_id
    return {"run_id": run_id, "work_id": item["id"], "generation": item["generation"]}


async def test_review_then_single_full_submission_and_verified_result(sample_execution):
    service, owner, run, files, document, read, inspect = sample_execution
    scope = claim(service, run["id"])
    calls = []

    async def execute(*args):
        calls.append(args)
        return {
            "is_error": False,
            "content": [],
            "details": {
                "task_id": "studio-1",
                "output_dir": "/outputs/full",
                "run_id": "pipeline-1",
            },
        }

    async def call(tool, arguments):
        return await service.role_call(
            owner, "cleaner", tool, arguments, scope, execute, read, inspect
        )

    with pytest.raises(PermissionError):
        await call("df_submit_pipeline", document["configuration"])
    result = await call("record_execution_checkpoint", document)
    checkpoint = result["checkpoint"]
    assert checkpoint["document"]["preview"][0]["input_records"] == [
        {"text": " a "},
        {"text": " b "},
        {"text": " c "},
    ]
    assert checkpoint["document"]["preview"][0]["output_records"] == [
        {"text": "a"},
        {"text": "b"},
        {"text": "c"},
    ]
    assert checkpoint["document"]["preview"][0]["output_records"] == [
        {"text": "a"},
        {"text": "b"},
        {"text": "c"},
    ]
    assert service.list(owner)[0].state == "waiting_user"
    service.store.continue_task(owner, run["id"], "early", "Discuss this sample")
    assert service.list(owner)[0].checkpoint.status == "pending"
    with pytest.raises(PermissionError, match="sample review is not approved"):
        await call(
            "record_experiment_result",
            {
                "run_id": run["id"],
                "summary": "Attempt to bypass the frozen sample review",
                "artifacts": [{"name": "data", "uri": "/source.jsonl"}],
            },
        )
    with pytest.raises(PermissionError):
        await call("df_submit_pipeline", document["configuration"])
    service.store.resolve_checkpoint(
        owner, run["id"], checkpoint["checkpoint_id"], "answer", True
    )
    service.store.resolve_checkpoint(
        owner, run["id"], checkpoint["checkpoint_id"], "answer", True
    )
    service.store.queue.finish(scope["work_id"], scope["generation"], {})
    work_id = service.store.continue_task(
        owner, run["id"], "full", "Use the approved sample"
    )
    assert (
        service.store.continue_task(owner, run["id"], "full", "Use the approved sample")
        == work_id
    )
    scope = claim(service, run["id"])
    files[document["script_path"]] += b"# changed"
    with pytest.raises(ValueError, match="changed"):
        await call("df_submit_pipeline", document["configuration"])
    files[document["script_path"]] = files[document["script_path"]].removesuffix(
        b"# changed"
    )
    with pytest.raises(PermissionError, match="approved sample"):
        await call("df_submit_pipeline", {**document["configuration"], "cpu": 99})
    await call("df_submit_pipeline", document["configuration"])
    await call("df_submit_pipeline", document["configuration"])
    assert len(calls) == 1 and service.list(owner)[0].state == "waiting_platform"
    service.store.queue.finish(scope["work_id"], scope["generation"], {})
    service.store.ingest("complete", "studio-1", "callback", {"status": "succeeded"})
    service.store.apply_observations()

    async def query(*_):
        return {"status": "succeeded"}

    await service.poll(query, read)
    scope = claim(service, run["id"], "generate_report")
    service.store.complete_agent_turn(
        owner, run["id"], scope, "Report failed", report=True
    )
    service.store.queue.finish(scope["work_id"], scope["generation"], {})
    service.store.queue_artifact_report(owner, run["id"])
    assert service.list(owner)[0].report_state == "failed"
    service.store.enqueue_report(
        owner, run["id"], "Retry artifact inspection", "retry-report"
    )
    scope = claim(service, run["id"], "generate_report")
    with pytest.raises(PermissionError, match="report retry"):
        await call("df_submit_pipeline", document["configuration"])
    report = f"public_data/experiment-runs/{run['id']}/public_data/report.md"
    files[report] = b"Full data verified."
    files["/outputs/full/clean.jsonl"] = files[document["sample_output"]]
    files["/outputs/full/report.json"] = json.dumps(
        {
            "status": "succeeded",
            "pipeline_exit_code": 0,
            "total_records_output": 3,
            "validation": {"status": "passed"},
            "source_integrity": {"unchanged": True},
        }
    ).encode()
    await call(
        "record_experiment_result",
        {
            "run_id": run["id"],
            "summary": "Verified clean output",
            "report_uri": report,
            "artifacts": [
                {"name": "data", "uri": "/outputs/full/clean.jsonl"},
                {"name": "report", "uri": "/outputs/full/report.json"},
            ],
        },
    )
    view = service.list(owner)[0]
    assert (view.state, view.report_state) == ("succeeded", "ready")
    assert {item["name"] for item in view.artifacts} == {
        "data",
        "report",
        "agent_report",
    }
    service.store.queue.finish(scope["work_id"], scope["generation"], {})
    assert service.store.queue.claim(kinds=("generate_report",)) is None
    assert len(calls) == 1
    train_request = RequestExperimentTask.model_validate(
        {
            "role_id": "trainer",
            "request_id": "next-stage",
            "task": {
                "name": "Train",
                "instruction": "Use the clean artifact",
                "artifact_refs": [
                    {
                        "input_name": "input_path",
                        "task_run_id": run["id"],
                        "artifact_name": "data",
                    }
                ],
            },
        }
    )
    next_run = service.request(owner, train_request, ("trainer",))
    assert (
        next_run["task_spec"]["task"]["inputs"]["input_path"]
        == "/outputs/full/clean.jsonl"
    )
    assert (
        next_run["task_spec"]["input_artifacts"]["input_path"]["sha256"]
        == hashlib.sha256(files["/outputs/full/clean.jsonl"]).hexdigest()
    )
    invalid = train_request.model_dump(mode="json")
    invalid["request_id"] = "wrong-artifact"
    invalid["task"]["artifact_refs"][0]["artifact_name"] = "invented"
    with pytest.raises(ValueError, match="unknown artifact"):
        service.request(
            owner, RequestExperimentTask.model_validate(invalid), ("trainer",)
        )


@pytest.mark.parametrize(
    "change", ["scope", "foreign", "bad_count", "validation", "large"]
)
async def test_sample_evidence_rejects_invalid_or_foreign_inputs(
    sample_execution, change
):
    service, owner, run, files, document, read, inspect = sample_execution
    scope = claim(service, run["id"])
    if change == "scope":
        document["sample_input"] = "public_data/other/input.jsonl"
    elif change == "foreign":
        document["run_id"] = "another-run"
    elif change == "bad_count":
        document["sample_count"] = 2
    elif change == "validation":
        document["validation"] = {"status": "failed"}
    else:
        files[document["sample_output"]] = b" " * 65537

    async def execute(*_):
        raise AssertionError("no platform execution during sample verification")

    with pytest.raises((ValueError, PermissionError)):
        await service.role_call(
            owner,
            "cleaner",
            "record_execution_checkpoint",
            document,
            scope,
            execute,
            read,
            inspect,
        )
    assert service.list(owner)[0].checkpoint is None


async def test_revised_sample_invalidates_old_approval_and_owner_isolation(
    sample_execution,
):
    service, owner, run, _, document, read, inspect = sample_execution
    scope = claim(service, run["id"])

    async def execute(*_):
        raise AssertionError("no platform task")

    async def checkpoint():
        return (
            await service.role_call(
                owner,
                "cleaner",
                "record_execution_checkpoint",
                document,
                scope,
                execute,
                read,
                inspect,
            )
        )["checkpoint"]

    first = await checkpoint()
    with pytest.raises(PermissionError):
        service.store.resolve_checkpoint(
            owner.model_copy(update={"profile_id": "other"}),
            run["id"],
            first["checkpoint_id"],
            "foreign",
            True,
        )
    service.store.resolve_checkpoint(
        owner, run["id"], first["checkpoint_id"], "revise", False
    )
    service.store.queue.finish(scope["work_id"], scope["generation"], {})
    service.store.continue_task(
        owner, run["id"], "revision", "Change the cleaning rule"
    )
    scope = claim(service, run["id"])
    document["summary"] = "Revised sample"
    second = await checkpoint()
    assert second["checkpoint_id"] != first["checkpoint_id"]
    with pytest.raises(ValueError, match="sample changed"):
        service.store.resolve_checkpoint(
            owner, run["id"], first["checkpoint_id"], "stale", True
        )
    assert service.list(owner)[0].checkpoint.status == "pending"


@pytest.mark.parametrize(
    "answer", ["confirm_full", "revise", "free_text", "cancelled", "stale"]
)
async def test_manager_ask_user_records_review_without_model_authority(
    sample_execution, tmp_path, answer
):
    import asyncio

    from pyromind_runtime.application.conversation_runtime import ConversationRuntime
    from pyromind_runtime.domain.commands import InputResponseCommand
    from pyromind_runtime.domain.context import RequestContext
    from pyromind_runtime.ports.harness import SessionSpec

    from .fake_adapter import FakeAdapter

    service, owner, run, _, document, read, inspect = sample_execution
    scope = claim(service, run["id"])

    async def execute(*_):
        raise AssertionError("sample does not submit")

    result = await service.role_call(
        owner,
        "cleaner",
        "record_execution_checkpoint",
        document,
        scope,
        execute,
        read,
        inspect,
    )
    adapter = FakeAdapter("pi")
    adapter.capabilities = adapter.capabilities.model_copy(
        update={"user_input_reply": True}
    )
    runtime = ConversationRuntime(
        tmp_path / "product",
        {"pi": adapter},
        default_harness_id="pi",
        experiment_plan_store=service.plans,
        experiment_execution_store=service.store,
        execution_roles=("cleaner",),
    )
    context = RequestContext(user_id=owner.user_id, profile_id=owner.profile_id)
    await runtime.create_conversation(
        SessionSpec(
            conversation_id=owner.conversation_id,
            user_id=owner.user_id,
            profile_id=owner.profile_id,
            workspace_root=str(tmp_path / "product" / owner.conversation_id),
            user_input_enabled=True,
        ),
        context,
    )
    try:
        reference = {
            "run_id": run["id"],
            "checkpoint_id": "old-checkpoint"
            if answer == "stale"
            else result["checkpoint"]["checkpoint_id"],
        }
        adapter.emit(
            owner.conversation_id,
            "input.requested",
            {
                "request_id": "sample-question",
                "operation_id": "ask-sample",
                "checkpoint": reference,
                "questions": [
                    {
                        "id": "sample_review",
                        "prompt": "确认三条清洗结果？",
                        "options": [
                            {"id": "confirm_full", "label": "确认并全量清洗"},
                            {"id": "revise", "label": "修改要求"},
                        ],
                    }
                ],
            },
        )
        for _ in range(100):
            if (
                await runtime.get_snapshot(owner.conversation_id, context)
            ).pending_inputs:
                break
            await asyncio.sleep(0.01)
        pending = (
            await runtime.get_snapshot(owner.conversation_id, context)
        ).pending_inputs[0]
        assert pending.checkpoint is not None and pending.checkpoint.run_id == run["id"]
        command = InputResponseCommand.model_validate(
            {
                "request_id": pending.request_id,
                "command_id": "answer-sample",
                "status": "cancelled" if answer == "cancelled" else "answered",
                "answers": []
                if answer == "cancelled"
                else [
                    {
                        "question_id": "sample_review",
                        **(
                            {"text": "looks good"}
                            if answer == "free_text"
                            else {
                                "option_id": "confirm_full"
                                if answer == "stale"
                                else answer
                            }
                        ),
                    }
                ],
            }
        )
        receipt = await runtime.submit_command(owner.conversation_id, command, context)
        assert receipt.status == "completed"
        if answer == "stale":
            assert service.list(owner)[0].checkpoint.status == "pending"
            assert not (
                await runtime.get_snapshot(owner.conversation_id, context)
            ).pending_inputs
            return
        for _ in range(100):
            if service.list(owner)[0].checkpoint.status != "pending":
                break
            await asyncio.sleep(0.01)
        assert service.list(owner)[0].checkpoint.status == (
            "approved" if answer == "confirm_full" else "revise"
        )
        assert service.list(owner)[0].platform_task_id is None
    finally:
        await runtime.close()


async def test_training_confirmation_uses_existing_permission_path_and_one_task(
    sample_execution, tmp_path
):
    from pyromind_runtime.application.conversation_runtime import ConversationRuntime
    from pyromind_runtime.domain.commands import PermissionResponseCommand
    from pyromind_runtime.domain.context import RequestContext
    from pyromind_runtime.ports.harness import SessionSpec

    from .fake_adapter import FakeAdapter
    from .test_workflow_execution import FakeSubmission

    service, owner, initial, _, _, _, _ = sample_execution
    initial_scope = claim(service, initial["id"])
    service.store.queue.finish(
        initial_scope["work_id"], initial_scope["generation"], {}
    )
    run = service.request(
        owner,
        RequestExperimentTask.model_validate(
            {
                "role_id": "trainer",
                "request_id": "training",
                "task": {"name": "Train", "instruction": "Train using confirmed data"},
            }
        ),
        ("trainer",),
        execution_policy={
            "submission_tool": "submit_workflow",
            "artifact_contract": "model",
        },
    )
    scope = claim(service, run["id"])
    submission = FakeSubmission()
    adapter = FakeAdapter("pi")
    runtime = ConversationRuntime(
        tmp_path / "product",
        {"pi": adapter},
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
    try:
        service.store.stage_workflow(owner, run["id"], scope, "training-request")
        prepared = await runtime.prepare_workflow_execution(
            owner.conversation_id,
            "/workflows/train.py",
            context,
            request_id="training-request",
            task_run_id=run["id"],
        )
        assert (
            prepared["status"] == "awaiting_confirmation" and not submission.submissions
        )
        snapshot = await runtime.get_snapshot(owner.conversation_id, context)
        permission = snapshot.pending_permissions[0].permission_id
        for command_id in ("approve-training", "double-click-training"):
            await runtime.submit_command(
                owner.conversation_id,
                PermissionResponseCommand(
                    command_id=command_id,
                    permission_id=permission,
                    decision="allow_once",
                ),
                context,
            )
        assert len(submission.submissions) == 1
        view = next(item for item in service.list(owner) if item.run_id == run["id"])
        assert (view.state, view.platform_task_id) == ("waiting_platform", "task-123")
        assert (
            len(
                next(
                    row for row in service.store.list(owner) if row["id"] == run["id"]
                )["submissions"]
            )
            == 1
        )
    finally:
        await runtime.close()


@pytest.mark.parametrize("output", ["valid", "foreign_task", "missing_weights"])
async def test_model_artifacts_require_weights_and_platform_provenance(
    sample_execution, output
):
    service, owner, initial, files, _, read, _ = sample_execution
    initial_scope = claim(service, initial["id"])
    service.store.queue.finish(
        initial_scope["work_id"], initial_scope["generation"], {}
    )
    run = service.request(
        owner,
        RequestExperimentTask.model_validate(
            {
                "role_id": "trainer",
                "request_id": "training-result",
                "task": {"name": "Train", "instruction": "Train"},
            }
        ),
        ("trainer",),
        execution_policy={
            "submission_tool": "submit_workflow",
            "artifact_contract": "model",
        },
    )
    scope = claim(service, run["id"])
    service.store.stage_workflow(owner, run["id"], scope, "workflow")
    submission_id = service.store.begin_workflow(
        owner, run["id"], "workflow", "digest", True
    )
    service.store.recover_managed_receipt(
        owner, run["id"], submission_id, {"task_id": "training-task"}
    )
    service.store.queue.finish(scope["work_id"], scope["generation"], {})
    service.store.ingest(
        "training-complete", "training-task", "callback", {"status": "succeeded"}
    )
    service.store.apply_observations()
    service.store.queue_artifact_report(owner, run["id"])
    scope = claim(service, run["id"], "generate_report")
    report = f"public_data/experiment-runs/{run['id']}/public_data/report.md"
    files[report] = b"Training output inspected."

    async def inspect(_owner, path, kind):
        assert _owner == owner and kind == "directory"
        return {
            "sha256": "a" * 64,
            "entries": ["config.json"]
            + ([] if output == "missing_weights" else ["model.safetensors"]),
            "scope": "immediate directory listing",
        }

    async def execute(_owner, role, _scope, tool, arguments):
        assert _owner == owner and role == "trainer" and tool == "training_analysis"
        assert arguments == {"operation": "artifacts", "task_id": "training-task"}
        return {
            "is_error": False,
            "details": {
                "result": {"training_nodes": [{"output_path": "/trained/this-task"}]}
            },
        }

    arguments = {
        "run_id": run["id"],
        "summary": "Verified training output",
        "report_uri": report,
        "artifacts": [
            {
                "name": "model",
                "uri": "/trained/other-task"
                if output == "foreign_task"
                else "/trained/this-task/checkpoint-10",
                "kind": "directory",
            }
        ],
    }
    call = service.role_call(
        owner,
        "trainer",
        "record_experiment_result",
        arguments,
        scope,
        execute,
        read,
        inspect,
    )
    if output == "valid":
        await call
        view = next(item for item in service.list(owner) if item.run_id == run["id"])
        assert (view.state, view.report_state) == ("succeeded", "ready")
    else:
        with pytest.raises(ValueError):
            await call
        assert (
            next(item for item in service.list(owner) if item.run_id == run["id"]).state
            == "waiting_artifacts"
        )


@pytest.mark.parametrize("key", ["input_path", "labels_source"])
async def test_checkpoint_source_must_match_frozen_resource_for_any_binding(
    sample_execution, key
):
    from pyromind_runtime.application.execution_checkpoints import verify_checkpoint

    service, owner, run, files, document, read, inspect = sample_execution
    row = {
        **run,
        "task_spec": {
            "task": {"inputs": {key: "/authorized/"}},
            "execution_policy": {"require_review": True},
        },
    }
    with pytest.raises(ValueError, match="source differs"):
        await verify_checkpoint(
            owner, row, {"work_id": "sample"}, document, read, inspect
        )
    row["task_spec"]["task"]["inputs"][key] = "/source.jsonl"
    assert (
        await verify_checkpoint(
            owner, row, {"work_id": "sample"}, document, read, inspect
        )
    )["status"] == "pending"


async def test_review_freezes_output_assembly_helpers(sample_execution):
    from pyromind_runtime.application.execution_checkpoints import (
        require_approved_checkpoint,
        verify_checkpoint,
    )

    service, owner, run, files, document, read, inspect = sample_execution
    helper = document["script_path"].replace("pipeline.py", "assemble.py")
    files[helper] = b"# assemble the final records\n"
    document["implementation_files"] = [helper]
    row = {
        **run,
        "task_spec": {
            "task": {"inputs": {}},
            "execution_policy": {"require_review": True},
        },
    }
    checkpoint = await verify_checkpoint(
        owner, row, {"work_id": "sample"}, document, read, inspect
    )
    assert checkpoint["evidence"][helper] == hashlib.sha256(files[helper]).hexdigest()
    row["attributes"] = {"checkpoint": {**checkpoint, "status": "approved"}}
    await require_approved_checkpoint(
        owner, row, document["configuration"], read, inspect
    )
    files[helper] = b"# changed assembly\n"
    with pytest.raises(ValueError, match="implementation changed"):
        await require_approved_checkpoint(
            owner, row, document["configuration"], read, inspect
        )
    document["implementation_files"] = ["public_data/another-task/assemble.py"]
    with pytest.raises(PermissionError, match="implementation files"):
        await verify_checkpoint(
            owner, row, {"work_id": "sample"}, document, read, inspect
        )
