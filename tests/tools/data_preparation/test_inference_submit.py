import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from harness_adapter.pi_adapter import business_tool_host
from harness_adapter.pi_adapter.business_tool_host import (
    PyromindBusinessToolHost,
    ToolExecutionContext,
    _ToolConversationFacade,
)
from pyromind_agent_server.external_task_registry import WorkflowExternalTaskRegistry
from pyromind_runtime.application.pipeline_runs import PipelineRuns
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.infrastructure.file_product_store import FileProductStore

from openhands.tools.data_preparation import inference as inference_module
from openhands.tools.data_preparation.inference import (
    InferenceOptions,
    InferencePipelineHandler,
    validate_pipeline_topology,
)
from openhands.tools.data_preparation.platform_submit import (
    DfSubmitPipelineAction,
    DfSubmitPipelineExecutor,
)
from openhands.tools.data_preparation.stop_task import (
    DfStopTaskAction,
    DfStopTaskExecutor,
    DfStopTaskObservation,
    DfStopTaskTool,
)
from openhands.tools.workflow.validate_workflow_dsl import (
    ValidateWorkflowDslObservation,
)


@pytest.fixture
def inference_submission(tmp_path, monkeypatch):
    conversation_id = "a" * 32
    workspace = tmp_path / conversation_id
    workspace.mkdir()
    store = FileProductStore(workspace)
    store.create(
        ConversationSnapshot(
            conversation_id=conversation_id,
            capabilities=HarnessCapabilities(),
        ),
        user_id="test",
    )
    registry = WorkflowExternalTaskRegistry(tmp_path)
    runs = PipelineRuns(conversation_id, store, registry)
    config = workspace / "public_data"
    config.mkdir()
    (config / "dataset.json").write_text(
        json.dumps(
            {
                "user_prompt_field": "prompt",
                "reference_field": "gt",
            }
        )
    )
    (config / "evaluation.json").write_text(
        json.dumps(
            {
                "mode": "agent_rubric",
                "rubrics": [
                    {
                        "name": "correct",
                        "criterion": "Matches GT",
                        "weight": 1,
                        "evaluator": {"type": "exact_match"},
                    }
                ],
            }
        )
    )
    context = ToolExecutionContext(
        conversation_id=conversation_id,
        workspace_root=workspace,
        request_context=RequestContext(
            user_id="test", authorization="Bearer test-token"
        ),
        model_configuration={},
    )
    conversation = _ToolConversationFacade(context)
    runtime = Path(__file__).parents[3] / ".agents/skills/inference-evaluation/scripts"
    validator = Mock(
        return_value=ValidateWorkflowDslObservation.from_text(
            text="valid",
            valid=True,
        )
    )
    handler = InferencePipelineHandler(runtime, runs, validator)
    executor = DfSubmitPipelineExecutor(
        env="pre",
        cluster="test",
        inference_handler=handler,
    )
    uploads = []
    monkeypatch.setattr(
        executor,
        "_stage_script",
        lambda path, output, conv, frozen_script_name: uploads.append(
            (frozen_script_name, Path(path).read_text())
        ),
    )
    create = Mock(return_value=object())
    submit = Mock(return_value=SimpleNamespace(task_id="task-1", status="Pending"))
    monkeypatch.setattr(inference_module, "create_workflow_api_client", create)
    monkeypatch.setattr(inference_module, "submit_workflow_task", submit)
    action = DfSubmitPipelineAction(
        input_path="/datasets/cases.jsonl",
        inference=InferenceOptions(
            model_path="/models/v1",
            dataset_config_path="public_data/dataset.json",
            evaluation_config_path="public_data/evaluation.json",
        ),
    )
    return SimpleNamespace(
        executor=executor,
        conversation=conversation,
        action=action,
        runs=runs,
        store=store,
        registry=registry,
        workspace=workspace,
        validator=validator,
        uploads=uploads,
        create=create,
        submit=submit,
    )


def test_fixed_two_node_submission_and_frozen_resume(inference_submission):
    ctx = inference_submission
    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    assert not result.is_error, result.text
    workflow = ctx.submit.call_args.kwargs["workflow"]
    validate_pipeline_topology(workflow)
    assert len(workflow["nodes"]) == 2
    assert "test_mode" not in ctx.submit.call_args.kwargs
    command = workflow["nodes"][1]["data"]["config"]["command"]
    assert '--endpoint "$param"' in command
    assert "pip install" not in command and "API_KEY" not in command
    assert len(ctx.uploads) == 3
    assert {name for name, _ in ctx.uploads} == {
        "evaluate_inference.py",
        "dataset_config.json",
        "evaluation_config.json",
    }
    assert ctx.conversation.signals[0]["type"] == "external_task.submitted"
    assert ctx.conversation.signals[0]["task"]["kind"] == "data_preparation"
    canvas = (ctx.workspace / "public_data/workflow_canvas/workflow.py").read_text()
    assert canvas == ctx.validator.call_args.args[0].dsl
    persisted = (ctx.store.directory / "pipeline-runs.json").read_text()
    assert "test-token" not in persisted
    assert not (ctx.workspace / ".pyromind_data_preparation_tasks").exists()
    ctx.registry.update_status(ctx.conversation.id, "task-1", "failed")
    ctx.submit.return_value.task_id = "task-2"
    resumed = ctx.executor(
        DfSubmitPipelineAction(
            input_path=ctx.action.input_path,
            mode="resume",
            resume_run_id=result.run_id,
        ),
        cast(Any, ctx.conversation),
    )
    assert not resumed.is_error, resumed.text
    assert resumed.output_dir == result.output_dir
    assert resumed.task_id == "task-2"
    assert resumed.execution_revision == 2


def test_platform_contract_failure_does_not_upload_or_submit(inference_submission):
    ctx = inference_submission
    ctx.validator.return_value = ValidateWorkflowDslObservation.from_text(
        text="ENUM_VALUE_INVALID: GPU unavailable",
        valid=False,
    )
    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    assert result.is_error
    assert "GPU unavailable" in result.text
    assert ctx.uploads == []
    ctx.submit.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("script_path", "custom.py"),
        ("model_profile", "text"),
        ("output_schema", "structured"),
        ("convert_format", "messages"),
    ],
)
def test_rejects_cleaning_options_for_inference(inference_submission, field, value):
    ctx = inference_submission
    action = ctx.action.model_copy(update={field: value})
    result = ctx.executor(action, cast(Any, ctx.conversation))
    assert result.is_error
    ctx.validator.assert_not_called()
    ctx.submit.assert_not_called()


def test_topology_rejects_extra_nodes_bad_handles_and_reversed_edges(
    inference_submission,
):
    ctx = inference_submission
    assert not ctx.executor(ctx.action, cast(Any, ctx.conversation)).is_error
    workflow = ctx.submit.call_args.kwargs["workflow"]
    for change in ("extra", "handle", "reversed", "disconnected", "unknown"):
        bad = deepcopy(workflow)
        if change == "extra":
            bad["nodes"].append(deepcopy(bad["nodes"][1]))
        elif change == "handle":
            bad["edges"][0]["sourceHandle"] = "result"
        elif change == "reversed":
            bad["edges"][0].update(source="2", target="1")
        elif change == "disconnected":
            bad["edges"] = []
        else:
            bad["nodes"][0]["data"]["nodeType"] = "ModelEvalApiNode"
        with pytest.raises(ValueError):
            validate_pipeline_topology(bad)


def test_stop_resolves_product_run_without_legacy_task_file(inference_submission):
    ctx = inference_submission
    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    tool = DfStopTaskTool.create(task_resolver=ctx.runs.task_for)[0]
    executor = tool.executor
    assert isinstance(executor, DfStopTaskExecutor)
    executor._post_stop = Mock(return_value={})
    stopped = tool(DfStopTaskAction(run_id=result.run_id), cast(Any, ctx.conversation))
    assert isinstance(stopped, DfStopTaskObservation)
    assert stopped.stopped
    executor._post_stop.assert_called_once_with(
        "task-1",
        {
            "accept": "*/*",
            "content-type": "application/json",
        },
    )


@pytest.mark.asyncio
async def test_pi_host_exposes_inference_through_existing_submit_tool(
    inference_submission,
    monkeypatch,
):
    ctx = inference_submission
    root = Path(__file__).parents[3]
    monkeypatch.setattr(
        business_tool_host,
        "ValidateWorkflowDslExecutor",
        lambda **kwargs: ctx.validator,
    )
    monkeypatch.setattr(
        DfSubmitPipelineExecutor, "_stage_script", lambda *args, **kwargs: None
    )
    host = PyromindBusinessToolHost(
        [],
        skills_directory=root / ".agents/skills",
        pipeline_runs=lambda conversation_id: ctx.runs,
    )
    names = {spec["name"] for spec in host.specs()}
    assert "df_submit_pipeline" in names and "run_workflow" not in names
    result = await host.execute(
        "df_submit_pipeline",
        ctx.action.model_dump(mode="json", exclude_unset=True),
        ToolExecutionContext(
            conversation_id=ctx.conversation.id,
            workspace_root=ctx.workspace,
            request_context=RequestContext(
                user_id="test", authorization="Bearer test-token", x_cluster="test#pre"
            ),
            model_configuration={},
        ),
    )
    assert not result["is_error"], result
    assert result["details"]["task_id"] == "task-1"
    assert result["signals"][0]["task"]["kind"] == "data_preparation"


def test_ambiguous_platform_submission_is_not_repeated(inference_submission):
    ctx = inference_submission
    ctx.submit.side_effect = TimeoutError("response lost")
    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    assert result.is_error and result.run_id
    assert "uncertain" in result.text
    resumed = ctx.executor(
        DfSubmitPipelineAction(
            mode="resume",
            resume_run_id=result.run_id,
            input_path=ctx.action.input_path,
        ),
        cast(Any, ctx.conversation),
    )
    assert resumed.is_error and "uncertain" in resumed.text
    ctx.submit.assert_called_once()
