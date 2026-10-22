import importlib.util
import json
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock
from uuid import UUID

import httpx
import pytest
from harness_adapter.pi_adapter import business_tool_host
from harness_adapter.pi_adapter.business_tool_host import (
    PyromindBusinessToolHost,
    ToolExecutionContext,
    _ToolConversationFacade,
)
from pyromind_agent_server.external_task_registry import WorkflowExternalTaskRegistry
from pyromind_runtime.application.pipeline_runs import PipelineRuns
from pyromind_runtime.application.versions import Versions, snapshot_result
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.infrastructure.file_artifact_snapshots import (
    FileArtifactSnapshots,
)
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.infrastructure.sqlite_version_store import SQLiteVersionStore

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
from openhands.tools.workflow.dsl_to_xyflow import convert_dsl_to_xyflow
from openhands.tools.workflow.validate_workflow_dsl import (
    ValidateWorkflowDslExecutor,
    ValidateWorkflowDslObservation,
    WorkflowValidationIssue,
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
        handler=handler,
    )


@pytest.fixture
def configure_model_source(inference_submission):
    ctx = inference_submission

    def configure(source):
        path = ctx.workspace / "public_data/evaluation.json"
        config = json.loads(path.read_text())
        config["model_source"] = source
        path.write_text(json.dumps(config))
        if source.get("type") != "storage":
            ctx.action.inference.model_path = "Unlisted/Example-Model"

    return configure


@pytest.mark.parametrize("operation", ["profile", "review", "report"])
def test_products_submit_cpu_only_without_model_or_scoring(
    operation, inference_submission
):
    ctx = inference_submission
    manifest = {
        "datasets": {
            "train": {"path": "/train", "sha256": "a" * 64},
            "validation": {"path": "/val", "sha256": "b" * 64},
        },
        "source": {"kind": "import", "path": "/badcases", "sha256": "c" * 64},
        "profile": {"path": "/profile", "sha256": "d" * 64},
    }
    (ctx.workspace / "public_data/products.json").write_text(json.dumps(manifest))
    (ctx.workspace / "public_data/extract.py").write_text(
        "def parse_record(raw, context): return raw\n"
    )
    action = DfSubmitPipelineAction.model_validate(
        {
            "input_path": "/badcases",
            "products": {
                "operation": operation,
                "manifest_path": "public_data/products.json",
                **(
                    {"extraction_script_path": "public_data/extract.py"}
                    if operation == "profile"
                    else {}
                ),
            },
        }
    )
    response = ctx.executor(action, ctx.conversation)
    assert not response.is_error, response.text
    workflow = ctx.submit.call_args.kwargs["workflow"]
    validate_pipeline_topology(workflow, require_inference=False)
    assert "VLLMInference" not in json.dumps(workflow)
    uploads = dict(ctx.uploads)
    assert "evaluation_products.py" in uploads
    assert "products_definition.json" in uploads
    assert "score_eval.py" not in uploads
    run = ctx.runs.resolve(response.run_id)
    assert run["definition"]["products_operation"] == operation


@pytest.mark.parametrize("with_test", [False, True])
def test_final_test_requires_selected_checkpoint_and_independent_data(
    inference_submission,
    configure_model_source,
    with_test,
):
    from uuid import uuid4

    from pyromind_runtime.domain.pipeline import PipelineRun

    ctx = inference_submission
    selection_id = str(uuid4())
    datasets = {
        "train": {"path": "/train", "sha256": "a" * 64},
        "validation": {"path": "/val", "sha256": "b" * 64},
    }
    if with_test:
        datasets["test"] = {"path": "/datasets/cases.jsonl", "sha256": "c" * 64}
    ctx.store.save_pipeline_run(
        PipelineRun(
            run_id=UUID(selection_id),
            conversation_id=str(ctx.conversation.id),
            definition=cast(
                JsonObject,
                {
                    "kind": "prediction_evaluation",
                    "source_manifest": {"datasets": datasets},
                    "scoring_script": (
                        "def score_case(case, spec): return {'passed': True}"
                    ),
                    "scoring_spec": {"business_rule": "frozen business rule"},
                },
            ),
            result={
                "selected_step": 100,
                "model": {
                    "path": "/models/checkpoint-100",
                    "kind": "lora",
                    "base_model": "/base",
                },
            },
            fingerprint="selected",
            output_dir="/selection",
        ),
        expected_revision=None,
    )
    configure_model_source(
        {"type": "checkpoint", "kind": "lora", "base_model": "/base"}
    )
    ctx.action.inference.model_path = "/models/checkpoint-100"
    ctx.action.inference.selection_run_id = selection_id
    result = ctx.executor(ctx.action, ctx.conversation)
    if not with_test:
        assert result.is_error
        assert "no test run" in result.text
        ctx.submit.assert_not_called()
        return
    assert not result.is_error, result.text
    definition = ctx.runs.resolve(result.run_id)["definition"]
    assert definition["dataset_role"] == "test"
    assert (
        definition["business_scoring"]["spec"]["business_rule"]
        == "frozen business rule"
    )
    assert "business_scoring.json" in dict(ctx.uploads)
    assert "checkpoint-100" in json.dumps(ctx.submit.call_args.kwargs["workflow"])
    # A caller cannot sneak a different checkpoint into the final-test path.
    ctx.action.inference.model_path = "/models/checkpoint-200"
    ctx.submit.reset_mock()
    rejected = ctx.executor(ctx.action, ctx.conversation)
    assert rejected.is_error and "validation-selected checkpoint" in rejected.text
    ctx.submit.assert_not_called()


@pytest.mark.parametrize(
    "parameters,invalid_fields",
    [
        (
            {"normalized": True, "image_size": [1000, 1000]},
            ["normalized", "image_size"],
        ),
        ({"min": 0}, ["min"]),
        ({"case_sensitive": True}, ["case_sensitive"]),
    ],
)
def test_invalid_evaluator_parameters_give_actionable_config_error(
    inference_submission, parameters, invalid_fields
):
    ctx = inference_submission
    path = ctx.workspace / "public_data/evaluation.json"
    config = json.loads(path.read_text())
    rubric = config["rubrics"][0]
    rubric["name"] = "boxes_iou"
    rubric["evaluator"] = {"type": "bbox_iou", **parameters}
    path.write_text(json.dumps(config))

    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))

    assert result.is_error
    assert result.failure_stage == "config_validation"
    assert result.retryable is False
    assert "public_data/evaluation.json" in result.text
    assert "rubrics.0.evaluator" in result.text
    assert "rubric='boxes_iou'" in result.text
    assert all(field in result.text for field in invalid_fields)
    assert "bbox_iou accepts: iou_threshold, prediction_path, reference_path, type" in (
        result.text
    )
    assert "errors.pydantic.dev" not in result.text
    assert ctx.store.load_pipeline_runs() == {}
    assert ctx.uploads == []
    ctx.validator.assert_not_called()
    ctx.create.assert_not_called()
    ctx.submit.assert_not_called()


def test_malformed_evaluation_json_identifies_file_and_location(inference_submission):
    ctx = inference_submission
    (ctx.workspace / "public_data/evaluation.json").write_text('{\n"mode": }')

    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))

    assert result.is_error
    assert result.failure_stage == "config_validation"
    assert result.retryable is False
    assert "public_data/evaluation.json" in result.text
    assert "JSON line 2, column 9" in result.text
    assert ctx.store.load_pipeline_runs() == {}
    assert ctx.uploads == []
    ctx.validator.assert_not_called()
    ctx.submit.assert_not_called()


def test_bbox_evaluation_preserves_frozen_default_fields_on_resume(
    inference_submission,
):
    ctx = inference_submission
    path = ctx.workspace / "public_data/evaluation.json"
    config = json.loads(path.read_text())
    config["rubrics"][0]["evaluator"] = {
        "type": "bbox_iou",
        "prediction_path": "boxes",
        "reference_path": "boxes",
        "iou_threshold": 0.7,
    }
    path.write_text(json.dumps(config))
    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    assert not result.is_error, result.text
    frozen = ctx.runs.resolve(result.run_id)
    evaluator = frozen["definition"]["evaluation_config"]["rubrics"][0]["evaluator"]
    assert evaluator["case_sensitive"] is False
    assert evaluator["prediction_paths"] == []
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
    assert resumed.task_id == "task-2"
    assert ctx.runs.resolve(result.run_id)["definition"] == frozen["definition"]
    assert ctx.runs.resolve(result.run_id)["fingerprint"] == frozen["fingerprint"]


def test_processing_cpu_topology_is_not_accepted_as_inference_evaluation():
    workflow = convert_dsl_to_xyflow(
        "processing = CustomCommandCPUNode(id=1, command='echo ok', cpu=2, memory=8)",
        name="processing",
    )
    validate_pipeline_topology(workflow)
    with pytest.raises(ValueError, match="requires inference"):
        validate_pipeline_topology(workflow, require_inference=True)


@pytest.mark.parametrize("source_type", ["storage", "clone", "download"])
def test_versioned_canvas_matches_submitted_workflow(
    inference_submission, configure_model_source, tmp_path, source_type
):
    ctx = inference_submission
    configure_model_source({"type": source_type})
    context = RequestContext(user_id="test")
    snapshots = FileArtifactSnapshots(tmp_path / "storage")
    for reference in ("/datasets/cases.jsonl", "/models/v1"):
        path = snapshots.path(reference, context)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source")
    versions = Versions(SQLiteVersionStore(tmp_path / "versions.sqlite3"), snapshots)
    ctx.handler.runs = PipelineRuns(
        ctx.conversation.id, ctx.store, ctx.registry, versions, context
    )
    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    assert not result.is_error, result.text
    workflow = ctx.submit.call_args.kwargs["workflow"]
    canvas = (ctx.workspace / "public_data/workflow_canvas/workflow.py").read_text()
    assert canvas == ctx.validator.call_args.args[0].dsl
    assert convert_dsl_to_xyflow(canvas, name=workflow["name"]) == workflow
    assert "/versions/" in canvas
    assert ctx.validator.call_count == 2
    project = versions.store.bound_project(ctx.conversation.id)
    assert project is not None
    record = versions.store.list_runs(project.id)[0]
    code = versions.store.get_version(project.id, record.spec.inputs["code"])
    assert snapshots.read_text(snapshot_result(code), context) == canvas


@pytest.mark.parametrize(
    "source,preparation_type",
    [
        ({"type": "storage"}, None),
        ({"type": "clone"}, "CloneAndCacheModel"),
        ({"type": "download"}, "DownloadAndCacheModel"),
        (
            {"type": "download", "download_source": "modelscope"},
            "DownloadAndCacheModel",
        ),
    ],
)
def test_model_sources_generate_bound_workflows_and_frozen_resume(
    inference_submission, configure_model_source, source, preparation_type
):
    ctx = inference_submission
    configure_model_source(source)
    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    assert not result.is_error, result.text
    workflow = ctx.submit.call_args.kwargs["workflow"]
    validate_pipeline_topology(workflow, require_inference=True)
    canvas = (ctx.workspace / "public_data/workflow_canvas/workflow.py").read_text()
    assert canvas == ctx.validator.call_args.args[0].dsl
    assert convert_dsl_to_xyflow(canvas, name=workflow["name"]) == workflow
    nodes = {node["data"]["nodeType"]: node for node in workflow["nodes"]}
    assert len(nodes) == (3 if preparation_type else 2)
    if preparation_type:
        preparation = nodes[preparation_type]
        inference = nodes["VLLMInference"]
        assert any(
            str(edge["source"]) == str(preparation["id"])
            and str(edge["target"]) == str(inference["id"])
            and edge["sourceHandle"] == edge["targetHandle"] == "model_path"
            for edge in workflow["edges"]
        )
        config = preparation["data"]["config"]
        if source["type"] == "clone":
            assert config["model"] == "Unlisted/Example-Model"
            assert config["target_path"] == "/workspace/models/"
        else:
            assert config["modelname"] == "Unlisted/Example-Model"
            assert config["cache_dir"] == "/workspace/models/Unlisted/Example-Model"
            assert config["download_source"] == source.get(
                "download_source", "huggingface"
            )
    frozen = json.loads(dict(ctx.uploads)["evaluation_config.json"])
    assert frozen["model_source"]["type"] == source["type"]
    prior = ctx.runs.resolve(result.run_id)
    configure_model_source({"type": "download", "download_source": "huggingface"})
    ctx.registry.update_status(ctx.conversation.id, "task-1", "failed")
    ctx.submit.return_value.task_id = "task-2"
    resumed = ctx.executor(
        DfSubmitPipelineAction(
            input_path=ctx.action.input_path, mode="resume", resume_run_id=result.run_id
        ),
        cast(Any, ctx.conversation),
    )
    assert not resumed.is_error, resumed.text
    assert ctx.submit.call_args.kwargs["workflow"] == workflow
    assert ctx.runs.resolve(result.run_id)["definition"] == prior["definition"]
    assert ctx.runs.resolve(result.run_id)["fingerprint"] == prior["fingerprint"]


@pytest.mark.parametrize(
    "source",
    [
        {"type": "unknown"},
        {"type": "download", "download_source": "unknown"},
        {"type": "clone", "download_source": "modelscope"},
    ],
)
def test_invalid_model_source_does_not_stage_or_submit(
    inference_submission, configure_model_source, source
):
    ctx = inference_submission
    configure_model_source(source)
    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    assert result.is_error and "model_source" in result.text
    assert ctx.uploads == []
    ctx.validator.assert_not_called()
    ctx.submit.assert_not_called()


@pytest.mark.parametrize(
    "source,reference",
    [
        ({"type": "storage"}, "Qwen/Qwen3-VL-2B-Instruct"),
        ({"type": "clone"}, "../model"),
        ({"type": "download"}, "/models/model"),
        ({"type": "download"}, " "),
    ],
)
def test_model_reference_error_identifies_field_and_source(
    inference_submission, configure_model_source, source, reference
):
    ctx = inference_submission
    configure_model_source(source)
    ctx.action.inference.model_path = reference
    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    assert result.is_error
    assert "inference.model_path" in result.text
    assert f"model_source.type={source['type']}" in result.text
    ctx.submit.assert_not_called()


@pytest.mark.parametrize("valid", [False, True])
def test_clone_validation_preserves_node_error_and_allows_corrected_identifier(
    inference_submission, configure_model_source, valid, caplog
):
    ctx = inference_submission
    configure_model_source({"type": "clone"})
    ctx.action.inference.model_path = "Qwen3-VL-4B-Instruct"
    issue = WorkflowValidationIssue(
        node_id="3",
        node_type="CloneAndCacheModel",
        field="model",
        code="ENUM_VALUE_INVALID",
        detail={"allowed_values": ["Qwen/Qwen3-VL-4B-Instruct"]},
    )
    ctx.validator.return_value = ValidateWorkflowDslObservation.from_text(
        text="ENUM_VALUE_INVALID: model is not available in this catalog",
        valid=valid,
        errors=[issue],
        failure_stage="platform_schema",
    )
    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    assert result.is_error
    assert "ENUM_VALUE_INVALID" in result.text
    assert "model_source.type=download" not in result.text
    assert result.validation_errors == [issue]
    assert result.failure_stage == "platform_schema"
    assert result.retryable is False
    llm_validation = json.loads(result.to_llm_content[-1].model_dump()["text"])
    assert llm_validation["validation_errors"] == [
        issue.model_dump(mode="json", exclude_none=True)
    ]
    assert llm_validation["retryable"] is False
    assert "dsl_sha256=" in caplog.text
    assert "test-token" not in caplog.text
    assert "Qwen3-VL-4B-Instruct" not in caplog.text
    ctx.validator.assert_called_once()
    assert ctx.uploads == []
    ctx.create.assert_not_called()
    ctx.submit.assert_not_called()

    ctx.action.inference.model_path = "Qwen/Qwen3-VL-4B-Instruct"
    warning = WorkflowValidationIssue(level="warning", message="Non-blocking warning")
    ctx.validator.return_value = ValidateWorkflowDslObservation.from_text(
        text="valid", valid=True, warnings=[warning]
    )
    corrected = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    assert not corrected.is_error, corrected.text
    assert corrected.validation_warnings == [warning]
    llm_validation = json.loads(corrected.to_llm_content[-1].model_dump()["text"])
    assert llm_validation["validation_warnings"] == [
        warning.model_dump(mode="json", exclude_none=True)
    ]
    workflow = ctx.submit.call_args.kwargs["workflow"]
    model = next(
        node
        for node in workflow["nodes"]
        if node["data"]["nodeType"] == "CloneAndCacheModel"
    )
    assert model["data"]["config"]["model"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert ctx.runs.resolve(corrected.run_id)["definition"]["evaluation_config"][
        "model_source"
    ] == {"type": "clone"}


@pytest.mark.parametrize("failure,retryable", [("timeout", True), ("auth", False)])
def test_validation_transport_failure_does_not_upload_or_submit(
    inference_submission, monkeypatch, failure, retryable
):
    ctx = inference_submission
    ctx.handler.validator = ValidateWorkflowDslExecutor(env="pre")

    def post(url, **kwargs):
        if failure == "timeout":
            raise httpx.ReadTimeout("validation timed out")
        return httpx.Response(401, text="authentication required")

    monkeypatch.setattr(httpx, "post", post)
    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    assert result.is_error
    assert result.failure_stage == "transport"
    assert result.retryable is retryable
    llm_validation = json.loads(result.to_llm_content[-1].model_dump()["text"])
    assert llm_validation["failure_stage"] == "transport"
    assert llm_validation["retryable"] is retryable
    assert ctx.uploads == []
    ctx.create.assert_not_called()
    ctx.submit.assert_not_called()


def test_final_dsl_validation_failure_does_not_submit(
    inference_submission, monkeypatch
):
    ctx = inference_submission

    def prepare_execution(run_id, workflow, dsl, script):
        final_dsl = dsl.replace("/workspace/models/v1", "/workspace/versions/model")
        return convert_dsl_to_xyflow(final_dsl, name=workflow["name"]), final_dsl

    monkeypatch.setattr(ctx.runs, "prepare_execution", prepare_execution)
    ctx.validator.side_effect = [
        ValidateWorkflowDslObservation.from_text(text="valid", valid=True),
        ValidateWorkflowDslObservation.from_text(
            text="final DSL rejected", valid=False, failure_stage="platform_schema"
        ),
    ]
    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    assert result.is_error and result.failure_stage == "platform_schema"
    assert "/workspace/versions/model" in ctx.validator.call_args.args[0].dsl
    assert ctx.validator.call_count == 2
    assert ctx.uploads == []
    ctx.create.assert_not_called()
    ctx.submit.assert_not_called()


@pytest.mark.parametrize("change", ["source", "model", "rubric"])
def test_resume_rejects_changed_model_source_reference_or_rubric(
    inference_submission, configure_model_source, change
):
    ctx = inference_submission
    configure_model_source({"type": "clone"})
    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    assert not result.is_error, result.text
    ctx.registry.update_status(ctx.conversation.id, "task-1", "failed")
    if change == "source":
        configure_model_source({"type": "download"})
    elif change == "model":
        ctx.action.inference.model_path = "Another/Model"
    else:
        path = ctx.workspace / "public_data/evaluation.json"
        config = json.loads(path.read_text())
        config["rubrics"][0]["criterion"] = "A different criterion"
        path.write_text(json.dumps(config))
    resumed = ctx.executor(
        ctx.action.model_copy(
            update={"mode": "resume", "resume_run_id": result.run_id}
        ),
        cast(Any, ctx.conversation),
    )
    assert resumed.is_error and "configuration changed" in resumed.text
    ctx.submit.assert_called_once()


def test_fixed_two_node_submission_and_frozen_resume(inference_submission):
    ctx = inference_submission
    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    assert not result.is_error, result.text
    workflow = ctx.submit.call_args.kwargs["workflow"]
    validate_pipeline_topology(workflow, require_inference=True)
    assert len(workflow["nodes"]) == 2
    assert "test_mode" not in ctx.submit.call_args.kwargs
    command = workflow["nodes"][1]["data"]["config"]["command"]
    assert '--endpoint "$param"' in command
    assert "--run-id" in command and result.run_id in command
    assert f"--report-assets-dir /workspace{result.output_dir}" in command
    assert "pip install" not in command and "API_KEY" not in command
    assert len(ctx.uploads) == 5
    assert {name for name, _ in ctx.uploads} == {
        "evaluate_inference.py",
        "evaluation_report.css",
        "evaluation_report.js",
        "dataset_config.json",
        "evaluation_config.json",
    }
    assert ctx.conversation.signals[0]["type"] == "external_task.submitted"
    assert ctx.conversation.signals[0]["task"]["kind"] == "data_preparation"
    canvas = (ctx.workspace / "public_data/workflow_canvas/workflow.py").read_text()
    assert canvas == ctx.validator.call_args.args[0].dsl
    persisted = (ctx.store.directory / "pipeline-runs.json").read_text()
    prior = ctx.runs.resolve(result.run_id)
    assert "model_source" not in prior["definition"]["evaluation_config"]
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
    assert ctx.runs.resolve(result.run_id)["definition"] == prior["definition"]
    assert ctx.runs.resolve(result.run_id)["fingerprint"] == prior["fingerprint"]


@pytest.mark.parametrize(
    "source", [{"type": "storage"}, {"type": "clone"}, {"type": "download"}]
)
def test_submission_reads_configs_from_sandbox_workspace(
    inference_submission, sandbox_workspace, configure_model_source, source
) -> None:
    """A sandbox conversation stages the evaluation configs off the remote FS."""
    ctx = inference_submission
    configure_model_source(source)
    public = sandbox_workspace.workspace_dir / "public_data"
    public.mkdir(parents=True)
    for name in ("dataset.json", "evaluation.json"):
        (public / name).write_text((ctx.workspace / "public_data" / name).read_text())
    conversation = _ToolConversationFacade(
        ToolExecutionContext(
            conversation_id=ctx.conversation.id,
            workspace_root=sandbox_workspace.workspace_dir,
            request_context=RequestContext(
                user_id="test", authorization="Bearer test-token"
            ),
            model_configuration={},
            workspace=sandbox_workspace,
        )
    )

    result = ctx.executor(ctx.action, cast(Any, conversation))

    assert not result.is_error, result.text
    staged = {name for name, _ in ctx.uploads}
    assert {"dataset_config.json", "evaluation_config.json"} <= staged


@pytest.mark.parametrize(
    "source", [{"type": "storage"}, {"type": "clone"}, {"type": "download"}]
)
def test_evaluation_canvas_reaches_mount_path_sandbox(
    inference_submission, mount_path_sandbox_workspace, configure_model_source, source
) -> None:
    """The canvas is published through the workspace port, not the mount path."""
    ctx = inference_submission
    configure_model_source(source)
    sandbox = mount_path_sandbox_workspace
    public = sandbox.host_workspace_dir / "public_data"
    public.mkdir(parents=True)
    for name in ("dataset.json", "evaluation.json"):
        (public / name).write_text((ctx.workspace / "public_data" / name).read_text())
    conversation = _ToolConversationFacade(
        ToolExecutionContext(
            conversation_id=ctx.conversation.id,
            workspace_root=sandbox.host_workspace_dir,
            request_context=RequestContext(
                user_id="test", authorization="Bearer test-token"
            ),
            model_configuration={},
            workspace=cast(Any, sandbox),
        )
    )

    result = ctx.executor(ctx.action, cast(Any, conversation))

    assert not result.is_error, result.text
    canvas = sandbox.host_workspace_dir / "public_data/workflow_canvas/workflow.py"
    assert canvas.read_text() == ctx.validator.call_args.args[0].dsl


def test_stale_report_ui_bundle_is_rejected(tmp_path):
    runtime = Path(__file__).parents[3] / ".agents/skills/inference-evaluation/scripts"
    copied_runtime = tmp_path / "runtime"
    shutil.copytree(runtime, copied_runtime)
    index = copied_runtime / "report_ui/index.js"
    index.write_text(index.read_text() + "\n// stale\n")

    with pytest.raises(ValueError, match="report UI bundle is stale"):
        inference_module._validate_report_ui_bundle(copied_runtime)


def _load_skill_script(name: str) -> Any:
    scripts = Path(__file__).parents[3] / ".agents/skills/inference-evaluation/scripts"
    spec = importlib.util.spec_from_file_location(
        name.removesuffix(".py"), scripts / name
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_ui_build_recipe_matches_consumers():
    build = _load_skill_script("build_report_ui.py")
    evaluation = _load_skill_script("evaluate_inference.py")

    assert build.SOURCE.name == "report_ui"
    inference_module._validate_report_ui_bundle(build.ROOT)
    assert set(inference_module.INFERENCE_RUNTIME_FILENAMES) == {
        "evaluate_inference.py",
        build.JAVASCRIPT_OUTPUT.name,
        build.CSS_OUTPUT.name,
    }
    assert set(evaluation.REPORT_ASSET_NAMES) == {
        build.JAVASCRIPT_OUTPUT.name,
        build.CSS_OUTPUT.name,
    }


def test_runtime_hash_matches_script_side_bundle_hash():
    scripts = Path(__file__).parents[3] / ".agents/skills/inference-evaluation/scripts"
    evaluation = _load_skill_script("evaluate_inference.py")

    assert evaluation._runtime_sha256() == inference_module.runtime_bundle_fingerprint(
        scripts, inference_module.INFERENCE_RUNTIME_FILENAMES
    )


def test_invalid_report_spec_json_names_the_file(inference_submission):
    ctx = inference_submission
    (ctx.workspace / "public_data" / "report_spec.json").write_text("{oops")
    ctx.action.inference.report_spec_path = "public_data/report_spec.json"

    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))

    assert result.is_error
    assert "invalid report_spec JSON in public_data/report_spec.json" in result.text
    assert ctx.uploads == []
    ctx.submit.assert_not_called()


def test_report_spec_is_staged_and_passed_to_command(inference_submission):
    ctx = inference_submission
    (ctx.workspace / "public_data" / "report_spec.json").write_text(
        json.dumps({"template_id": "evaluation-basic-v1"})
    )
    ctx.action.inference.report_spec_path = "public_data/report_spec.json"
    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    assert not result.is_error, result.text
    workflow = ctx.submit.call_args.kwargs["workflow"]
    command = workflow["nodes"][1]["data"]["config"]["command"]
    assert "--report-spec" in command
    assert len(ctx.uploads) == 6
    assert {name for name, _ in ctx.uploads} == {
        "evaluate_inference.py",
        "evaluation_report.css",
        "evaluation_report.js",
        "dataset_config.json",
        "evaluation_config.json",
        "report_spec.json",
    }


def test_report_spec_is_frozen_into_definition_for_resume(inference_submission):
    ctx = inference_submission
    spec_file = ctx.workspace / "public_data" / "report_spec.json"
    spec_file.write_text(json.dumps({"template_id": "evaluation-basic-v1"}))
    ctx.action.inference.report_spec_path = "public_data/report_spec.json"
    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    assert not result.is_error, result.text
    staged = dict(ctx.uploads)["report_spec.json"]
    assert json.loads(staged)["template_id"] == "evaluation-basic-v1"
    # Mutate the workspace copy after submission: resume must reuse the frozen
    # spec content from the definition, not re-read the (now different) file.
    spec_file.write_text(json.dumps({"template_id": "evaluation-media-v1"}))
    ctx.uploads.clear()
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
    assert dict(ctx.uploads)["report_spec.json"] == staged
    command = ctx.submit.call_args.kwargs["workflow"]["nodes"][1]["data"]["config"][
        "command"
    ]
    assert "--report-spec" in command


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


@pytest.mark.parametrize("source_type", ["storage", "clone", "download"])
def test_topology_rejects_extra_nodes_bad_handles_and_reversed_edges(
    inference_submission, configure_model_source, source_type
):
    ctx = inference_submission
    configure_model_source({"type": source_type})
    assert not ctx.executor(ctx.action, cast(Any, ctx.conversation)).is_error
    workflow = ctx.submit.call_args.kwargs["workflow"]
    for change in (
        "extra",
        "handle",
        "reversed",
        "disconnected",
        "unknown",
        "duplicate_id",
    ):
        bad = deepcopy(workflow)
        if change == "extra":
            bad["nodes"].append(deepcopy(bad["nodes"][1]))
        elif change == "handle":
            bad["edges"][0]["sourceHandle"] = "result"
        elif change == "reversed":
            bad["edges"][0].update(source="2", target="1")
        elif change == "disconnected":
            bad["edges"] = []
        elif change == "unknown":
            bad["nodes"][0]["data"]["nodeType"] = "ModelEvalApiNode"
        else:
            bad["nodes"][0]["id"] = bad["nodes"][1]["id"]
        with pytest.raises(ValueError):
            validate_pipeline_topology(bad, require_inference=True)


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
    assert "df_submit_pipeline" in names
    assert "df_save_annotations" not in names
    assert "run_workflow" not in names
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


def test_saved_prediction_submission_is_cpu_only_and_freezes_scoring(
    inference_submission,
):
    from openhands.tools.data_preparation.prediction_evaluation import (
        PredictionEvaluationOptions,
    )

    ctx = inference_submission
    public = ctx.workspace / "public_data"
    manifest: dict[str, Any] = {
        "evaluations": [
            {"step": 100, "path": "/train/eval_step_100.jsonl", "sha256": "a" * 64}
        ],
        "checkpoints": [{"step": 100, "path": "/train/checkpoint-100"}],
    }
    manifest["generation"] = {"max_tokens": 128, "temperature": 0}
    spec = {
        "business_rule": "confirmed rule",
        "validation_examples": [
            {"record": {}, "passed": True},
            {"record": {}, "passed": False},
        ],
    }
    (public / "source.json").write_text(json.dumps(manifest))
    (public / "spec.json").write_text(json.dumps(spec))
    script = (
        "def parse_record(raw, context): return raw\n"
        'def score_case(case, spec): return {"passed": True}\n'
    )
    (public / "score.py").write_text(script)
    action = DfSubmitPipelineAction(
        input_path="/train",
        evaluation=PredictionEvaluationOptions(
            source_manifest_path="public_data/source.json",
            scoring_spec_path="public_data/spec.json",
            scoring_script_path="public_data/score.py",
        ),
    )
    result = ctx.executor(action, cast(Any, ctx.conversation))
    assert not result.is_error, result.text
    workflow = ctx.submit.call_args.kwargs["workflow"]
    assert [node["data"]["nodeType"] for node in workflow["nodes"]] == [
        "CustomCommandCPUNode"
    ]
    assert workflow["edges"] == []
    frozen = ctx.runs.resolve(result.run_id)
    assert frozen["definition"]["scoring_script"] == script
    assert frozen["definition"]["source_manifest"] == manifest
    assert {"score_eval.py", "evaluate_predictions.py", "training_artifacts.py"} <= {
        name for name, _ in ctx.uploads
    }
    assert len(ctx.store.load_pipeline_runs()) == 1
    # A changed script cannot silently mutate a reserved scoring run.
    (public / "score.py").write_text(script + "# changed rule\n")
    changed = action.model_copy(
        update={"mode": "resume", "resume_run_id": result.run_id}
    )
    failed = ctx.executor(changed, cast(Any, ctx.conversation))
    assert failed.is_error
    assert ctx.submit.call_count == 1


@pytest.mark.parametrize("kind", ["full", "lora"])
def test_selected_checkpoint_preparation_binds_exact_step(inference_submission, kind):
    ctx = inference_submission
    path = ctx.workspace / "public_data/evaluation.json"
    config = json.loads(path.read_text())
    config["model_source"] = {"type": "checkpoint", "kind": kind}
    if kind == "lora":
        config["model_source"]["base_model"] = "/models/base"
    path.write_text(json.dumps(config))
    ctx.action.inference.model_path = "/train/checkpoint-100"
    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    assert not result.is_error, result.text
    workflow = ctx.submit.call_args.kwargs["workflow"]
    by_type = {n["data"]["nodeType"]: n for n in workflow["nodes"]}
    if kind == "lora":
        merged = by_type["ModelMergeLoraNode"]["data"]["config"]
        assert merged["lora_path"] == "/workspace/train/checkpoint-100"
        assert merged["model_path"] == "/workspace/models/base"
        assert merged["output_path"].endswith("/merged-model")
        assert any(e["sourceHandle"] == "merged_model_path" for e in workflow["edges"])
    else:
        assert (
            by_type["VLLMInference"]["data"]["config"]["model_path"]
            == "/workspace/train/checkpoint-100"
        )
        assert "ModelMergeLoraNode" not in by_type


def test_checkpoint_versioning_never_rebinds_to_default_model(
    inference_submission, tmp_path
):
    from pyromind_runtime.domain.versions import SaveVersion, SnapshotSource

    ctx = inference_submission
    context = RequestContext(user_id="test")
    snapshots = FileArtifactSnapshots(tmp_path / "storage")
    for reference in (
        "/datasets/cases.jsonl",
        "/train/checkpoint-100",
        "/models/base",
        "/models/default",
    ):
        path = snapshots.path(reference, context)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source")
    versions = Versions(SQLiteVersionStore(tmp_path / "versions.sqlite3"), snapshots)
    project = versions.project(ctx.conversation.id, context)
    assert project is not None
    default = versions.save(
        project,
        "default-model",
        SaveVersion(
            kind="model",
            logical_key="model",
            source=SnapshotSource(text="default model fixture", filename="model.bin"),
        ),
        context,
    )
    versions.store.select(
        project.id, "model", default.id, "select-default", "test", project.revision
    )
    ctx.handler.runs = PipelineRuns(
        ctx.conversation.id, ctx.store, ctx.registry, versions, context
    )
    config_file = ctx.workspace / "public_data/evaluation.json"
    config = json.loads(config_file.read_text())
    config["model_source"] = {
        "type": "checkpoint",
        "kind": "lora",
        "base_model": "/models/base",
    }
    config_file.write_text(json.dumps(config))
    ctx.action.inference.model_path = "/train/checkpoint-100"
    result = ctx.executor(ctx.action, cast(Any, ctx.conversation))
    assert not result.is_error, result.text
    canvas = (ctx.workspace / "public_data/workflow_canvas/workflow.py").read_text()
    assert "/train/checkpoint-100" in canvas
    assert default.manifest.storage_ref not in canvas
    record = versions.store.list_runs(project.id)[0]
    assert {"selected_checkpoint", "base_model"} <= set(record.spec.inputs)
