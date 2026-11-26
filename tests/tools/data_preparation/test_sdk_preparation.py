from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from pyromind_sdk.client.studio import StudioClient
from pyromind_sdk.client.workflow import DslConverter

from pyromind_common.platform.credentials import materialize_runtime_environment
from pyromind_common.platform.evaluation import prepare_evaluation, stage_evaluation
from pyromind_common.platform.evaluation_core import (
    EvaluationOptions,
    ValidationSelection,
)
from pyromind_common.platform.paths import StoragePathError, to_storage_api_path
from pyromind_common.platform.preparation import (
    StagingDirectory,
    prepare_cleaning,
    prepare_edp_aggregate,
    prepare_edp_render,
    prepare_pipeline,
)
from pyromind_common.platform.scene_options import (
    AggregateOptions,
    CleaningOptions,
    PipelineOptions,
    RenderOptions,
    ReuseAssessment,
)
from pyromind_runtime.domain import storage_paths as product_paths


def test_mounted_staging_uses_one_directory_for_local_and_platform_paths():
    directory = "/workspace/data/runs/unique-run"
    staging = StagingDirectory.mounted(directory)
    assert staging.local == Path(directory)
    assert staging.storage_path("pipeline.py") == "/data/runs/unique-run/pipeline.py"
    for invalid in ("/data/run", "public_data/run", "/workspace/../other"):
        with pytest.raises(ValueError):
            StagingDirectory.mounted(invalid)


def test_new_pipeline_run_needs_a_new_local_freeze_directory(
    sdk_files, sdk_staging, sdk_run_id, processing_scripts
):
    script = sdk_files.root / "pipeline.py"
    script.write_text("print('pipeline')\n")
    options = PipelineOptions(
        script_path="pipeline.py",
        input_path="/workspace/input.jsonl",
        model_profile="none",
        mode="full",
    )
    prepare_pipeline(
        options,
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=processing_scripts / "preparation",
        run_id=sdk_run_id,
    )
    previous = (sdk_staging.local / "pipeline-inputs.json").read_bytes()
    next_run = uuid4()
    reused_local = StagingDirectory(sdk_staging.local, f"/workspace/jobs/{next_run}")
    with pytest.raises(ValueError, match="new local staging directory") as caught:
        prepare_pipeline(
            options,
            files=sdk_files,
            staging=reused_local,
            runtime_dir=processing_scripts / "preparation",
            run_id=next_run,
        )
    assert str(sdk_staging.local.resolve()) in str(caught.value)
    assert (sdk_staging.local / "pipeline-inputs.json").read_bytes() == previous
    new_local = StagingDirectory(
        sdk_staging.local.parent / str(next_run), reused_local.storage
    )
    graph = prepare_pipeline(
        options,
        files=sdk_files,
        staging=new_local,
        runtime_dir=processing_scripts / "preparation",
        run_id=next_run,
    )
    assert graph["nodes"]
    assert (new_local.local / "pipeline-inputs.json").is_file()


@pytest.mark.parametrize(
    "path",
    [
        "/workspace/data/a.jsonl",
        "/data/a.jsonl",
        "/.pyromind-agent/run",
        "data/a",
        "/",
        "/workspace",
        "/target-workspace/data",
        "/workspace/../outside",
        "public_data/a",
        "/public_data/a",
        "/workspace/data\\a",
        "/workspace/data\nkey",
    ],
)
def test_platform_path_contract_matches_product_paths(path):
    try:
        expected = product_paths.to_storage_api_path(path)
    except product_paths.StoragePathError:
        with pytest.raises(StoragePathError):
            to_storage_api_path(path)
    else:
        assert to_storage_api_path(path) == expected


def test_plain_helpers_import_without_host_tool_or_runtime_packages():
    code = """
import importlib.abc
import sys

class BlockHost(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('pyromind_common.tools', 'pyromind_common.sdk',
                                'pyromind_runtime', 'harness_adapter', 'litellm')):
            raise ImportError(fullname)

sys.meta_path.insert(0, BlockHost())
import pyromind_common.platform.workflows
import pyromind_common.platform.preparation
import pyromind_common.platform.evaluation
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_pipeline_freezes_inputs_and_rejects_changed_resume(
    sdk_files, sdk_staging, sdk_run_id, processing_scripts
):
    (sdk_files.root / "pipeline.py").write_text("print('pipeline')\n")
    (sdk_files.root / "support.json").write_text('{"labels": ["ok"]}')
    options = PipelineOptions(
        script_path="pipeline.py",
        input_path="/workspace/input.jsonl",
        model_profile="none",
        output_schema="text",
        support_file_path="support.json",
    )
    workflow = prepare_pipeline(
        options,
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=processing_scripts / "preparation",
        run_id=sdk_run_id,
    )
    command = workflow["nodes"][0]["data"]["config"]["command"]
    assert "/target-workspace/jobs/one/pipeline.py" in command
    assert (sdk_staging.local / "runtime/source_fingerprint.py").is_file()
    assert (sdk_staging.local / "pipeline.py").read_text() == "print('pipeline')\n"
    resume = PipelineOptions(
        mode="resume", resume_run_id=sdk_run_id, input_path="/workspace/input.jsonl"
    )
    resumed = prepare_pipeline(
        resume,
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=processing_scripts / "preparation",
        run_id=sdk_run_id,
    )
    resumed_command = resumed["nodes"][0]["data"]["config"]["command"]
    assert "DF_OUTPUT_SCHEMA=text" in resumed_command
    assert "/target-workspace/jobs/one/job-spec.json" in resumed_command
    changed = PipelineOptions(
        mode="resume",
        resume_run_id=sdk_run_id,
        input_path="/workspace/different.jsonl",
        model_profile="none",
    )
    with pytest.raises(ValueError, match="Frozen input changed"):
        prepare_pipeline(
            changed,
            files=sdk_files,
            staging=sdk_staging,
            runtime_dir=processing_scripts / "preparation",
            run_id=sdk_run_id,
        )


@pytest.mark.parametrize("profile", ["text", "vision"])
def test_pipeline_requires_model_configuration_before_freezing(
    sdk_files, sdk_staging, sdk_run_id, processing_scripts, monkeypatch, profile
):
    monkeypatch.delenv("PYROMIND_DATAFLOW_PROFILES", raising=False)
    options = PipelineOptions(
        script_path="pipeline.py",
        input_path="/workspace/input.jsonl",
        model_profile=profile,
    )
    with pytest.raises(ValueError, match=f"no usable {profile} model profile"):
        prepare_pipeline(
            options,
            files=sdk_files,
            staging=sdk_staging,
            runtime_dir=processing_scripts / "preparation",
            run_id=sdk_run_id,
        )
    assert not sdk_staging.local.exists()


@pytest.mark.parametrize("profile", ["text", "vision"])
def test_pipeline_binds_model_credentials_only_to_the_transport_graph(
    sdk_files, sdk_staging, sdk_run_id, processing_scripts, sdk_model_profiles, profile
):
    (sdk_files.root / "pipeline.py").write_text("print('model pipeline')\n")
    graph = prepare_pipeline(
        PipelineOptions(
            script_path="pipeline.py",
            input_path="/workspace/input.jsonl",
            model_profile=profile,
        ),
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=processing_scripts / "preparation",
        run_id=sdk_run_id,
    )
    key = sdk_model_profiles[profile]["DF_API_KEY"]
    assert key not in json.dumps(graph)
    assert all(
        key.encode() not in path.read_bytes()
        for path in sdk_staging.local.rglob("*")
        if path.is_file()
    )
    transported = materialize_runtime_environment(graph)
    command = transported["nodes"][0]["data"]["config"]["command"]
    assert f"export DF_API_KEY={key}" in command
    assert sdk_model_profiles[profile]["DF_MODEL_NAME"] in command
    assert key not in graph["nodes"][0]["data"]["config"]["command"]


@pytest.mark.parametrize(
    ("change", "dimension"),
    [
        ("pipeline", "pipeline"),
        ("prompt", "prompt"),
        ("model", "model"),
        ("schema", "schema"),
        ("runtime", "runtime"),
        ("support", "schema"),
    ],
)
def test_pipeline_compatible_resume_requires_evidence_and_keeps_prior_assets(
    sdk_files,
    sdk_staging,
    sdk_run_id,
    processing_scripts,
    sdk_model_profiles,
    change,
    dimension,
):
    script = sdk_files.root / "pipeline.py"
    script.write_text("print('original')\n")
    support = sdk_files.root / "support.json"
    support.write_text('{"labels": ["ok"]}')
    runtime = processing_scripts / "preparation"
    prepare_pipeline(
        PipelineOptions(
            script_path="pipeline.py",
            input_path="/workspace/input.jsonl",
            model_profile="none",
            output_schema="text",
            support_file_path="support.json",
        ),
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=runtime,
        run_id=sdk_run_id,
    )
    first_manifest = (sdk_staging.local / "pipeline-inputs.json").read_bytes()
    changes = {}
    if change == "pipeline":
        script.write_text("print('compatible revision')\n")
        changes["script_path"] = "pipeline.py"
    elif change == "prompt":
        changes["prompt_fingerprint"] = "prompt-revision-two"
    elif change == "model":
        changes["model_profile"] = "text"
    elif change == "schema":
        changes["output_schema"] = "dpo"
    elif change == "runtime":
        runtime = shutil.copytree(runtime, sdk_files.root / "updated-runtime")
        with (runtime / "generate_report.py").open("a") as output:
            output.write("\n# Runtime revision\n")
        changes["script_path"] = "pipeline.py"
    else:
        support.write_text('{"labels": ["ok", "warning"]}')
        changes["support_file_path"] = "support.json"
    resume = PipelineOptions(
        mode="resume",
        resume_run_id=sdk_run_id,
        input_path="/workspace/input.jsonl",
        **changes,
    )
    with pytest.raises(ValueError, match=f"reuse_assessment for: {dimension}"):
        prepare_pipeline(
            resume,
            files=sdk_files,
            staging=sdk_staging,
            runtime_dir=runtime,
            run_id=sdk_run_id,
        )
    assert not (sdk_staging.local / "pipeline-revisions").exists()
    assessed = PipelineOptions.model_validate(
        {
            **resume.model_dump(),
            "reuse_assessment": {
                "changed_dimensions": [dimension],
                "change_summary": "Verified compatible change",
                "reason": "Previously committed output still meets the contract",
                "verification_samples": ["sample-1"],
                "verification_result": "passed",
            },
        }
    )
    graph = prepare_pipeline(
        assessed,
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=runtime,
        run_id=sdk_run_id,
    )
    command = graph["nodes"][0]["data"]["config"]["command"]
    assert "DF_EXECUTION_REVISION=2" in command
    assert "DF_REUSE_ASSESSMENT_JSON=" in command
    assert (sdk_staging.local / "pipeline-inputs.json").read_bytes() == first_manifest
    assert (sdk_staging.local / "pipeline.py").read_text() == "print('original')\n"
    snapshot = json.loads(
        (sdk_staging.local / "pipeline-revisions/00000002.json").read_text()
    )
    assert snapshot["reuse_assessment"]["changed_dimensions"] == [dimension]
    assert snapshot["execution_revision"] == 2
    third = prepare_pipeline(
        PipelineOptions(
            mode="resume", resume_run_id=sdk_run_id, input_path="/workspace/input.jsonl"
        ),
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=runtime,
        run_id=sdk_run_id,
    )
    assert "DF_EXECUTION_REVISION=3" in third["nodes"][0]["data"]["config"]["command"]


def test_pipeline_strict_resume_uses_frozen_script_and_runtime(
    sdk_files, sdk_staging, sdk_run_id, processing_scripts
):
    script = sdk_files.root / "pipeline.py"
    script.write_text("print('original')\n")
    runtime = shutil.copytree(
        processing_scripts / "preparation", sdk_files.root / "runtime"
    )
    prepare_pipeline(
        PipelineOptions(
            script_path="pipeline.py",
            input_path="/workspace/input.jsonl",
            model_profile="none",
        ),
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=runtime,
        run_id=sdk_run_id,
    )
    script.write_text("raise RuntimeError('unreviewed change')\n")
    with (runtime / "generate_report.py").open("a") as output:
        output.write("\n# New runtime\n")
    resumed = prepare_pipeline(
        PipelineOptions(
            mode="resume", resume_run_id=sdk_run_id, input_path="/workspace/input.jsonl"
        ),
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=runtime,
        run_id=sdk_run_id,
    )
    command = resumed["nodes"][0]["data"]["config"]["command"]
    assert "/target-workspace/jobs/one/pipeline.py" in command
    assert "runtime-r2" not in command
    assert (sdk_staging.local / "pipeline.py").read_text() == "print('original')\n"
    first = json.loads((sdk_staging.local / "pipeline-inputs.json").read_text())
    second = json.loads(
        (sdk_staging.local / "pipeline-revisions/00000002.json").read_text()
    )
    assert first["runtime_sha256"] == second["runtime_sha256"]


@pytest.mark.parametrize("initial", ["text", "structured"])
def test_pipeline_resume_cannot_switch_structured_output_contract(
    sdk_files, sdk_staging, sdk_run_id, processing_scripts, initial
):
    (sdk_files.root / "pipeline.py").write_text(
        "from image_utils import ImagePipelineConfig, run_image_pipeline_from_cli\n"
    )
    prepare_pipeline(
        PipelineOptions(
            script_path="pipeline.py",
            input_path="/workspace/input.jsonl",
            model_profile="none",
            output_schema=initial,
        ),
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=processing_scripts / "preparation",
        run_id=sdk_run_id,
    )
    with pytest.raises(ValueError, match="Cannot change structured output contract"):
        prepare_pipeline(
            PipelineOptions(
                mode="resume",
                resume_run_id=sdk_run_id,
                input_path="/workspace/input.jsonl",
                output_schema="text" if initial == "structured" else "structured",
                reuse_assessment=ReuseAssessment(
                    changed_dimensions=["schema"],
                    change_summary="Switch output contract",
                    reason="User requested change",
                    verification_samples=["sample-1"],
                    verification_result="passed",
                ),
            ),
            files=sdk_files,
            staging=sdk_staging,
            runtime_dir=processing_scripts / "preparation",
            run_id=sdk_run_id,
        )


@pytest.mark.parametrize("profile", ["none", "text"])
def test_pipeline_recipe_reuses_frozen_business_builders(
    sdk_files, sdk_staging, sdk_run_id, processing_scripts, sdk_model_profiles, profile
):
    for name in ("pipeline.py", "export.py"):
        (sdk_files.root / name).write_text("print('recipe')\n")
    options = PipelineOptions(
        script_path="pipeline.py",
        sft_export_script_path="export.py",
        input_path="/workspace/input.jsonl",
        model_profile=profile,
    )
    graph = prepare_pipeline(
        options,
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=processing_scripts / "preparation",
        run_id=sdk_run_id,
    )
    command = graph["nodes"][0]["data"]["config"]["command"]
    assert "sha256sum" in command
    assert list((sdk_staging.local / "recipe-assets").glob("*/verifier.py"))
    recipe = json.loads((sdk_staging.local / "dataset-recipe.json").read_text())
    assert {asset["role"] for asset in recipe["assets"]} == {
        "pipeline",
        "export",
        "verifier",
    }
    if profile == "text":
        key = sdk_model_profiles["text"]["DF_API_KEY"]
        assert key not in json.dumps(recipe)
        assert key not in command
        assert 'export DF_API_KEY="$param"' in recipe["command_template"]
        transported = materialize_runtime_environment(graph)
        assert key in transported["nodes"][0]["data"]["config"]["command"]


def test_saved_prediction_scoring_runs_examples_before_freezing(
    sdk_files, sdk_staging, sdk_run_id, sdk_validator, processing_scripts
):
    manifest = {
        "evaluations": [
            {"step": 100, "path": "/train/eval_step_100.jsonl", "sha256": "a" * 64}
        ]
    }
    spec = {
        "business_rule": "Answers must match",
        "validation_examples": [
            {
                "record": {"id": "pass", "reference": "yes", "prediction": "yes"},
                "passed": True,
            },
            {
                "record": {"id": "fail", "reference": "yes", "prediction": "no"},
                "passed": False,
            },
        ],
    }
    (sdk_files.root / "source.json").write_text(json.dumps(manifest))
    (sdk_files.root / "score.json").write_text(json.dumps(spec))
    (sdk_files.root / "score.py").write_text(
        "def parse_record(raw, context): return raw\n"
        "def score_case(case, spec):\n"
        "    return {'passed': case['reference'] == case['prediction']}\n"
    )
    options = EvaluationOptions(
        operation="score",
        input_path="/workspace/input.jsonl",
        source_manifest_path="source.json",
        scoring_spec_path="score.json",
        scoring_script_path="score.py",
    )
    graph = prepare_evaluation(
        options,
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=processing_scripts.parent.parent / "inference-evaluation/scripts",
        run_id=sdk_run_id,
        validator=sdk_validator,
    )
    assert [node["data"]["nodeType"] for node in graph["nodes"]] == [
        "CustomCommandCPUNode"
    ]
    frozen = json.loads((sdk_staging.local / "prediction_evaluation.json").read_text())
    assert frozen["source_manifest"] == manifest
    assert frozen["scoring_spec"] == spec


def test_cleaning_preflight_freezes_script_without_implying_resume(
    sdk_files, sdk_staging, sdk_run_id, processing_scripts
):
    (sdk_files.root / "clean.py").write_text("print('clean')\n")
    options = CleaningOptions(
        script_path="clean.py", input_path="/workspace/input.jsonl"
    )
    graph = prepare_cleaning(
        options,
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=processing_scripts / "cleaning",
        run_id=sdk_run_id,
    )
    command = graph["nodes"][0]["data"]["config"]["command"]
    assert "--resume" not in command
    assert "cp " not in command
    assert (sdk_staging.local / "clean_script.py").read_text() == "print('clean')\n"
    other_run = UUID("22222222-2222-2222-2222-222222222222")
    with pytest.raises(ValueError, match="Frozen input changed"):
        prepare_cleaning(
            CleaningOptions(input_path=options.input_path, resume_run_id=other_run),
            files=sdk_files,
            staging=sdk_staging,
            runtime_dir=processing_scripts / "cleaning",
            run_id=other_run,
        )
    (sdk_files.root / "clean.py").write_text(
        "from cleaning_utils import imaginary_api\n"
    )
    with pytest.raises(ValueError, match="unsupported cleaning_utils APIs"):
        prepare_cleaning(
            options,
            files=sdk_files,
            staging=sdk_staging,
            runtime_dir=processing_scripts / "cleaning",
            run_id=sdk_run_id,
        )


def test_edp_render_and_aggregate_prepare_without_auth_tokens(
    sdk_files, sdk_staging, sdk_run_id, processing_scripts
):
    (sdk_files.root / "template.json").write_text(
        json.dumps({"fields": {"task_id": "id"}})
    )
    render = RenderOptions(
        template_path="template.json", data_source="/workspace/input.parquet"
    )
    graph = prepare_edp_render(
        render,
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=processing_scripts / "edp",
        run_id=sdk_run_id,
    )
    assert "render_manifest.py" in graph["nodes"][0]["data"]["config"]["command"]
    aggregate = AggregateOptions(
        run_dirs=["/workspace/jobs/shard-1"], out_dir=sdk_staging.storage
    )
    graph = prepare_edp_aggregate(
        aggregate,
        staging=sdk_staging,
        runtime_dir=processing_scripts / "edp",
        run_id=sdk_run_id,
    )
    command = graph["nodes"][0]["data"]["config"]["command"]
    assert "/target-workspace/jobs/shard-1" in command
    assert "AUTH_TOKEN" not in command


def test_evaluation_freezes_inference_and_reuses_original_configs(
    sdk_files, sdk_staging, sdk_run_id, sdk_validator, processing_scripts
):
    (sdk_files.root / "dataset.json").write_text(
        json.dumps({"user_prompt_field": "prompt", "reference_field": "answer"})
    )
    (sdk_files.root / "evaluation.json").write_text(
        json.dumps(
            {
                "mode": "agent_rubric",
                "rubrics": [
                    {
                        "name": "exact",
                        "criterion": "match",
                        "weight": 1,
                        "evaluator": {"type": "exact_match"},
                    }
                ],
            }
        )
    )
    options = EvaluationOptions(
        operation="inference",
        input_path="/workspace/input.jsonl",
        model_path="/workspace/model",
        dataset_config_path="dataset.json",
        evaluation_config_path="evaluation.json",
    )
    runtime = processing_scripts.parent.parent / "inference-evaluation/scripts"
    graph = prepare_evaluation(
        options,
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=runtime,
        run_id=sdk_run_id,
        validator=sdk_validator,
    )
    assert len(graph["nodes"]) == 2
    original = (sdk_staging.local / "evaluation_config.json").read_bytes()
    (sdk_files.root / "evaluation.json").write_text("changed")
    resume = EvaluationOptions(
        operation="resume",
        input_path="/workspace/input.jsonl",
        resume_run_id=sdk_run_id,
        cpu=8,
        memory=64,
    )
    resumed_graph = prepare_evaluation(
        resume,
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=runtime,
        run_id=sdk_run_id,
        validator=sdk_validator,
    )
    assert (sdk_staging.local / "evaluation_config.json").read_bytes() == original
    worker = next(
        node
        for node in resumed_graph["nodes"]
        if node["data"]["nodeType"] == "CustomCommandCPUNode"
    )
    assert worker["data"]["config"]["cpu"] == 8
    assert worker["data"]["config"]["memory"] == 64
    assert len(list((sdk_staging.local / "workflows").glob("*.py"))) == 2
    other_run = UUID("22222222-2222-2222-2222-222222222222")
    with pytest.raises(ValueError, match="belongs to another run"):
        prepare_evaluation(
            resume.model_copy(update={"resume_run_id": other_run}),
            files=sdk_files,
            staging=sdk_staging,
            runtime_dir=runtime,
            run_id=other_run,
            validator=sdk_validator,
        )


@pytest.mark.parametrize("with_reward", [False, True])
def test_held_out_evaluation_freezes_selection_and_reuses_it_on_resume(
    sdk_files,
    sdk_staging,
    sdk_run_id,
    sdk_validator,
    processing_scripts,
    sdk_validation_selection,
    sdk_held_out_options,
    with_reward,
):
    selection = sdk_validation_selection
    reward = None
    if with_reward:
        reward = {"manifest_sha256": "d" * 64}
        selection = selection.model_copy(update={"frozen_reward": reward})
        config_path = sdk_files.root / "evaluation.json"
        config = json.loads(config_path.read_text())
        config["rubrics"] = []
        config_path.write_text(json.dumps(config))
    runtime = processing_scripts.parent.parent / "inference-evaluation/scripts"
    graph = prepare_evaluation(
        sdk_held_out_options,
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=runtime,
        run_id=sdk_run_id,
        validator=sdk_validator,
        selection=selection,
        frozen_reward=reward,
    )
    assert len(graph["nodes"]) == 3
    definition = json.loads(
        (sdk_staging.local / "evaluation-definition.json").read_text()
    )
    assert definition["dataset_role"] == "test"
    assert definition["test_source"]["sha256"] == "c" * 64
    contract_path = sdk_staging.local / (
        "frozen_reward.json" if with_reward else "business_scoring.json"
    )
    contract = json.loads(contract_path.read_text())
    assert contract["test_source"] == definition["test_source"]
    prepare_evaluation(
        EvaluationOptions(
            operation="resume",
            input_path=sdk_held_out_options.input_path,
            resume_run_id=sdk_run_id,
        ),
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=runtime,
        run_id=sdk_run_id,
        validator=sdk_validator,
        frozen_reward=reward,
    )
    assert json.loads(contract_path.read_text()) == contract


@pytest.mark.parametrize(
    "change,expected",
    [
        ("input", "frozen test dataset"),
        ("checkpoint", "selected checkpoint"),
        ("selection", "matching validation selection"),
        ("limit", "complete test dataset"),
        ("reward", "frozen Reward"),
    ],
)
def test_held_out_evaluation_rejects_changed_project_evidence(
    sdk_files,
    sdk_staging,
    sdk_run_id,
    sdk_validator,
    processing_scripts,
    sdk_validation_selection,
    sdk_held_out_options,
    change,
    expected,
):
    options = sdk_held_out_options
    reward = None
    if change == "input":
        options = options.model_copy(
            update={"input_path": "/workspace/datasets/other.jsonl"}
        )
    elif change == "checkpoint":
        options = options.model_copy(
            update={"model_path": "/workspace/models/checkpoint-200"}
        )
    elif change == "selection":
        options = options.model_copy(update={"selection_run_id": "another-selection"})
    else:
        path = sdk_files.root / "evaluation.json"
        config = json.loads(path.read_text())
        if change == "limit":
            config["generation"] = {"limit": 1}
        else:
            reward = {"manifest_sha256": "d" * 64}
            config["rubrics"] = []
        path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=expected):
        prepare_evaluation(
            options,
            files=sdk_files,
            staging=sdk_staging,
            runtime_dir=processing_scripts.parent.parent
            / "inference-evaluation/scripts",
            run_id=sdk_run_id,
            validator=sdk_validator,
            selection=sdk_validation_selection,
            frozen_reward=reward,
        )
    assert not sdk_staging.local.exists()


@pytest.mark.parametrize("collision", ["path", "sha256"])
def test_validation_selection_requires_independent_frozen_test_data(
    sdk_validation_selection,
    collision,
):
    evidence = sdk_validation_selection.model_dump()
    evidence["test"][collision] = evidence["validation"][collision]
    with pytest.raises(ValueError, match="independent of training and validation"):
        ValidationSelection.model_validate(evidence)


def test_sdk_only_evaluation_staging_uses_no_http_or_create(
    sdk_files,
    sdk_staging,
    sdk_run_id,
    processing_scripts,
    sdk_validation_selection,
    sdk_held_out_options,
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Staging must not validate over HTTP or create tasks")

    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(StudioClient, "create", forbidden)
    prepared = stage_evaluation(
        sdk_held_out_options,
        files=sdk_files,
        staging=sdk_staging,
        runtime_dir=processing_scripts.parent.parent / "inference-evaluation/scripts",
        run_id=sdk_run_id,
        selection=sdk_validation_selection,
    )
    assert prepared.workflow_path == sdk_staging.local / "workflow.py"
    assert prepared.workflow_path.read_text() == prepared.dsl
    assert prepared.name == "agent-evaluation-11111111"
    assert "checkpoint-100" in prepared.dsl
    assert prepared.frozen_files and all(
        path.is_file() for path in prepared.frozen_files
    )


def test_evaluation_wrapper_returns_the_actual_server_graph(
    sdk_files,
    sdk_staging,
    sdk_run_id,
    processing_scripts,
    sdk_validation_selection,
    sdk_held_out_options,
):
    returned = []
    resources = []

    def validator(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if request.url.path.endswith("dsl/validate"):
            graph = DslConverter().from_python(payload["dsl"], name=payload["name"])
            worker = next(
                node
                for node in graph["nodes"]
                if node["data"]["nodeType"] == "CustomCommandCPUNode"
            )
            worker["data"]["config"]["cpu"] = 12
            returned.append(graph)
            return httpx.Response(
                200, json={"success": True, "data": {"valid": True, "workflow": graph}}
            )
        resources.append(payload["workflow"])
        return httpx.Response(200, json={"success": True, "data": {"valid": True}})

    with httpx.Client(
        base_url="https://validation.test/workflows/",
        transport=httpx.MockTransport(validator),
    ) as client:
        graph = prepare_evaluation(
            sdk_held_out_options,
            files=sdk_files,
            staging=sdk_staging,
            runtime_dir=processing_scripts.parent.parent
            / "inference-evaluation/scripts",
            run_id=sdk_run_id,
            selection=sdk_validation_selection,
            validator=client,
        )
    assert graph == returned[0] == resources[0]


@pytest.mark.parametrize("scene", ["cleaning", "render", "aggregate"])
def test_prepared_cpu_scenes_preserve_business_parameters_through_sdk_dsl(
    sdk_files,
    sdk_staging,
    sdk_run_id,
    processing_scripts,
    scene,
):
    if scene == "cleaning":
        (sdk_files.root / "clean.py").write_text("print('clean')\n")
        graph = prepare_cleaning(
            CleaningOptions(
                script_path="clean.py",
                input_path="/workspace/input.jsonl",
                cpu=7,
                memory=24,
            ),
            files=sdk_files,
            staging=sdk_staging,
            runtime_dir=processing_scripts / "cleaning",
            run_id=sdk_run_id,
        )
    elif scene == "render":
        (sdk_files.root / "template.json").write_text('{"fields": {"task_id": "id"}}')
        graph = prepare_edp_render(
            RenderOptions(
                template_path="template.json",
                data_source="/workspace/source.parquet",
                cpu=7,
                memory=24,
            ),
            files=sdk_files,
            staging=sdk_staging,
            runtime_dir=processing_scripts / "edp",
            run_id=sdk_run_id,
        )
    else:
        graph = prepare_edp_aggregate(
            AggregateOptions(
                run_dirs=["/workspace/shards/one"],
                out_dir=sdk_staging.storage,
                cpu=7,
                memory=24,
            ),
            staging=sdk_staging,
            runtime_dir=processing_scripts / "edp",
            run_id=sdk_run_id,
        )
    converter = DslConverter()
    dsl = converter.to_python(graph)
    round_trip = converter.from_python(dsl, name=graph["name"])
    assert f"# workflow: {graph['name']}" in dsl
    assert round_trip["name"] == graph["name"]
    assert (
        round_trip["nodes"][0]["data"]["config"] == graph["nodes"][0]["data"]["config"]
    )
