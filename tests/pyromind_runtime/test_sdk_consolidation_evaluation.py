import json
from hashlib import sha256
from unittest.mock import Mock
from uuid import UUID

import pytest

from pyromind_agent_server.echomind_sdk_eval_consolidation import sdk_evaluation_assets
from pyromind_common.platform.evaluation import staged_evaluation_digest
from pyromind_common.platform.evaluation_core import INFERENCE_RUNTIME_FILENAMES
from pyromind_common.platform.scene_core import runtime_bundle_fingerprint
from pyromind_runtime.domain.pipeline import (
    PipelineRun,
    pipeline_definition_fingerprint,
)
from pyromind_runtime.domain.task_correlation import TaskCorrelation
from pyromind_runtime.domain.versions import fingerprint
from pyromind_runtime.domain.workflow_execution import WorkflowExecutionRequest
from pyromind_runtime.infrastructure.file_product_store import FileProductStore


@pytest.fixture
def sdk_eval(tmp_path):
    conversation = "a" * 32
    run_id = str(UUID(int=3))
    directory = (
        f"/workspace/.pyromind-agent/{conversation}/public_data/scoring/{run_id}"
    )
    files = {name: tmp_path / name for name in INFERENCE_RUNTIME_FILENAMES}
    for name, path in files.items():
        path.write_text(name)
    correlation = TaskCorrelation.model_validate(
        {
            "conversation_id": conversation,
            "task_kind": "workflow_run",
            "signature": "a" * 64,
            "execution": {
                "task_id": "scoring",
                "project_id": "proj",
                "plan_sequence": 1,
                "attempt": 1,
            },
        }
    )
    receipt = WorkflowExecutionRequest(
        request_id="sdk:7",
        conversation_id=conversation,
        workflow_path="version://v1",
        fingerprint="777",
        workflow={},
        summary={},
        status="submitted",
        correlation=correlation,
        task_id="7",
    )
    store = Mock(spec_set=FileProductStore)
    store.verify_task_correlation.return_value = True
    store.load_snapshot.return_value = Mock(
        conversation_id=conversation,
        external_tasks=[Mock(task_id="7", status="succeeded", correlation=correlation)],
    )
    store.load_pipeline_runs.return_value = {}
    arguments = {
        "--run-id": run_id,
        "--output-dir": directory,
        "--model-reference": "/workspace/models/merged",
        "script": f"{directory}/evaluate_inference.py",
        "--dataset-config": f"{directory}/dataset_config.json",
        "--evaluation-config": f"{directory}/evaluation_config.json",
    }
    definition = {
        "run_id": run_id,
        "dataset_config": {"label": "判定"},
        "evaluation_config": {"metric": "exact"},
        "runtime_sha256": staged_evaluation_digest(files),
        "inference": {"model_path": "/workspace/models/merged"},
    }
    metrics = {"scores": [1.0]}
    documents = {
        "evaluation-definition.json": definition,
        "metrics.json": metrics,
        "run_manifest.json": {
            "run_id": run_id,
            "model_reference": "/models/merged",
            "dataset_config": definition["dataset_config"],
            "evaluation_config_sha256": fingerprint(definition["evaluation_config"]),
            "runtime_sha256": runtime_bundle_fingerprint(
                tmp_path, INFERENCE_RUNTIME_FILENAMES
            ),
        },
        "report.json": {
            "metrics": metrics,
            "status": "succeeded",
            "evaluation_status": "validated",
            "report_status": "ready",
            "artifacts": {
                name: f"{directory}/{name}"
                for name in ("predictions.jsonl", "metrics.json")
            },
        },
    }
    return store, receipt, arguments, documents


@pytest.mark.parametrize(
    "case", ["current", "historical", "metrics", "signature", "script", "manifest"]
)
def test_sdk_eval_recovers_staged_definition_and_pins_scoring_assets(
    sdk_eval, tmp_path, case
):
    store, receipt, arguments, documents = sdk_eval
    if case == "metrics":
        documents["metrics.json"] = {"scores": [0.5]}
    elif case == "signature":
        store.verify_task_correlation.return_value = False
    elif case in {"historical", "script", "manifest"}:
        (tmp_path / "evaluate_inference.py").write_text("new local scorer")
        if case == "manifest":
            documents["run_manifest.json"]["runtime_sha256"] = "a" * 64

    def read(path):
        assert path.startswith(arguments["--output-dir"] + "/")
        name = path.rsplit("/", 1)[1]
        if name in INFERENCE_RUNTIME_FILENAMES:
            assert case in {"historical", "script", "manifest"}
            if case == "script" and name == "evaluate_inference.py":
                return b"changed remote scorer"
            return name.encode()
        return json.dumps(documents[name]).encode()

    if case not in ("current", "historical"):
        with pytest.raises(ValueError, match="指标|签名|SHA-256"):
            sdk_evaluation_assets(store, receipt, arguments, tmp_path, read)
        return
    definition, root, assets = sdk_evaluation_assets(
        store, receipt, arguments, tmp_path, read
    )
    assert definition == documents["evaluation-definition.json"]
    assert assets["runtime_script"] == {
        "path": root + "/evaluate_inference.py",
        "sha256": sha256(b"evaluate_inference.py").hexdigest(),
    }
    assert set(assets) == {
        "runtime_script",
        "dataset_config",
        "evaluation_config",
        "evaluation_report.js",
        "evaluation_report.css",
    }
    staged_config = json.dumps(
        definition["dataset_config"], ensure_ascii=False, sort_keys=True
    ).encode()
    assert assets["dataset_config"] == {
        "path": root + "/dataset_config.json",
        "sha256": sha256(staged_config).hexdigest(),
    }


@pytest.mark.parametrize("attempt", [1, 2])
def test_sdk_eval_durable_result_is_bound_to_exact_attempt(sdk_eval, tmp_path, attempt):
    store, receipt, arguments, documents = sdk_eval
    root = arguments["--output-dir"]
    source = {
        "request_id": receipt.request_id,
        "workflow_path": receipt.workflow_path,
        "fingerprint": receipt.fingerprint,
        "project_id": "proj",
        "plan_sequence": 1,
        "attempt": attempt,
        "plan_task_id": "scoring",
        "external_task_id": "7",
        "report_path": root + "/report.json",
        "metrics_path": root + "/metrics.json",
        "definition_path": root + "/evaluation-definition.json",
    }
    run = PipelineRun(
        run_id=UUID(arguments["--run-id"]),
        conversation_id=receipt.conversation_id,
        definition=documents["evaluation-definition.json"],
        result={
            "status": "succeeded",
            "evaluation_status": "validated",
            "report_status": "ready",
            "receipt_source": source,
        },
        fingerprint=pipeline_definition_fingerprint(
            documents["evaluation-definition.json"]
        ),
        output_dir=root,
        task_id="7",
        submission_status="submitted",
    )
    store.load_pipeline_runs.return_value = {str(run.run_id): run}
    if attempt != 1:
        with pytest.raises(ValueError, match="回执"):
            sdk_evaluation_assets(store, receipt, arguments, tmp_path, None)
        return
    definition, _, assets = sdk_evaluation_assets(
        store, receipt, arguments, tmp_path, None
    )
    assert definition == run.definition and "runtime_script" in assets
