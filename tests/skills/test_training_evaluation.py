"""Saved predictions are scored once, compared by provenance, and reviewable."""

import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from pyromind_runtime.application.training_evaluations import TrainingEvaluations
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.infrastructure.file_product_store import FileProductStore


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / ".agents/skills/inference-evaluation/scripts"
TRAINING = ROOT / ".agents/skills/training-analysis/scripts"


@pytest.fixture
def prediction_runtime(monkeypatch):
    monkeypatch.syspath_prepend(str(RUNTIME))
    monkeypatch.syspath_prepend(str(TRAINING))
    spec = importlib.util.spec_from_file_location(
        "prediction_worker_test", RUNTIME / "evaluate_predictions.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT = """
def parse_record(raw, context):
    return {"id": raw["id"], "reference": raw.get("gt", raw.get("target")),
            "prediction": raw.get("answer", raw.get("prediction")),
            "original_label": raw.get("gt", raw.get("target")),
            "sample": {"prompt": raw.get("prompt", "inspect image")}}

def score_case(case, spec):
    mapping = {"pass": "pass", "defect": "pass", "fail": "fail"}
    gt = mapping[case["reference"]]
    prediction = mapping.get(case["prediction"])
    return {"passed": gt == prediction, "reference_label": gt,
            "prediction_label": prediction, "evidence": ["binary business verdict"]}
"""


@pytest.fixture
def saved_batch(tmp_path, prediction_runtime):
    mount = tmp_path / "storage"
    root = mount / "training"
    root.mkdir(parents=True)
    dataset = [
        {"id": "a", "gt": "defect"},
        {"id": "b", "gt": "fail"},
        {"id": "c", "gt": "pass"},
    ]
    for source in ("train", "validation"):
        (root / f"{source}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in dataset)
        )
    entries = []
    for step, predictions in [
        (20, ["pass", "pass", "pass"]),
        (100, ["pass", "fail", None]),
        (200, ["pass", "fail", "pass"]),
    ]:
        rows = [
            {"id": row["id"], "target": row["gt"], "prediction": prediction}
            for row, prediction in zip(dataset, predictions)
        ]
        path = root / f"eval_step_{step}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        entries.append(
            {
                "step": step,
                "path": "/training/" + path.name,
                "sha256": prediction_runtime._file_digest(path),
            }
        )
    checkpoints = []
    for step in (20, 100):
        checkpoint = root / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "Qwen/base"})
        )
        (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
        checkpoints.append({"step": step, "path": "/training/" + checkpoint.name})
    manifest = {
        "task_id": "1051",
        "training_node_id": "train",
        "evaluations": entries,
        "checkpoints": checkpoints,
        "generation": {"temperature": 0, "max_tokens": 256},
        "datasets": {
            source: {
                "path": f"/training/{source}.jsonl",
                "sha256": prediction_runtime._file_digest(root / f"{source}.jsonl"),
            }
            for source in ("train", "validation")
        },
    }
    spec = {
        "business_rule": "pass and defect are accepted; fail is rejected",
        "selection": {
            "constraints": [{"metric": "pass_rate", "operator": "ge", "value": 0.8}]
        },
        "validation_examples": [
            {
                "record": {"id": "good", "gt": "defect", "answer": "pass"},
                "passed": True,
            },
            {
                "record": {"id": "bad", "gt": "fail", "answer": "defect"},
                "passed": False,
            },
        ],
    }
    script = tmp_path / "score_eval.py"
    script.write_text(SCRIPT)
    run_id = str(uuid4())
    output_ref = f"/.pyromind-agent/test/data_preparation/{run_id}"
    output = mount / output_ref.lstrip("/")
    return prediction_runtime, manifest, spec, script, run_id, mount, output, output_ref


class DiskStorage:
    def __init__(self, root):
        self.root = root

    def read(self, path):
        return (self.root / path.lstrip("/")).read_bytes()

    def write(self, path, payload):
        target = self.root / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    def list(self, prefix):
        return {path.name for path in (self.root / prefix.lstrip("/")).iterdir()}


def review_manifest(worker, manifest, output, mount, *, step=20):
    directory = output / "steps" / str(step)
    binding = json.loads((directory / "run_manifest.json").read_text())

    def ref(path):
        return {
            "path": "/" + str(path.relative_to(mount)),
            "sha256": worker._file_digest(path),
        }

    return {
        "source": {
            **ref(directory / "predictions.jsonl"),
            "kind": "scoring",
            "scoring_run_id": binding["run_id"],
            "scoring_spec_sha256": binding["evaluation_config_sha256"],
        },
        "scoring_manifest": ref(directory / "run_manifest.json"),
        "metrics": ref(directory / "metrics.json"),
        "profile": ref(output / "dataset_profile.json"),
        "datasets": manifest["datasets"],
    }


def test_independent_review_uses_existing_ui_and_fresh_annotations(saved_batch):
    worker, manifest, spec, script, run_id, mount, output, _ = saved_batch
    products = importlib.import_module("evaluation_products")

    manifest["render_reports"] = False
    worker.evaluate_batch(manifest, spec, script, output, run_id, mount)
    assert not (output / "steps/20/evaluation_report.html").exists()
    inputs = review_manifest(worker, manifest, output, mount)
    first_id, second_id = str(uuid4()), str(uuid4())
    first, second = mount / first_id, mount / second_id
    definition = {"products_operation": "review", "products_manifest": inputs}
    task = products.run_products(definition, first, first_id, mount)
    assert task["case_count"] == 1
    assert (first / "evaluation_report.html").exists()
    records = [
        json.loads(line)
        for line in (first / "evaluation_annotations.r0.jsonl").read_text().splitlines()
    ]
    assert records[0]["case_id"] == "b"
    assert records[0]["run_id"] == first_id
    assert records[0]["human_annotation"] is None
    (first / "evaluation_annotations.r1.jsonl").write_text("old human feedback\n")
    before = (first / "evaluation_annotations.r1.jsonl").read_bytes()
    products.run_products(definition, second, second_id, mount)
    assert (first / "evaluation_annotations.r1.jsonl").read_bytes() == before
    assert first_id not in (second / "evaluation_annotations.r0.jsonl").read_text()
    assert not (second / "evaluation_annotations.r1.jsonl").exists()
    payload = json.loads((second / "predictions.jsonl").read_text())
    assert payload["details"]["distribution_context"]["train"]["count"] == 1


def test_review_requires_matching_complete_distribution(saved_batch):
    worker, manifest, spec, script, run_id, mount, output, _ = saved_batch
    products = importlib.import_module("evaluation_products")

    worker.evaluate_batch(manifest, spec, script, output, run_id, mount)
    inputs = review_manifest(worker, manifest, output, mount)
    inputs["datasets"] = {**inputs["datasets"], "train": {"path": "/different"}}
    target = mount / "review"
    with pytest.raises(ValueError, match="different dataset"):
        products.run_products(
            {"products_operation": "review", "products_manifest": inputs},
            target,
            str(uuid4()),
            mount,
        )
    assert not (target / "evaluation_annotations.r0.jsonl").exists()


def test_import_badcases_needs_no_scoring_or_full_metric_denominator(saved_batch):
    worker, manifest, spec, script, run_id, mount, output, _ = saved_batch
    products = importlib.import_module("evaluation_products")

    profile_dir = mount / "profile"
    profile_dir.mkdir()
    (profile_dir / "extract_dataset.py").write_text(SCRIPT)
    products.run_products(
        {
            "products_operation": "profile",
            "products_manifest": {"datasets": manifest["datasets"]},
        },
        profile_dir,
        str(uuid4()),
        mount,
    )
    source = mount / "external.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "external-a",
                "input": "inspect this image",
                "original_label": "defect",
                "reference": "defect",
                "prediction": "fail",
                "failure_reason": "false alarm",
                "human_annotation": {"comment": "old"},
            }
        )
        + "\n"
    )
    inputs = {
        "source": {
            "kind": "import",
            "path": "/external.jsonl",
            "sha256": worker._file_digest(source),
        },
        "profile": {
            "path": "/profile/dataset_profile.json",
            "sha256": worker._file_digest(profile_dir / "dataset_profile.json"),
        },
        "datasets": manifest["datasets"],
    }
    target = mount / "review"
    task = products.run_products(
        {
            "products_operation": "review",
            "products_manifest": inputs,
        },
        target,
        str(uuid4()),
        mount,
    )
    assert task["source_kind"] == "import" and task["case_count"] == 1
    index = json.loads((target / "evaluation_cases_manifest.json").read_text())
    cases = json.loads((target / index["chunks"][0]["name"]).read_text())["items"]
    assert cases[0]["input"]["prompt"] == "inspect this image"
    assert cases[0]["detail"]["details"]["distribution_context"]["train"] == {
        "count": 1,
        "ratio": 1 / 3,
    }
    html = (target / "evaluation_report.html").read_text()
    assert '"metrics_available": false' in html
    assert '"pass_rate": null' in html
    assert "false alarm" in (target / "predictions.jsonl").read_text()
    assert (
        '"human_annotation": null'
        in (target / "evaluation_annotations.r0.jsonl").read_text()
    )


def test_rescoring_changes_badcases_without_inference_or_old_annotation_reuse(
    saved_batch,
    monkeypatch,
):
    worker, manifest, spec, script, run_id, mount, output, _ = saved_batch
    evaluate_inference = importlib.import_module("evaluate_inference")
    products = importlib.import_module("evaluation_products")

    def forbidden(*args, **kwargs):
        pytest.fail("rescoring must not call a model")

    monkeypatch.setattr(evaluate_inference, "_call_api", forbidden)
    worker.evaluate_batch(manifest, spec, script, output, run_id, mount)
    source_bytes = (mount / "training/eval_step_20.jsonl").read_bytes()
    first = mount / "review-old"
    products.run_products(
        {
            "products_operation": "review",
            "products_manifest": review_manifest(worker, manifest, output, mount),
        },
        first,
        str(uuid4()),
        mount,
    )
    old_annotations = (first / "evaluation_annotations.r0.jsonl").read_bytes()
    script.write_text(SCRIPT.replace('"defect": "pass"', '"defect": "fail"'))
    spec["business_rule"] = "defect now belongs to fail"
    spec["validation_examples"] = [
        {"record": {"id": "ok", "gt": "pass", "answer": "pass"}, "passed": True},
        {"record": {"id": "bad", "gt": "fail", "answer": "pass"}, "passed": False},
    ]
    second_id = str(uuid4())
    second = mount / "score-new"
    batch = worker.evaluate_batch(manifest, spec, script, second, second_id, mount)
    assert batch["steps"][0]["metrics"]["pass_rate"] == pytest.approx(1 / 3, abs=1e-6)
    new_review = mount / "review-new"
    task = products.run_products(
        {
            "products_operation": "review",
            "products_manifest": review_manifest(worker, manifest, second, mount),
        },
        new_review,
        str(uuid4()),
        mount,
    )
    assert task["case_count"] == 2
    assert (first / "evaluation_annotations.r0.jsonl").read_bytes() == old_annotations
    assert (mount / "training/eval_step_20.jsonl").read_bytes() == source_bytes


def test_report_failure_does_not_invalidate_saved_scores(saved_batch, monkeypatch):
    worker, manifest, spec, script, run_id, mount, output, _ = saved_batch
    evaluate_inference = importlib.import_module("evaluate_inference")

    def broken(*args, **kwargs):
        raise OSError("renderer unavailable")

    monkeypatch.setattr(evaluate_inference, "_build_html", broken)
    batch = worker.evaluate_batch(manifest, spec, script, output, run_id, mount)
    assert batch["steps"][0]["eligible"]
    report = json.loads((output / "steps/20/report.json").read_text())
    assert report["report_status"] == "failed"
    assert report["status"] == "succeeded"


@pytest.mark.parametrize("inference_error", [False, True])
def test_independent_test_infers_then_uses_validation_scoring_script(
    saved_batch,
    monkeypatch,
    inference_error,
):
    worker, _, spec, _, _, mount, _, _ = saved_batch
    evaluate_inference = importlib.import_module("evaluate_inference")

    dataset = mount / "test.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "independent-case",
                "prompt": "inspect image",
                "gt": "pass",
            }
        )
        + "\n"
    )
    calls = []

    def predict(*args):
        calls.append(args)
        if inference_error:
            raise RuntimeError("model unavailable")
        return {"choices": [{"message": {"content": "defect"}}]}, 0.1

    monkeypatch.setattr(evaluate_inference, "_call_api", predict)
    config = {
        "mode": "agent_rubric",
        "rubrics": [
            {
                "name": "exact",
                "criterion": "GT equality",
                "weight": 1,
                "required": True,
                "evaluator": {"type": "exact_match"},
            }
        ],
    }
    with (
        pytest.raises(RuntimeError, match="no successfully scored predictions")
        if inference_error
        else nullcontext()
    ):
        evaluate_inference.evaluate_inference(
            endpoint="http://unused",
            model="default",
            model_reference="/model/checkpoint-100",
            dataset_path=str(dataset),
            output_dir=str(mount / "test-result"),
            dataset_config_json=json.dumps(
                {
                    "reference_field": "gt",
                    "user_prompt_field": "prompt",
                }
            ),
            evaluation_config_json=json.dumps(config),
            run_id=str(uuid4()),
            max_retries=0,
            business_scoring={
                "script": SCRIPT,
                "spec": spec,
                "selection_run_id": "selected",
                "test_source": {
                    "path": "/test.jsonl",
                    "sha256": worker._file_digest(dataset),
                },
            },
        )
    result = json.loads((mount / "test-result/report.json").read_text())
    assert len(calls) == 1
    assert result["metrics"]["pass_rate"] == (0 if inference_error else 1)
    assert result["metrics"]["dataset_role"] == "test"
    assert result["metrics"]["classification"]["per_class"]["pass"]["support"] == 1
    assert result["evaluation_status"] == (
        "invalid" if inference_error else "validated"
    )
    binding = json.loads((mount / "test-result/run_manifest.json").read_text())
    assert binding["evaluation_config_sha256"] == worker._config_digest(spec)
    assert not (mount / "test-result/evaluation_annotations.r0.jsonl").exists()


def test_distribution_failure_preserves_valid_validation_selection(saved_batch):
    worker, manifest, spec, script, run_id, mount, output, _ = saved_batch
    (mount / "training/train.jsonl").write_text("broken JSON\n")
    manifest["datasets"]["train"]["sha256"] = worker._file_digest(
        mount / "training/train.jsonl"
    )
    result = worker.evaluate_batch(manifest, spec, script, output, run_id, mount)
    assert any(row["eligible"] for row in result["steps"])
    profile = json.loads((output / "dataset_profile.json").read_text())
    assert profile["status"] == "incomplete"
    assert profile["distribution"]["validation"]["total"] == 3
    assert any("train_dataset_profile_error" in issue for issue in profile["issues"])
    products = importlib.import_module("evaluation_products")

    inputs = review_manifest(worker, manifest, output, mount)
    with pytest.raises(ValueError, match="complete dataset distribution"):
        products.run_products(
            {"products_operation": "review", "products_manifest": inputs},
            mount / "blocked-review",
            str(uuid4()),
            mount,
        )


def test_review_runtime_and_api_reuse_annotation_protocol_and_refresh_context(
    saved_batch,
    tmp_path,
):
    from unittest.mock import MagicMock

    products = importlib.import_module("evaluation_products")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pyromind_agent_server.api.evaluation_runs import (
        create_evaluation_runs_router,
        get_evaluation_storage,
    )
    from pyromind_runtime.application.conversation_runtime import ConversationRuntime

    worker, manifest, spec, script, run_id, mount, output, output_ref = saved_batch
    worker.evaluate_batch(manifest, spec, script, output, run_id, mount)
    conversations = tmp_path / "conversations"
    root = conversations / "test"
    root.mkdir(parents=True)
    store = FileProductStore(root)
    store.create(
        ConversationSnapshot(
            conversation_id="test",
            capabilities=HarnessCapabilities(),
        ),
        user_id="user",
    )
    store.save_pipeline_run(
        PipelineRun(
            run_id=run_id,
            conversation_id="test",
            definition={
                "kind": "prediction_evaluation",
                "source_manifest": manifest,
                "scoring_spec": spec,
                "scoring_script": SCRIPT,
            },
            fingerprint="score",
            output_dir=output_ref,
            submission_status="submitted",
        ),
        expected_revision=None,
    )
    storage = DiskStorage(mount)
    TrainingEvaluations(store, storage).finalize(run_id)
    inputs = review_manifest(worker, manifest, output, mount)
    review_id = str(uuid4())
    definition = {
        "kind": "badcase_review",
        "products_operation": "review",
        "products_manifest": inputs,
    }
    target = mount / review_id
    products.run_products(definition, target, review_id, mount)
    store.save_pipeline_run(
        PipelineRun(
            run_id=UUID(review_id),
            conversation_id="test",
            definition=definition,
            fingerprint="review",
            output_dir="/" + review_id,
            submission_status="submitted",
        ),
        expected_revision=None,
    )
    app = FastAPI()
    app.state.product_runtime = ConversationRuntime(conversations, MagicMock())
    app.include_router(create_evaluation_runs_router())
    app.dependency_overrides[get_evaluation_storage] = lambda: storage
    client = TestClient(app)
    url = f"/api/v2/pyromind/conversations/test/review-tasks/{review_id}"
    assert client.get(url).json()["review_id"] == review_id
    assert client.get(url + "/report").status_code == 200
    state = client.get(url + "/annotations").json()
    assert state["unreviewed"] == 1 and state["revision"] == 0
    page = client.get(url + "/cases").json()
    assert page["items"][0]["id"] == "b"
    report_before = (output / "evaluation_report.html").read_bytes()
    response = client.patch(
        url + "/annotations",
        json={
            "save_id": str(uuid4()),
            "base_revision": 0,
            "changes": [
                {
                    "case_id": "b",
                    "human_annotation": {
                        "decision": "confirmed_badcase",
                        "disposition": "needs_analysis",
                        "failure_modes": ["model_miss"],
                        "severity": "P1",
                        "comment": "possible coverage gap; not an established cause",
                        "updated_at": "2026-10-22T00:00:00+00:00",
                    },
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    context = json.loads((output / "iteration_context.json").read_text())
    assert context["human_review"]["by_task"][review_id]["coverage"] == 1
    assert (output / "evaluation_report.html").read_bytes() == report_before
    # Completion callback replay preserves feedback and the original task.
    TrainingEvaluations(store, storage).finalize(run_id)
    context = json.loads((output / "iteration_context.json").read_text())
    assert context["human_review"]["by_task"][review_id]["coverage"] == 1
    assert client.get(url.replace("/test/", "/other/")).status_code == 404


def test_score_every_step_reuse_predictions_select_saved_model_and_register_reports(
    saved_batch, tmp_path
):
    worker, manifest, spec, script, run_id, mount, output, output_ref = saved_batch
    batch = worker.evaluate_batch(manifest, spec, script, output, run_id, mount)
    assert [row["step"] for row in batch["steps"]] == [20, 100, 200]
    assert all(row["metrics"]["request_count"] == 0 for row in batch["steps"])
    assert batch["steps"][1]["metrics"]["pass_rate"] == pytest.approx(2 / 3)
    assert (
        batch["steps"][1]["metrics"]["classification"]["per_class"]["pass"]["recall"]
        == 0.5
    )
    assert batch["steps"][2]["checkpoint"] is None
    assert batch["dataset_distribution"]["train"]["labels"]["defect"]["count"] == 1
    (tmp_path / "conversation").mkdir()
    store = FileProductStore(tmp_path / "conversation")
    store.create(
        ConversationSnapshot(
            conversation_id="test", capabilities=HarnessCapabilities()
        ),
        user_id="user",
    )
    parent = PipelineRun(
        run_id=run_id,
        conversation_id="test",
        definition={
            "kind": "prediction_evaluation",
            "source_manifest": manifest,
            "scoring_spec": spec,
            "scoring_script": SCRIPT,
        },
        fingerprint="frozen",
        output_dir=output_ref,
        submission_status="submitted",
    )
    store.save_pipeline_run(parent, expected_revision=None)
    service = TrainingEvaluations(store, DiskStorage(mount))
    result = service.finalize(run_id)
    # 20 and 100 tie on pass rate; the declared default favors the earlier step.
    assert result["selected_step"] == 20
    assert result["best_evaluated_step"] == 200
    assert result["meets_target"] is False
    assert result["model"]["kind"] == "lora"
    assert len(store.load_pipeline_runs()) == 4
    assert service.finalize(run_id) == result
    assert "Checkpoint evaluation" in (output / "evaluation_report.html").read_text()
    assert not (output / "steps/100/evaluation_annotations.r0.jsonl").exists()
    assert (
        "distribution_context" in (output / "steps/100/predictions.jsonl").read_text()
    )
    assert (
        json.loads((output / "iteration_context.json").read_text())["human_review"][
            "status"
        ]
        == "unreviewed"
    )


def test_source_changes_and_unknown_generation_never_silently_compare(saved_batch):
    worker, manifest, spec, script, run_id, mount, output, _ = saved_batch
    (mount / "training/eval_step_20.jsonl").write_text("{}\n")
    manifest["evaluations"][1]["generation"] = {}
    result = worker.evaluate_batch(manifest, spec, script, output, run_id, mount)
    assert not result["steps"][0]["eligible"]
    assert (output / "steps/20/evaluation_report.html").exists()
    assert "generation_configuration_unverified" in result["steps"][1]["issues"]
    assert result["steps"][2]["eligible"]


def test_scoring_contract_must_pass_examples_before_full_run(saved_batch):
    worker, manifest, spec, script, run_id, mount, output, _ = saved_batch
    spec["validation_examples"][0]["passed"] = False
    with pytest.raises(ValueError, match="passing and failing"):
        worker.evaluate_batch(manifest, spec, script, output, run_id, mount)
    assert not (output / "batch_result.json").exists()


def test_different_business_rule_creates_new_run_reuses_original_predictions(
    saved_batch,
):
    worker, manifest, spec, script, run_id, mount, output, _ = saved_batch
    worker.evaluate_batch(manifest, spec, script, output, run_id, mount)
    spec["business_rule"] = "updated rule description"
    with pytest.raises(ValueError, match="frozen inputs changed"):
        worker.evaluate_batch(manifest, spec, script, output, run_id, mount)
    second = str(uuid4())
    result = worker.evaluate_batch(
        manifest, spec, script, output.parent / second, second, mount
    )
    assert all(row["metrics"]["request_count"] == 0 for row in result["steps"])


@pytest.mark.parametrize(
    "damage", ["duplicate", "missing_gt", "missing_sample", "bad_line"]
)
def test_incomplete_step_has_diagnostics_and_no_review_initialization(
    saved_batch, damage
):
    worker, manifest, spec, script, run_id, mount, output, _ = saved_batch
    path = mount / "training/eval_step_20.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if damage == "duplicate":
        rows.append(rows[0])
    elif damage == "missing_gt":
        rows[0].pop("target")
    elif damage == "missing_sample":
        rows.pop()
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(payload + ("{bad\n" if damage == "bad_line" else ""))
    manifest["evaluations"][0]["sha256"] = worker._file_digest(path)
    batch = worker.evaluate_batch(manifest, spec, script, output, run_id, mount)
    assert not batch["steps"][0]["eligible"]
    assert batch["steps"][1]["eligible"]
    assert (
        json.loads((output / "steps/20/report.json").read_text())["evaluation_status"]
        == "invalid"
    )
    assert not (output / "steps/20/evaluation_annotations.r0.jsonl").exists()


def test_empty_predictions_are_failures_and_remain_reviewable(saved_batch):
    worker, manifest, spec, script, run_id, mount, output, _ = saved_batch
    path = mount / "training/eval_step_20.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["prediction"] = ""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest["evaluations"][0]["sha256"] = worker._file_digest(path)
    batch = worker.evaluate_batch(manifest, spec, script, output, run_id, mount)
    assert batch["steps"][0]["eligible"]
    assert batch["steps"][0]["metrics"]["total"] == 3
    assert batch["steps"][0]["metrics"]["pass_rate"] == 0
    assert not (output / "steps/20/evaluation_annotations.r0.jsonl").exists()


def test_real_agent_replay_transport_smoke(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from openhands.tools.data_preparation.platform_submit import DfSubmitPipelineAction
    from tests.integration.tests import b09_saved_prediction_flywheel as replay

    test = replay.SavedPredictionFlywheelTest.__new__(
        replay.SavedPredictionFlywheelTest
    )
    test.workspace = str(tmp_path)
    test.agent = cast(Any, SimpleNamespace(tools=[], model_copy=Mock()))
    monkeypatch.setattr(replay, "LocalConversation", Mock())
    test.setup()
    artifact = replay.ReplayTraining(test)(
        replay.TrainingAnalysisAction(operation="artifacts", task_id="1051")
    )
    manifest = artifact.result
    manifest["datasets"] = {
        name: {
            "path": path,
            "sha256": replay.hashlib.sha256(
                (test.mount / path.lstrip("/")).read_bytes()
            ).hexdigest(),
        }
        for name, path in manifest["datasets"].items()
    }
    spec = {
        "business_rule": "pass/defect accepted, fail rejected",
        "selection": {
            "constraints": [{"metric": "pass_rate", "operator": "ge", "value": 0.95}]
        },
        "validation_examples": [
            {
                "record": {"sample_key": "a", "label": "pass", "completion": "defect"},
                "passed": True,
            },
            {
                "record": {"sample_key": "b", "label": "fail", "completion": "defect"},
                "passed": False,
            },
        ],
    }
    script = """import json
def parse_record(raw, context):
    row = raw.get("case", raw)
    prediction = (json.loads(raw["model_response"])["verdict"]
                  if "model_response" in raw else raw.get("completion"))
    return {"id": row["sample_key"], "reference": row["label"],
            "original_label": row["label"], "prediction": prediction}
def score_case(case, spec):
    def label(v):
        if v in ("pass", "defect"): return "accepted"
        return "rejected" if v == "fail" else None
    gt, pred = label(case["reference"]), label(case["prediction"])
    return {"passed": gt == pred, "reference_label": gt, "prediction_label": pred}
"""
    for name, value in [("sources.json", manifest), ("spec.json", spec)]:
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "scorer.py").write_text(script)
    result = replay.ReplaySubmit(test)(
        DfSubmitPipelineAction.model_validate(
            {
                "input_path": "/training/1051",
                "evaluation": {
                    "source_manifest_path": "sources.json",
                    "scoring_spec_path": "spec.json",
                    "scoring_script_path": "scorer.py",
                },
            }
        )
    )
    assert not result.is_error, result.text
    assert test.selections[0]["selected_step"] == 100
    assert test.selections[0]["best_evaluated_step"] == 200
    assert test.selections[0]["meets_target"] is False


def test_human_feedback_updates_iteration_context_without_replacing_machine_scores(
    saved_batch, tmp_path
):
    from datetime import UTC, datetime

    from pyromind_runtime.application.conversation_runtime import ConversationRuntime
    from pyromind_runtime.domain.context import RequestContext
    from pyromind_runtime.domain.evaluation import EvaluationCommitState

    from tests.pyromind_runtime.fake_adapter import FakeAdapter

    worker, manifest, spec, script, run_id, mount, output, output_ref = saved_batch
    batch = worker.evaluate_batch(manifest, spec, script, output, run_id, mount)
    (tmp_path / "test").mkdir()
    store = FileProductStore(tmp_path / "test")
    store.create(
        ConversationSnapshot(
            conversation_id="test", capabilities=HarnessCapabilities()
        ),
        user_id="owner",
    )
    store.save_pipeline_run(
        PipelineRun(
            run_id=run_id,
            conversation_id="test",
            definition={
                "kind": "prediction_evaluation",
                "source_manifest": manifest,
                "scoring_spec": spec,
                "scoring_script": SCRIPT,
            },
            fingerprint="frozen",
            output_dir=output_ref,
        ),
        expected_revision=None,
    )
    storage = DiskStorage(mount)
    service = TrainingEvaluations(store, storage)
    original = service.finalize(run_id)
    row = batch["steps"][0]
    artifact = "evaluation_annotations.r1.jsonl"
    storage.write(
        row["output_dir"] + "/" + artifact,
        json.dumps(
            {
                "case_id": "b",
                "review_status": "reviewed",
                "human_annotation": {
                    "decision": "confirmed_badcase",
                    "resolution_code": "needs_more_data",
                    "disposition": "needs_analysis",
                    "failure_modes": ["model_miss"],
                    "severity": "P1",
                    "updated_at": "2026-10-22T00:00:00+00:00",
                },
            }
        ).encode(),
    )
    store.save_evaluation_commit(
        EvaluationCommitState(
            run_id=row["run_id"],
            revision=1,
            annotations_object=artifact,
            labeled_object="labeled_evaluation_results.r1.jsonl",
            updated_at=datetime.now(UTC),
        )
    )
    runtime = ConversationRuntime(tmp_path, FakeAdapter())
    # Storage-authenticated report routes intentionally do not require local ownership.
    runtime._refresh_iteration_review(
        "test", row["run_id"], RequestContext(user_id="anonymous"), storage
    )
    evidence = json.loads((output / "iteration_context.json").read_text())
    review = evidence["human_review"]["by_run"][row["run_id"]]
    assert review["coverage"] == 1
    assert review["conclusions"] == {"needs_more_data": 1}
    assert review["scope"] == "machine_failed_cases_only"
    assert service.finalize(run_id) == original
    assert store.load_pipeline_runs()[run_id].result == original
    assert json.loads((output / "steps/20/metrics.json").read_text())[
        "pass_rate"
    ] == pytest.approx(2 / 3)
