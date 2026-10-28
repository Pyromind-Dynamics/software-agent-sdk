"""Exercise frozen platform packaging and the real CPU evaluation worker."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from harness_adapter.checkpoint_evaluation import PlatformCheckpointEvaluation
from pyromind_runtime.application.checkpoint_evaluations import fingerprint
from pyromind_runtime.application.test_evaluation import final_test_definition
from pyromind_runtime.domain.base import utc_now
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.checkpoint_evaluation import (
    CheckpointCandidate,
    CheckpointEvaluationRequest,
    CheckpointEvaluationState,
)
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.domain.training_evaluation import CheckpointArtifact
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.storage import EvaluationStorageMissingError


SKILLS = Path(__file__).resolve().parents[2] / ".agents/skills"


class MountedStorage:
    def __init__(self, root):
        self.root = root

    def read(self, path):
        try:
            return (self.root / path.lstrip("/")).read_bytes()
        except FileNotFoundError as error:
            raise EvaluationStorageMissingError(path) from error

    def write(self, path: str, payload: bytes) -> None:
        file = self.root / path.lstrip("/")
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(payload)

    def list(self, prefix: str) -> set[str]:
        return {file.name for file in (self.root / prefix.lstrip("/")).iterdir()}


@pytest.fixture
def checkpoint_platform(tmp_path, monkeypatch):
    storage = MountedStorage(tmp_path)
    storage.write("/val.jsonl", b'{"id":"case","prompt":"inspect","gt":"pass"}\n')
    storage.write("/train.jsonl", b'{"id":"train","prompt":"train","gt":"fail"}\n')
    storage.write("/test.jsonl", b'{"id":"test","prompt":"test","gt":"pass"}\n')
    platform = PlatformCheckpointEvaluation(SKILLS, tmp_path, lambda context: storage)
    task = SimpleNamespace(status="Succeeded", name="test", error_message=None)
    monkeypatch.setattr(
        platform,
        "client",
        lambda context: SimpleNamespace(
            studio=SimpleNamespace(get_task=lambda task_id: task)
        ),
    )
    monkeypatch.setattr(
        platform,
        "_observer",
        lambda *args: {
            "output_path": "/training",
            "inputs": {"training_config": {"save_total_limit": 300}},
        },
    )
    monkeypatch.setattr(
        "harness_adapter.checkpoint_evaluation.ValidateWorkflowDslExecutor",
        lambda **kwargs: lambda *args: SimpleNamespace(valid=True, is_error=False),
    )
    return platform, storage, task


def request(custom=False):
    template = {
        "input_path": "/val.jsonl",
        "dataset_config": {
            "reference_field": "gt",
            "user_prompt_field": "prompt",
            "id_field": "id",
        },
        "evaluation_config": {
            "mode": "agent_rubric",
            "rubrics": [
                {
                    "name": "verdict",
                    "criterion": "Exact business verdict",
                    "weight": 1,
                    "required": True,
                    "evaluator": {"type": "exact_match"},
                }
            ],
        },
        "inference": {"gpu_product": "NVIDIA-L40S", "gpu_count": 1},
        "datasets": {
            "train": {"path": "/train.jsonl"},
            "test": {"path": "/test.jsonl"},
        },
    }
    if custom:
        template["scoring_script"] = """
def parse_record(raw, context):
    assert context.get('validation_example') or context['source'] == 'validation'
    return {'id': raw['id'], 'reference': raw['gt'],
            'prediction': raw.get('prediction')}
def score_case(case, spec):
    return {'passed': case['reference'] == case['prediction']}
"""
        template["scoring_spec"] = {
            "business_rule": "exact verdict",
            "validation_examples": [
                {
                    "record": {"id": "x", "gt": "pass", "prediction": "pass"},
                    "passed": True,
                },
                {
                    "record": {"id": "y", "gt": "pass", "prediction": "fail"},
                    "passed": False,
                },
            ],
        }
    return CheckpointEvaluationRequest(
        training_task_id="123", training_node_id="train", template=template
    )


def child_run(parent):
    artifact = CheckpointArtifact(
        step=100, path="/training/checkpoint-100", kind="full", complete=True
    )
    template = parent.definition["template"]
    definition = {
        **template,
        "inference": {**template["inference"], "model_path": artifact.path},
        "evaluation_config": {
            **template["evaluation_config"],
            "model_source": {"type": "checkpoint", "kind": "full", "base_model": None},
        },
    }
    identity = uuid4()
    child = PipelineRun(
        run_id=identity,
        conversation_id=parent.conversation_id,
        definition=definition,
        fingerprint=fingerprint(definition),
        output_dir=f"/outputs/{identity}",
    )
    return child, artifact


@pytest.mark.parametrize("custom", [False, True])
async def test_frozen_validation_inference_and_custom_scoring_produce_eligible_steps(
    checkpoint_platform, monkeypatch, custom
):
    platform, storage, task = checkpoint_platform
    context = RequestContext(user_id="owner", cookie="not-persisted")
    definition = await platform.freeze(request(custom), "/selection", context)
    parent = PipelineRun(
        run_id=uuid4(),
        conversation_id="a" * 32,
        definition=definition,
        fingerprint="selection",
        output_dir="/selection",
    )
    child, artifact = child_run(parent)
    output = (
        f"/.pyromind-agent/{parent.conversation_id}/data_preparation/{child.run_id}"
    )
    child = child.model_copy(update={"output_dir": output})
    workflow = await platform.prepare(parent, child, context)
    assert "nodes" in workflow
    assert "/training/checkpoint-100" in storage.read(output + "/workflow.py").decode()
    runtime = SKILLS / "inference-evaluation/scripts"
    monkeypatch.syspath_prepend(str(runtime))
    spec = importlib.util.spec_from_file_location(
        "checkpoint_cpu_worker", runtime / "evaluate_inference.py"
    )
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    monkeypatch.setattr(
        worker,
        "_call_api",
        lambda *args, **kwargs: ({"choices": [{"message": {"content": "pass"}}]}, 0.01),
    )
    template = child.definition
    worker.evaluate_inference(
        endpoint="http://unused",
        model="default",
        model_reference=artifact.path,
        dataset_path=str(storage.root / str(template["input_path"]).lstrip("/")),
        output_dir=str(storage.root / output.lstrip("/")),
        dataset_config_json=json.dumps(template["dataset_config"]),
        evaluation_config_json=json.dumps(template["evaluation_config"]),
        run_id=str(child.run_id),
        report_assets_dir=str(runtime),
        max_retries=0,
        business_scoring=template.get("business_scoring"),
        dataset_provenance=template["dataset_provenance"],
    )
    candidate = CheckpointCandidate(
        artifact=artifact, run_id=child.run_id, task_id="456"
    )
    result = await platform.inspect(parent, candidate, context)
    assert result.evaluation is not None
    assert result.evaluation.eligible, result.evaluation.issues
    assert result.evaluation.dataset_role == "validation"
    assert result.evaluation.metrics["pass_rate"] == 1
    # The original JSONL can change without changing the frozen validation input.
    storage.write("/val.jsonl", b"changed")
    assert (
        hashlib.sha256(storage.read(str(template["input_path"]))).hexdigest()
        == definition["datasets"]["validation"]["sha256"]
    )
    metrics_path = output + "/metrics.json"
    metrics = json.loads(storage.read(metrics_path))
    metrics["evaluated_cases"] = 0
    storage.write(metrics_path, json.dumps(metrics).encode())
    rejected = await platform.inspect(parent, candidate, context)
    assert rejected.evaluation is not None and not rejected.evaluation.eligible


def test_checkpoint_observer_never_uses_wandb_log_index_as_training_step(
    tmp_path, monkeypatch
):
    scripts = SKILLS / "training-analysis/scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "checkpoint_observer_test", scripts / "checkpoint_observer.py"
    )
    assert spec is not None and spec.loader is not None
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    source = SimpleNamespace(
        extract_creds=lambda *args: {"key": "ephemeral"},
        connect=lambda *args: None,
        get_run_data=lambda *args, **kwargs: {
            "state": "running",
            "summary": {"_step": 999, "train/loss": 0.5},
        },
    )
    monkeypatch.setattr(observer, "WandbDataSource", lambda: source)
    client = SimpleNamespace(
        task_workflow_result=lambda *args: {},
        node_log=lambda *args: "https://wandb.ai/entity/project/runs/abc123",
    )
    assert observer.telemetry(client, "123", "train")["training_step"] is None
    checkpoint = tmp_path / "training/checkpoint-100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "part1.safetensors", "b": "part2.safetensors"}})
    )
    (checkpoint / "part1.safetensors").write_bytes(b"partial")
    first = observer.observe_files(tmp_path, "/training", None)[0]
    assert not first["artifact"]["complete"] and first["fingerprint"] is None
    (checkpoint / "part2.safetensors").write_bytes(b"complete")
    ready = observer.observe_files(tmp_path, "/training", None)[0]
    assert ready["artifact"]["complete"] and ready["fingerprint"]


@pytest.mark.parametrize("custom", [False, True])
async def test_final_test_accepts_completed_continuous_selection_only(
    checkpoint_platform, custom
):
    platform, storage, task = checkpoint_platform
    context = RequestContext(user_id="owner", cookie="ephemeral")
    definition = await platform.freeze(request(custom), "/selection", context)
    definition["kind"] = "checkpoint_evaluation"
    model = CheckpointArtifact(
        step=100, path="/training/checkpoint-100", kind="full", complete=True
    )
    state = CheckpointEvaluationState(status="completed", next_poll_at=utc_now())
    parent = PipelineRun(
        run_id=uuid4(),
        conversation_id="a" * 32,
        definition=definition,
        result={
            "model": model.model_dump(mode="json"),
            "checkpoint_evaluation": state.model_dump(mode="json"),
        },
        fingerprint="frozen",
        output_dir="/selection",
    )
    candidate, artifact = child_run(parent)
    inference = candidate.definition["inference"]
    assert isinstance(inference, dict)
    test_definition = {
        **candidate.definition,
        "input_path": "/test.jsonl",
        "inference": {
            **inference,
            "selection_run_id": str(parent.run_id),
        },
    }
    root = storage.root / "conversation"
    root.mkdir()
    store = FileProductStore(root)
    store.create(
        ConversationSnapshot(
            conversation_id=parent.conversation_id, capabilities=HarnessCapabilities()
        ),
        user_id="owner",
    )
    store.save_pipeline_run(parent, expected_revision=None)
    result = final_test_definition(store, parent.conversation_id, test_definition)
    assert result["dataset_role"] == "test"
    assert (
        result["dataset_provenance"]["sha256"]
        == definition["datasets"]["test"]["sha256"]
    )
    if custom:
        assert result["business_scoring"]["dataset_role"] == "test"
        assert (
            result["business_scoring"]["script"]
            == definition["template"]["business_scoring"]["script"]
        )
    state.status = "watching"
    parent = parent.model_copy(
        update={
            "result": {
                "model": model.model_dump(mode="json"),
                "checkpoint_evaluation": state.model_dump(mode="json"),
            }
        }
    )
    parent = parent.model_copy(update={"revision": parent.revision + 1})
    store.save_pipeline_run(parent, expected_revision=parent.revision - 1)
    with pytest.raises(ValueError, match="finish"):
        final_test_definition(store, parent.conversation_id, test_definition)
