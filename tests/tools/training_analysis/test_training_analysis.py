import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError

from openhands.sdk.conversation.impl.local_conversation import LocalConversation
from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.tools.training_analysis import (
    TrainingAnalysisAction,
    TrainingAnalysisExecutor,
    TrainingAnalysisTool,
    _worker_environment,
)


_SKILL_SCRIPTS = (
    Path(__file__).resolve().parents[3] / ".agents/skills/training-analysis/scripts"
)
sys.path.insert(0, str(_SKILL_SCRIPTS))
TrainingAnalysisService = importlib.import_module(
    "train_analysis"
).TrainingAnalysisService


def _conversation(workspace: Path) -> LocalConversation:
    return cast(
        LocalConversation,
        SimpleNamespace(
            workspace=SimpleNamespace(working_dir=str(workspace)),
            state=SimpleNamespace(
                agent_state={},
                secret_registry=SecretRegistry(),
            ),
        ),
    )


def _remote_conversation(workspace: object) -> LocalConversation:
    return cast(
        LocalConversation,
        SimpleNamespace(
            workspace=workspace,
            state=SimpleNamespace(
                agent_state={},
                secret_registry=SecretRegistry(),
            ),
        ),
    )


def test_training_analysis_publishes_report_into_remote_workspace(
    tmp_path: Path,
    sandbox_workspace,
) -> None:
    """A sandbox session runs the worker in staging and publishes the report back."""
    worker = tmp_path / "training_analysis_worker.py"
    worker.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "payload = json.load(sys.stdin)\n"
        "report = Path(payload['output_path'])\n"
        "report.parent.mkdir(parents=True, exist_ok=True)\n"
        "report.write_text('report body', encoding='utf-8')\n"
        "print(json.dumps({'ok': True, 'target': {}, 'result': {}, "
        "'report_path': payload['output_relative']}))\n",
        encoding="utf-8",
    )
    conversation = _remote_conversation(sandbox_workspace)
    executor = TrainingAnalysisExecutor(runtime_dir=str(tmp_path))

    observation = executor(
        TrainingAnalysisAction(operation="report", task_id="1050"),
        conversation,
    )

    assert not observation.is_error, observation.text
    assert observation.report_path == "public_data/training-analysis/1050/report.md"
    published = sandbox_workspace.workspace_dir / observation.report_path
    assert published.read_text(encoding="utf-8") == "report body"


@pytest.mark.parametrize("remote", [False, True], ids=["local", "sandbox-echo"])
@pytest.mark.parametrize("available", [False, True], ids=["missing", "available"])
def test_training_analysis_discovers_artifacts_in_execution_workspace(
    tmp_path: Path, sandbox_workspace, truncated_downloads, remote, available
) -> None:
    workspace = sandbox_workspace.workspace_dir
    (workspace / "storage").symlink_to(
        sandbox_workspace.storage_dir, target_is_directory=True
    )
    sample = {"prediction": "正常", "label": "正常"}
    if available:
        output = sandbox_workspace.storage_dir / "runs/training"
        checkpoint = output / "checkpoint-20"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}")
        (checkpoint / "model.safetensors").write_bytes(b"weights")
        (output / "eval_step_20.jsonl").write_text(json.dumps(sample) + "\n")
    response = {
        "ok": True,
        "target": {"task_id": "9175"},
        "result": {
            "training_nodes": [
                {"output_path": "/runs/training", "inputs": {}, "issues": []}
            ]
        },
    }
    (tmp_path / "training_analysis_worker.py").write_text(
        "import json\nprint(json.dumps(" + repr(response) + "))\n"
    )
    (tmp_path / "training_artifacts.py").write_text(
        (_SKILL_SCRIPTS / "training_artifacts.py").read_text()
    )
    conversation = (
        _remote_conversation(
            truncated_downloads(sandbox_workspace, limit=1_000_000, echo_commands=True)
        )
        if remote
        else _conversation(workspace)
    )

    observation = TrainingAnalysisExecutor(runtime_dir=str(tmp_path))(
        TrainingAnalysisAction(operation="artifacts", task_id="9175"),
        conversation,
    )

    assert not observation.is_error, observation.text
    assert json.loads(observation.text) == observation.result
    node = observation.result["training_nodes"][0]
    if available:
        assert node["issues"] == []
        assert node["checkpoints"][0]["step"] == 20
        assert node["checkpoints"][0]["complete"]
        assert node["evaluations"][0]["samples"] == [sample]
    else:
        assert len(node["issues"]) == 1
        assert "Training output directory is unavailable" in node["issues"][0]


def test_training_analysis_schema_requires_a_target() -> None:
    with pytest.raises(ValidationError):
        TrainingAnalysisAction.model_validate({"operation": "probe"})
    with pytest.raises(ValidationError):
        TrainingAnalysisAction.model_validate(
            {"operation": "probe", "run_url": "https://wandb.ai/e/p/runs/r1"}
        )
    schema = TrainingAnalysisTool.create()[0].to_mcp_tool()["inputSchema"]
    assert set(schema["properties"]) == {
        "operation",
        "training_node_id",
        "task_id",
        "run_url",
        "data_source",
        "metric",
        "keys",
        "output_path",
    }
    assert "api_key" not in schema["properties"]
    assert schema["required"] == ["operation", "task_id"]


def test_training_analysis_worker_uses_stdin_and_redacts_secret(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "training_analysis_worker.py"
    worker.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "secret = payload['headers']['cookie']\n"
        "print(json.dumps({'ok': True, 'target': {'task_id': payload['task_id']}, "
        "'result': {'nested': {'plain': secret, 'cookie': secret}}}))\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    conversation = _conversation(workspace)
    secret = "cookie-value-that-must-not-leak"
    conversation.state.secret_registry.update_secrets({"COOKIE_SECRET": secret})
    executor = TrainingAnalysisExecutor(
        runtime_dir=str(tmp_path),
        secret_headers={"cookie": "COOKIE_SECRET"},
    )

    observation = executor(
        TrainingAnalysisAction(operation="analyze", task_id="1050"),
        conversation,
    )

    assert not observation.is_error
    assert secret not in observation.text
    assert observation.result["nested"]["cookie"] == "[REDACTED]"


def test_training_analysis_tool_resolves_api_base_precedence(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("PYROMIND_API_BASE", "https://env.example/")
    env_executor = TrainingAnalysisTool.create()[0].executor
    assert isinstance(env_executor, TrainingAnalysisExecutor)
    assert env_executor.api_base == "https://env.example/"

    configured_executor = TrainingAnalysisTool.create(
        api_base="https://configured.example/"
    )[0].executor
    assert isinstance(configured_executor, TrainingAnalysisExecutor)
    assert configured_executor.api_base == "https://configured.example/"


def test_training_analysis_worker_environment_drops_auth(monkeypatch) -> None:
    monkeypatch.setenv("PYROMIND_VALIDATE_AUTH_COOKIE", "cookie")
    monkeypatch.setenv("PYROMIND_X_CLUSTER", "cluster")
    monkeypatch.setenv("PYROMIND_VALIDATE_AUTHORIZATION", "authorization")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-key")
    environment = _worker_environment()
    assert "PYROMIND_VALIDATE_AUTH_COOKIE" not in environment
    assert "PYROMIND_X_CLUSTER" not in environment
    assert "PYROMIND_VALIDATE_AUTHORIZATION" not in environment
    assert "WANDB_API_KEY" not in environment


def test_training_analysis_confines_report_output(tmp_path: Path) -> None:
    observation = TrainingAnalysisExecutor(runtime_dir=str(tmp_path))(
        TrainingAnalysisAction(
            operation="report",
            task_id="1050",
            output_path="../../credentials.json",
        ),
        _conversation(tmp_path),
    )

    assert observation.is_error
    assert observation.failure_stage == "input_resolution"
    assert observation.error_code == "invalid_training_target"


def test_training_analysis_rejects_symlinked_report_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "public_data").symlink_to(outside, target_is_directory=True)

    action = TrainingAnalysisAction(operation="report", task_id="1050")
    with pytest.raises(ValueError, match="public_data/training-analysis"):
        TrainingAnalysisExecutor._output_path(action, workspace)


def test_training_analysis_reports_missing_worker(tmp_path: Path) -> None:
    observation = TrainingAnalysisExecutor(runtime_dir=str(tmp_path))(
        TrainingAnalysisAction(operation="analyze", task_id="1050"),
        _conversation(tmp_path),
    )

    assert observation.is_error
    assert observation.failure_stage == "runtime_dependency"
    assert observation.error_code == "training_worker_missing"


@pytest.mark.parametrize("task_id", ["pcb_avi_recheck_sft_4b", "-1", "1.0", "１０５０"])
def test_training_analysis_rejects_invalid_id_before_starting_worker(
    tmp_path: Path, task_id: str
) -> None:
    observation = TrainingAnalysisExecutor(runtime_dir=str(tmp_path))(
        TrainingAnalysisAction(operation="probe", task_id=task_id),
        _conversation(tmp_path),
    )
    assert observation.is_error
    assert observation.error_code == "invalid_platform_task_id"
    assert observation.failure_stage == "input_resolution"
    assert "product_versions" in (observation.error_message or "")


def test_training_worker_rejects_invalid_id_without_platform_request() -> None:
    result = subprocess.run(
        [sys.executable, str(_SKILL_SCRIPTS / "training_analysis_worker.py")],
        input=json.dumps({"operation": "probe", "task_id": "pcb_avi_recheck_sft_4b"}),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    response = json.loads(result.stdout)
    assert result.returncode == 2
    assert response["error_code"] == "invalid_platform_task_id"
    assert response["failure_stage"] == "input_resolution"
    assert "product_versions" in response["error_message"]


def test_training_analysis_report_redacts_nested_analysis(
    monkeypatch, tmp_path: Path
) -> None:
    secret = "training-report-secret"
    creds_file = tmp_path / "creds.json"
    creds_file.write_text(json.dumps({"api_key": secret}), encoding="utf-8")

    def fake_analyze(self, **_kwargs):
        return {
            "run": "run-1",
            "display_name": f"api_key={secret}",
            "state": "finished",
            "config": {"nested": {"authorization": secret, "plain": secret}},
            "summary": {},
            "metric": "loss",
            "metric_stats": {"count": 1},
            "diagnostics": {
                "error": f"authorization: {secret}",
                "nested": {"secret": secret},
            },
        }

    monkeypatch.setattr(TrainingAnalysisService, "analyze_run", fake_analyze)
    report = TrainingAnalysisService().report(
        creds_file=creds_file,
        data_source="wandb",
        entity_project="entity/project",
        run_id="run-1",
        metric="loss",
        output_path=None,
    )

    assert secret not in report
    assert "[REDACTED]" in report


@pytest.mark.parametrize("kind", ["ModelTrainSFTNode", "ModelTrainGRPONode"])
def test_artifacts_resolve_executed_inputs_and_multi_node_selection(kind):
    resolve_artifacts = importlib.import_module("training_artifacts").resolve_artifacts

    class Client:
        def task_workflow_result(self, task_id):
            assert task_id == "1050"
            return {
                "data": {
                    "workflow": {
                        "nodes": [
                            {
                                "id": "train",
                                "data": {
                                    "nodeType": kind,
                                    "resolved_inputs": {
                                        "output_path": {
                                            "node_id": "path",
                                            "output": "path",
                                        },
                                        "model_config": "model_path: /models/base",
                                        "dataset_config": (
                                            "train_data_path: /datasets/train.jsonl"
                                        ),
                                    },
                                },
                            },
                            {"id": "other", "data": {"nodeType": kind, "config": {}}},
                        ]
                    }
                }
            }

        def node_output(self, node_id, task_id):
            return (
                {"data": {"path": "/workspace/models/run"}} if node_id == "path" else {}
            )

    result = resolve_artifacts(Client(), "1050")
    assert result["requires_node_selection"]
    assert result["training_nodes"][0]["output_path"] == "/models/run"
    assert result["training_nodes"][0]["inputs"]["model_path"] == "/models/base"
    assert result["training_nodes"][1]["output_path"] is None
    chosen = resolve_artifacts(Client(), "1050", "train")
    assert not chosen["requires_node_selection"]
    assert len(chosen["training_nodes"]) == 1
    with pytest.raises(ValueError, match="No matching"):
        resolve_artifacts(Client(), "1050", "absent")


def test_discover_sorts_numeric_steps_and_checks_lora_and_shards(tmp_path):
    discover = importlib.import_module("training_artifacts").discover

    root = tmp_path / "run"
    root.mkdir()
    for step in (100, 20, 300):
        cp = root / f"checkpoint-{step}"
        cp.mkdir()
        if step == 20:
            (cp / "adapter_config.json").write_text("{}")
            (cp / "adapter_model.safetensors").write_bytes(b"weights")
        else:
            (cp / "config.json").write_text("{}")
            (cp / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": {"a": "part1.safetensors", "b": "part2.safetensors"}}
                )
            )
            (cp / "part1.safetensors").write_bytes(b"weights")
            if step == 100:
                (cp / "part2.safetensors").write_bytes(b"weights")
    (root / "eval_step_20.jsonl").write_text('{"custom_gt":"pass"}\n{bad\n')
    result = discover("/workspace/run", str(tmp_path))
    assert [cp["step"] for cp in result["checkpoints"]] == [20, 100, 300]
    assert [cp["complete"] for cp in result["checkpoints"]] == [False, True, False]
    assert "missing_base_model" in result["checkpoints"][0]["issues"]
    assert result["evaluations"][0]["samples"][0] == {"custom_gt": "pass"}
    assert result["evaluations"][0]["errors"]


def test_sparse_metric_queries_keep_actual_steps_and_distinguish_missing_nan():
    helpers = importlib.import_module("analysis_helpers")
    diagnose_run, metric_series = helpers.diagnose_run, helpers.metric_series
    WandbDataSource = importlib.import_module("data_sources.wandb").WandbDataSource

    history = [
        {"_step": 1, "loss": 0.5, "trainer/global_step": 100},
        {"_step": 2, "val_loss": 0.7, "trainer/global_step": 150},
        {"_step": 3, "loss": 0.2, "trainer/global_step": 200},
        {"_step": 4, "loss": float("nan"), "trainer/global_step": 250},
        {"_step": 5, "loss": 0.4},
    ]
    calls = []

    class Run:
        def scan_history(self, keys, page_size):
            calls.append(keys)
            return [
                {k: row[k] for k in keys}
                for row in history
                if all(k in row for k in keys)
            ]

    result = WandbDataSource().scan_history(Run(), ["loss", "val_loss"])
    assert all(len(keys) <= 2 for keys in calls)
    diagnosis = diagnose_run({"history": result, "summary": {}}, "loss", "val_loss")
    assert diagnosis["min_value_step"] == 200
    assert diagnosis["min_value_log_step"] == 3
    assert diagnosis["observations"] == 4
    assert diagnosis["has_nan"] and diagnosis["step_alignment"] == "unverified"
    assert metric_series(result, "loss")[-1]["step"] is None
    assert "converged" not in diagnosis
    assert not diagnose_run({"history": history[:3], "summary": {}}, "loss")["has_nan"]


def test_artifact_graph_edges_use_upstream_actual_output_in_nested_snapshot():
    resolve_artifacts = importlib.import_module("training_artifacts").resolve_artifacts

    class Client:
        def task_workflow_result(self, _task):
            return {
                "data": {
                    "workflow": {
                        "nodes": [
                            {"id": "train", "data": {"nodeType": "ModelTrainSFTNode"}},
                            {"id": "directory", "data": {"nodeType": "InputNode"}},
                            {
                                "id": "config",
                                "data": {"nodeType": "TrainingConfigBuilderNode"},
                            },
                        ],
                        "edges": [
                            {
                                "source": "directory",
                                "sourceHandle": "path",
                                "target": "train",
                                "targetHandle": "output_path",
                            }
                        ],
                    }
                }
            }

        def node_output(self, source, _task):
            assert source == "directory"
            return {
                "data": {"entries": [{"m": '{"path":"/workspace/training/actual"}'}]}
            }

    result = resolve_artifacts(Client(), "123")
    assert not result["requires_node_selection"]
    assert result["training_nodes"][0]["output_path"] == "/training/actual"


def test_numeric_generation_parameters_survive_both_redaction_layers():
    from openhands.tools.training_analysis import _redact_value

    _sanitized = importlib.import_module("training_analysis_worker")._sanitized
    raw = {
        "generation": {"max_tokens": 512, "temperature": 0},
        "api_key": "secret-value",
        "authorization": "Bearer secret-value",
    }
    value = _redact_value(_sanitized(raw), {})
    assert value["generation"] == {"max_tokens": 512, "temperature": 0}
    assert value["api_key"] == "[REDACTED]"
    assert "secret-value" not in json.dumps(value)


@pytest.fixture
def terminal_node_output():
    def render(values):
        separator = "-" * 90
        lines = ["Output Values:", separator]
        for name, value in values.items():
            if len(f"{name}::  {value}") > 90:
                lines.extend([f"\x1b[1;33m{name}:\x1b[0m", value])
            else:
                lines.append(f"\x1b[1;33m{name}:\x1b[0m:  {value}")
            lines.append(separator)
        return {"entries": [{"m": line + "\n"} for line in lines]}

    return render


@pytest.mark.parametrize("wrapped", [False, True])
def test_artifacts_decode_platform_terminal_output(terminal_node_output, wrapped):
    resolve_artifacts = importlib.import_module("training_artifacts").resolve_artifacts

    class Client:
        def task_workflow_result(self, _task):
            return {
                "nodes": [{"id": "12", "data": {"nodeType": "ModelTrainSFTNode"}}],
                "edges": [
                    {
                        "source": "5",
                        "sourceHandle": "dataset_config",
                        "target": "12",
                        "targetHandle": "dataset_config",
                    }
                ],
            }

        def node_output(self, source, _task):
            values = (
                {
                    "dataset_config": "train_data_path: /workspace/data/train.jsonl\n"
                    "val_data_path: /workspace/data/validation.jsonl\n"
                }
                if source == "5"
                else {"model_output_path": "/workspace/models/checkpoints/run"}
            )
            output = terminal_node_output(values)
            return {"raw": json.dumps(output)} if wrapped else output

    node = resolve_artifacts(Client(), "9175")["training_nodes"][0]
    assert node["output_path"] == "/models/checkpoints/run"
    assert node["inputs"]["dataset_config"] == {
        "train_data_path": "/workspace/data/train.jsonl",
        "val_data_path": "/workspace/data/validation.jsonl",
    }
    assert node["issues"] == []


def test_wandb_config_output_preserves_multiline_values(terminal_node_output):
    WandbDataSource = importlib.import_module("data_sources.wandb").WandbDataSource

    output = terminal_node_output(
        {
            "wandb_config": "WANDB_API_KEY: fixture-key\n"
            "WANDB_PROJECT: pcb-avi-recheck\n"
            "WANDB_ENTITY: pengtao-shi-pyromind\n",
        }
    )
    assert WandbDataSource()._collect_output_env(output) == {
        "WANDB_API_KEY": "fixture-key",
        "WANDB_PROJECT": "pcb-avi-recheck",
        "WANDB_ENTITY": "pengtao-shi-pyromind",
    }
    assert WandbDataSource()._collect_output_env(
        {"entries": [{"m": "WANDB_API_KEY: fixture-key\nWANDB_PROJECT: project\n"}]}
    ) == {"WANDB_API_KEY": "fixture-key", "WANDB_PROJECT": "project"}


@pytest.mark.parametrize(
    "text",
    [
        "View run at https://wandb.ai/team/project/runs/wgr7gawp",
        "\x1b[1;34mwandb\x1b[0m: 🚀 View run \x1b[33mtraining-name\x1b[0m "
        "at: \x1b[34mhttps://wandb.ai/team/project/runs/wgr7gawp\x1b[0m\n"
        "wandb: Find logs at: wandb/run-20261022_032237-wgr7gawp/logs",
    ],
)
def test_wandb_run_url_with_named_run_and_terminal_colors(text):
    WandbDataSource = importlib.import_module("data_sources.wandb").WandbDataSource

    assert WandbDataSource().extract_run_ref(text) == ("team", "project", "wgr7gawp")


def test_unreadable_output_returns_missing_provenance_without_yaml_error():
    parse_node_output = importlib.import_module("platform_output").parse_node_output

    assert parse_node_output({"entries": [{"m": "\x1b[Aunreadable\x00"}]}) == {}


def test_artifact_fingerprint_supports_sandbox_without_file_digest(
    tmp_path, monkeypatch
):
    file_digest = importlib.import_module("training_artifacts").file_digest

    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    content = b"prediction\n" * 120_000
    path = tmp_path / "eval_step_20.jsonl"
    path.write_bytes(content)
    assert file_digest(path) == hashlib.sha256(content).hexdigest()


@pytest.fixture
def training_target_platform(monkeypatch):
    PlatformClient = importlib.import_module("train_analysis").PlatformClient

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    logs = []
    calls = []

    def get(_client, path, params):
        calls.append((path, params))
        if path == "api/task_workflow_result":
            assert params == {"task_id": "9175"}
            return {
                "nodes": [
                    {"id": "12", "data": {"nodeType": "ModelTrainSFTNode"}},
                    {
                        "id": "11",
                        "data": {
                            "nodeType": "WandbConfigBuilderNode",
                            "config": {
                                "WANDB_ENTITY": "pengtao-shi-pyromind",
                                "WANDB_PROJECT": "pcb-avi-recheck",
                            },
                        },
                    },
                ]
            }
        assert path == "internal/logs/node/raw"
        assert params == {"nodeId": "12", "taskId": "9175"}
        return logs.pop(0)

    monkeypatch.setattr(PlatformClient, "_get", get)
    return logs, calls


@pytest.mark.parametrize("node_id", [None, "12"])
@pytest.mark.parametrize(
    "run_url", ["", "https://wandb.ai/pyromind/pcb_avi_recheck_sft_4b"]
)
def test_same_task_resolves_same_wandb_run_with_optional_node_and_invalid_hint(
    training_target_platform, node_id, run_url
):
    logs, calls = training_target_platform
    logs.append(
        {
            "entries": [
                {
                    "m": "wandb: View run at https://wandb.ai/"
                    "pengtao-shi-pyromind/pcb-avi-recheck/runs/wgr7gawp\n"
                }
            ]
        }
    )
    target = TrainingAnalysisService().resolve_target(
        task_id="9175", training_node_id=node_id, run_url=run_url
    )
    assert (target["entity"], target["project"], target["run_id"]) == (
        "pengtao-shi-pyromind",
        "pcb-avi-recheck",
        "wgr7gawp",
    )
    assert len(calls) == 2


def test_training_target_retries_empty_logs_and_decodes_wrapped_terminal_response(
    training_target_platform,
):
    logs, calls = training_target_platform
    logs.extend(
        [
            {"entries": []},
            {
                "data": json.dumps(
                    {"entries": [{"m": "wandb: setting up run wgr7gawp\n"}]}
                )
            },
        ]
    )
    target = TrainingAnalysisService().resolve_target(
        task_id="9175", training_node_id="12"
    )
    assert target["run_id"] == "wgr7gawp"
    assert len(calls) == 3


def test_training_target_reports_unavailable_logs_without_claiming_run_absent(
    training_target_platform,
):
    WandbAnalysisError = importlib.import_module("train_analysis").WandbAnalysisError

    logs, _ = training_target_platform
    logs.extend([{"entries": []}, {"entries": []}])
    with pytest.raises(WandbAnalysisError) as error:
        TrainingAnalysisService().resolve_target(task_id="9175", training_node_id="12")
    message = str(error.value)
    assert "missing run_id" in message
    assert "node=12, log_chars=0, reads=2, run_ref_found=False" in message
    assert "do not prove" in message


def test_valid_run_hint_resolves_without_fetching_logs(training_target_platform):
    _, calls = training_target_platform
    target = TrainingAnalysisService().resolve_target(
        task_id="9175",
        run_url="https://wandb.ai/pengtao-shi-pyromind/pcb-avi-recheck/runs/wgr7gawp",
    )
    assert target["run_id"] == "wgr7gawp"
    assert len(calls) == 1
