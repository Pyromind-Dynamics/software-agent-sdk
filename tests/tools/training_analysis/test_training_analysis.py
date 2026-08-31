import importlib
import json
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
    Path(__file__).resolve().parents[3]
    / ".agents/skills/training-analysis/scripts"
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
        TrainingAnalysisAction(operation="analyze", task_id="task-1"),
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
            task_id="task-1",
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

    action = TrainingAnalysisAction(operation="report", task_id="task-1")
    with pytest.raises(ValueError, match="public_data/training-analysis"):
        TrainingAnalysisExecutor._output_path(action, workspace)


def test_training_analysis_reports_missing_worker(tmp_path: Path) -> None:
    observation = TrainingAnalysisExecutor(runtime_dir=str(tmp_path))(
        TrainingAnalysisAction(operation="analyze", task_id="task-1"),
        _conversation(tmp_path),
    )

    assert observation.is_error
    assert observation.failure_stage == "runtime_dependency"
    assert observation.error_code == "training_worker_missing"


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
