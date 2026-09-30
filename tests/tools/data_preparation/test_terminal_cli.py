"""DataFlow's public terminal entry, without the removed business Tool."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".agents/skills/data-processing/scripts/preparation/sandbox_sample.py"
)


@pytest.fixture
def runtime():
    spec = importlib.util.spec_from_file_location("sample_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    # Stub only the dependency probe; pipeline, validation and report subprocesses
    # still execute normally. No model or DataFlow operators are used here.
    dist = tmp_path / "open_dataflow-1.0.10.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: open-dataflow\nVersion: 1.0.10\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("PYROMIND_SANDBOX_DATAFLOW_PYTHON", sys.executable)
    (tmp_path / "public_data").mkdir()
    (tmp_path / "storage").mkdir()
    (tmp_path / "storage/input.jsonl").write_text(
        '{"id":"1","system_prompt":"system","user_prompt":"hello","gt":"hi"}\n'
    )
    (tmp_path / "public_data/pipeline.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "Path(sys.argv[2]).write_text(Path(sys.argv[1]).read_text())\n"
    )
    return tmp_path


def config():
    return {
        "pipeline_path": "public_data/pipeline.py",
        "args": ["storage/input.jsonl", "public_data/output.jsonl"],
        "model_profile": "none",
        "output_schema": "text",
    }


def test_cli_executes_validates_and_generates_report(workspace):
    path = workspace / "public_data/run.json"
    path.write_text(json.dumps(config()))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(path)],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    envelope = json.loads(result.stdout.splitlines()[-1])
    assert envelope["rc"] == 0 and envelope["validation_rc"] == 0
    assert (workspace / "public_data/report.json").is_file()
    assert (workspace / "public_data/output.jsonl").read_text() == (
        workspace / "storage/input.jsonl"
    ).read_text()


@pytest.mark.parametrize("profile", ["text", "vision"])
def test_cli_model_configuration_is_environment_only(
    runtime, workspace, monkeypatch, profile
):
    if profile == "vision":
        (workspace / "public_data/pipeline.py").write_text(
            "from image_utils import ImagePipelineConfig, run_image_pipeline_from_cli\n"
        )
    profiles = {
        "text": {"DF_API_KEY": "text-secret", "DF_MODEL_NAME": "text-model"},
        "vision": {"DF_API_KEY": "vision-secret", "DF_MODEL_NAME": "vision-model"},
    }
    monkeypatch.setenv("PYROMIND_DATAFLOW_PROFILES", json.dumps(profiles))
    value = {**config(), "model_profile": profile}
    spec = runtime.cli_spec(value, workspace)
    assert spec["env"]["DF_API_KEY"] == f"{profile}-secret"
    assert "secret" not in json.dumps(value)
    none = runtime.cli_spec(config(), workspace)
    assert "DF_API_KEY" not in none["env"]
    with pytest.raises(ValueError, match="credentials"):
        runtime.cli_spec({**value, "env": {"DF_API_KEY": "bad"}}, workspace)
    with pytest.raises(ValueError, match="read-only"):
        runtime.cli_spec(
            {**config(), "args": ["storage/input.jsonl", "storage/output.jsonl"]},
            workspace,
        )


def test_cli_timeout_and_validation_failures_return_nonzero(runtime, workspace):
    pipeline = workspace / "public_data/pipeline.py"
    pipeline.write_text("import time; time.sleep(10)")
    spec = runtime.cli_spec({**config(), "timeout": 1}, workspace)
    result = runtime._run(spec)
    assert result["rc"] != 0 and result["failure_stage"] == "timeout"
    pipeline.write_text(
        'import sys\nfrom pathlib import Path\nPath(sys.argv[2]).write_text("not-json")'
    )
    result = runtime._run(runtime.cli_spec(config(), workspace))
    assert result["rc"] != 0 and result["failure_stage"] == "schema_validation"


def test_cli_rejects_handwritten_vision_transport(runtime, workspace, monkeypatch):
    monkeypatch.setenv("PYROMIND_DATAFLOW_PROFILES", '{"vision":{"DF_API_KEY":"test"}}')
    (workspace / "public_data/pipeline.py").write_text("import requests\n")
    with pytest.raises(ValueError, match="image_utils"):
        runtime.cli_spec({**config(), "model_profile": "vision"}, workspace)
