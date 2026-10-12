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
    assert spec is not None and spec.loader is not None
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


def interpreter_spec(venv, **overrides):
    spec = {
        "python": None,
        "venv": str(venv),
        "dataflow_version": "1.0.10",
        "packages": ["open-dataflow==1.0.10"],
        "system_python": sys.executable,
    }
    spec.update(overrides)
    return spec


def test_check_reports_a_missing_runtime_without_installing(
    runtime, tmp_path, monkeypatch
):
    monkeypatch.setattr(runtime, "_image_interpreters", lambda: ())
    monkeypatch.setattr(runtime, "_system_python", lambda spec: sys.executable)
    monkeypatch.setattr(runtime, "_interpreter_version", lambda python: None)
    venv = tmp_path / "venv"
    report = runtime._interpreter_report(interpreter_spec(venv), prepare=False)
    assert report["ready"] is False
    assert report["marker"] == "check"
    assert "--prepare" in report["detail"]
    assert not venv.exists()


def test_configured_interpreter_missing_inside_the_sandbox_falls_back(
    runtime, tmp_path, monkeypatch
):
    monkeypatch.setattr(runtime, "_image_interpreters", lambda: ())
    monkeypatch.setattr(runtime, "_interpreter_version", lambda python: None)
    spec = interpreter_spec(
        tmp_path / "venv", python=str(tmp_path / "absent" / "python")
    )
    python, detail = runtime._resolve_interpreter(spec, bootstrap=False)
    assert python is None
    assert "configured interpreter" not in detail
    assert "--prepare" in detail


def test_configured_interpreter_with_the_wrong_version_is_an_error(
    runtime, tmp_path, monkeypatch
):
    monkeypatch.setattr(runtime, "_interpreter_version", lambda python: "0.9.9")
    python, detail = runtime._resolve_interpreter(
        interpreter_spec(tmp_path / "venv", python=sys.executable), bootstrap=False
    )
    assert python is None
    assert "0.9.9" in detail and "expected 1.0.10" in detail


def test_cached_venv_is_reused_only_for_the_pinned_version(runtime, tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").write_text("")
    stamp = venv / "open-dataflow-version"
    stamp.write_text("0.9.0")
    assert runtime._cached_venv_python(interpreter_spec(venv)) is None
    stamp.write_text("1.0.10")
    assert runtime._cached_venv_python(interpreter_spec(venv)) == str(
        venv / "bin/python"
    )


def test_prepare_installs_once_and_then_reuses_the_cached_venv(
    runtime, tmp_path, monkeypatch
):
    monkeypatch.setattr(runtime, "_image_interpreters", lambda: ())
    # A stub system interpreter keeps the bootstrap contract under test without
    # reaching PyPI: it only has to materialize the venv's own interpreter.
    system_python = tmp_path / "system-python"
    system_python.write_text(
        "#!/bin/sh\n"
        'venv="$3"\n'
        'mkdir -p "$venv/bin"\n'
        "printf '#!/bin/sh\\nexit 0\\n' > \"$venv/bin/python\"\n"
        'chmod +x "$venv/bin/python"\n'
    )
    system_python.chmod(0o755)
    venv = tmp_path / "venv"
    spec = interpreter_spec(venv, system_python=str(system_python))
    first = runtime._interpreter_report(spec, prepare=True)
    assert first["ready"] is True and first["source"] == "installed"
    assert (venv / "open-dataflow-version").read_text() == "1.0.10"
    second = runtime._interpreter_report(spec, prepare=True)
    assert second["ready"] is True and second["source"] == "reused"
