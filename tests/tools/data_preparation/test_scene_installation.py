import json
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from pyromind_common.platform.scene_core import (
    _LEGACY_RUNTIME_PACKAGES,
    RUNTIME_FILENAMES,
    _build_dataflow_command,
    build_render_command,
    dataflow_pip_install_commands,
)


@pytest.mark.parametrize("matching_metadata", [True, False])
def test_vision_validation_uses_the_matching_runtime_image_root(
    tmp_path, matching_metadata
):
    source = tmp_path / "input.jsonl"
    source.touch()
    metadata_path = tmp_path / "runtime_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_path": str(
                    source if matching_metadata else tmp_path / "other.jsonl"
                ),
                "output_format": "vision",
                "image_root": "/workspace",
            }
        )
    )
    command = _build_dataflow_command(
        input_path="/workspace/data/input.jsonl",
        output_dir="/workspace/output/run",
        output_schema="vision",
        llm_env={},
        convert_format="none",
    )
    resolve_command = command.split("image_root=$(", 1)[1].split("); ", 1)[0]
    arguments = shlex.split(resolve_command)
    result = subprocess.run(
        [sys.executable, *arguments[1:3], str(source), str(metadata_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    expected = "/workspace" if matching_metadata else str(tmp_path)
    assert result.stdout.strip() == expected


@pytest.mark.parametrize("cluster", ["cn-east-1", "cn-east-1#pre2", "us-west-2"])
def test_dataflow_installation_uses_the_execution_cluster(monkeypatch, cluster):
    monkeypatch.setenv("PYROMIND_CLUSTER", cluster)
    commands = dataflow_pip_install_commands(
        pip_path="/tmp/df-venv/bin/pip",
        requirements_path="/runtime/requirements-cpu.txt",
    )
    for command in commands:
        args = shlex.split(command)
        assert args[args.index("--timeout") + 1] == "20"
        assert args[args.index("--retries") + 1] == "2"
        assert args[args.index("--progress-bar") + 1] == "off"
        if cluster.startswith("cn-east-1"):
            assert args[args.index("--index-url") + 1] == (
                "https://mirrors.aliyun.com/pypi/simple/"
            )
        else:
            assert "--index-url" not in args
    assert "--no-deps" in shlex.split(commands[1])
    assert "open-dataflow==1.0.13" in shlex.split(commands[1])


@pytest.mark.parametrize("timeout", [False, True])
def test_installation_wrapper_limits_runtime_and_keeps_the_exit_code(
    monkeypatch, capsys, timeout
):
    command = dataflow_pip_install_commands(
        pip_path="pip", requirements_path="cpu.txt"
    )[0]
    args = shlex.split(command)
    run = Mock(return_value=subprocess.CompletedProcess(args, 23))
    if timeout:
        run.side_effect = subprocess.TimeoutExpired(args, 600)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["-c", *args[3:]])
    with pytest.raises(SystemExit) as error:
        exec(compile(args[2], "<dependency-install>", "exec"), {})
    assert error.value.code == (124 if timeout else 23)
    assert run.call_args.kwargs["timeout"] == 600
    if timeout:
        assert "Dependency installation timed out after 600s" in capsys.readouterr().err


def test_legacy_bundle_keeps_the_cpu_manifest_and_no_deps_dataflow():
    manifest = (
        Path(__file__).resolve().parents[3]
        / ".agents/skills/data-processing/scripts/preparation/requirements-cpu.txt"
    )
    requirements = tuple(
        line
        for line in manifest.read_text().splitlines()
        if line and not line.startswith("#")
    )
    assert _LEGACY_RUNTIME_PACKAGES == requirements
    command = _build_dataflow_command(
        input_path="/input",
        output_dir="/output",
        llm_env={},
        convert_format="none",
        reuse_runtime=True,
        runtime_dir_name="runtime-r1",
        cluster="cn-east-1",
    )
    assert command.count("--index-url") == 4
    assert command.count("install --no-deps open-dataflow==1.0.13") == 2
    assert "torch" not in command
    render = build_render_command(
        output_dir="/workspace/output",
        data_source="/workspace/input",
        shard_size=10,
        limit=None,
        cluster="cn-east-1",
    )
    assert "--index-url https://mirrors.aliyun.com/pypi/simple/" in render


def test_setup_failure_stops_before_pipeline_and_source_integrity(tmp_path):
    output = tmp_path / "output"
    runtime = output / "runtime-r1"
    runtime.mkdir(parents=True)
    for filename in RUNTIME_FILENAMES:
        (runtime / filename).write_text("")
    marker = output / "pipeline-ran"
    (output / "pipeline.py").write_text(f"open({str(marker)!r}, 'w').write('ran')")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    pip = venv / "bin/pip"
    pip.write_text("#!/bin/sh\necho 'dependency download failed' >&2\nexit 23\n")
    pip.chmod(0o700)
    command = (
        _build_dataflow_command(
            input_path="/input",
            output_dir="/output",
            llm_env={},
            convert_format="none",
            runtime_dir_name="runtime-r1",
            cluster="cn-east-1",
        )
        .replace("/target-workspace", str(tmp_path))
        .replace("/tmp/df-venv", str(venv))
    )
    python = shlex.quote(sys.executable)
    shim = (
        'python3() { if [ "$1" = "-m" ] && [ "$2" = "venv" ]; '
        f'then return 0; fi; command {python} "$@"; }}\n'
    )
    result = subprocess.run(
        ["bash", "-c", shim + command], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 23
    assert "Data processing setup failed" in result.stderr
    assert "dependency download failed" in result.stderr
    assert not marker.exists()
    assert not (output / "source_integrity.json").exists()
