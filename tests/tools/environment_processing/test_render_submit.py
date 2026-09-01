"""Tests for the edp_render platform render submitter."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openhands.tools.environment_processing.render_submit import (
    EdpRenderAction,
    EdpRenderExecutor as RenderExecutor,
    EdpRenderTool,
    build_render_command,
)


def test_build_render_command_paths_and_args() -> None:
    cmd = build_render_command(
        output_dir="/edp/render/r1",
        data_source="datasets/tmax/data/train-1.parquet",
        shard_size=500,
        limit=10,
    )
    assert "python3 -m venv /tmp/edp-venv" in cmd
    assert "pip install --quiet" in cmd
    assert "> /tmp/edp-render-install.log 2>&1" in cmd
    assert "-r /target-workspace/edp/render/r1/render_requirements.txt" in cmd
    assert "--template /target-workspace/edp/render/r1/render_template.json" in cmd
    assert "--data-source /target-workspace/datasets/tmax/data/train-1.parquet" in cmd
    assert "--output-root /target-workspace/edp/render/r1" in cmd
    assert "--shard-size 500" in cmd
    assert "--limit 10" in cmd


def test_build_render_command_without_limit() -> None:
    cmd = build_render_command(
        output_dir="/edp/render/r1",
        data_source="datasets/tmax/data/train-1.parquet",
        shard_size=500,
        limit=None,
    )
    assert "--limit" not in cmd


def _conversation_with_secrets(
    secrets: dict[str, str], working_dir: Path | None = None
) -> MagicMock:
    registry = MagicMock()
    registry.get_secret_value.side_effect = lambda name: secrets.get(name)
    state = SimpleNamespace(secret_registry=registry, agent_state={})
    conversation = MagicMock()
    conversation.state = state
    conversation.id = "conv-1"
    conversation.workspace = SimpleNamespace(
        working_dir=str(working_dir if working_dir is not None else Path.cwd())
    )
    return conversation


def _render_runtime(tmp_path: Path) -> Path:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "render_manifest.py").write_text("print('render')\n")
    pod = scripts / "pod_runtime"
    pod.mkdir()
    (pod / "render_requirements.txt").write_text("pandas>=2.0\n")
    return scripts


def _patch_submission(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    module = sys.modules["openhands.tools.environment_processing.render_submit"]
    upload = MagicMock()
    monkeypatch.setattr(module, "upload_local_file_to_pyromind", upload)
    # preflight downloads fail open by default so existing tests never hit storage
    monkeypatch.setattr(
        module,
        "download_tail_from_pyromind",
        MagicMock(side_effect=ValueError("storage unavailable in test")),
    )
    response = SimpleNamespace(task_id="task-render-1", status="Pending")
    submit = MagicMock(return_value=response)
    monkeypatch.setattr(module, "submit_workflow_task", submit)
    monkeypatch.setattr(
        module, "create_workflow_api_client", MagicMock(return_value=MagicMock())
    )
    return SimpleNamespace(upload=upload, submit=submit, response=response)


def _executor(runtime_dir: str | None = None) -> Any:
    return RenderExecutor(
        env="pre",
        cluster="us-west-1",
        headers={},
        runtime_dir=runtime_dir,
        output_root=None,
        storage_base_url=None,
        storage_headers={},
        storage_secret_headers={},
        timeout=10,
    )


def test_executor_submits_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mocks = _patch_submission(monkeypatch)
    template = tmp_path / "render_template.json"
    template.write_text(json.dumps({"fields": {"task_id": "task_id"}}))
    conversation = _conversation_with_secrets({"auth_token": "tok"})

    obs = _executor(runtime_dir=str(_render_runtime(tmp_path)))(
        EdpRenderAction(
            template_path=str(template),
            data_source="datasets/tmax/data/train.parquet",
            shard_size=100,
        ),
        conversation,
    )

    assert obs.status == "Pending"
    assert obs.task_id == "task-render-1"
    assert obs.output_dir and "/edp_render/" in obs.output_dir
    # render script + template + requirements staged
    assert mocks.upload.call_count == 3
    names = {
        str(c.kwargs["local_path"]).split("/")[-1] for c in mocks.upload.call_args_list
    }
    assert {
        "render_manifest.py",
        "render_template.json",
        "render_requirements.txt",
    } == names
    workflow = mocks.submit.call_args.kwargs["workflow"]
    assert workflow["nodes"][0]["data"]["nodeType"] == "CustomCommandCPUNode"
    command = workflow["nodes"][0]["data"]["config"]["command"]
    assert "render_manifest.py" in command
    conversation.register_active_long_task.assert_called_once()


def test_executor_missing_template_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_submission(monkeypatch)
    conversation = _conversation_with_secrets(
        {"auth_token": "tok"}, working_dir=tmp_path
    )
    obs = _executor(runtime_dir=str(_render_runtime(tmp_path)))(
        EdpRenderAction(
            template_path=str(tmp_path / "nope.json"),
            data_source="datasets/tmax/data/train.parquet",
        ),
        conversation,
    )
    assert obs.status == "Failed"
    assert "template not found" in obs.text


def test_executor_resolves_workspace_relative_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """file_editor 写的 workspace 相对路径直接可用,不再靠 agent 摸绝对路径。

    回归守卫(对话 ebda2d49):agent 用 apply_patch 写 'public_data/xx.json' 后
    edp_render 连续 6 次 not found,直到拼出服务端 cwd 相对路径才成功。
    """
    mocks = _patch_submission(monkeypatch)
    workspace = tmp_path / "ws"
    (workspace / "public_data").mkdir(parents=True)
    template = workspace / "public_data" / "render_template.json"
    template.write_text(json.dumps({"fields": {"task_id": "task_id"}}))
    conversation = _conversation_with_secrets(
        {"auth_token": "tok"}, working_dir=workspace
    )

    obs = _executor(runtime_dir=str(_render_runtime(tmp_path)))(
        EdpRenderAction(
            template_path="public_data/render_template.json",
            data_source="datasets/tmax/data/train.parquet",
        ),
        conversation,
    )

    assert obs.status == "Pending"
    assert mocks.upload.call_count == 3
    staged_template = [
        c.kwargs["local_path"]
        for c in mocks.upload.call_args_list
        if str(c.kwargs["local_path"]).endswith("render_template.json")
    ]
    assert staged_template and Path(str(staged_template[0])).read_text() == (
        template.read_text()
    )


def test_edp_render_tool_rejects_unknown_params() -> None:
    with pytest.raises(ValueError, match="unknown params"):
        EdpRenderTool.create(bogus=1)


def _parquet_bytes() -> bytes:
    table = pa.table(
        {
            "task_id": ["t1"],
            "description": ["do the thing"],
            "env_config": [{"task_id": "t1"}],
        }
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def _patch_preflight(
    monkeypatch: pytest.MonkeyPatch, parquet: bytes | None
) -> MagicMock:
    module = sys.modules["openhands.tools.environment_processing.render_submit"]

    def fake_download(**kwargs: Any) -> tuple[bytes, int]:
        if parquet is None:
            raise ValueError("storage unavailable in test")
        return parquet, len(parquet)

    mock = MagicMock(side_effect=fake_download)
    monkeypatch.setattr(module, "download_tail_from_pyromind", mock)
    return mock


def test_preflight_rejects_missing_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """列名写错秒级拦住,不再浪费一轮平台任务(对话 b32487cc)。"""
    mocks = _patch_submission(monkeypatch)
    _patch_preflight(monkeypatch, _parquet_bytes())
    template = tmp_path / "render_template.json"
    template.write_text(
        json.dumps({"fields": {"task_id": "task_id", "prompt": "nope"}})
    )
    conversation = _conversation_with_secrets({"auth_token": "tok"})

    obs = _executor(runtime_dir=str(_render_runtime(tmp_path)))(
        EdpRenderAction(
            template_path=str(template),
            data_source="datasets/tmax/data/train.parquet",
        ),
        conversation,
    )

    assert obs.status == "Failed"
    assert "top-level column 'nope' which does not exist" in obs.text
    assert "available columns" in obs.text
    assert "nothing was submitted" in obs.text
    mocks.submit.assert_not_called()
    mocks.upload.assert_not_called()


def test_preflight_rejects_missing_nested_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mocks = _patch_submission(monkeypatch)
    _patch_preflight(monkeypatch, _parquet_bytes())
    template = tmp_path / "render_template.json"
    template.write_text(json.dumps({"fields": {"task_id": "env_config.missing"}}))
    conversation = _conversation_with_secrets({"auth_token": "tok"})

    obs = _executor(runtime_dir=str(_render_runtime(tmp_path)))(
        EdpRenderAction(
            template_path=str(template),
            data_source="datasets/tmax/data/train.parquet",
        ),
        conversation,
    )

    assert obs.status == "Failed"
    assert "missing nested field" in obs.text
    mocks.submit.assert_not_called()


def test_preflight_accepts_valid_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mocks = _patch_submission(monkeypatch)
    download = _patch_preflight(monkeypatch, _parquet_bytes())
    template = tmp_path / "render_template.json"
    template.write_text(
        json.dumps(
            {
                "fields": {
                    "task_id": "task_id",
                    "prompt": "description",
                    "workdir": {"field": "env_config.task_id"},
                }
            }
        )
    )
    conversation = _conversation_with_secrets({"auth_token": "tok"})

    obs = _executor(runtime_dir=str(_render_runtime(tmp_path)))(
        EdpRenderAction(
            template_path=str(template),
            data_source="datasets/tmax/data/train.parquet",
        ),
        conversation,
    )

    assert obs.status == "Pending"
    assert mocks.submit.call_count == 1
    download.assert_called_once()


def test_preflight_skips_glob_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_submission(monkeypatch)
    download = _patch_preflight(monkeypatch, _parquet_bytes())
    template = tmp_path / "render_template.json"
    template.write_text(
        json.dumps({"fields": {"task_id": "task_id", "prompt": "nope"}})
    )
    conversation = _conversation_with_secrets({"auth_token": "tok"})

    obs = _executor(runtime_dir=str(_render_runtime(tmp_path)))(
        EdpRenderAction(
            template_path=str(template),
            data_source="datasets/tmax/data/train-*.parquet",
        ),
        conversation,
    )

    assert obs.status == "Pending"
    download.assert_not_called()


def test_preflight_fails_open_on_storage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """预检自身失败不阻塞提交(节点端渲染仍会 fail fast)。"""
    mocks = _patch_submission(monkeypatch)
    _patch_preflight(monkeypatch, None)
    template = tmp_path / "render_template.json"
    template.write_text(
        json.dumps({"fields": {"task_id": "task_id", "prompt": "nope"}})
    )
    conversation = _conversation_with_secrets({"auth_token": "tok"})

    obs = _executor(runtime_dir=str(_render_runtime(tmp_path)))(
        EdpRenderAction(
            template_path=str(template),
            data_source="datasets/tmax/data/train.parquet",
        ),
        conversation,
    )

    assert obs.status == "Pending"
    assert mocks.submit.call_count == 1


def test_preflight_parses_schema_through_sparse_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """大文件场景:tail 只含文件尾部,头部 magic 虚拟提供、中间为稀疏间隙。"""
    from openhands.tools.environment_processing.render_submit import (
        _preflight_template,
    )

    small = _parquet_bytes()
    # 模拟 10MB 大文件:真实数据块不下载(稀疏零字节),tail 即整个小 parquet
    total = len(small) + 10 * 1024 * 1024

    def fake_download(**kwargs: Any) -> tuple[bytes, int]:
        return small, total

    monkeypatch.setattr(
        sys.modules["openhands.tools.environment_processing.render_submit"],
        "download_tail_from_pyromind",
        MagicMock(side_effect=fake_download),
    )

    assert (
        _preflight_template(
            {"fields": {"task_id": "task_id", "prompt": "description"}},
            "datasets/tmax/data/train.parquet",
            storage_base_url="http://storage",
            headers={},
            timeout=5.0,
        )
        is None
    )
    message = _preflight_template(
        {"fields": {"prompt": "nope"}},
        "datasets/tmax/data/train.parquet",
        storage_base_url="http://storage",
        headers={},
        timeout=5.0,
    )
    assert message is not None
    assert "top-level column 'nope'" in message
