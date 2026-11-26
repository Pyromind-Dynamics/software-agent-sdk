from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from starlette.requests import Request

from harness_adapter.pi_adapter.actor_replay import RecordedActorReplayReader
from harness_adapter.pi_adapter.actor_workspace import ActorTransport, ActorWorkspace
from harness_adapter.pi_adapter.adapter import PiAdapter
from harness_adapter.pi_adapter.fixed_actor import FixedActorExecutionManager
from harness_adapter.pi_adapter.persistence import PiSessionFiles, _atomic_json
from pyromind_agent_server.api import echomind_projects as api
from pyromind_common.platform.dataset_replay import DatasetReplayPaths
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.storage import (
    DatasetReplayReader,
    EvaluationStorageUnavailableError,
)


async def test_replay_reader_requires_owner_cluster_and_recorded_workspace(
    tmp_path, monkeypatch
):
    cid = "sample"
    root = tmp_path / cid
    root.mkdir()
    store = FileProductStore(root)
    store.create(
        ConversationSnapshot(conversation_id=cid, capabilities=HarnessCapabilities()),
        user_id="a",
    )
    store.save_x_cluster("east")
    monkeypatch.setenv("PYROMIND_SANDBOX_ACTOR_URL", "https://router.test")
    monkeypatch.setenv("PYROMIND_SANDBOX_ACTOR_TARGET", "user/actor")
    adapter = PiAdapter(tmp_path, terminal_backend="sandbox")
    runtime = ConversationRuntime(tmp_path, adapter)
    ctx = RequestContext(user_id="a", x_cluster="east")
    assert await runtime.dataset_replay_reader(cid, ctx) is None
    sandbox = adapter._sandbox
    assert isinstance(sandbox, FixedActorExecutionManager)
    files = PiSessionFiles(root)
    workspace = sandbox._workspace(cid)

    _atomic_json(files.directory / "actor-workspace.json", sandbox._endpoint(workspace))
    reader = await runtime.dataset_replay_reader(cid, ctx)
    assert isinstance(reader, RecordedActorReplayReader)
    assert sandbox.execution_workspace_path(cid) is None
    with pytest.raises(PermissionError):
        await runtime.dataset_replay_reader(cid, replace(ctx, user_id="b"))
    assert (
        await runtime.dataset_replay_reader(cid, replace(ctx, x_cluster="west")) is None
    )
    changed = sandbox._endpoint(workspace) | {"base_url": "https://other.test"}
    _atomic_json(files.directory / "actor-workspace.json", changed)
    assert await runtime.dataset_replay_reader(cid, ctx) is None


def test_recorded_replay_restricts_paths_and_checks_file_identity(monkeypatch):
    workspace = ActorWorkspace(
        working_dir="/workspace/.pyromind-agent/self",
        transport=ActorTransport("https://router.test", "owner/actor"),
    )
    storage = RecordedActorReplayReader(workspace)
    execute = Mock(
        return_value=SimpleNamespace(exit_code=0, stdout=sha256(b"ok").hexdigest())
    )
    read = Mock(return_value=b"changed")
    monkeypatch.setattr(ActorTransport, "execute", execute)
    monkeypatch.setattr(ActorTransport, "read_file", read)
    with pytest.raises(ValueError):
        storage.read("/.pyromind-agent/other/asset")
    valid = DatasetReplayPaths(
        "/datasets/approved", "/datasets/b", "/.pyromind-agent/self/out"
    )
    with pytest.raises(ValueError):
        storage.verify_splits(
            replace(valid, fixed_validation="/.pyromind-agent/other/val")
        )
    execute.assert_not_called()
    read.assert_not_called()
    with pytest.raises(EvaluationStorageUnavailableError, match="changed"):
        storage.read("/.pyromind-agent/self/asset")
    read.assert_called_once_with(
        "/workspace/.pyromind-agent/self/asset", max_bytes=8 * 1024 * 1024
    )


@pytest.mark.parametrize("entry", ["compile", "generate"])
async def test_replay_reader_reaches_each_workbench_compiler(
    entry, tmp_path, monkeypatch
):
    ctx = RequestContext(user_id="a", x_cluster="east")
    reader = Mock(spec=DatasetReplayReader)
    reader.read.return_value = b"verified-file"
    recipe = Mock()
    recipe.source_path, recipe.output_directory, recipe.fixed_validation_path = (
        "/workspace/source/b",
        "/workspace/.pyromind-agent/self/output",
        None,
    )
    snapshot = ConversationSnapshot(
        conversation_id="self", capabilities=HarnessCapabilities()
    )
    runtime = SimpleNamespace(
        versions=None,
        dataset_replay_reader=AsyncMock(return_value=reader),
        development_storage_proxy=AsyncMock(return_value=None),
        consolidation_command_state=AsyncMock(return_value=((), snapshot)),
    )
    consolidation = SimpleNamespace(
        source_batch="/workspace/data/approved",
        evidence=SimpleNamespace(conversation_id="self"),
        evidence_sha256="digest",
    )
    project, store = Mock(), SimpleNamespace(root=tmp_path)
    monkeypatch.setattr(api, "_load_project", lambda *a, **kw: (ctx, project, store))
    monkeypatch.setattr(
        api, "_load_consolidation_context", AsyncMock(return_value=consolidation)
    )
    monkeypatch.setattr(api, "get_product_runtime", lambda req: runtime)
    monkeypatch.setattr(
        api,
        "get_conversation_service",
        lambda req: SimpleNamespace(conversations_dir=tmp_path),
    )
    monkeypatch.setattr(api, "load_script_compilation", lambda *a: None)
    monkeypatch.setattr(
        api, "_read_repair_recipes", AsyncMock(return_value=(None, [], {}))
    )
    monkeypatch.setattr(api, "build_runtime_credential_store", lambda *args: None)
    monkeypatch.setattr(api, "build_bound_evaluation_storage", lambda **kw: None)
    approved = json.dumps(
        {
            "schema_version": 1,
            "source_kind": "jsonl",
            **{
                field: "a" * 64
                for field in (
                    "source_sha256",
                    "train_sha256",
                    "train_identity_sha256",
                    "val_sha256",
                )
            },
            "train_rows": 2,
            "val_rows": 0,
            "approved_source_member": True,
        }
    )
    reader.verify_splits.return_value = (0, approved, "")

    def compile_inputs(*args, read_replay_file, check_replay, **kwargs):
        assert (
            read_replay_file("/workspace/.pyromind-agent/self/asset")
            == b"verified-file"
        )
        assert b'"approved_source_member":true' in check_replay(recipe)
        raise RuntimeError("verified compiler inputs")

    monkeypatch.setattr(api, "build_script_compilation", compile_inputs)
    req = Request(
        {"type": "http", "app": SimpleNamespace(state=SimpleNamespace(config=None))}
    )
    with pytest.raises(RuntimeError, match="verified compiler inputs"):
        if entry == "compile":
            await api.compile_echomind_script_plan(
                "project", api.CompileConsolidation(), req
            )
        else:
            await api.generate_echomind_consolidation(
                "project", api.GenerateConsolidation(), req
            )
    runtime.dataset_replay_reader.assert_awaited_once_with("self", ctx)
    reader.verify_splits.assert_called_once_with(
        DatasetReplayPaths(
            "/data/approved", "/source/b", "/.pyromind-agent/self/output"
        )
    )
