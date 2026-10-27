from __future__ import annotations

import asyncio
import base64
import json
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest
from harness_adapter.agentgenome_bridge import (
    BridgeTool,
    ExecutionHost,
    _storage_alias,
    _tool_path,
)


MOUNT = "/target-workspace"


def test_platform_storage_spelling_gains_the_sandbox_mount() -> None:
    assert (
        _storage_alias("/datasets/test_100/images", MOUNT)
        == "/target-workspace/datasets/test_100/images"
    )
    assert (
        _storage_alias("datasets/test_100/prelabel_manifest.jsonl", MOUNT)
        == "/target-workspace/datasets/test_100/prelabel_manifest.jsonl"
    )


def test_sandbox_aliases_are_already_resolvable() -> None:
    assert _storage_alias("public_data/label-studio/field_map.json", MOUNT) is None
    assert _storage_alias(".agents/skills/data-processing/SKILL.md", MOUNT) is None
    assert _storage_alias("knowledge/nodes/x.md", MOUNT) is None
    assert _storage_alias("", MOUNT) is None
    # No mount in the local execution plane, so there is nothing to add.
    assert _storage_alias("/datasets/x.jsonl", None) is None


def test_the_removed_storage_prefix_is_not_remapped() -> None:
    """The alias is gone, so the policy's own removal error must surface.

    Remapping it to the mount here would turn a clear "alias was removed" into a
    "file not found" somewhere further along.
    """

    assert _storage_alias("storage/datasets/x.jsonl", MOUNT) is None


def test_a_tool_reads_back_the_path_the_graph_declared() -> None:
    declared = {"/target-workspace/datasets/x.jsonl": "/datasets/x.jsonl"}

    assert (
        _tool_path(
            "/target-workspace/datasets/x.jsonl",
            declared,
            "/target-workspace/.pyromind-agent/abc",
            "public_data/workflow-runs/run/package",
        )
        == "/datasets/x.jsonl"
    )


def test_run_directory_paths_become_workspace_relative() -> None:
    workspace = "/target-workspace/.pyromind-agent/abc"
    package = "public_data/workflow-runs/run/package"

    assert (
        _tool_path(f"{workspace}/{package}/scripts/pcb.py", {}, workspace, package)
        == f"{package}/scripts/pcb.py"
    )
    assert (
        _tool_path("scripts/pcb_prelabel_pipeline.py", {}, workspace, package)
        == f"{package}/scripts/pcb_prelabel_pipeline.py"
    )
    assert _tool_path("vision", {}, workspace, package) == "vision"


def test_a_removed_alias_path_is_not_rewritten_for_a_tool() -> None:
    """The alias is gone, so the bridge must not invent a platform path from it.

    Rewriting `storage/...` here would hide the removal behind a lookup that
    fails later; leaving it alone lets the tool report the removal itself.
    """
    workspace = "/target-workspace/.pyromind-agent/abc"
    package = "public_data/workflow-runs/run/package"

    assert (
        _tool_path("storage/datasets/test_100/review.jsonl", {}, workspace, package)
        == "storage/datasets/test_100/review.jsonl"
    )
    # A path the graph already declared keeps that spelling.
    declared = {
        "storage/datasets/test_100/review.jsonl": "datasets/test_100/review.jsonl"
    }
    assert (
        _tool_path(
            "storage/datasets/test_100/review.jsonl",
            declared,
            workspace,
            package,
        )
        == "datasets/test_100/review.jsonl"
    )


def test_a_storage_artifact_loses_the_mount_prefix_for_a_tool() -> None:
    workspace = "/target-workspace/.pyromind-agent/abc"
    package = "public_data/workflow-runs/run/package"
    mount = "/target-workspace"

    assert (
        _tool_path(
            f"{mount}/datasets/test_100/manifest_10.jsonl",
            {},
            workspace,
            package,
            mount,
        )
        == "/datasets/test_100/manifest_10.jsonl"
    )
    # A run-directory artifact stays workspace-relative, not mount-relative.
    assert (
        _tool_path(
            f"{workspace}/{package}/artifacts/head.out", {}, workspace, package, mount
        )
        == f"{package}/artifacts/head.out"
    )


def test_dataflow_receives_the_conversation_scoped_storage_path() -> None:
    workspace = "/target-workspace/.pyromind-agent/conversation"
    artifact = f"{workspace}/public_data/workflow-runs/run/artifacts/head.out"
    calls: list[dict[str, Any]] = []

    async def request(method: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert method == "execution.tool"
        calls.append(payload)
        return {"is_error": False, "content": [], "details": {}}

    async def invoke() -> None:
        host = ExecutionHost(request, lambda *_: None)
        host.workspace = workspace
        host.root = f"{workspace}/public_data/workflow-runs/run"
        host.package = "public_data/workflow-runs/run/package"
        await asyncio.to_thread(
            BridgeTool(host).call,
            "df_submit_pipeline",
            {"input_path": artifact, "script_path": "scripts/pipeline.py"},
            execution_id="node-clean:1:call-1",
        )

    asyncio.run(invoke())
    assert calls[0]["arguments"] == {
        "input_path": artifact.removeprefix("/target-workspace"),
        "script_path": "public_data/workflow-runs/run/package/scripts/pipeline.py",
    }
    assert calls[0]["tool_call_id"] == "node-clean:1:call-1"
    # DataFlow's pod path adds the mount back; it must find the *same* file.
    assert "/target-workspace" + calls[0]["arguments"]["input_path"] == artifact


def test_tool_cancellation_forwards_only_the_exact_execution_id() -> None:
    calls: list[dict[str, Any]] = []

    async def request(method: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append({"method": method, **payload})
        return {
            "requested": True,
            "confirmed": False,
            "detail": "stop requested; termination unconfirmed",
        }

    async def invoke() -> Any:
        host = ExecutionHost(request, lambda *_: None)
        return await asyncio.to_thread(BridgeTool(host).cancel, "node-clean:1:call-1")

    cancelled = asyncio.run(invoke())
    assert cancelled.confirmed is False
    assert cancelled.detail == "stop requested; termination unconfirmed"
    assert calls == [
        {
            "method": "execution.cancel_tool",
            "tool_call_id": "node-clean:1:call-1",
        }
    ]


@pytest.mark.parametrize("model_path", ["/models/checkpoint-100", "Qwen/Qwen3-32B"])
def test_evaluation_maps_configs_but_keeps_the_model_reference(model_path):
    workspace = "/target-workspace/.pyromind-agent/conversation"
    package = "public_data/workflow-runs/run/package"
    calls = []

    async def request(method, payload):
        assert method == "execution.tool"
        calls.append(payload)
        return {"is_error": False, "content": [], "details": {}}

    async def invoke():
        host = ExecutionHost(request, lambda *_: None)
        host.workspace = workspace
        host.root = f"{workspace}/public_data/workflow-runs/run"
        host.package = package
        await asyncio.to_thread(
            BridgeTool(host).call,
            "df_submit_evaluation",
            {
                "operation": "inference",
                "input_path": "/target-workspace/datasets/test.jsonl",
                "model_path": model_path,
                "dataset_config_path": "configs/dataset.json",
                "evaluation_config_path": "configs/evaluation.json",
                "report_spec_path": "configs/report.json",
                "served_model_name": "default",
            },
            execution_id="evaluate:1",
        )

    asyncio.run(invoke())
    assert calls[0]["arguments"] == {
        "operation": "inference",
        "input_path": "/datasets/test.jsonl",
        "model_path": model_path,
        "dataset_config_path": f"{package}/configs/dataset.json",
        "evaluation_config_path": f"{package}/configs/evaluation.json",
        "report_spec_path": f"{package}/configs/report.json",
        "served_model_name": "default",
    }


def test_tool_bridge_rejects_an_empty_execution_id() -> None:
    async def request(method: str, payload: dict[str, Any]) -> Any:
        raise AssertionError(f"no bridge call expected: {method} {payload}")

    async def invoke() -> None:
        bridge = BridgeTool(ExecutionHost(request, lambda *_: None))
        with pytest.raises(ValueError, match="execution_id"):
            await asyncio.to_thread(
                bridge.call,
                "df_get_task",
                {"task_id": "task-1"},
                execution_id=" ",
            )
        with pytest.raises(ValueError, match="execution_id"):
            await asyncio.to_thread(bridge.cancel, "")

    asyncio.run(invoke())


def test_dataflow_refuses_a_graph_artifact_outside_storage() -> None:
    async def request(method: str, payload: dict[str, Any]) -> Any:
        raise AssertionError(f"no bridge call expected: {method} {payload}")

    async def invoke() -> None:
        host = ExecutionHost(request, lambda *_: None)
        host.workspace = "/tmp/conversation"
        host.root = "/tmp/conversation/public_data/workflow-runs/run"
        with pytest.raises(ValueError, match="shared Storage mount"):
            host.platform_input_path(f"{host.root}/artifacts/head.out")

    asyncio.run(invoke())


class FakePlane:
    """The two bridge calls sampling makes: resolve, then one bounded head."""

    def __init__(self, *, scope_error_for: str = "", stdout: str = "") -> None:
        self.scope_error_for = scope_error_for
        self.stdout = stdout
        self.resolved: list[str] = []
        self.commands: list[str] = []
        self.runs: list[dict[str, Any]] = []

    async def request(self, method: str, payload: dict[str, Any]) -> Any:
        if method == "execution.resolve":
            path = str(payload["path"])
            if self.scope_error_for and path.startswith(self.scope_error_for):
                raise RuntimeError(
                    "request_failed: PATH_SCOPE_ERROR: read paths must stay within "
                    "public_data/, the user Storage mount, configured skill "
                    "directories, or knowledge/"
                )
            self.resolved.append(path)
            return {
                "path": path if path.startswith("/") else f"/target-workspace/{path}"
            }
        if method == "execution.run":
            self.commands.append(str(payload["command"]))
            self.runs.append(dict(payload))
            return {"rc": 0, "stdout": self.stdout, "truncated": False}
        raise AssertionError(f"unexpected bridge call: {method}")


def sample(plane: FakePlane, path: str) -> bytes:
    """``head`` runs from the genome worker thread, so drive it off the loop thread."""

    async def call() -> bytes:
        host = ExecutionHost(plane.request, lambda *_: None)
        host.workspace = "/target-workspace/.pyromind-agent/conv"
        return await asyncio.to_thread(host.head, path, 64)

    return asyncio.run(call())


def test_head_retries_the_storage_spelling_a_tool_declared() -> None:
    plane = FakePlane(
        scope_error_for="datasets/", stdout=base64.b64encode(b"{}").decode()
    )

    assert sample(plane, "datasets/test_100/manifest.jsonl") == b"{}"
    assert plane.resolved == ["/target-workspace/datasets/test_100/manifest.jsonl"]
    assert "/target-workspace/datasets/test_100/manifest.jsonl" in plane.commands[0]


def test_head_reports_a_directory_instead_of_reading_it() -> None:
    plane = FakePlane(stdout="<dir/>")

    with pytest.raises(IsADirectoryError):
        sample(plane, "storage/datasets/test_100/images")
    assert "[ -d " in plane.commands[0]


def test_a_graph_shell_step_outlives_the_runtime_default(monkeypatch) -> None:
    """A step that waits on a platform job cannot die at the runtime's 300s default."""
    monkeypatch.setenv("PYROMIND_AGENTGENOME_SHELL_TIMEOUT_SECONDS", "900")
    plane = FakePlane()

    async def call() -> None:
        host = ExecutionHost(plane.request, lambda *_: None)
        await asyncio.to_thread(host.execute, "true", Path("/tmp"))

    asyncio.run(call())
    assert plane.runs[0]["timeout"] == 900


class RecordingPlane:
    """A plane that records the spelling of every path it is asked to touch."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.commands: list[str] = []

    async def request(self, method: str, payload: dict[str, Any]) -> Any:
        if method == "execution.run":
            self.commands.append(str(payload["command"]))
            return {
                "rc": 0,
                "stdout": base64.b64encode(b"row\n").decode("ascii"),
                "truncated": False,
            }
        path = str(payload["path"])
        self.calls.append((method, path))
        if method == "execution.resolve":
            if path.startswith("/"):
                return {"path": path}
            if path.startswith("storage/"):
                return {"path": f"/target-workspace/{path.removeprefix('storage/')}"}
            return {"path": f"/target-workspace/.pyromind-agent/conv/{path}"}
        if method == "execution.write":
            return {}
        if method == "execution.read":
            return {"content": base64.b64encode(b"row\n").decode("ascii")}
        raise AssertionError(f"unexpected bridge call: {method}")


class SandboxShapePlane:
    """A remote plane with the same two roots as a sandbox conversation."""

    workspace = "/target-workspace/.pyromind-agent/conv"
    mount = "/target-workspace"

    def __init__(self, run_result: dict[str, Any] | None = None) -> None:
        self.run_result = run_result
        self.resolved: list[str] = []
        self.runs: list[dict[str, Any]] = []

    async def request(self, method: str, payload: dict[str, Any]) -> Any:
        if method == "execution.resolve":
            path = str(payload["path"])
            self.resolved.append(path)
            if path.startswith("/"):
                return {"path": path}
            if path.startswith("storage/"):
                return {"path": f"{self.mount}/{path.removeprefix('storage/')}"}
            return {"path": f"{self.workspace}/{path}"}
        if method == "execution.run":
            self.runs.append(dict(payload))
            return self.run_result or {
                "rc": 0,
                "stdout": json.dumps(
                    {
                        "exists": True,
                        "is_file": True,
                        "is_dir": False,
                        "size": 12,
                    }
                ),
                "truncated": False,
            }
        raise AssertionError(f"unexpected bridge call: {method}")


def test_metadata_probes_session_files_and_storage_at_their_sandbox_roots() -> None:
    plane = SandboxShapePlane()

    async def call() -> tuple[dict[str, Any], dict[str, Any]]:
        host = ExecutionHost(plane.request, lambda *_: None)
        return (
            await asyncio.to_thread(host.metadata, "public_data/scripts/x.py"),
            await asyncio.to_thread(
                host.metadata, "storage/datasets/test_100/manifest.jsonl"
            ),
        )

    session_file, storage_file = asyncio.run(call())
    assert session_file["exists"] is True
    assert storage_file["exists"] is True
    assert plane.resolved == [
        "public_data/scripts/x.py",
        "storage/datasets/test_100/manifest.jsonl",
    ]
    assert f"{plane.workspace}/public_data/scripts/x.py" in plane.runs[0]["command"]
    assert f"{plane.mount}/datasets/test_100/manifest.jsonl" in plane.runs[1]["command"]
    assert all(run["cwd"] is None for run in plane.runs)


def test_metadata_reports_a_failed_probe_instead_of_claiming_the_file_is_missing() -> (
    None
):
    plane = SandboxShapePlane(
        {
            "rc": 1,
            "stdout": "python3: command not found",
            "truncated": False,
        }
    )

    async def call() -> dict[str, Any]:
        host = ExecutionHost(plane.request, lambda *_: None)
        return await asyncio.to_thread(host.metadata, "public_data/scripts/x.py")

    with pytest.raises(RuntimeError, match=r"metadata probe.*rc=1.*command not found"):
        asyncio.run(call())


def test_an_artifact_pinned_beside_a_storage_input_travels_under_the_mount() -> None:
    """The path policy admits Storage by its mount path, as the file already is."""
    plane = RecordingPlane()
    pinned = "/target-workspace/datasets/m/manifest_10.jsonl"

    async def call() -> tuple[str, str]:
        host = ExecutionHost(plane.request, lambda *_: None)
        host.workspace = "/target-workspace/.pyromind-agent/conv"
        host.root = f"{host.workspace}/public_data/workflow-runs/run"
        path = await asyncio.to_thread(host.path, pinned)
        await asyncio.to_thread(host.write_text, pinned, "row\n")
        text = await asyncio.to_thread(host.read_text, pinned)
        return path, text

    path, text = asyncio.run(call())
    assert path == pinned
    assert ("execution.write", pinned) in plane.calls
    assert ("execution.read", pinned) in plane.calls
    assert text == "row\n"


def test_a_pinned_artifact_is_sampled_under_the_mount() -> None:
    """Sampling and preflights resolve the mount path the same way reads do."""
    plane = RecordingPlane()
    pinned = "/target-workspace/datasets/m/manifest_10.jsonl"

    async def call() -> bytes:
        host = ExecutionHost(plane.request, lambda *_: None)
        host.workspace = "/target-workspace/.pyromind-agent/conv"
        host.root = f"{host.workspace}/public_data/workflow-runs/run"
        return await asyncio.to_thread(host.head, pinned, 64)

    assert asyncio.run(call()) == b"row\n"
    assert plane.calls[0] == ("execution.resolve", pinned)
    assert pinned in plane.commands[0]


def test_a_run_directory_artifact_keeps_its_absolute_path() -> None:
    """Artifacts inside the conversation workspace need no alias."""
    plane = RecordingPlane()

    async def call() -> tuple[str, str]:
        host = ExecutionHost(plane.request, lambda *_: None)
        host.workspace = "/target-workspace/.pyromind-agent/conv"
        host.root = f"{host.workspace}/public_data/workflow-runs/run"
        name = "artifacts/run--clean.jsonl"
        path = await asyncio.to_thread(host.path, name)
        await asyncio.to_thread(host.write_text, name, "row\n")
        return path, name

    path, name = asyncio.run(call())
    absolute = (
        f"/target-workspace/.pyromind-agent/conv/public_data/workflow-runs/run/{name}"
    )
    assert path == absolute
    assert ("execution.write", absolute) in plane.calls


class LocalPlane:
    """An execution plane backed by a real directory, so re-entry is real.

    The bridge's host-local fixes are about the second visit to a run directory,
    which a canned response cannot reproduce: the commands have to actually run.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.commands: list[str] = []
        self.tools: list[dict[str, Any]] = []

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    async def request(self, method: str, payload: dict[str, Any]) -> Any:
        if method == "execution.capabilities":
            return {"interactive": True}
        if method == "execution.resolve":
            # The plane's root plays the sandbox's Storage mount, so the alias
            # the path policy admits maps back onto it.
            value = str(payload["path"])
            if value.startswith("storage/"):
                value = value.removeprefix("storage/")
            return {"path": str(self._path(value))}
        if method == "execution.write":
            target = self._path(str(payload["path"]))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(str(payload["content"])))
            return {}
        if method == "execution.read":
            return {
                "content": base64.b64encode(
                    self._path(str(payload["path"])).read_bytes()
                ).decode("ascii")
            }
        if method == "execution.run":
            self.commands.append(str(payload["command"]))
            done = subprocess.run(
                str(payload["command"]),
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
            )
            return {"rc": done.returncode, "stdout": done.stdout, "truncated": False}
        if method == "execution.tool":
            self.tools.append(dict(payload))
            return {"is_error": False, "content": [{"text": "ok"}], "details": {}}
        raise AssertionError(f"unexpected bridge call: {method}")


def _package(root: Path, parameters: dict[str, dict[str, Any]] | None = None) -> Path:
    package = root / "package"
    package.mkdir(parents=True)
    manifest: dict[str, Any] = {"revision": {}}
    if parameters is not None:
        manifest["parameters"] = parameters
    (package / "manifest.json").write_text(json.dumps(manifest))
    return package


def test_prepare_survives_the_second_visit_a_wakeup_makes(tmp_path) -> None:
    """A resumed run re-enters prepare over its own run directory."""
    plane = LocalPlane(tmp_path / "plane")
    package = _package(tmp_path)

    async def call() -> None:
        host = ExecutionHost(plane.request, lambda *_: None)
        for _ in range(2):
            await asyncio.to_thread(
                host.prepare,
                "run-1",
                package,
                {},
                [],
                threading.Event(),
                lambda _event: None,
            )

    asyncio.run(call())
    assert (plane.root / "public_data/workflow-runs/run-1/artifacts").is_dir()


def test_prepare_reports_the_execution_domain_and_preserves_scalar_types(
    tmp_path,
) -> None:
    plane = LocalPlane(tmp_path / "plane")
    package = _package(
        tmp_path,
        {
            "data_file": {"type": "path"},
            "limit": {"type": "number"},
            "enabled": {"type": "boolean"},
            "records": {"type": "records"},
        },
    )
    events: list[dict[str, Any]] = []

    async def call() -> dict[str, Any]:
        host = ExecutionHost(plane.request, lambda *_: None)
        _, resolved = await asyncio.to_thread(
            host.prepare,
            "run-1",
            package,
            {
                "data_file": "storage/datasets/data.csv",
                "limit": 10,
                "enabled": True,
                "records": 5,
            },
            ["data_file", "limit", "enabled", "records"],
            threading.Event(),
            events.append,
        )
        return resolved

    resolved = asyncio.run(call())
    assert resolved["data_file"] == str(plane.root / "datasets/data.csv")
    assert resolved["limit"] == 10
    assert resolved["enabled"] is True
    assert resolved["records"] == 5
    assert events[-1] == {
        "event": "execution_prepared",
        "execution": {
            "execution_domain": "sandbox",
            "execution_path": str(plane.root / "public_data/workflow-runs/run-1"),
            "workspace": str(plane.root),
            "package": "public_data/workflow-runs/run-1/package",
            "artifacts_dir": str(
                plane.root / "public_data/workflow-runs/run-1/artifacts"
            ),
        },
    }


def test_metadata_probes_the_execution_plane_and_reports_missing_paths(
    tmp_path,
) -> None:
    plane = LocalPlane(tmp_path / "plane")
    data = plane.root / "datasets/data.csv"
    data.parent.mkdir(parents=True)
    data.write_text("question\nhello\n")

    async def call() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        host = ExecutionHost(plane.request, lambda *_: None)
        return (
            await asyncio.to_thread(host.metadata, "storage/datasets/data.csv"),
            await asyncio.to_thread(host.metadata, "storage/datasets"),
            await asyncio.to_thread(host.metadata, "storage/datasets/missing.csv"),
        )

    file, directory, missing = asyncio.run(call())
    assert file == {
        "exists": True,
        "is_file": True,
        "is_dir": False,
        "size": data.stat().st_size,
    }
    assert directory == {
        "exists": True,
        "is_file": False,
        "is_dir": True,
        "size": (plane.root / "datasets").stat().st_size,
    }
    assert missing == {"exists": False}


def test_metadata_falls_through_to_the_plane_when_host_file_is_absent(
    tmp_path,
) -> None:
    plane = LocalPlane(tmp_path / "plane")
    data = plane.root / "datasets/data.csv"
    data.parent.mkdir(parents=True)
    data.write_text("question\nhello\n")

    async def call() -> dict[str, Any]:
        host = ExecutionHost(
            plane.request,
            lambda *_: None,
            tmp_path / "conversation",
        )
        return await asyncio.to_thread(
            host.metadata,
            "storage/datasets/data.csv",
        )

    metadata = asyncio.run(call())
    assert metadata == {
        "exists": True,
        "is_file": True,
        "is_dir": False,
        "size": data.stat().st_size,
    }


def test_a_wakeup_lands_the_persons_export_on_the_declared_artifact(tmp_path) -> None:
    """The person exports outside the run directory; the gate reads it there."""
    plane = LocalPlane(tmp_path / "plane")
    workspace = plane.root / ".pyromind-agent/conv"
    artifact = str(
        workspace / "public_data/workflow-runs/run-1/artifacts/capture-ab--review.out"
    )
    # The person exports under the Storage alias, which the mount maps to the
    # agent directory beside the conversation.
    export = plane.root / ".pyromind-agent/label-studio/p/export/x.json"
    export.parent.mkdir(parents=True)
    export.write_text('{"records": []}')

    async def call() -> Any:
        host = ExecutionHost(plane.request, lambda *_: None)
        host.root = str(workspace / "public_data/workflow-runs/run-1")
        host.workspace = str(workspace)
        host.resuming = "capture-ab.review"
        host.human_artifact = "storage/.pyromind-agent/label-studio/p/export/x.json"
        approved = await asyncio.to_thread(
            host.ask,
            "复核",
            {"node": "capture-ab.review", "artifact": artifact},
        )
        parked = await asyncio.to_thread(
            host.ask,
            "复核",
            {"node": "capture-ab.review2", "artifact": artifact},
        )
        return approved, parked

    approved, parked = asyncio.run(call())
    assert approved.approve is True
    assert parked.pending is True
    assert Path(artifact).read_text() == '{"records": []}'


def test_a_host_local_export_reaches_the_gate_without_the_plane(tmp_path) -> None:
    """A local conversation's export is already on this host; copy it straight in."""
    plane = LocalPlane(tmp_path / "plane")
    artifact = str(
        plane.root / "public_data/workflow-runs/run-1/artifacts/capture-ab--review.out"
    )
    conversation = tmp_path / "conversation"
    export = conversation / "storage/export/label_studio_export.json"
    export.parent.mkdir(parents=True)
    export.write_text('{"records": [1]}')

    async def call() -> Any:
        host = ExecutionHost(plane.request, lambda *_: None, conversation)
        host.root = str(plane.root / "public_data/workflow-runs/run-1")
        host.resuming = "capture-ab.review"
        host.human_artifact = "storage/export/label_studio_export.json"
        return await asyncio.to_thread(
            host.ask, "复核", {"node": "capture-ab.review", "artifact": artifact}
        )

    assert asyncio.run(call()).approve is True
    assert Path(artifact).read_text() == '{"records": [1]}'


def _submission(plane: LocalPlane, manifest: Path):
    """Submit one manifest from a sandbox-shaped host.

    The plane's root plays the Storage mount, so the conversation workspace sits
    under it and a pinned artifact is mount-absolute, as in the sandbox.
    """
    workspace = plane.root / ".pyromind-agent/conv"

    async def call():
        host = ExecutionHost(plane.request, lambda *_: None)
        host.root = str(workspace / "public_data/workflow-runs/run")
        host.workspace = str(workspace)
        return await asyncio.to_thread(
            BridgeTool(host).call,
            "df_submit_pipeline",
            {"input_path": str(manifest), "script_path": "scripts/pipeline.py"},
            execution_id="node-prelabel:1:call-1",
        )

    return asyncio.run(call())


def test_a_submission_stops_before_the_platform_when_images_are_elsewhere(
    tmp_path,
) -> None:
    """A manifest that left its images behind fails here, not minutes later."""
    plane = LocalPlane(tmp_path / "plane")
    manifest = (
        plane.root
        / ".pyromind-agent/conv/public_data/workflow-runs/run/artifacts/subset.jsonl"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"images": ["1-0805Y/1/B0.bmp"], "text": "x"}\n')

    result = _submission(plane, manifest)
    assert result.is_error is True
    assert "1-0805Y/1/B0.bmp" in result.text
    assert plane.tools == []


def test_a_submission_reaches_the_platform_when_the_images_travel_with_it(
    tmp_path,
) -> None:
    """A manifest pinned beside its images submits under the Storage spelling."""
    plane = LocalPlane(tmp_path / "plane")
    manifest = plane.root / "datasets/m/manifest_10.jsonl"
    manifest.parent.mkdir(parents=True)
    image = manifest.parent / "1-0805Y/1/B0.bmp"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"bmp")
    manifest.write_text('{"images": ["1-0805Y/1/B0.bmp"], "text": "x"}\n')

    result = _submission(plane, manifest)
    assert result.is_error is False
    assert [item["name"] for item in plane.tools] == ["df_submit_pipeline"]
    assert plane.tools[0]["arguments"]["input_path"] == "/datasets/m/manifest_10.jsonl"


def test_df_submit_maps_storage_and_package_paths_independently(tmp_path) -> None:
    """Storage input and workspace package files keep their own path contracts."""
    plane = LocalPlane(tmp_path / "plane")
    workspace = plane.root / ".pyromind-agent/conv"
    package = workspace / "public_data/workflow-runs/run/package"
    manifest = plane.root / "datasets/m/manifest.jsonl"
    script = package / "scripts/pipeline.py"
    support = package / "analysis_spec.json"
    for path in (manifest, script, support):
        path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('{"text": "x"}\n')
    script.write_text("print('pipeline')\n")
    support.write_text('{"limit": 10}\n')

    async def call() -> Any:
        host = ExecutionHost(plane.request, lambda *_: None)
        host.root = str(workspace / "public_data/workflow-runs/run")
        host.workspace = str(workspace)
        host.package = "public_data/workflow-runs/run/package"
        return await asyncio.to_thread(
            BridgeTool(host).call,
            "df_submit_pipeline",
            {
                "input_path": str(manifest),
                "script_path": str(script),
                "support_file_path": str(support),
                "model_profile": "scripts/is-not-a-path",
            },
            execution_id="node-submit:1:call-1",
        )

    result = asyncio.run(call())
    assert result.is_error is False
    arguments = plane.tools[0]["arguments"]
    assert arguments == {
        "input_path": "/datasets/m/manifest.jsonl",
        "script_path": "public_data/workflow-runs/run/package/scripts/pipeline.py",
        "support_file_path": (
            "public_data/workflow-runs/run/package/analysis_spec.json"
        ),
        "model_profile": "scripts/is-not-a-path",
    }
    assert plane.root / arguments["input_path"].lstrip("/") == manifest
    assert workspace / arguments["script_path"] == script
    assert workspace / arguments["support_file_path"] == support


def test_metadata_many_falls_back_to_the_plane_when_the_host_has_no_file(
    tmp_path,
) -> None:
    """A Storage path never lives under the conversation directory.

    With `local_root` set, a path absent on the host disk used to be answered
    "missing" without asking the plane — which turned every Storage input into a
    blocking "输入不存在" during a plan review.
    """
    mount = tmp_path / "mount"
    plane = LocalPlane(mount)
    data = mount / "datasets/data.csv"
    data.parent.mkdir(parents=True)
    data.write_text("question\nhello\n")
    conversation = mount / ".pyromind-agent" / "conversation"
    conversation.mkdir(parents=True)

    async def call() -> dict[str, Any]:
        host = ExecutionHost(plane.request, lambda *_: None, conversation)
        # The third argument is the host-local root; the workspace is what tells
        # the bridge where Storage is mounted.
        host.workspace = str(conversation)
        return await asyncio.to_thread(
            host.metadata_many,
            # The platform spelling of the file, and one that really is missing.
            ["/datasets/data.csv", "/datasets/missing.csv"],
        )

    answers = asyncio.run(call())
    assert answers["/datasets/data.csv"]["exists"] is True
    assert answers["/datasets/missing.csv"]["exists"] is False


def test_metadata_many_maps_object_paths_before_any_workflow_has_run() -> None:
    """A distill probes its inputs before `workspace` is set by a run.

    The mount has to come from the execution plane; without it every Storage
    object path is probed literally, reads as missing, and the plan can never
    pass its review.
    """
    resolved: list[str] = []
    spellings: list[list[str]] = []

    async def request(method: str, payload: dict[str, Any]) -> Any:
        if method == "execution.resolve":
            path = str(payload["path"])
            resolved.append(path)
            return {"path": f"/target-workspace/.pyromind-agent/conv/{path}"}
        if method == "execution.run":
            probed = json.loads(shlex.split(str(payload["command"]))[-1])
            spellings.append(probed)
            answers = {
                spelling: {"exists": spelling.startswith("/target-workspace/")}
                for spelling in probed
            }
            return {"rc": 0, "stdout": json.dumps(answers), "truncated": False}
        raise AssertionError(f"unexpected bridge call: {method}")

    async def run() -> tuple[dict[str, Any], str]:
        host = ExecutionHost(request, lambda *_: None)
        assert host.workspace == ""
        answers = await asyncio.to_thread(host.metadata_many, ["/datasets/a.jsonl"])
        # One round trip for the whole session, not one per probe.
        assert host._probe_mount() == "/target-workspace"
        return answers, host.workspace

    answers, workspace = asyncio.run(run())

    assert answers["/datasets/a.jsonl"]["exists"] is True
    assert "/target-workspace/datasets/a.jsonl" in spellings[0]
    assert workspace == "/target-workspace/.pyromind-agent/conv"
    assert resolved == ["public_data"]


def test_storage_mount_stays_unset_on_a_plane_without_one() -> None:
    """The local execution plane has no mount; nothing may be invented."""

    async def request(method: str, payload: dict[str, Any]) -> Any:
        assert method == "execution.resolve"
        return {"path": f"/Users/someone/conversations/conv/{payload['path']}"}

    async def run() -> tuple[str | None, str]:
        host = ExecutionHost(request, lambda *_: None)
        return await asyncio.to_thread(host._probe_mount), host.workspace

    mount, workspace = asyncio.run(run())

    assert mount is None
    assert workspace == ""


def test_a_failed_mount_probe_is_retried_instead_of_sticking() -> None:
    """A transient resolve failure must not blind the rest of the session."""
    attempts: list[str] = []

    async def request(method: str, payload: dict[str, Any]) -> Any:
        assert method == "execution.resolve"
        attempts.append(str(payload["path"]))
        if len(attempts) == 1:
            raise RuntimeError("request_failed: sandbox is starting")
        return {"path": "/target-workspace/.pyromind-agent/conv/public_data"}

    async def run() -> tuple[str | None, str | None]:
        host = ExecutionHost(request, lambda *_: None)
        first = await asyncio.to_thread(host._probe_mount)
        second = await asyncio.to_thread(host._probe_mount)
        return first, second

    first, second = asyncio.run(run())

    assert first is None
    assert second == "/target-workspace"
    assert len(attempts) == 2
