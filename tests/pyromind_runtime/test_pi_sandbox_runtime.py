from __future__ import annotations

import asyncio
import io
import json
import re
import tarfile
from pathlib import Path

import pytest
from harness_adapter.pi_adapter.business_tool_host import ToolExecutionContext
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from harness_adapter.pi_adapter.sandbox_runtime import (
    _WORKSPACE_LAYOUT_VERSION,
    DEFAULT_CPU,
    DEFAULT_MEMORY,
    DEFAULT_MOUNT_PATH,
    PYROMIND_AGENT_STORAGE_ROOT,
    STORAGE_ALIAS,
    STORAGE_HOST_PATH,
    SandboxExecutionManager,
    SandboxSettings,
)
from harness_adapter.pi_adapter.sandbox_workspace import SandboxWorkspace
from pyromind_runtime.domain.context import RequestContext
from pyromind_sdk.client.models import (
    ResourceConfig,
    SandboxRequest,
    SandboxResponse,
    SandboxType,
    VolumeMount,
)


class _FakeSandboxClient:
    def __init__(
        self,
        *,
        status: str = "running",
        api_key: str = "access-key",
        existing: SandboxResponse | None = None,
        lookup_error: Exception | None = None,
        list_result: list[SandboxResponse] | None = None,
        create_error: Exception | None = None,
        files: dict[str, bytes] | None = None,
    ) -> None:
        self.base_url = "https://pre-api.pyromind.ai/api/v1"
        self.api_key = api_key
        self.cluster = "us-west-1"
        self.calls: list[str] = []
        self.requests: list[SandboxRequest] = []
        self.files: dict[str, bytes] = dict(files or {})
        self._status = status
        self._existing = existing
        self._lookup_error = lookup_error
        self._list_result = list(list_result or [])
        self._create_error = create_error

    @property
    def lifecycle_calls(self) -> list[str]:
        """Lifecycle calls only; script and file transfers are left out."""
        return [call for call in self.calls if not call.startswith(("read:", "write:"))]

    def _sandbox(self, status: str) -> SandboxResponse:
        return SandboxResponse(
            id="sbx-1",
            name="pi-conv-1",
            type=SandboxType.CUSTOM,
            status=status,
        )

    def get_sandbox(self, sandbox_id: str) -> SandboxResponse:
        self.calls.append(f"get:{sandbox_id}")
        if self._lookup_error is not None:
            raise self._lookup_error
        return self._existing or self._sandbox(self._status)

    def list(self) -> list[SandboxResponse]:
        self.calls.append("list")
        return list(self._list_result)

    def create_and_wait(
        self, request: SandboxRequest, *, target_status: str, timeout: int
    ) -> SandboxResponse:
        self.calls.append(f"create:{target_status}")
        self.requests.append(request)
        if self._create_error is not None:
            raise self._create_error
        return self._sandbox("running")

    def resume(self, sandbox_id: str) -> SandboxResponse:
        self.calls.append(f"resume:{sandbox_id}")
        return self._sandbox("running")

    def delete(self, sandbox_id: str) -> None:
        self.calls.append(f"delete:{sandbox_id}")

    def pause(self, sandbox_id: str) -> SandboxResponse:
        self.calls.append(f"pause:{sandbox_id}")
        return self._sandbox("paused")

    def read_file(self, sandbox_id: str, path: str) -> bytes:
        self.calls.append(f"read:{path}")
        try:
            return self.files[path]
        except KeyError as exc:
            raise RuntimeError(f"file not found: {path}") from exc

    def write_file(
        self, sandbox_id: str, path: str, source: object
    ) -> dict[str, object]:
        self.calls.append(f"write:{path}")
        content = (
            Path(source).read_bytes() if isinstance(source, (str, Path)) else source
        )
        assert isinstance(content, bytes)
        self.files[path] = content
        return {"path": path, "size": len(content)}


def _context(tmp_path: Path, extra: dict | None = None) -> ToolExecutionContext:
    return ToolExecutionContext(
        conversation_id="conv-1",
        workspace_root=tmp_path / "conv-1",
        request_context=RequestContext(
            user_id="user-1",
            x_cluster="us-west-1#pre",
        ),
        model_configuration={},
        extra=extra or {},
    )


def _uploaded_scripts(client: _FakeSandboxClient) -> list[str]:
    """Scripts this manager shipped into the sandbox, in upload order."""
    paths = [path for path in client.files if path.startswith("/tmp/pi-sandbox-")]
    return [client.files[path].decode("utf-8") for path in paths]


def _uploaded_script(client: _FakeSandboxClient) -> str:
    """Script this manager shipped into the sandbox for its last command."""
    scripts = _uploaded_scripts(client)
    assert scripts, "no sandbox script was uploaded"
    return scripts[-1]


def _unpack_resource_archives(client: _FakeSandboxClient) -> None:
    """Apply the resource archive the way the sandbox's own tar would."""
    for path in [name for name in client.files if name.startswith("/tmp/pi-sandbox-")]:
        match = re.search(
            r"tar -xzf (\S+) -C (\S+)", client.files[path].decode("utf-8")
        )
        if match is None:
            continue
        archive, workspace = match.group(1), match.group(2)
        with tarfile.open(fileobj=io.BytesIO(client.files[archive])) as tar:
            for member in tar.getmembers():
                extracted = tar.extractfile(member) if member.isfile() else None
                if extracted is None:
                    continue
                client.files[f"{workspace}/{member.name}"] = extracted.read()
        client.files.pop(archive, None)


@pytest.fixture
def files(tmp_path: Path) -> PiSessionFiles:
    session_files = PiSessionFiles(tmp_path / "conv-1")
    session_files.initialize({"session_id": "conv-1"})
    return session_files


def test_sandbox_settings_read_session_config_then_environment(monkeypatch) -> None:
    monkeypatch.setenv("PYROMIND_SANDBOX_CPU", "8")
    settings = SandboxSettings.from_extra(
        {"sandbox": {"memory": "16Gi", "gpu_card": "L40S"}}
    )
    assert settings.cpu == "8"
    assert settings.memory == "16Gi"
    assert settings.gpu_card == "L40S"
    assert settings.mount_path == DEFAULT_MOUNT_PATH
    assert settings.host_path == STORAGE_HOST_PATH
    assert SandboxSettings.from_extra({}).mount_path == DEFAULT_MOUNT_PATH
    assert (
        SandboxSettings.from_extra({"sandbox": {"mount_path": "/data"}}).mount_path
        == "/data"
    )
    assert (
        SandboxSettings.from_extra({"sandbox": {"host_path": "/mnt"}}).host_path
        == "/mnt"
    )


def test_sandbox_settings_default_to_platform_resources(monkeypatch) -> None:
    for name in (
        "PYROMIND_SANDBOX_CPU",
        "PYROMIND_SANDBOX_MEMORY",
        "PYROMIND_SANDBOX_GPU",
        "PYROMIND_SANDBOX_GPU_CARD",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = SandboxSettings.from_extra({})

    assert settings.cpu == DEFAULT_CPU
    assert settings.memory == DEFAULT_MEMORY
    assert SandboxSettings.from_extra({"sandbox": {"cpu": "8"}}).memory == "16Gi"


def test_ensure_creates_and_prepares_a_sandbox(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    client = _FakeSandboxClient()
    commands: list[str] = []
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        lambda **kwargs: (commands.append(kwargs["command"]) or "", 0, False),
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    endpoint = asyncio.run(
        manager.ensure(
            _context(tmp_path, {"sandbox": {"cpu": "4", "memory": "8Gi"}}), files
        )
    )

    assert client.lifecycle_calls == ["create:running"]
    assert client.requests[0].sandbox_type is SandboxType.CUSTOM
    assert client.requests[0].resources == ResourceConfig(cpu="4", memory="8Gi")
    assert client.requests[0].volume_mounts == [
        VolumeMount(
            host_path=STORAGE_HOST_PATH,
            mount_path=DEFAULT_MOUNT_PATH,
            read_only=False,
        )
    ]
    assert endpoint == {
        "base_url": "https://pre-api.pyromind.ai/api/v1",
        "ws_base_url": "https://pre-api.pyromind.ai/api/v1",
        "sandbox_id": "sbx-1",
        "api_key": "access-key",
        "workspace_path": f"{DEFAULT_MOUNT_PATH}{PYROMIND_AGENT_STORAGE_ROOT}/conv-1",
        "storage_path": DEFAULT_MOUNT_PATH,
        "cluster": "us-west-1",
    }
    # The TTY bridge echoes the command line, so the script must be shipped as a
    # file and the command itself must stay free of markers.
    assert commands[0].startswith("sh /tmp/pi-sandbox-")
    assert "__PM_" not in commands[0]
    script = _uploaded_script(client)
    assert f"mkdir -p {endpoint['workspace_path']}" in script
    assert (
        f"ln -s {DEFAULT_MOUNT_PATH} {endpoint['workspace_path']}/{STORAGE_ALIAS}"
    ) in script
    record = files.load_sandbox()
    assert record is not None
    assert "api_key" not in json.dumps(record)
    assert record["workspace_path"] == endpoint["workspace_path"]
    assert record["workspace_version"] == _WORKSPACE_LAYOUT_VERSION
    assert record["storage_host_path"] == STORAGE_HOST_PATH


def test_ensure_requests_default_resources_without_session_overrides(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    for name in ("PYROMIND_SANDBOX_CPU", "PYROMIND_SANDBOX_MEMORY"):
        monkeypatch.delenv(name, raising=False)
    client = _FakeSandboxClient()
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        lambda **kwargs: ("", 0, False),
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    asyncio.run(manager.ensure(_context(tmp_path), files))

    assert client.requests[0].resources == ResourceConfig(
        cpu=DEFAULT_CPU, memory=DEFAULT_MEMORY
    )


def test_ensure_reuses_a_running_sandbox_without_repairing(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    files.save_sandbox(
        {
            "sandbox_id": "sbx-1",
            "workspace_path": f"{DEFAULT_MOUNT_PATH}/.pyromind-agent/conv-1",
            "mount_path": DEFAULT_MOUNT_PATH,
            "storage_host_path": STORAGE_HOST_PATH,
            "created_at": "2026-01-01T00:00:00+00:00",
            "workspace_version": _WORKSPACE_LAYOUT_VERSION,
        }
    )
    client = _FakeSandboxClient(status="running")
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        lambda **kwargs: pytest.fail("a reused sandbox must not be re-probed"),
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    endpoint = asyncio.run(manager.ensure(_context(tmp_path), files))

    assert client.calls == ["get:sbx-1"]
    assert endpoint["sandbox_id"] == "sbx-1"
    record = files.load_sandbox()
    assert record is not None
    assert record["created_at"] == "2026-01-01T00:00:00+00:00"


def test_ensure_refreshes_a_cached_endpoint_bundle_on_request(
    tmp_path: Path, files: PiSessionFiles
) -> None:
    files.save_sandbox(
        {
            "sandbox_id": "sbx-1",
            "workspace_path": f"{DEFAULT_MOUNT_PATH}/.pyromind-agent/conv-1",
            "mount_path": DEFAULT_MOUNT_PATH,
            "storage_host_path": STORAGE_HOST_PATH,
            "created_at": "2026-01-01T00:00:00+00:00",
            "workspace_version": _WORKSPACE_LAYOUT_VERSION,
        }
    )
    clients: list[_FakeSandboxClient] = []

    def factory(context: ToolExecutionContext) -> _FakeSandboxClient:
        client = _FakeSandboxClient(status="running", api_key=f"key-{len(clients)}")
        clients.append(client)
        return client

    manager = SandboxExecutionManager(client_factory=factory)
    context = _context(tmp_path)

    first = asyncio.run(manager.ensure(context, files))
    cached = asyncio.run(manager.ensure(context, files))
    refreshed = asyncio.run(manager.ensure(context, files, refresh=True))

    # A cached call reuses the bundle; a refresh re-reads the sandbox and mints
    # the client whose access key replaces the one the runner is holding.
    assert len(clients) == 2
    assert clients[1].lifecycle_calls == ["get:sbx-1"]
    assert first["api_key"] == cached["api_key"] == "key-0"
    assert refreshed["api_key"] == "key-1"
    assert refreshed["sandbox_id"] == first["sandbox_id"] == "sbx-1"


def test_ensure_upgrades_a_running_sandbox_to_the_storage_alias_layout(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    files.save_sandbox(
        {
            "sandbox_id": "sbx-1",
            "workspace_path": f"{DEFAULT_MOUNT_PATH}/.pyromind-agent/conv-1",
            "mount_path": DEFAULT_MOUNT_PATH,
            "storage_host_path": STORAGE_HOST_PATH,
        }
    )
    client = _FakeSandboxClient(status="running")
    commands: list[str] = []
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        lambda **kwargs: (commands.append(kwargs["command"]) or "", 0, False),
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    endpoint = asyncio.run(manager.ensure(_context(tmp_path), files))

    assert endpoint["storage_path"] == DEFAULT_MOUNT_PATH
    assert len(commands) == 1
    assert "ln -s" in _uploaded_script(client)
    record = files.load_sandbox()
    assert record is not None
    assert record["workspace_version"] == _WORKSPACE_LAYOUT_VERSION


def test_ensure_resumes_a_paused_sandbox(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    files.save_sandbox(
        {
            "sandbox_id": "sbx-1",
            "mount_path": DEFAULT_MOUNT_PATH,
            "storage_host_path": STORAGE_HOST_PATH,
        }
    )
    client = _FakeSandboxClient(status="paused")
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        lambda **kwargs: ("", 0, False),
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    endpoint = asyncio.run(manager.ensure(_context(tmp_path), files))

    assert client.lifecycle_calls == ["get:sbx-1", "resume:sbx-1"]
    assert endpoint["sandbox_id"] == "sbx-1"


def test_ensure_recreates_a_sandbox_that_failed(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    files.save_sandbox(
        {
            "sandbox_id": "sbx-1",
            "mount_path": DEFAULT_MOUNT_PATH,
            "storage_host_path": STORAGE_HOST_PATH,
        }
    )
    client = _FakeSandboxClient(status="error")
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        lambda **kwargs: ("", 0, False),
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    asyncio.run(manager.ensure(_context(tmp_path), files))

    assert client.lifecycle_calls == ["get:sbx-1", "delete:sbx-1", "create:running"]


def test_ensure_recreates_a_sandbox_recorded_without_the_storage_mount(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    # Records written before the mount moved into the create request describe a
    # container that can never expose /target-workspace.
    files.save_sandbox(
        {
            "sandbox_id": "sbx-1",
            "workspace_path": f"{DEFAULT_MOUNT_PATH}/.pyromind-agent/conv-1",
            "mount_path": DEFAULT_MOUNT_PATH,
        }
    )
    client = _FakeSandboxClient(status="running")
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        lambda **kwargs: ("", 0, False),
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    asyncio.run(manager.ensure(_context(tmp_path), files))

    assert client.lifecycle_calls == ["get:sbx-1", "delete:sbx-1", "create:running"]
    assert client.requests[0].volume_mounts == [
        VolumeMount(
            host_path=STORAGE_HOST_PATH,
            mount_path=DEFAULT_MOUNT_PATH,
            read_only=False,
        )
    ]


def test_ensure_fails_fast_when_storage_is_not_mounted(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    client = _FakeSandboxClient(lookup_error=RuntimeError("not found"))
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        lambda **kwargs: ("", 3, False),
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    with pytest.raises(RuntimeError, match="PI_SANDBOX_MOUNT_MISSING.*exit_code=3"):
        asyncio.run(manager.ensure(_context(tmp_path), files))


def test_ensure_keeps_the_container_when_prepare_fails(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    client = _FakeSandboxClient()
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        lambda **kwargs: ("", 3, False),
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    with pytest.raises(RuntimeError, match="PI_SANDBOX_MOUNT_MISSING"):
        asyncio.run(manager.ensure(_context(tmp_path), files))

    # The container stays recorded but unprepared, so the next attempt repairs
    # it instead of creating a second sandbox with the same name.
    assert client.lifecycle_calls == ["create:running"]
    record = files.load_sandbox()
    assert record is not None
    assert record["sandbox_id"] == "sbx-1"
    assert "workspace_version" not in record


def test_ensure_adopts_the_container_left_behind_by_a_failed_create(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    orphan = SandboxResponse(
        id="sbx-orphan",
        name="pi-conv-1",
        type=SandboxType.CUSTOM,
        status="running",
    )
    client = _FakeSandboxClient(
        list_result=[orphan],
        create_error=RuntimeError("INSTANCE_EXIST: instance pi-conv-1 already exists"),
    )
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        lambda **kwargs: ("", 0, False),
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    endpoint = asyncio.run(manager.ensure(_context(tmp_path), files))

    assert endpoint["sandbox_id"] == "sbx-orphan"
    assert client.lifecycle_calls == ["create:running", "list"]
    record = files.load_sandbox()
    assert record is not None
    assert record["sandbox_id"] == "sbx-orphan"
    assert record["workspace_version"] == _WORKSPACE_LAYOUT_VERSION


def test_ensure_unpacks_resources_from_one_archive(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    # The skills tree is hundreds of files, so it travels as one archive that
    # the sandbox unpacks, which also carries the empty __init__.py files the
    # chunked upload API cannot write.
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "__init__.py").write_bytes(b"")
    (knowledge / "notes.md").write_bytes(b"hello\n")
    client = _FakeSandboxClient()
    commands: list[str] = []

    def run_terminal_command(**kwargs: object) -> tuple[str, int, bool]:
        commands.append(str(kwargs["command"]))
        _unpack_resource_archives(client)
        return ("", 0, False)

    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        run_terminal_command,
    )
    manager = SandboxExecutionManager(
        client_factory=lambda context: client,
        resource_roots=[("knowledge", knowledge)],
    )

    asyncio.run(manager.ensure(_context(tmp_path), files))

    workspace = f"{DEFAULT_MOUNT_PATH}{PYROMIND_AGENT_STORAGE_ROOT}/conv-1"
    assert client.files[f"{workspace}/knowledge/notes.md"] == b"hello\n"
    assert client.files[f"{workspace}/knowledge/__init__.py"] == b""
    # The tree arrives in one archive: nothing is written file by file.
    assert [
        call for call in client.calls if call.startswith(f"write:{workspace}/")
    ] == []
    assert client.calls.count("write:/tmp/pi-session-resources.tgz") == 1
    assert "/tmp/pi-session-resources.tgz" not in client.files


def test_ensure_reports_prepare_failures_that_are_not_a_missing_mount(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    client = _FakeSandboxClient()
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        lambda **kwargs: ("mkdir: permission denied", 1, False),
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    with pytest.raises(RuntimeError, match="PI_SANDBOX_PREPARE_FAILED"):
        asyncio.run(manager.ensure(_context(tmp_path), files))

    record = files.load_sandbox()
    assert record is not None
    assert record["sandbox_id"] == "sbx-1"


def test_ensure_applies_pending_fork_from_materialized_source(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    files.save_pending_sandbox_fork("conv-source", "nodes: []\n")
    client = _FakeSandboxClient()
    commands: list[str] = []

    def run_terminal_command(**kwargs: object) -> tuple[str, int, bool]:
        commands.append(str(kwargs["command"]))
        return ("", 0, False)

    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        run_terminal_command,
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    asyncio.run(manager.ensure(_context(tmp_path), files))

    scripts = _uploaded_scripts(client)
    assert any("cp -a" in script for script in scripts)
    assert "ln -s" in scripts[-1]
    assert (
        client.files[
            f"{DEFAULT_MOUNT_PATH}{PYROMIND_AGENT_STORAGE_ROOT}/conv-1"
            "/public_data/workflow_canvas/workflow.py"
        ]
        == b"nodes: []\n"
    )
    assert files.load_pending_sandbox_fork() is None


def test_ensure_skips_copy_when_fork_source_never_materialized(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    files.save_pending_sandbox_fork(None, "nodes: []\n")
    client = _FakeSandboxClient()
    commands: list[str] = []

    def run_terminal_command(**kwargs: object) -> tuple[str, int, bool]:
        commands.append(str(kwargs["command"]))
        return ("", 0, False)

    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        run_terminal_command,
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    asyncio.run(manager.ensure(_context(tmp_path), files))

    assert not any("cp -a" in script for script in _uploaded_scripts(client))
    assert all(command.startswith("sh /tmp/pi-sandbox-") for command in commands)
    assert (
        client.files[
            f"{DEFAULT_MOUNT_PATH}{PYROMIND_AGENT_STORAGE_ROOT}/conv-1"
            "/public_data/workflow_canvas/workflow.py"
        ]
        == b"nodes: []\n"
    )
    assert files.load_pending_sandbox_fork() is None


def test_sandbox_workspace_maps_paths_for_upload_download_and_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _FakeSandboxClient()
    workspace_root = f"{DEFAULT_MOUNT_PATH}{PYROMIND_AGENT_STORAGE_ROOT}/conv-1"
    workspace = SandboxWorkspace(
        working_dir=workspace_root,
        storage_path=DEFAULT_MOUNT_PATH,
        sandbox_id="sbx-1",
        ws_base_url="https://pre-api.pyromind.ai/api/v1",
        client=client,
    )
    assert workspace.storage_path == DEFAULT_MOUNT_PATH
    commands: list[str] = []
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_workspace.run_terminal_command",
        lambda **kwargs: (commands.append(kwargs["command"]) or "ok", 0, False),
    )
    source = tmp_path / "script.py"
    source.write_text("print('hi')\n", encoding="utf-8")
    upload = workspace.file_upload(source, "scripts/script.py")
    assert upload.success
    assert client.files[f"{workspace_root}/scripts/script.py"] == b"print('hi')\n"

    download = workspace.file_download("scripts/script.py", tmp_path / "out.py")
    assert download.success
    assert (tmp_path / "out.py").read_bytes() == b"print('hi')\n"

    result = workspace.execute_command("python script.py", cwd="scripts")
    assert result.exit_code == 0
    assert commands[-1] == f"cd {workspace_root}/scripts && python script.py"
