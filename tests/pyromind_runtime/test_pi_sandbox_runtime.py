from __future__ import annotations

import asyncio
import io
import json
import logging
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
    LEGACY_STORAGE_ALIAS,
    PYROMIND_AGENT_RESOURCES_ROOT,
    PYROMIND_AGENT_STORAGE_ROOT,
    STORAGE_HOST_PATH,
    SandboxExecutionManager,
    SandboxSettings,
    _sandbox_client,
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

    def wait_for_sandbox_status(
        self,
        sandbox_id: str,
        target_status: str,
        timeout: int = 300,
        check_interval: int = 3,
        intermediate_statuses: list[str] | None = None,
    ) -> bool:
        self.calls.append(f"wait:{sandbox_id}:{target_status}")
        if self._status.strip().lower() in {"error", "failed"}:
            return False
        self._status = target_status
        return True

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


def _context(
    tmp_path: Path, extra: dict | None = None, *, conversation_id: str = "conv-1"
) -> ToolExecutionContext:
    return ToolExecutionContext(
        conversation_id=conversation_id,
        workspace_root=tmp_path / conversation_id,
        request_context=RequestContext(
            user_id="user-1",
            x_cluster="us-west-1#pre",
        ),
        model_configuration={},
        extra=extra or {},
    )


def test_default_sandbox_client_requires_platform_credentials(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="auth_token cookie or Bearer"):
        _sandbox_client(_context(tmp_path))


def _uploaded_scripts(client: _FakeSandboxClient) -> list[str]:
    """Scripts this manager shipped into the sandbox, in upload order."""
    paths = [path for path in client.files if path.startswith("/tmp/pi-sandbox-")]
    return [client.files[path].decode("utf-8") for path in paths]


def _uploaded_script(client: _FakeSandboxClient) -> str:
    """Script this manager shipped into the sandbox for its last command."""
    scripts = _uploaded_scripts(client)
    assert scripts, "no sandbox script was uploaded"
    return scripts[-1]


_SCRIPT_COMMAND = re.compile(r"sh (/tmp/pi-sandbox-[0-9a-f]+\.sh)")
_MOVE_LINE = re.compile(r"^mv (\S+) (\S+)$", re.MULTILINE)
_TAR_LINE = re.compile(r"tar -xzf (\S+) -C (\S+)")
_PROBE_SCRIPT = re.compile(r"test -f (\S+)\n")


def _run_sandbox_script(
    client: _FakeSandboxClient, script: str
) -> tuple[str, int, bool]:
    """Apply one uploaded script the way the sandbox's own shell would.

    Only the shapes the resource staging uses are modelled: the cache probe,
    the rename that publishes an uploaded archive, and the unpack. Everything
    else is a success, which is all the tests assert about it.
    """
    probe = _PROBE_SCRIPT.fullmatch(script)
    if probe is not None:
        return ("", 0 if probe.group(1) in client.files else 1, False)
    for source, destination in _MOVE_LINE.findall(script):
        if source not in client.files:
            return (f"mv: cannot stat '{source}'", 1, False)
        client.files[destination] = client.files.pop(source)
    for archive, workspace in _TAR_LINE.findall(script):
        if archive not in client.files:
            return (f"tar: {archive}: Cannot open", 1, False)
        with tarfile.open(fileobj=io.BytesIO(client.files[archive])) as tar:
            for member in tar.getmembers():
                extracted = tar.extractfile(member) if member.isfile() else None
                if extracted is None:
                    continue
                client.files[f"{workspace}/{member.name}"] = extracted.read()
    return ("", 0, False)


def _run_sandbox_command(
    client: _FakeSandboxClient, command: str
) -> tuple[str, int, bool]:
    """Answer one manager command through the script it uploaded."""
    match = _SCRIPT_COMMAND.match(command)
    if match is None:
        return ("", 0, False)
    return _run_sandbox_script(client, client.files[match.group(1)].decode("utf-8"))


def _dataflow_skills(tmp_path: Path) -> Path:
    """Minimal skills tree the warm-up looks for."""
    skills = tmp_path / "skills"
    script = skills / "data-processing" / "scripts" / "preparation"
    script.mkdir(parents=True)
    (script / "sandbox_sample.py").write_text("", encoding="utf-8")
    return skills


def _warmup_command(commands: list[str]) -> str:
    """The DataFlow warm-up command among the sandbox commands."""
    warmups = [command for command in commands if "sandbox_sample.py" in command]
    assert len(warmups) == 1, commands
    return warmups[0]


@pytest.fixture
def files(tmp_path: Path) -> PiSessionFiles:
    session_files = PiSessionFiles(tmp_path / "conv-1")
    session_files.initialize({"session_id": "conv-1"})
    return session_files


@pytest.fixture(autouse=True)
def _terminal_file_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirror terminal file writes into the fake client's file store.

    The manager writes scripts and staged files over the terminal WebSocket;
    the sandbox-runtime tests keep inspecting ``client.files``, so the write
    helper is redirected to the same store here. The terminal transport itself
    is covered by the sandbox tool tests.
    """

    def write(
        client: _FakeSandboxClient,
        *,
        ws_base_url: str,
        sandbox_id: str,
        path: str,
        source: bytes | str | Path,
        timeout_seconds: int = 60,
    ) -> None:
        content = (
            Path(source).read_bytes() if isinstance(source, (str, Path)) else source
        )
        client.files[path] = content
        client.calls.append(f"write:{path}")

    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.write_sandbox_file", write
    )


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
    assert f"mkdir -p {endpoint['workspace_path']}/public_data" in script
    # Storage is addressed by its mount path, so the old symlink must be gone:
    # the script only clears one an older runtime left behind.
    assert "ln -s" not in script
    assert (f"if [ -L {endpoint['workspace_path']}/{LEGACY_STORAGE_ALIAS} ]") in script
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
    probes: list[dict[str, object]] = []

    def terminal(**kwargs: object) -> tuple[str, int, bool]:
        probes.append(kwargs)
        return "", 0, False

    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command", terminal
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    endpoint = asyncio.run(manager.ensure(_context(tmp_path), files))

    assert client.calls == ["get:sbx-1"]
    assert [probe["command"] for probe in probes] == ["true"]
    assert endpoint["sandbox_id"] == "sbx-1"
    record = files.load_sandbox()
    assert record is not None
    assert record["created_at"] == "2026-01-01T00:00:00+00:00"


def test_ensure_rebuilds_a_running_sandbox_whose_terminal_is_dead(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    """The control plane can report "running" for a container whose data plane
    stopped answering; reusing it turns every later call into a hung connect."""
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

    def terminal(**kwargs: object) -> tuple[str, int, bool]:
        # The probe never reaches the container; the prepare script does.
        if kwargs["command"] == "true":
            raise TimeoutError("connect timed out")
        return "", 0, False

    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command", terminal
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    asyncio.run(manager.ensure(_context(tmp_path), files))

    assert "delete:sbx-1" in client.calls
    assert "create:running" in client.calls


def test_prewarm_logs_why_the_sandbox_could_not_be_prepared(
    tmp_path: Path, files: PiSessionFiles, caplog
) -> None:
    """The warning used to name only the exception type, which hid the reason."""
    client = _FakeSandboxClient(
        create_error=RuntimeError("quota exceeded for us-west-1")
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    with caplog.at_level("WARNING"):
        asyncio.run(manager._prewarm(_context(tmp_path), files))

    assert "quota exceeded for us-west-1" in caplog.text


def test_ensure_refreshes_a_cached_endpoint_bundle_on_request(
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
    clients: list[_FakeSandboxClient] = []

    def factory(context: ToolExecutionContext) -> _FakeSandboxClient:
        client = _FakeSandboxClient(status="running", api_key=f"key-{len(clients)}")
        clients.append(client)
        return client

    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        lambda **kwargs: ("", 0, False),
    )
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


def test_ensure_clears_the_legacy_storage_alias_from_a_running_sandbox(
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
    script = _uploaded_script(client)
    assert "ln -s" not in script
    assert (f"if [ -L {endpoint['workspace_path']}/{LEGACY_STORAGE_ALIAS} ]") in script
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

    # resume() only accepts the request, so the manager polls until the pod is
    # running before it touches the workspace.
    assert client.lifecycle_calls == [
        "get:sbx-1",
        "resume:sbx-1",
        "wait:sbx-1:running",
        "get:sbx-1",
    ]
    assert endpoint["sandbox_id"] == "sbx-1"


def test_ensure_waits_for_a_sandbox_that_is_still_starting(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    files.save_sandbox(
        {
            "sandbox_id": "sbx-1",
            "mount_path": DEFAULT_MOUNT_PATH,
            "storage_host_path": STORAGE_HOST_PATH,
        }
    )

    client = _FakeSandboxClient(status="Pending")
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        lambda **kwargs: ("", 0, False),
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    endpoint = asyncio.run(manager.ensure(_context(tmp_path), files))

    # A pod that is still coming up must not be torn down and recreated.
    assert client.lifecycle_calls == [
        "get:sbx-1",
        "wait:sbx-1:running",
        "get:sbx-1",
    ]
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
    # The skills tree is hundreds of files, so it travels as one archive the
    # sandbox unpacks, which also carries the empty __init__.py files the
    # chunked upload API cannot write. The archive is cached in the Storage
    # mount, so the next conversation of this user does not upload it again.
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "__init__.py").write_bytes(b"")
    (knowledge / "notes.md").write_bytes(b"hello\n")
    cache = knowledge / "__pycache__"
    cache.mkdir()
    (cache / "notes.cpython-313.pyc").write_bytes(b"\x00")
    client = _FakeSandboxClient()
    commands: list[str] = []

    def run_terminal_command(**kwargs: object) -> tuple[str, int, bool]:
        command = str(kwargs["command"])
        commands.append(command)
        return _run_sandbox_command(client, command)

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
    # Bytecode cached next to host scripts is not part of the sandbox tree.
    assert (
        f"{workspace}/knowledge/__pycache__/notes.cpython-313.pyc" not in client.files
    )
    # The tree arrives in one archive: nothing is written file by file.
    assert [
        call for call in client.calls if call.startswith(f"write:{workspace}/")
    ] == []
    cached = [
        call
        for call in client.calls
        if call.startswith(
            f"write:{DEFAULT_MOUNT_PATH}{PYROMIND_AGENT_RESOURCES_ROOT}/"
        )
    ]
    assert len(cached) == 1
    assert ".tgz.partial-" in cached[0]
    published = [
        name
        for name in client.files
        if name.startswith(f"{DEFAULT_MOUNT_PATH}{PYROMIND_AGENT_RESOURCES_ROOT}/")
    ]
    assert len(published) == 1
    assert published[0].endswith("/pi-session-resources.tgz")


def test_ensure_reuses_the_cached_resource_archive(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    # The archive is named after the content of the resource tree, so a second
    # conversation of the same user unpacks it from Storage instead of paying
    # the upload again.
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "notes.md").write_bytes(b"hello\n")
    client = _FakeSandboxClient()

    def run_terminal_command(**kwargs: object) -> tuple[str, int, bool]:
        return _run_sandbox_command(client, str(kwargs["command"]))

    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        run_terminal_command,
    )
    manager = SandboxExecutionManager(
        client_factory=lambda context: client,
        resource_roots=[("knowledge", knowledge)],
    )

    async def exercise() -> None:
        await manager.ensure(_context(tmp_path), files)
        second = PiSessionFiles(tmp_path / "conv-2")
        second.initialize({"session_id": "conv-2"})
        await manager.ensure(_context(tmp_path, conversation_id="conv-2"), second)

    asyncio.run(exercise())

    uploads = [
        call
        for call in client.calls
        if call.startswith(
            f"write:{DEFAULT_MOUNT_PATH}{PYROMIND_AGENT_RESOURCES_ROOT}/"
        )
    ]
    assert len(uploads) == 1
    workspace = f"{DEFAULT_MOUNT_PATH}{PYROMIND_AGENT_STORAGE_ROOT}/conv-2"
    assert client.files[f"{workspace}/knowledge/notes.md"] == b"hello\n"


def test_prewarm_starts_the_sandbox_before_the_first_touch(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    # The container and its workspace are prepared while the model reads the
    # first message, so the first tool call reuses them instead of waiting.
    client = _FakeSandboxClient()
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        lambda **kwargs: _run_sandbox_command(client, str(kwargs["command"])),
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    async def exercise() -> dict:
        manager.prewarm(_context(tmp_path), files)
        # A second request while the first one is in flight is not a second
        # start.
        manager.prewarm(_context(tmp_path), files)
        await asyncio.sleep(0.1)
        return await manager.ensure(_context(tmp_path), files)

    endpoint = asyncio.run(exercise())

    assert endpoint["sandbox_id"] == "sbx-1"
    assert client.calls.count("create:running") == 1
    assert manager._prewarms == {}


def test_prewarm_can_be_switched_off(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    monkeypatch.setenv("PYROMIND_SANDBOX_PREWARM", "0")
    client = _FakeSandboxClient()
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    async def exercise() -> None:
        manager.prewarm(_context(tmp_path), files)
        await asyncio.sleep(0.05)

    asyncio.run(exercise())

    assert manager._prewarms == {}
    assert client.lifecycle_calls == []
    assert files.load_sandbox() is None


def test_ensure_warms_the_dataflow_runtime_after_resources(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    # The sandbox image ships without open-dataflow, so preparing the cached
    # venv at sandbox start keeps the first data-processing run from paying for
    # the install. The warm-up runs the same entry point a run would use.
    monkeypatch.setenv("PYROMIND_SANDBOX_SYSTEM_PYTHON", "python3.12")
    monkeypatch.setenv("PYROMIND_SANDBOX_DATAFLOW_VENV", "/tmp/dataflow-venv")
    monkeypatch.delenv("PYROMIND_SANDBOX_DATAFLOW_PYTHON", raising=False)
    client = _FakeSandboxClient()
    commands: list[str] = []

    def run_terminal_command(**kwargs: object) -> tuple[str, int, bool]:
        command = str(kwargs["command"])
        commands.append(command)
        if "sandbox_sample.py" in command:
            return ('{"ready": true, "source": "installed"}', 0, False)
        return _run_sandbox_command(client, command)

    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        run_terminal_command,
    )
    skills = _dataflow_skills(tmp_path)
    manager = SandboxExecutionManager(
        client_factory=lambda context: client,
        resource_roots=[(".agents/skills", skills)],
    )

    async def exercise() -> None:
        await manager.ensure(_context(tmp_path), files)
        # The warm-up is a background task; let it finish before asserting.
        await asyncio.sleep(0.1)

    asyncio.run(exercise())

    workspace = f"{DEFAULT_MOUNT_PATH}{PYROMIND_AGENT_STORAGE_ROOT}/conv-1"
    command = _warmup_command(commands)
    assert "--prepare" in command
    assert (
        f"{workspace}/.agents/skills/data-processing/scripts/preparation/"
        "sandbox_sample.py"
    ) in command
    assert "env PYROMIND_SANDBOX_DATAFLOW_VENV=/tmp/dataflow-venv" in command
    assert "PYROMIND_SANDBOX_SYSTEM_PYTHON=python3.12" in command
    # The install yields CPU and disk to the interactive session instead of
    # competing with the first command.
    assert "nice -n 19" in command
    assert "ionice -c 3 -n 7" in command
    # The install is silent for minutes; the heartbeat keeps the TTY alive.
    assert "sleep 20" in command
    assert manager._warmups == {}


def test_ensure_skips_the_dataflow_warmup_when_disabled(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    monkeypatch.setenv("PYROMIND_SANDBOX_DATAFLOW_WARMUP", "0")
    client = _FakeSandboxClient()
    commands: list[str] = []
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        lambda **kwargs: (commands.append(kwargs["command"]) or "", 0, False),
    )
    manager = SandboxExecutionManager(
        client_factory=lambda context: client,
        resource_roots=[(".agents/skills", _dataflow_skills(tmp_path))],
    )

    asyncio.run(manager.ensure(_context(tmp_path), files))

    assert "sandbox_sample.py" not in "\n".join(commands)
    assert manager._warmups == {}


def test_ensure_survives_a_failed_dataflow_warmup(
    tmp_path: Path, files: PiSessionFiles, monkeypatch, caplog
) -> None:
    # A warm-up failure is not fatal: the run path still installs on demand.
    client = _FakeSandboxClient()

    def run_terminal_command(**kwargs: object) -> tuple[str, int, bool]:
        if "sandbox_sample.py" in str(kwargs["command"]):
            return ("could not install", 1, False)
        return ("", 0, False)

    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        run_terminal_command,
    )
    manager = SandboxExecutionManager(
        client_factory=lambda context: client,
        resource_roots=[(".agents/skills", _dataflow_skills(tmp_path))],
    )

    async def exercise() -> dict:
        endpoint = await manager.ensure(_context(tmp_path), files)
        await asyncio.sleep(0.1)
        return endpoint

    with caplog.at_level(logging.WARNING):
        endpoint = asyncio.run(exercise())

    assert endpoint["sandbox_id"] == "sbx-1"
    assert "Pi DataFlow warm-up did not finish" in caplog.text
    assert manager._warmups == {}


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
    files.save_pending_sandbox_staging(
        source_conversation_id="conv-source", workflow_dsl="nodes: []\n"
    )
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
    assert "ln -s" not in scripts[-1]
    assert (
        client.files[
            f"{DEFAULT_MOUNT_PATH}{PYROMIND_AGENT_STORAGE_ROOT}/conv-1"
            "/public_data/workflow_canvas/workflow.py"
        ]
        == b"nodes: []\n"
    )
    assert files.load_pending_sandbox_staging() is None


def test_ensure_skips_copy_when_fork_source_never_materialized(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    files.save_pending_sandbox_staging(
        source_conversation_id=None, workflow_dsl="nodes: []\n"
    )
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
    assert files.load_pending_sandbox_staging() is None


def test_ensure_flushes_staged_workflow_into_a_running_sandbox(
    tmp_path: Path, files: PiSessionFiles, monkeypatch
) -> None:
    client = _FakeSandboxClient()
    files.save_sandbox(
        {
            "sandbox_id": "sbx-1",
            "workspace_path": (
                f"{DEFAULT_MOUNT_PATH}{PYROMIND_AGENT_STORAGE_ROOT}/conv-1"
            ),
            "mount_path": DEFAULT_MOUNT_PATH,
            "storage_host_path": STORAGE_HOST_PATH,
            "workspace_version": _WORKSPACE_LAYOUT_VERSION,
        }
    )
    files.save_pending_sandbox_staging(
        source_conversation_id=None, workflow_dsl="nodes: []\n"
    )
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_runtime.run_terminal_command",
        lambda **kwargs: ("", 0, False),
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)

    asyncio.run(manager.ensure(_context(tmp_path), files))

    assert "create:running" not in client.calls
    assert (
        client.files[
            f"{DEFAULT_MOUNT_PATH}{PYROMIND_AGENT_STORAGE_ROOT}/conv-1"
            "/public_data/workflow_canvas/workflow.py"
        ]
        == b"nodes: []\n"
    )
    assert files.load_pending_sandbox_staging() is None


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
    monkeypatch.setattr(
        "harness_adapter.pi_adapter.sandbox_workspace.write_sandbox_file",
        lambda client, **kwargs: client.files.__setitem__(
            kwargs["path"], Path(kwargs["source"]).read_bytes()
        ),
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
