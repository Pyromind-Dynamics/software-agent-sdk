"""Execute DataFlow sample runs inside the sandbox that owns the workspace.

A sandbox workspace has no host filesystem view, so the default sample path
stages inputs onto the agent-server host and publishes the results back. When
the workspace is served by a platform sandbox that can run commands and mount
user Storage, the run belongs where the data already is: this module uploads the
run spec and the runtime helpers, executes ``sandbox_sample.py`` through the
workspace command port, and reads back one JSON envelope.

Only the envelope and the run's log tails cross the boundary; the input, the
pipeline script, and the produced artifacts never leave the sandbox.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import shlex
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, runtime_checkable

from openhands.sdk.workspace.models import CommandResult
from openhands.tools.utils.workspace_staging import (
    STORAGE_ALIAS,
    WorkspaceStagingError,
    is_remote_workspace,
    resolve_workspace_path,
)


SANDBOX_RUNNER_FILENAME = "sandbox_sample.py"
SANDBOX_RUNTIME_ROOT = ".agent_tmp/dataflow-runtime"
SANDBOX_RUN_ROOT = ".agent_tmp/dataflow-runs"
ENV_SANDBOX_DATAFLOW_PYTHON = "PYROMIND_SANDBOX_DATAFLOW_PYTHON"
ENV_SANDBOX_DATAFLOW_VENV = "PYROMIND_SANDBOX_DATAFLOW_VENV"
ENV_SANDBOX_SYSTEM_PYTHON = "PYROMIND_SANDBOX_SYSTEM_PYTHON"
DEFAULT_SANDBOX_SYSTEM_PYTHON = "python3"
DEFAULT_DATAFLOW_VENV = "/tmp/pyromind-dataflow-venv"
_RUNTIME_READY_MARKER = ".ready"
_BOOTSTRAP_BUDGET_SECONDS = 1800
_CACHE_TIMEOUT_SECONDS = 120.0
_ENVELOPE_BEGIN = "__PM_DF_BEGIN__"
_ENVELOPE_END = "__PM_DF_END__"


class SandboxExecutionError(RuntimeError):
    """Raised when the sandbox cannot run a sample pipeline."""


@runtime_checkable
class SandboxCommandWorkspace(Protocol):
    """The slice of a workspace needed to run a sample inside its sandbox."""

    working_dir: str
    storage_path: str

    def execute_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
    ) -> CommandResult: ...

    def file_upload(
        self,
        source_path: str | Path,
        destination_path: str | Path,
    ) -> Any: ...


def supports_sandbox_execution(workspace: Any) -> bool:
    """True when a remote workspace can host the run instead of the agent host.

    Platform sandbox workspaces expose both a command port and the Storage mount
    the conversation addresses as ``storage/``. A remote workspace without the
    mount keeps the staging path, because a run there could not reach user
    Storage.
    """

    if not is_remote_workspace(workspace):
        return False
    if not isinstance(workspace, SandboxCommandWorkspace):
        return False
    return bool(str(workspace.working_dir)) and bool(str(workspace.storage_path))


@dataclass(frozen=True)
class SandboxTarget:
    """One workspace path as the sandbox sees it."""

    absolute: PurePosixPath
    relative: PurePosixPath
    display: str
    from_storage: bool


def sandbox_target(workspace: Any, path: str) -> SandboxTarget:
    """Resolve a workspace path for execution inside the sandbox.

    ``storage/...`` addresses the Storage mount, everything else the
    conversation workspace, mirroring the layout the sandbox shell sees.
    """

    relative, from_storage = resolve_workspace_path(workspace, path)
    if not relative.parts:
        raise WorkspaceStagingError(
            f"Path must name a file or directory inside the workspace: {path!r}"
        )
    root = (
        PurePosixPath(str(workspace.storage_path))
        if from_storage
        else PurePosixPath(str(workspace.working_dir))
    )
    display = (
        PurePosixPath(STORAGE_ALIAS, *relative.parts).as_posix()
        if from_storage
        else relative.as_posix()
    )
    return SandboxTarget(
        absolute=root / relative,
        relative=relative,
        display=display,
        from_storage=from_storage,
    )


def sandbox_target_from_absolute(
    workspace: Any, absolute: PurePosixPath
) -> SandboxTarget:
    """Describe an already-resolved sandbox path without re-validating it.

    Legacy pipeline arguments may name an output anywhere the workspace allows,
    so the result of joining such an argument to a resolved base has to come back
    as a target without the ``public_data/`` rule standard outputs carry.
    """

    root = PurePosixPath(str(workspace.working_dir))
    try:
        relative = absolute.relative_to(root)
    except ValueError:
        return SandboxTarget(
            absolute=absolute,
            relative=absolute,
            display=absolute.as_posix(),
            from_storage=False,
        )
    return SandboxTarget(
        absolute=absolute,
        relative=relative,
        display=relative.as_posix(),
        from_storage=False,
    )


def sandbox_execute(
    workspace: Any,
    command: str,
    *,
    timeout: float,
    cwd: str | Path | None = None,
) -> CommandResult:
    """Run one shell command inside the sandbox behind ``workspace``."""

    return workspace.execute_command(command, cwd=cwd, timeout=timeout)


def require_sandbox_success(result: CommandResult, action: str) -> str:
    """Return a command's stdout, or raise when the sandbox command failed."""

    if result.timeout_occurred:
        raise SandboxExecutionError(f"{action} timed out in the sandbox")
    if result.exit_code != 0:
        detail = (result.stdout or "").strip()[-2000:]
        raise SandboxExecutionError(
            f"{action} failed in the sandbox (exit {result.exit_code}): {detail}"
        )
    return result.stdout or ""


def sandbox_exists(workspace: Any, path: str) -> bool:
    """True when the sandbox has ``path``."""

    result = sandbox_execute(
        workspace, f"test -e {shlex.quote(path)}", timeout=_CACHE_TIMEOUT_SECONDS
    )
    return result.exit_code == 0


def sandbox_is_dir(workspace: Any, path: str) -> bool:
    """True when the sandbox path is a directory."""

    result = sandbox_execute(
        workspace, f"test -d {shlex.quote(path)}", timeout=_CACHE_TIMEOUT_SECONDS
    )
    return result.exit_code == 0


def sandbox_read_text(workspace: Any, path: str, *, limit: int = 1_000_000) -> str:
    """Read a bounded text file out of the sandbox."""

    result = sandbox_execute(
        workspace,
        f"head -c {int(limit)} {shlex.quote(path)}",
        timeout=_CACHE_TIMEOUT_SECONDS,
    )
    if result.exit_code != 0:
        raise SandboxExecutionError(
            f"cannot read {path} from the sandbox: "
            f"{(result.stdout or '').strip()[-500:]}"
        )
    return result.stdout or ""


def sandbox_upload(workspace: Any, source: Path, destination: str) -> None:
    """Upload one host file into the sandbox, raising when it does not arrive."""

    uploaded = workspace.file_upload(str(source), destination)
    if not getattr(uploaded, "success", False):
        detail = getattr(uploaded, "error", None) or "upload failed"
        raise SandboxExecutionError(
            f"cannot upload {source.name} into the sandbox: {detail}"
        )


class SandboxSampleExecutor:
    """Run one DataFlow sample through a platform sandbox command port."""

    def __init__(
        self,
        workspace: Any,
        *,
        runtime_dir: Path,
        runtime_filenames: Sequence[str],
        runtime_fingerprint: str,
    ) -> None:
        self._workspace = workspace
        self._runtime_dir = runtime_dir
        self._runtime_filenames = tuple(runtime_filenames)
        self._runtime_fingerprint = runtime_fingerprint
        self._run_id: str | None = None

    @property
    def _runner_source(self) -> Path:
        return self._runtime_dir / SANDBOX_RUNNER_FILENAME

    @property
    def runtime_fingerprint(self) -> str:
        return self._runtime_fingerprint

    def runtime_path(self) -> str:
        """Directory the sandbox keeps the runtime helpers in."""
        return str(
            PurePosixPath(str(self._workspace.working_dir))
            / SANDBOX_RUNTIME_ROOT
            / self._runtime_fingerprint
        )

    def _run_dir_path(self, run_id: str) -> str:
        return str(
            PurePosixPath(str(self._workspace.working_dir)) / SANDBOX_RUN_ROOT / run_id
        )

    def ensure_runtime(self) -> str:
        """Upload the runtime helpers once per fingerprint and return the dir."""
        target = self.runtime_path()
        marker = f"{target}/{_RUNTIME_READY_MARKER}"
        result = sandbox_execute(
            self._workspace,
            f"cat {shlex.quote(marker)} 2>/dev/null",
            timeout=_CACHE_TIMEOUT_SECONDS,
        )
        if result.exit_code == 0 and (result.stdout or "").strip() == (
            self._runtime_fingerprint
        ):
            return target
        sources = [self._runtime_dir / name for name in self._runtime_filenames]
        sources.append(self._runner_source)
        missing = [str(path) for path in sources if not path.is_file()]
        if missing:
            raise SandboxExecutionError(
                "DataFlow runtime files are missing on the agent host: "
                + ", ".join(missing)
            )
        require_sandbox_success(
            sandbox_execute(
                self._workspace, f"mkdir -p {shlex.quote(target)}", timeout=60
            ),
            "creating the sandbox runtime directory",
        )
        for source in sources:
            sandbox_upload(self._workspace, source, f"{target}/{source.name}")
        require_sandbox_success(
            sandbox_execute(
                self._workspace,
                f"printf '%s' {shlex.quote(self._runtime_fingerprint)} > "
                f"{shlex.quote(marker)}",
                timeout=60,
            ),
            "stamping the sandbox runtime directory",
        )
        return target

    def run(self, spec: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
        """Execute the sample in the sandbox and return its JSON envelope."""
        runtime_dir = self.ensure_runtime()
        run_id = uuid.uuid4().hex
        self._run_id = run_id
        run_dir = self._run_dir_path(run_id)
        marker = uuid.uuid4().hex
        system_python = (
            os.environ.get(ENV_SANDBOX_SYSTEM_PYTHON) or DEFAULT_SANDBOX_SYSTEM_PYTHON
        )
        payload = {
            **spec,
            "runtime_dir": runtime_dir,
            "marker": marker,
            "system_python": system_python,
        }
        try:
            with tempfile.TemporaryDirectory(prefix="pyromind-df-spec-") as staging:
                spec_path = Path(staging) / "run.json"
                spec_path.write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
                require_sandbox_success(
                    sandbox_execute(
                        self._workspace,
                        f"mkdir -p {shlex.quote(run_dir)}",
                        timeout=60,
                    ),
                    "creating the sandbox run directory",
                )
                destination = f"{run_dir}/run.json"
                sandbox_upload(self._workspace, spec_path, destination)
            command = (
                f"chmod 600 {shlex.quote(destination)}; "
                f"{shlex.quote(system_python)} "
                f"{shlex.quote(runtime_dir + '/' + SANDBOX_RUNNER_FILENAME)} "
                f"--spec {shlex.quote(destination)}; rc=$?; "
                f"rm -rf {shlex.quote(run_dir)}; (exit $rc)"
            )
            result = sandbox_execute(
                self._workspace,
                command,
                timeout=timeout + _BOOTSTRAP_BUDGET_SECONDS,
            )
        finally:
            self._run_id = None
        if result.timeout_occurred:
            raise SandboxExecutionError(
                f"the sandbox sample run timed out after "
                f"{int(timeout + _BOOTSTRAP_BUDGET_SECONDS)}s"
            )
        return _parse_envelope(result.stdout or "", marker)

    def interrupt(self) -> None:
        """Best-effort cancel of the running sample inside the sandbox."""
        run_id = self._run_id
        if run_id is None:
            return
        try:
            sandbox_execute(
                self._workspace,
                f"pkill -f {shlex.quote(run_id)} >/dev/null 2>&1; true",
                timeout=30,
            )
        except Exception:  # noqa: BLE001 - interruption is best effort
            return


def _parse_envelope(output: str, marker: str) -> dict[str, Any]:
    begin = f"{_ENVELOPE_BEGIN}{marker}"
    end = f"{_ENVELOPE_END}{marker}"
    start = output.rfind(begin)
    if start < 0:
        raise SandboxExecutionError(
            "the sandbox sample run returned no result envelope: "
            + output.strip()[-2000:]
        )
    stop = output.find(end, start)
    if stop < 0:
        raise SandboxExecutionError(
            "the sandbox sample result envelope was cut off: "
            + output[start:].strip()[-2000:]
        )
    # The runner wraps its base64 payload, and the terminal bridge may wrap it
    # again, so every line break between the markers is transport noise.
    body = "".join(output[start + len(begin) : stop].split())
    try:
        decoded = base64.b64decode(body, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SandboxExecutionError(
            f"the sandbox sample result envelope is not valid base64: {exc}"
        ) from exc
    try:
        payload = json.loads(decoded)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SandboxExecutionError(
            f"the sandbox sample result envelope is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SandboxExecutionError(
            "the sandbox sample result envelope is not a JSON object"
        )
    return payload


def sandbox_dataflow_venv(version: str) -> str:
    """Interpreter cache the sandbox creates when the image has no DataFlow."""
    configured = os.environ.get(ENV_SANDBOX_DATAFLOW_VENV)
    if configured:
        return configured
    return f"{DEFAULT_DATAFLOW_VENV}-{version}"
