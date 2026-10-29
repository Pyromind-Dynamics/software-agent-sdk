from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import shlex
import threading
import time
from collections.abc import Callable, Coroutine, Mapping, Sequence
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from uuid import uuid4

from agentgenome import GenomeService, RunCancelled
from agentgenome.dispatch import invoke as invoke_genome
from core.ports import (
    Answer,
    AwaitResult,
    PortError,
    Ports,
    ShellResult,
    ToolCancellation,
    ToolResult,
)
from pyromind_runtime.domain.content import JsonObject, JsonValue
from pyromind_runtime.ports.workflows import WorkflowEventSink

from harness_adapter.pi_adapter.runner import PiRunnerError
from openhands.tools.utils.storage_paths import REMOVED_ALIAS


logger = logging.getLogger(__name__)

# A planning probe resolves one child of `public_data` because the workspace root
# itself is refused as conversation metadata: the policy only opens entries that
# live *inside* it. Nothing is read; only the mount prefix is wanted.
_MOUNT_PROBE = "public_data/.agentgenome-mount-probe"


type Request = Callable[[str, JsonObject], Coroutine[Any, Any, Any]]
type Subscribe = Callable[[str, WorkflowEventSink | None], None]


@dataclass(frozen=True)
class ProbeRead:
    content: bytes | None
    error: str | None = None


def _is_missing_file(error: Exception) -> bool:
    """Whether the execution plane answered "that file is not there".

    The sandbox file API reports a missing file as a 400 request error, and only the
    portal's own wording says which failure it was.
    """
    message = str(error)
    return "FILE_READ_ERROR" in message and "No such file or directory" in message


def _missing_read(error: str | None) -> bool:
    if not error:
        return False
    message = error.lower()
    return (
        "no such file" in message
        or "not found" in message
        or "does not exist" in message
        or "cannot read file" in message
    )


def _oversized_read(error: str | None) -> bool:
    return bool(error) and "file exceeds size limit" in error.lower()


_SANDBOX_ALIASES = ("public_data", ".agents", "knowledge")
# Agent workspaces live at ``<storage mount>/.pyromind-agent/<conversation>``, so
# the mount is whatever sits above that directory.
_AGENT_DIRECTORY = "/.pyromind-agent/"
_SLOW_CALLS = frozenset({"execution.write", "execution.protect", "execution.read_many"})
# A finished platform run's own words for "done". The probe used to accept only
# `succeeded`, so any other spelling parked the run for ever.
_PLATFORM_SUCCEEDED_STATUSES = frozenset(
    {
        "succeeded",
        "success",
        "finished",
        "completed",
        "complete",
        "done",
        "ok",
        "passed",
    }
)
_PLATFORM_FAILED_STATUSES = frozenset(
    {
        "failed",
        "failure",
        "cancelled",
        "canceled",
        "error",
        "terminated",
        "timeout",
        "killed",
    }
)
# Not base64, so a real head can never be mistaken for the directory report.
_DIRECTORY_MARK = "<dir/>"
# One execution-plane read window; mirrors the runtime's per-file ceiling
# (512 KiB encodes under the 1 MiB frame limit).
_READ_LIMIT = 512 * 1024
_STATUS_REPORT = "report.status.json"
# Vision pipelines resolve the images a manifest names relative to the manifest
# itself, so a manifest staged away from its images only fails minutes later on
# the platform. The preflight reports that before the submission is paid for,
# reading the same bounded head (50 records, 200 images) the data-preparation
# tool stages, and running in the plane because that is where the files live.
_MANIFEST_PRECHECK = """\
import json, sys
from pathlib import Path

suffixes = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
record_limit = 50
image_limit = 200


def strings(value, depth=0):
    if depth > 3:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from strings(nested, depth + 1)
    elif isinstance(value, list):
        for nested in value:
            yield from strings(nested, depth + 1)


manifest = Path(sys.argv[1])
root = manifest.parent
missing, seen, records = [], set(), 0
try:
    handle = open(manifest, encoding="utf-8")
except OSError:
    print(json.dumps({"records": 0}))
    raise SystemExit(0)
with handle:
    for index, line in enumerate(handle):
        if index >= record_limit:
            break
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        records += 1
        for value in strings(record):
            candidate = value.strip()
            if (
                not candidate
                or "://" in candidate
                or candidate.startswith("/")
                or candidate in seen
            ):
                continue
            if Path(candidate).suffix.lower() not in suffixes:
                continue
            seen.add(candidate)
            if len(seen) > image_limit:
                break
            if not (root / candidate).exists():
                missing.append(candidate)
print(json.dumps({"records": records, "missing": missing[:5]}))
"""
_METADATA_PROBE_MANY = """\
import json, sys
from pathlib import Path

answers = {}
for raw in json.loads(sys.argv[1]):
    path = Path(raw)
    try:
        if not path.exists():
            answers[raw] = {"exists": False}
            continue
        stat = path.stat()
        answers[raw] = {
            "exists": True,
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "size": stat.st_size,
        }
    except OSError as error:
        answers[raw] = {"error": f"{type(error).__name__}: {error}"}
print(json.dumps(answers))
"""


_METADATA_PROBE = """\
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
try:
    exists = p.exists()
except OSError as error:
    print(json.dumps({"error": f"{type(error).__name__}: {error}"}))
    raise SystemExit(2)
if not exists:
    print(json.dumps({"exists": False}))
else:
    try:
        stat = p.stat()
    except OSError as error:
        print(json.dumps({"error": f"{type(error).__name__}: {error}"}))
        raise SystemExit(2)
    print(json.dumps({
        "exists": True,
        "is_file": p.is_file(),
        "is_dir": p.is_dir(),
        "size": stat.st_size,
    }))
"""


def _storage_alias(path: str, mount: str | None) -> str | None:
    """The sandbox spelling of a Storage path that a business tool declared.

    Platform tools address Storage without the mount prefix (`/datasets/...`,
    `datasets/...`), while the sandbox reads Storage under its mount
    (`/target-workspace/datasets/...`). A path the sandbox can already resolve
    is left alone.
    """
    if mount is None:
        return None
    stripped = path.lstrip("/")
    if not stripped:
        return None
    head = stripped.split("/", 1)[0]
    if not path.startswith("/") and head in _SANDBOX_ALIASES:
        return None
    if not path.startswith("/") and head == REMOVED_ALIAS:
        # The sandbox path policy reports the removal itself; remapping it here
        # would hide that error behind a "file not found".
        return None
    return f"{mount.rstrip('/')}/{stripped}"


def _tool_path(
    value: str,
    declared: Mapping[str, str],
    workspace: str,
    package: str,
    mount: str | None = None,
) -> str:
    """Restate one graph path in the vocabulary a business tool reads.

    A run step hands the same value to shell commands and to business tools, and
    those read paths in different vocabularies: the shell needs a sandbox path,
    while a tool resolves `public_data/...` itself. Give each
    tool back the spelling the graph declared, and make run-directory paths
    workspace-relative so they survive a remote workspace. A Storage artifact
    pinned beside its input is absolute in the sandbox, so it loses the mount
    prefix the platform tool never sees.
    """
    original = declared.get(value)
    if original is not None:
        return original
    if (
        mount
        and value.startswith(f"{mount}/")
        and not value.startswith(f"{workspace}/")
    ):
        return value[len(mount) :]
    if workspace and value.startswith(f"{workspace}/"):
        return value[len(workspace) + 1 :]
    if not value.startswith("/") and value.split("/", 1)[0] == "scripts":
        return f"{package}/{value}"
    return value


class _AwaitHost(Protocol):
    workspace: str

    def read_probe(self, path: str, limit: int) -> ProbeRead: ...

    def metadata(self, path: str) -> dict[str, Any]: ...

    def _resolve(self, path: str, write: bool) -> str: ...


class BridgeAwait:
    """Answer one await probe from the platform's own terminal report.

    The graph parks on the receipt; this port is the only place that knows a
    Pyromind run writes ``report.json`` next to its product. Every probe is a
    couple of short reads, so a parked run costs nothing while it waits.
    """

    def __init__(self, host: _AwaitHost) -> None:
        self.host = host

    def probe(self, receipt_path: str) -> AwaitResult:
        receipt, error = self._json_probe(receipt_path, 65536)
        if error is not None:
            return AwaitResult(
                "failed",
                error=f"receipt is unreadable: {error}",
                detail={"receipt": receipt_path},
            )
        if not isinstance(receipt, dict):
            return AwaitResult("pending", detail={"receipt": receipt_path})
        task_id = receipt.get("task_id")
        task_id = task_id if isinstance(task_id, str) and task_id else None
        directory = receipt.get("output_dir")
        if not isinstance(directory, str) or not directory.strip():
            return AwaitResult(
                "pending", detail={"receipt": receipt_path, "task_id": task_id}
            )
        target = self._sandbox_path(directory.strip().rstrip("/"))
        report, error = self._json_probe(f"{target}/{_STATUS_REPORT}", 65536)
        if error is not None:
            return AwaitResult(
                "failed",
                error=f"platform status report is unreadable: {error}",
                detail={"output_dir": target, "task_id": task_id},
            )
        source = "status"
        if not isinstance(report, dict):
            report, error = self._json_probe(f"{target}/report.json")
            source = "report"
            if error is not None:
                if _oversized_read(error):
                    return AwaitResult(
                        "failed",
                        error=(
                            "platform report exceeds the 512 KiB host read limit "
                            "and report.status.json is missing"
                        ),
                        detail={"output_dir": target, "task_id": task_id},
                    )
                if _missing_read(error):
                    return AwaitResult(
                        "pending",
                        detail={"output_dir": target, "task_id": task_id},
                    )
                return AwaitResult(
                    "failed",
                    error=f"platform report is unreadable: {error}",
                    detail={"output_dir": target, "task_id": task_id},
                )
        if not isinstance(report, dict):
            # A running platform task can expose a valid JSON placeholder before
            # the final report object is written. Only malformed JSON is a probe
            # failure; a non-object is still an in-flight state.
            return AwaitResult(
                "pending",
                detail={
                    "output_dir": target,
                    "task_id": task_id,
                    "report_type": type(report).__name__,
                    "status_source": source,
                },
            )
        status = str(report.get("status") or "").strip().lower()
        detail = {
            "output_dir": target,
            "status": status,
            "task_id": task_id,
            "status_source": source,
        }
        if status in _PLATFORM_FAILED_STATUSES:
            return AwaitResult("failed", error=f"platform run {status}", detail=detail)
        if status not in _PLATFORM_SUCCEEDED_STATUSES:
            return AwaitResult("pending", detail=detail)
        product = self._report_product(report, target)
        try:
            product_metadata = self.host.metadata(product)
        except Exception as error:  # noqa: BLE001 - a probe must report its cause
            if _missing_read(str(error)):
                return AwaitResult("pending", detail={**detail, "product": product})
            return AwaitResult(
                "failed",
                error=f"succeeded platform run product metadata is unreadable: {error}",
                detail={**detail, "product": product},
            )
        if not product_metadata.get("exists"):
            return AwaitResult("pending", detail={**detail, "product": product})
        return AwaitResult("completed", product=product, detail=detail)

    def _report_product(self, report: dict[str, Any], target: str) -> str:
        """The file a finished run leaves behind.

        The platform names its product in the report; the old hard-coded
        ``processed.jsonl`` only matched one pipeline shape. Fall back to it so
        existing chains keep working.
        """
        for key in (
            "product",
            "product_path",
            "output_path",
            "result_path",
            "processed_path",
            "artifact",
        ):
            value = report.get(key)
            if isinstance(value, str) and value.strip():
                path = value.strip()
                return (
                    self._sandbox_path(path)
                    if path.startswith("/")
                    else (f"{target}/{path.lstrip('./')}")
                )
        return f"{target}/processed.jsonl"

    def _json_probe(
        self, path: str, limit: int = _READ_LIMIT
    ) -> tuple[Any, str | None]:
        """Read a small JSON file, telling "absent" apart from "unreadable"."""
        read = self.host.read_probe(path, limit)
        if read.error is not None:
            if _missing_read(read.error):
                return None, None
            return None, f"{path} 无法读取：{read.error}"[:300]
        if read.content is None:
            return None, None
        try:
            return json.loads(read.content.decode("utf-8")), None
        except (UnicodeDecodeError, ValueError) as exc:
            return None, f"{path} 无法解析：{exc}"[:300]

    def _sandbox_path(self, directory: str) -> str:
        """A Storage-absolute path as the sandbox sees it.

        The run directory sits at ``<mount>/.pyromind-agent/<conversation>``, so the
        mount is whatever precedes that segment; anything else is already a path the
        execution plane can read.
        """
        if not directory.startswith("/"):
            return self.host._resolve(directory, False)
        marker = "/.pyromind-agent/"
        if marker in directory:
            workspace = self.host.workspace or ""
            mount = workspace.split(marker, 1)[0]
            if mount and not directory.startswith(f"{mount}/"):
                return mount + directory
        return directory


class BridgeTool:
    """Run a graph's fixed tool call through the same bridge the agent's tools use.

    The call travels Python → Pi runtime → adapter, so the tool executes with the
    conversation's own context, permissions and signals instead of a second
    implementation of the same business tool.
    """

    def __init__(self, host: ExecutionHost) -> None:
        self.host = host

    def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        execution_id: str,
    ) -> ToolResult:
        if not execution_id.strip():
            raise ValueError("execution_id must be a non-empty string")
        if name == "df_submit_pipeline":
            input_path = arguments.get("input_path")
            if isinstance(input_path, str):
                problem = self.host.precheck_submission(input_path)
                if problem is not None:
                    return ToolResult(True, {}, problem)
            tool_arguments = self.host.df_submit_arguments(arguments)
        elif name == "df_submit_evaluation":
            tool_arguments = self.host.evaluation_submit_arguments(arguments)
        else:
            tool_arguments = self.host.tool_arguments(arguments)
        payload: JsonObject = {
            "name": name,
            "arguments": tool_arguments,
            "tool_call_id": execution_id,
        }
        if self.host.graph_run_id:
            payload["origin_run_id"] = self.host.graph_run_id
        response = self.host._call("execution.tool", payload)
        if not isinstance(response, dict) or not isinstance(
            response.get("is_error"), bool
        ):
            raise PortError(
                "tool bridge returned an invalid response", kind="bad_output"
            )
        details = response.get("details")
        text = "\n".join(
            block["text"]
            for block in response.get("content") or []
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ).strip()
        return ToolResult(
            response["is_error"], details if isinstance(details, dict) else {}, text
        )

    def cancel(self, execution_id: str) -> ToolCancellation:
        if not execution_id.strip():
            raise ValueError("execution_id must be a non-empty string")
        response = self.host._call(
            "execution.cancel_tool",
            {"tool_call_id": execution_id},
        )
        if not isinstance(response, dict):
            raise PortError(
                "tool bridge returned an invalid cancellation response",
                kind="bad_output",
            )
        confirmed = response.get("confirmed") is True
        detail = response.get("detail")
        return ToolCancellation(
            confirmed,
            detail if isinstance(detail, str) else "",
        )


class ExecutionHost:
    def __init__(
        self, request: Request, subscribe: Subscribe, local_root: Path | None = None
    ) -> None:
        self.request = request
        self.subscribe = subscribe
        self.local_root = local_root
        self.loop = asyncio.get_running_loop()
        self.cancel = threading.Event()
        self.call_timeout = float(
            os.getenv("PYROMIND_AGENTGENOME_CALL_TIMEOUT_SECONDS", "180")
        )
        self.slow_call_timeout = float(
            os.getenv("PYROMIND_AGENTGENOME_SLOW_CALL_TIMEOUT_SECONDS", "600")
        )
        # A graph step may wait for an asynchronous platform job, which outlives the
        # runtime's 300s default for one command.
        self.shell_timeout = float(
            os.getenv("PYROMIND_AGENTGENOME_SHELL_TIMEOUT_SECONDS", "1800")
        )
        self.emit: Callable[[dict[str, Any]], None] = lambda _event: None
        self.root = ""
        self.workspace = ""
        self.package = ""
        self.graph_run_id: str | None = None
        self.verification = None
        self.model_judge = False
        self.verification_files: dict[str, str] = {}
        self.resuming: str | None = None
        self.human_artifact: str | None = None
        self._output_lock = asyncio.Lock()
        self._reads: dict[str, bytes] = {}
        self._missing: set[str] = set()
        self._declared_paths: dict[str, str] = {}
        self._mount_cache: str | None = None
        self._mount_probed = False

    def _call(self, method: str, payload: JsonObject) -> Any:
        """One bounded bridge round trip from the genome worker thread.

        The execution plane can answer a file request by retrying transports for
        minutes, so an unbounded wait turns one wedged read into a distill that
        never returns. Optional probes treat the timeout as "not there" and a
        required read reports it. Uploads carry the whole package and stay
        bounded by the slow budget instead.
        """
        timeout = self.slow_call_timeout if method in _SLOW_CALLS else self.call_timeout
        future = asyncio.run_coroutine_threadsafe(
            self.request(method, payload), self.loop
        )
        try:
            return future.result(timeout)
        except FuturesTimeoutError:
            future.cancel()
            raise TimeoutError(
                f"{method} did not answer within {timeout:.0f}s; "
                "the execution plane is unreachable or still preparing"
            ) from None

    def prepare(
        self,
        run_id: str,
        package: Path,
        params: dict[str, Any],
        path_parameters: list[str],
        cancel: threading.Event,
        emit: Callable[[dict[str, Any]], None],
        *,
        verification_package: Path | None = None,
        resuming: str | None = None,
        human_artifact: str | None = None,
        **options: Any,  # noqa: ARG002
    ) -> tuple[Ports, dict[str, Any]]:
        self.cancel, self.emit = cancel, emit
        self.graph_run_id = run_id
        # The parked node a wakeup already answered: this attempt approves exactly
        # that gate and parks again on any later one.
        self.resuming = resuming
        # The file the person produced out of band, landed on the declared artifact
        # when this attempt re-enters the gate they answered.
        self.human_artifact = human_artifact
        run_root = f"public_data/workflow-runs/{run_id}"
        self.root = self._resolve(run_root, True)
        self.workspace = (
            self.root[: -len(run_root) - 1]
            if self.root.endswith(f"/{run_root}")
            else ""
        )
        self.package = f"{run_root}/package"
        self._declared_paths = {}
        cwd = f"{self.root}/package"
        capabilities = self._call("execution.capabilities", {})
        manifest = json.loads((package / "manifest.json").read_text())
        declared_parameters = manifest.get("parameters")
        path_parameters = [
            key
            for key in path_parameters
            if isinstance(declared_parameters, Mapping)
            and isinstance(declared_parameters.get(key), Mapping)
            and declared_parameters[key].get("type") == "path"
        ]
        protected = (
            set(manifest.get("revision", {}).get("verification_resources", []))
            if capabilities.get("protected_verification")
            else set()
        )
        for file in sorted(package.rglob("*")):
            if file.is_symlink():
                raise ValueError("package symlinks are not supported")
            if (
                not file.is_file()
                or "__pycache__" in file.parts
                or file.suffix == ".pyc"
                or file.name == ".DS_Store"
            ):
                continue
            if file.relative_to(package).as_posix() in protected:
                continue
            target = f"{cwd}/{file.relative_to(package).as_posix()}"
            self._write(target, file.read_bytes())
        # A resumed run re-enters prepare over an existing run directory, so the
        # artifact directory has to tolerate the second visit.
        self._python(
            "from pathlib import Path\n"
            f"Path({self.root + '/artifacts'!r}).mkdir(parents=True, exist_ok=True)"
        )
        resolved = {
            key: (
                self._resolve_param(value)
                if key in path_parameters and isinstance(value, str) and value
                else value
            )
            for key, value in params.items()
        }
        verification = None
        if capabilities.get("protected_verification"):
            verification = Path(self._protect(run_id, verification_package or package))
        self.verification = str(verification) if verification else None
        self.model_judge = capabilities.get("model_judge", False)
        emit(
            {
                "event": "execution_prepared",
                "execution": {
                    "execution_domain": "sandbox",
                    "execution_path": self.root,
                    "workspace": self.workspace,
                    "package": self.package,
                    "artifacts_dir": f"{self.root}/artifacts",
                },
            }
        )
        return Ports(
            shell=self,
            artifacts=self,
            cwd=Path(cwd),
            verification_cwd=verification,
            tool=BridgeTool(self),
            human=self,
            await_port=BridgeAwait(self),
        ), resolved

    def ask(self, question: str, evidence: dict[str, Any]) -> Answer:  # noqa: ARG002
        """A human node pauses the run; the person answers out of band.

        Nothing is asked through the bridge: the run parks with the question and the
        artifact path in its waiting payload, the conversation tells the user what to
        do, and the resume call comes back through ``prepare(resuming=...)``.
        """
        if self.resuming is not None and evidence.get("node") == self.resuming:
            artifact = evidence.get("artifact")
            if self.human_artifact and isinstance(artifact, str):
                self._land_human_artifact(artifact)
            return Answer(approve=True)
        return Answer(pending=True)

    def _land_human_artifact(self, artifact: str) -> None:
        """Move the file the person exported onto the artifact the graph declared.

        The person works outside the run directory, so their export has to be in
        place where the graph's downstream steps read it before the gate opens.
        """
        source = self.human_artifact
        assert source is not None
        if source == artifact:
            return
        export = self._local_export(source)
        if export is not None:
            self._write(artifact, export.read_bytes())
            return
        resolved = self._resolve_aliased(source)
        target = PurePosixPath(artifact)
        result = self._run(
            f"mkdir -p {shlex.quote(str(target.parent))} && "
            f"cp -f {shlex.quote(resolved)} {shlex.quote(str(target))}",
            None,
            stream=False,
        )
        if result.get("rc") != 0:
            raise FileNotFoundError(
                f"human artifact {source!r} could not be copied to the gate: "
                f"{result.get('stdout')}"
            )

    def capabilities(self) -> dict[str, Any]:
        return {"interactive": True}

    def _protect(self, run_id: str, package: Path) -> str:
        protected = ""
        for file in sorted(package.rglob("*")):
            if file.is_symlink():
                raise ValueError("package symlinks are not supported")
            if not file.is_file() or "__pycache__" in file.parts:
                continue
            content = file.read_bytes()
            if len(content) > 256 * 1024:
                raise ValueError("protected package file exceeds 256 KiB")
            protected = self._call(
                "execution.protect",
                {
                    "run_id": run_id,
                    "files": [
                        {
                            "path": file.relative_to(package).as_posix(),
                            "content": base64.b64encode(content).decode("ascii"),
                        }
                    ],
                },
            )["path"]
            self.verification_files[
                f"{protected}/{file.relative_to(package).as_posix()}"
            ] = hashlib.sha256(content).hexdigest()
        return protected

    def prepare_revision(
        self, revision_id: str, package: Path, allowed: list[str]
    ) -> dict[str, Any]:
        capabilities = self._call("execution.capabilities", {})
        if not capabilities.get("revision"):
            raise ValueError("host cannot prepare and collect revision files")
        revision_root = f"public_data/workflow-revisions/{revision_id}"
        directory = self._resolve(revision_root, True)
        self.root = directory
        self.workspace = (
            directory[: -len(revision_root) - 1]
            if directory.endswith(f"/{revision_root}")
            else ""
        )
        self.package = revision_root
        self._declared_paths = {}
        protected = (
            self._protect(revision_id, package)
            if capabilities.get("protected_verification")
            else None
        )
        # Without OS protection, the full copy provides context for the Agent.
        # Submission still collects only allowlisted business scripts.
        names = (
            allowed
            if protected
            else [
                file.relative_to(package).as_posix()
                for file in sorted(package.rglob("*"))
                if file.is_file()
            ]
        )
        for name in names:
            file = package / name
            if not file.is_file() or file.is_symlink():
                raise ValueError("editable script is not a regular package file")
            self._write(f"{directory}/{name}", file.read_bytes())
        return {
            "editable_dir": directory,
            "verification_dir": protected or directory,
            "verification_protection": "read_only" if protected else "prompt",
        }

    def collect_revision(
        self, draft: dict[str, Any], allowed: list[str]
    ) -> dict[str, bytes]:
        return {
            name: base64.b64decode(
                self._call(
                    "execution.read",
                    {
                        "path": f"{draft['editable_dir']}/{name}",
                        "limit": _READ_LIMIT,
                    },
                )["content"],
                validate=True,
            )
            for name in allowed
        }

    def _resolve(self, path: str, write: bool) -> str:
        return self._call("execution.resolve", {"path": path, "write": write})["path"]

    def _resolve_aliased(self, value: str) -> str:
        """Resolve a read path, retrying the Storage spelling a business tool declared.

        Platform tools name Storage without the mount prefix (`/datasets/...`),
        which the sandbox path policy rejects, so retry those under the mount.
        An artifact pinned beside a Storage input is mount-absolute for the same
        reason and takes the prefix before the first attempt.
        """
        try:
            return self._resolve(value, False)
        except Exception as error:
            alias = _storage_alias(value, self._probe_mount())
            if alias is None or "PATH_SCOPE_ERROR" not in str(error):
                raise
            try:
                return self._resolve(alias, False)
            except Exception as aliased:
                raise aliased from error

    def _resolve_param(self, value: str) -> str:
        """Resolve one graph path for the sandbox and keep its tool spelling."""
        path = self._resolve_aliased(value)
        self._declared_paths[path] = value
        return path

    def tool_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Restate a graph's fixed tool call in the vocabulary the tool reads."""
        return {key: self._tool_value(value) for key, value in arguments.items()}

    def df_submit_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Adapt the DataFlow submission's three path fields independently.

        ``input_path`` names a Storage object, while the script and support file
        are workspace files that the submission tool stages. Running every string
        through the generic graph-path adapter rewrites ``input_path`` twice and
        can give the two package paths the wrong base.
        """
        adapted = dict(arguments)
        input_path = arguments.get("input_path")
        if isinstance(input_path, str):
            adapted["input_path"] = self.platform_input_path(input_path)
        for key in ("script_path", "support_file_path"):
            value = arguments.get(key)
            if isinstance(value, str):
                adapted[key] = _tool_path(
                    value,
                    {},
                    self.workspace,
                    self.package,
                )
        return adapted

    def evaluation_submit_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Map configs and Storage inputs while preserving model references."""
        adapted = dict(arguments)
        input_path = arguments.get("input_path")
        if isinstance(input_path, str):
            adapted["input_path"] = self.platform_input_path(input_path)
        for key in (
            "dataset_config_path",
            "evaluation_config_path",
            "report_spec_path",
            "source_manifest_path",
            "scoring_spec_path",
            "scoring_script_path",
            "manifest_path",
            "extraction_script_path",
        ):
            value = arguments.get(key)
            if isinstance(value, str):
                path = _tool_path(value, {}, self.workspace, self.package)
                if path == value and not value.startswith(
                    ("/", "storage/", "public_data/", ".agents/", "knowledge/")
                ):
                    path = f"{self.package}/{value}"
                adapted[key] = path
        return adapted

    def platform_input_path(self, value: str) -> str:
        """Give DataFlow a Storage path to a graph-produced input.

        The remote conversation workspace is a directory within the platform's
        Storage mount. DataFlow interprets input_path relative to that mount,
        not to the conversation workspace, so an absolute Storage path is
        restated mount-relative and a run-directory artifact keeps its
        conversation prefix instead of turning into a nonexistent /public_data/...
        file.
        """
        mount = self._storage_mount()
        if mount is not None and value.startswith(f"{mount}/"):
            return value[len(mount) :]
        if value.startswith(f"{self.root}/"):
            raise ValueError(
                "DataFlow cannot access a graph-produced input outside the shared "
                "Storage mount; stage the input in Storage before submission"
            )
        return self._tool_value(value)

    def precheck_submission(self, value: str) -> str | None:
        """Why the platform would reject this submission's own input, if known.

        A vision pipeline resolves the images a manifest names relative to the
        manifest itself, so a manifest that left its images behind only fails
        minutes later inside the platform run. Read a bounded head here and report
        the missing images while the submission is still local.
        """
        try:
            path = self._resolve_aliased(value)
            result = self._run(
                f"python3 -c {shlex.quote(_MANIFEST_PRECHECK)} {shlex.quote(path)}",
                None,
                stream=False,
            )
        except RunCancelled:
            raise
        except Exception:  # noqa: BLE001 - a probe, not a required read
            return None
        if result.get("rc") != 0 or result.get("truncated"):
            return None
        try:
            report = json.loads(result.get("stdout") or "")
        except json.JSONDecodeError:
            return None
        missing = report.get("missing")
        if not isinstance(missing, list) or not missing:
            return None
        shown = "、".join(str(item) for item in missing[:3])
        return (
            f"提交前预检失败：清单 `{value}` 中相对图片路径找不到文件（例如 {shown}）。"
            "视觉 pipeline 按清单所在目录解析图片：请让清单和图片待在同一个 "
            "Storage 目录，或让产出该清单的节点用 artifact.at 把产物钉在输入旁边。"
        )

    def _tool_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return _tool_path(
                value,
                self._declared_paths,
                self.workspace,
                self.package,
                self._storage_mount(),
            )
        if isinstance(value, list):
            return [self._tool_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._tool_value(item) for key, item in value.items()}
        return value

    def _storage_mount(self) -> str | None:
        """The Storage mount root, when this session has one mounted.

        A sandbox conversation lives under ``<mount>/.pyromind-agent/<id>``; the
        local execution plane has no mount and answers ``None``.
        """
        head, separator, _ = self.workspace.partition(_AGENT_DIRECTORY)
        return head if separator else None

    def _probe_mount(self) -> str | None:
        """The mount for a planning probe, which runs before any workflow does.

        A distill probes the plan's inputs while ``workspace`` is still empty,
        so asking the execution plane for the conversation root once keeps
        Storage object paths (`/datasets/...`) resolvable there. Without it every
        Storage input reads as missing and the plan can never pass its review.
        """
        mount = self._storage_mount()
        if mount is not None:
            return mount
        if self._mount_probed:
            return self._mount_cache
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            # A blocking round trip here would deadlock the loop thread.
            logger.info("Pi storage mount probe skipped: called on the event loop")
            return None
        try:
            resolved = self._resolve(_MOUNT_PROBE, False)
        except Exception as error:  # noqa: BLE001 - a hiccup must not stick
            logger.info(
                "Pi storage mount probe failed error=%s: %s",
                type(error).__name__,
                error,
            )
            return None
        self._mount_probed = True
        leaf = "/" + _MOUNT_PROBE.rsplit("/", 1)[1]
        if not resolved.endswith(leaf):
            logger.info("Pi storage mount probe found no mount resolved=%s", resolved)
            return None
        workspace = resolved[: -len(leaf)]
        head, separator, _ = workspace.partition(_AGENT_DIRECTORY)
        if not separator or not workspace.endswith("/public_data"):
            logger.info("Pi storage mount probe found no mount resolved=%s", resolved)
            return None
        self.workspace = workspace[: -len("/public_data")]
        self._mount_cache = head
        logger.info("Pi storage mount probe resolved mount=%s", head)
        return head

    def path(self, name: str) -> str:
        """The sandbox path of a graph artifact, inside or outside the run directory.

        Most artifacts are named relative to the run directory. A node can also pin
        its product beside a Storage input, and that name is already absolute.
        """
        path = PurePosixPath(name)
        if ".." in path.parts:
            raise ValueError("artifact path escapes run directory")
        if not path.is_absolute():
            return str(PurePosixPath(self.root) / path)
        return self._resolve(name, True)

    def describe(self, name: str) -> dict[str, Any]:
        result = self._python(
            "import json,hashlib\nfrom pathlib import Path\n"
            f"p=Path({self.path(name)!r})\n"
            "if not p.is_file():\n print(json.dumps({'missing': True}))\n"
            "else:\n h=hashlib.sha256()\n with p.open('rb') as f:\n"
            "  for block in iter(lambda: f.read(65536), b''): h.update(block)\n"
            f" print(json.dumps({{'path': {name!r}, 'bytes': p.stat().st_size, "
            "'sha256': h.hexdigest()}))"
        )
        value = json.loads(result)
        if value.get("missing"):
            raise FileNotFoundError(name)
        value["execution_path"] = self.path(name)
        return value

    def read_text(self, name: str) -> str:
        path = self.path(name)
        result = self._call("execution.read", {"path": path, "limit": 65536})
        return base64.b64decode(result["content"], validate=True).decode("utf-8")

    def write_text(self, name: str, text: str) -> None:
        self._write(self.path(name), text.encode("utf-8"))

    def link(self, name: str, source: str) -> None:
        """Land an external product on a graph artifact path.

        A hardlink keeps a large product out of the run directory; a filesystem
        boundary falls back to a copy, which is what the old wait script did.
        """
        target = self.path(name)
        self._python(
            "import os,shutil\n"
            f"t={target!r}\ns={source!r}\n"
            "os.makedirs(os.path.dirname(t), exist_ok=True)\n"
            "if os.path.exists(t): os.remove(t)\n"
            "try: os.link(s,t)\n"
            "except OSError: shutil.copyfile(s,t)\n"
        )

    def read_optional(self, path: str, limit: int) -> bytes | None:
        """Read a file the execution plane may not have written yet."""
        return self.read_probe(path, limit).content

    def read_probe(self, path: str, limit: int) -> ProbeRead:
        """Read a file while preserving whether it was absent or unreadable."""
        try:
            result = self._call(
                "execution.read",
                {"path": path, "limit": limit},
            )
        except Exception as exc:
            return ProbeRead(None, f"{type(exc).__name__}: {exc}"[:300])
        try:
            return ProbeRead(
                base64.b64decode(result["content"], validate=True),
                None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return ProbeRead(None, f"invalid execution.read response: {exc}"[:300])

    def exists(self, path: str) -> bool:
        return self.read_probe(path, 1).content is not None

    def read(self, path: str) -> bytes:
        """Read a workspace file, answering host-local exports from disk.

        Pi exports traces with its own file APIs next to the conversation, while the
        execution plane can be a remote sandbox. Those exports are already on this
        host, so read them here; mirroring them into the execution plane first only
        paid a second transport for bytes the reader already holds.
        """
        export = self._local_export(path)
        if export is not None:
            return export.read_bytes()
        if path in self._reads:
            return self._reads[path]
        try:
            result = self._call(
                "execution.read",
                {"path": self._resolve(path, False), "limit": _READ_LIMIT},
            )
        except PiRunnerError as error:
            # Callers probe for optional files; keep the exception faithful to the
            # filesystem so a probe does not have to know how the portal words it.
            if _is_missing_file(error):
                raise FileNotFoundError(path) from error
            raise
        content = base64.b64decode(result["content"], validate=True)
        self._reads[path] = content
        return content

    def read_many(self, paths: Sequence[str]) -> dict[str, bytes | None]:
        """Answer many workspace reads with as few bridge round trips as possible.

        Freezing a chain reads every script it runs, and a chain is built twice
        (draft and capture), so a read per file turns one slow sandbox into dozens
        of them. Answer from what this host already read, ask for the rest in
        batches, and report a file the plane could not read as ``None``.
        """
        wanted = list(dict.fromkeys(str(path) for path in paths))
        answers: dict[str, bytes | None] = {}
        pending: list[str] = []
        for path in wanted:
            export = self._local_export(path)
            if export is not None:
                answers[path] = export.read_bytes()
            elif path in self._reads:
                answers[path] = self._reads[path]
            elif path in self._missing:
                answers[path] = None
            else:
                pending.append(path)
        while pending:
            pending_paths: list[JsonValue] = [str(path) for path in pending]
            result = self._call(
                "execution.read_many", {"paths": pending_paths, "limit": _READ_LIMIT}
            )
            answered = 0
            for item in result.get("files") or []:
                path = item.get("path")
                if not isinstance(path, str):
                    continue
                answered += 1
                content = item.get("content")
                if isinstance(content, str):
                    decoded = base64.b64decode(content, validate=True)
                    self._reads[path] = decoded
                    answers[path] = decoded
                else:
                    self._missing.add(path)
                    answers[path] = None
            remaining = [str(path) for path in (result.get("remaining") or [])]
            if not remaining or answered == 0:
                break
            pending = remaining
        return answers

    def head(self, path: str, limit: int) -> bytes:
        """Read at most ``limit`` bytes from the start of a workspace file.

        The file read API rejects anything larger than the requested window, so a
        bounded head of a real dataset has to be cut in the execution plane; that
        keeps capture-time sampling from pulling gigabytes across the bridge.
        A root input may be a directory (a folder of images): the same command
        reports that instead of asking the plane to read a directory as a file.
        """
        if not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        export = self._local_export(path)
        if export is not None:
            with export.open("rb") as stream:
                return stream.read(limit)
        resolved = self._resolve_aliased(path)
        quoted = shlex.quote(resolved)
        result = self._run(
            f"if [ -d {quoted} ]; then printf '%s' {shlex.quote(_DIRECTORY_MARK)}; "
            f"else head -c {max(limit, 0)} {quoted} | base64 | tr -d '\\n'; fi",
            None,
            stream=False,
        )
        if result.get("truncated"):
            raise RuntimeError("execution head output was truncated")
        if result.get("rc") != 0:
            raise FileNotFoundError(path)
        content = result.get("stdout") or ""
        if content.strip() == _DIRECTORY_MARK:
            raise IsADirectoryError(path)
        return base64.b64decode(content, validate=True)

    def metadata_many(self, paths: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Describe many paths in one execution-plane call.

        `metadata` is a full sandbox command, and a plan review probes every root
        input; at sandbox latency a handful of files costs minutes. One probe for
        the whole batch keeps the review in the millisecond range.
        """
        wanted = list(dict.fromkeys(str(path) for path in paths))
        answers: dict[str, dict[str, Any]] = {}
        pending: list[str] = []
        for path in wanted:
            local = self._local_candidate(path)
            if local is None:
                pending.append(path)
                continue
            try:
                stat = local.stat()
            except OSError:
                # Not on the host disk does not mean absent: Storage paths live in
                # the mount the execution plane sees, so ask the plane.
                pending.append(path)
                continue
            answers[path] = {
                "exists": True,
                "is_file": local.is_file(),
                "is_dir": local.is_dir(),
                "size": stat.st_size,
            }
        if not pending:
            return answers
        # Resolving each path is itself a bridge round trip, and a plan review
        # probes a dozen of them; at sandbox latency that is minutes. The mount
        # prefix a path may need is pure string logic, so try both spellings
        # inside one probe and keep whichever exists.
        mount = self._probe_mount()
        candidates: dict[str, list[str]] = {}
        for path in pending:
            options = [path]
            alias = _storage_alias(path, mount)
            if alias is not None and alias != path:
                options.append(alias)
            candidates[path] = options
        flat = [spelling for options in candidates.values() for spelling in options]
        result = self._run(
            f"python3 -c {shlex.quote(_METADATA_PROBE_MANY)} "
            f"{shlex.quote(json.dumps(flat))}",
            None,
            stream=False,
        )
        if result.get("rc") != 0 or result.get("truncated"):
            stdout = str(result.get("stdout") or "")[:500]
            raise RuntimeError(
                f"metadata batch probe failed; rc={result.get('rc')!r}, "
                f"stdout={stdout!r}"
            )
        try:
            mapping = json.loads(result.get("stdout") or "")
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"metadata batch probe returned no JSON: {error}"
            ) from error
        for original, options in candidates.items():
            found = next(
                (
                    mapping[spelling]
                    for spelling in options
                    if isinstance(mapping.get(spelling), dict)
                    and mapping[spelling].get("exists")
                ),
                None,
            )
            answers[original] = found or {"exists": False}
        return answers

    def metadata(self, path: str) -> dict[str, Any]:
        """Describe an execution-plane path without reading its contents."""
        local = self._local_candidate(path)
        if local is not None:
            try:
                stat = local.stat()
            except OSError:
                pass
            else:
                return {
                    "exists": True,
                    "is_file": local.is_file(),
                    "is_dir": local.is_dir(),
                    "size": stat.st_size,
                }
        try:
            resolved = self._resolve_aliased(path)
            result = self._run(
                f"python3 -c {shlex.quote(_METADATA_PROBE)} {shlex.quote(resolved)}",
                None,
                stream=False,
            )
        except Exception as error:  # noqa: BLE001 - optional planning probe
            if "PATH_SCOPE_ERROR" in str(error) or _is_missing_file(error):
                return {"exists": False}
            raise
        stdout = result.get("stdout")
        if result.get("truncated"):
            raise RuntimeError(
                f"metadata probe for {path!r} was truncated; "
                f"resolved path={resolved!r}, rc={result.get('rc')!r}, "
                f"stdout={str(stdout or '')[:500]!r}"
            )
        rc = result.get("rc")
        if rc != 0:
            raise RuntimeError(
                f"metadata probe for {path!r} failed; "
                f"resolved path={resolved!r}, rc={rc!r}, "
                f"stdout={str(stdout or '')[:500]!r}"
            )
        try:
            value = json.loads(stdout or "")
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"metadata probe for {path!r} returned invalid JSON; "
                f"resolved path={resolved!r}, stdout={str(stdout or '')[:500]!r}"
            ) from error
        if not isinstance(value, dict) or "exists" not in value:
            raise RuntimeError(
                f"metadata probe for {path!r} returned an invalid result; "
                f"resolved path={resolved!r}, value={value!r}"
            )
        detail = value.get("error")
        if isinstance(detail, str) and detail:
            raise RuntimeError(
                f"metadata probe for {path!r} failed; "
                f"resolved path={resolved!r}, error={detail}"
            )
        return value

    def _local_candidate(self, path: str) -> Path | None:
        if self.local_root is None:
            return None
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts:
            return None
        return self.local_root / relative

    def _local_export(self, path: str) -> Path | None:
        candidate = self._local_candidate(path)
        if candidate is None:
            return None
        return candidate if candidate.is_file() else None

    def _write(self, path: str, content: bytes) -> None:
        if self.cancel.is_set():
            raise RunCancelled()
        # Keep each JSONL request below the bridge's 1 MiB frame limit.
        chunk_size = 256 * 1024
        if len(content) > chunk_size:
            transfer = f"{self.root}/.host/transfers/{uuid4().hex}"
            count = (len(content) + chunk_size - 1) // chunk_size
            for index in range(count):
                self._write(
                    f"{transfer}/{index}",
                    content[index * chunk_size : (index + 1) * chunk_size],
                )
            self._python(
                "from pathlib import Path\n"
                f"target=Path({path!r})\n"
                "target.parent.mkdir(parents=True, exist_ok=True)\n"
                f"parts=Path({transfer!r})\n"
                "with target.open('wb') as out:\n"
                f" for index in range({count}):\n"
                "  part=parts/str(index)\n"
                "  out.write(part.read_bytes())\n"
                "  part.unlink()\n"
                "parts.rmdir()\n"
            )
            return
        self._call(
            "execution.write",
            {"path": path, "content": base64.b64encode(content).decode("ascii")},
        )

    def _python(self, source: str) -> str:
        script = f"{self.root}/.host/{uuid4().hex}.py"
        self._write(script, source.encode("utf-8"))
        command = f"python3 {shlex.quote(script)}"
        result = self._run(command, None, stream=False)
        if result.get("rc") != 0 or result.get("truncated"):
            raise RuntimeError(
                f"Execution environment operation failed: {result.get('stdout')}"
            )
        return result["stdout"]

    def execute(self, command: str, cwd: Path) -> ShellResult:
        started = time.monotonic()
        if str(cwd) == self.verification:
            for path, digest in self.verification_files.items():
                content = base64.b64decode(
                    self._call("execution.read", {"path": path, "limit": _READ_LIMIT})[
                        "content"
                    ],
                    validate=True,
                )
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError("frozen verification resource changed")
        result = self._run(command, str(cwd), stream=True)
        if result.get("rc") is None:
            raise RuntimeError("command exited without a confirmed exit status")
        return ShellResult(
            command,
            result["rc"],
            result["stdout"],
            "",
            round((time.monotonic() - started) * 1000),
        )

    def _run(self, command: str, cwd: str | None, *, stream: bool) -> dict[str, Any]:
        if self.cancel.is_set():
            raise RunCancelled()
        return asyncio.run_coroutine_threadsafe(
            self._run_async(command, cwd, stream=stream), self.loop
        ).result()

    async def _run_async(
        self, command: str, cwd: str | None, *, stream: bool
    ) -> dict[str, Any]:
        execution_id = uuid4().hex
        await asyncio.to_thread(
            self.emit,
            {
                "event": "execution",
                "execution": {
                    "request_id": execution_id,
                    "phase": "executing" if stream else "preparing",
                },
            },
        )

        async def output(event: JsonObject) -> None:
            async with self._output_lock:
                await asyncio.to_thread(
                    self.emit, {"event": "output", "text": event.get("text", "")}
                )

        self.subscribe(execution_id, output)
        task = asyncio.create_task(
            self.request(
                "execution.run",
                {
                    "id": execution_id,
                    "command": command,
                    "cwd": cwd,
                    "stream": stream,
                    "timeout": self.shell_timeout,
                    **(
                        {"judge_directory": f"{self.root}/.judge/{execution_id}"}
                        if self.model_judge and cwd == self.verification
                        else {}
                    ),
                },
            )
        )
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=0.1)
                if self.cancel.is_set() and not task.done():
                    await asyncio.wait_for(
                        self.request("execution.cancel", {"id": execution_id}), 5
                    )
                    await asyncio.wait_for(asyncio.shield(task), 10)
                    break
            result = await task
            if isinstance(result.get("execution"), dict):
                await asyncio.to_thread(
                    self.emit,
                    {"event": "execution", "execution": result["execution"]},
                )
            if self.cancel.is_set():
                if result.get("stopped") is True or isinstance(result.get("rc"), int):
                    raise RunCancelled()
                raise RuntimeError(
                    "Cancellation requested; command termination is unconfirmed"
                )
            if result.get("error"):
                raise RuntimeError(
                    f"Execution infrastructure interrupted: {result['error']}"
                )
            return result
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self.subscribe(execution_id, None)


class AgentGenomeBackend:
    def __init__(self, root: Path, host: Callable[[str], ExecutionHost]) -> None:
        self.service = GenomeService(root)
        self.host = host
        self.service.claim_execution_owner()
        self.service.recover()

    async def close(self) -> None:
        self.service.release_execution_owner()

    async def invoke(
        self, scope: str, action: str, arguments: JsonObject, request_id: str
    ) -> JsonObject:
        host = self.host(scope)
        capabilities = set()
        if action in {
            "get",
            "status",
            "run",
            "prepare_run",
            "confirm_run",
            "prepare_revision",
        }:
            available = await host.request("execution.capabilities", {})
            capabilities = {key for key, value in available.items() if value is True}
        return await asyncio.to_thread(
            invoke_genome,
            self.service,
            scope,
            action,
            arguments,
            request_id,
            capabilities=capabilities,
            host=host,
        )

    async def execute(
        self, scope: str, run_id: str, emit: WorkflowEventSink
    ) -> JsonObject:
        loop = asyncio.get_running_loop()
        host = self.host(scope)

        def event(payload: dict[str, Any]) -> Any:
            return asyncio.run_coroutine_threadsafe(emit(payload), loop).result()

        return await asyncio.to_thread(self.service.execute, scope, run_id, host, event)
