"""The df_run_pipeline tool: run agent-authored Python data pipelines.

``df_run_pipeline`` executes a pipeline script in an isolated subprocess.
DataFlow and the managed AVI runtime are optional libraries selected by the
script.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

from pydantic import Field
from rich.text import Text

from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.llm.message import TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.sdk.workspace.workspace import LocalWorkspace
from openhands.tools.data_preparation.runner import (
    DATAFLOW_RUNTIME_PACKAGES,
    SUPPORTED_DATAFLOW_VERSION,
    LabelingModelGateway,
    ProcessLocalSampleExecutor,
    build_dataflow_env,
    check_dataflow_installed,
    check_dataflow_version,
    preflight_dataflow_llm,
    resolve_dataflow_python,
    runtime_bundle_fingerprint,
    runtime_public_names,
    summarize_dataflow_env,
    validate_managed_image_pipeline,
)
from openhands.tools.data_preparation.sandbox_execution import (
    ENV_SANDBOX_DATAFLOW_PYTHON,
    SandboxExecutionError,
    SandboxSampleExecutor,
    SandboxTarget,
    sandbox_dataflow_venv,
    sandbox_exists,
    sandbox_is_dir,
    sandbox_read_text,
    sandbox_target,
    sandbox_target_from_absolute,
    supports_sandbox_execution,
)
from openhands.tools.data_preparation.workspace_paths import resolve_workspace_file
from openhands.tools.utils import default_path_access_policy
from openhands.tools.utils.workspace_staging import (
    STORAGE_ALIAS,
    WorkspaceArchiveTruncatedError,
    WorkspaceStagingError,
    is_remote_workspace,
    publish_workspace_path,
    resolve_workspace_path,
    stage_workspace_members,
    stage_workspace_path,
)


RUNTIME_FILENAMES = (
    "avi_pcb_runtime.py",
    "df_logging.py",
    "generate_report.py",
    "image_utils.py",
    "preparation_runtime.py",
    "source_fingerprint.py",
    "validate_prepared_data.py",
)

_LOG_TAIL_CHARS = 6000
_MANIFEST_RECORD_LIMIT = 50
_MANIFEST_IMAGE_LIMIT = 200
_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
OutputSchema = Literal[
    "text",
    "dpo",
    "vision",
    "structured",
    "multiturn",
    "function_call",
    "quality_evaluation",
    "text2sql",
    "artifacts",
]


def _resolve_in_workspace(conversation: Any, path: str) -> Path:
    workspace = conversation.workspace
    if not isinstance(workspace, LocalWorkspace):
        raise ValueError(
            "This operation is only supported for local conversation workspaces."
        )
    workspace_dir = Path(workspace.working_dir).resolve()
    candidate = Path(path)

    # If path is absolute, check if it's within workspace_dir
    if candidate.is_absolute():
        resolved = candidate.resolve()
        try:
            resolved.relative_to(workspace_dir)
        except ValueError as exc:
            raise ValueError(
                f"Path is outside the conversation workspace: {path}"
            ) from exc
        return resolved

    # If path is relative, resolve it relative to workspace_dir
    resolved = (workspace_dir / candidate).resolve()
    try:
        resolved.relative_to(workspace_dir)
    except ValueError as exc:
        raise ValueError(f"Path is outside the conversation workspace: {path}") from exc
    return resolved


def _resolve_input_path(conversation: Any, path: str) -> Path:
    """Resolve an existing readable file or directory inside the workspace."""

    resolved = _resolve_in_workspace(conversation, path)
    workspace_dir = Path(conversation.workspace.working_dir).resolve()
    policy = default_path_access_policy(workspace_dir)
    if not policy.check(resolved, "read") or not resolved.exists():
        raise ValueError(f"Missing or unreadable workspace input: {path}")
    return resolved


def _resolve_output_path(conversation: Any, path: str) -> Path:
    """Resolve a writable output path inside the workspace."""

    resolved = _resolve_in_workspace(conversation, path)
    workspace_dir = Path(conversation.workspace.working_dir).resolve()
    policy = default_path_access_policy(workspace_dir)
    if not policy.check(resolved, "write"):
        raise ValueError(f"Path is not writable by the agent: {path}")
    return resolved


def _source_fingerprint(path: Path) -> str:
    """Return a stable byte fingerprint for a source file or directory."""

    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = (
        [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    )
    for item in files:
        if path.is_dir():
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _resolve_legacy_input_arg(
    run: _WorkspaceRun,
    arg: str,
    pipeline: Path,
    *,
    with_siblings: bool = False,
) -> tuple[Path, Path] | None:
    """Resolve a legacy pipeline input argument.

    Legacy arguments were historically interpreted by the child process
    relative to the pipeline script directory, which is inconsistent with the
    workspace-root resolution used everywhere else in this tool family. Resolve
    ``arg`` from the workspace root when it exists there, otherwise fall back to
    the pipeline directory. Returns the resolved input and the base directory
    used to anchor the matching output argument, or ``None`` when neither exists
    so script-defined non-path arguments pass through unchanged.
    """
    try:
        input_path = run.input_path(arg, with_siblings=with_siblings)
    except WorkspaceArchiveTruncatedError:
        # A cut transfer is a staging failure, not an argument that is not a path.
        raise
    except ValueError:
        input_path = None
    if input_path is not None:
        return input_path, run.stage_root
    base = pipeline.parent
    candidate = base / arg
    if candidate.exists():
        return candidate.resolve(), base
    return None


# ---------------------------------------------------------------------------
# df_run_pipeline
# ---------------------------------------------------------------------------


class DfRunPipelineAction(Action):
    pipeline_path: str = Field(
        description=(
            "Workspace-relative path of the Python pipeline script to run, "
            "e.g. 'public_data/data-preparation/pipeline.py'."
        )
    )
    args: list[str] = Field(
        default_factory=list,
        description=(
            "Positional arguments forwarded to the pipeline script. Their meaning "
            "is defined by that script; standard pipelines use input path and "
            "output path. Sampling is determined by the prepared input, never "
            "by this executor. When output_schema is set, args[0] "
            "and args[1] are workspace-relative input/output paths and are "
            "normalized before execution. With model_profile='vision', pass the "
            "manifest file as args[0] and keep its images beside it, either in "
            "the same workspace directory or the same 'storage/...' directory."
        ),
    )
    support_file_path: str | None = Field(
        default=None,
        description=(
            "Optional workspace JSON file frozen into the run's state directory "
            "and passed to the pipeline as its third positional argument."
        ),
    )
    timeout: int = Field(default=3600, ge=60, le=7200, description="Timeout seconds.")
    python: str | None = Field(
        default=None,
        description=(
            "Python interpreter override. Sandbox runs resolve it inside the "
            "sandbox (set $PYROMIND_SANDBOX_DATAFLOW_PYTHON to change the "
            "default); host runs default to $DATAFLOW_PYTHON or the current "
            "interpreter. text/vision profiles require `open-dataflow`; the "
            "none profile does not."
        ),
    )
    output_schema: OutputSchema | None = Field(
        default=None,
        description=(
            "Canonical JSONL schema to validate after a successful run: text, dpo, "
            "vision, structured, multiturn, function_call, quality_evaluation, "
            "text2sql, or artifacts. Use structured for direct image annotations "
            "and vision for training messages; match "
            "ImagePipelineConfig.output_format. "
            "For the standard pipeline contract, args[1] is treated as the output "
            "path. Omit only for legacy pipelines with non-standard outputs."
        ),
    )
    model_profile: Literal["none", "text", "vision"] = Field(
        default="text",
        description=(
            "Use none for pure Python/AVI work without model credentials, text "
            "for the conversation model, or vision for the managed image model."
        ),
    )
    labeling_gateway: LabelingModelGateway | None = Field(
        default=None,
        description=(
            "Optional user-supplied OpenAI-compatible gateway for the image "
            "labeling model: api_url (or base_url), model, and api_key. When set "
            "it replaces the managed vision model for this run and requires "
            "model_profile='vision'. Collect these values from the user in "
            "writing before falling back to the platform default."
        ),
    )


def _read_output_records(
    output_path: Path, *, limit: int = 3
) -> tuple[list[dict[str, Any]], int]:
    """Read up to ``limit`` output JSONL records plus the total record count."""
    records: list[dict[str, Any]] = []
    total = 0
    try:
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total += 1
                if len(records) < limit:
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        records.append(value)
    except OSError:
        return [], 0
    return records, total


def _read_report_failure(report_path: Path | None) -> dict[str, Any]:
    """Read the runtime failure captured in ``report.json`` when available."""

    if report_path is None or not report_path.is_file():
        return {}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(report, dict):
        return {}
    failure = report.get("failure")
    if not isinstance(failure, dict):
        return {}
    # generate_report.py currently preserves the complete failure.json payload,
    # whose actionable fields live under its own ``failure`` key.
    nested = failure.get("failure")
    return nested if isinstance(nested, dict) else failure


class DfRunPipelineObservation(Observation):
    exit_code: int = Field(default=-1)
    stdout_tail: str = Field(default="")
    stderr_tail: str = Field(default="")
    execution: str | None = Field(
        default=None,
        description=(
            "Where the Sample ran: 'sandbox' inside the conversation's platform "
            "sandbox, 'host' in an agent-server subprocess."
        ),
    )
    failure_stage: str | None = Field(
        default=None, description="Stable stage at which the local Sample failed."
    )
    error_code: str | None = Field(
        default=None, description="Machine-readable local Sample failure code."
    )
    error_message: str | None = Field(
        default=None, description="Controlled human-readable failure reason."
    )
    report_path: str | None = Field(default=None)
    output_path: str | None = Field(default=None)
    record_count: int | None = Field(
        default=None, description="Total records written to the output JSONL."
    )
    sample_records: list[dict[str, Any]] = Field(
        default_factory=list,
        description="First output records for inspection (local sample preview).",
    )

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        original_text = "\n".join(
            item.text for item in self.content if isinstance(item, TextContent)
        ).strip()
        if self.is_error:
            error_message = (
                self.error_message or original_text or "DataFlow pipeline failed."
            )
            lines = [
                self.ERROR_MESSAGE_HEADER,
                f"failure_stage={self.failure_stage or 'pipeline_execution'}",
                f"error_code={self.error_code or 'dataflow_pipeline_failed'}",
                f"error_message={error_message}",
                f"exit_code={self.exit_code}",
            ]
            if self.execution:
                lines.append(f"execution={self.execution}")
            lines.append(f"report_path={self.report_path or 'none'}")
            if (
                original_text
                and original_text != self.error_message
                and not (self.stdout_tail or self.stderr_tail)
            ):
                lines.extend(["--- diagnostic ---", original_text])
            if self.stdout_tail:
                lines.extend(["--- stdout (tail) ---", self.stdout_tail])
            if self.stderr_tail:
                lines.extend(["--- stderr (tail) ---", self.stderr_tail])
            return [TextContent(text="\n".join(lines))]

        location = f"execution={self.execution}\n" if self.execution else ""
        content = [
            TextContent(
                text=(
                    f"{location}"
                    f"exit_code={self.exit_code}\n"
                    f"report_path={self.report_path or 'none'}\n"
                    f"--- stdout (tail) ---\n{self.stdout_tail}\n"
                    f"--- stderr (tail) ---\n{self.stderr_tail}"
                )
            )
        ]
        if self.sample_records:
            preview = "\n".join(
                json.dumps(record, ensure_ascii=False) for record in self.sample_records
            )
            shown = self.record_count or len(self.sample_records)
            content.append(
                TextContent(text=f"--- output records ({shown}) ---\n{preview}")
            )
        return content

    @property
    def visualize(self) -> Text:
        text = Text()
        style = "green" if self.exit_code == 0 else "red"
        text.append(f"Pipeline finished (exit {self.exit_code})\n", style=style)
        if self.sample_records:
            shown = self.record_count or len(self.sample_records)
            text.append(f"Output records ({shown}):\n", style="bold cyan")
            for record in self.sample_records:
                text.append(json.dumps(record, ensure_ascii=False) + "\n")
        tail = (self.stdout_tail or self.stderr_tail)[-500:]
        if tail:
            text.append(tail)
        return text


def _df_failure(
    *,
    stage: str,
    code: str,
    message: str,
    exit_code: int = 2,
    stdout_tail: str = "",
    stderr_tail: str = "",
    report_path: str | None = None,
    output_path: str | None = None,
) -> DfRunPipelineObservation:
    """Build a structured error without losing the diagnostic text."""

    return DfRunPipelineObservation.from_text(
        text=message,
        is_error=True,
        exit_code=exit_code,
        failure_stage=stage,
        error_code=code,
        error_message=message,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        report_path=report_path,
        output_path=output_path,
    )


def _truncated_input_failure(exc: Exception) -> DfRunPipelineObservation:
    return _df_failure(
        stage="input_resolution",
        code="workspace_input_staging_truncated",
        message=(
            f"Staging the pipeline input from the execution workspace was cut "
            f"off: {exc}. Retry the call once; if it repeats, pass an input "
            "that stages fewer files."
        ),
    )


class _WorkspaceRun:
    """Resolve pipeline paths for one DataFlow run.

    Local workspaces resolve against the host filesystem exactly as before.
    A remote workspace that can run commands and mount user Storage executes the
    run where the data already is, so its paths stay sandbox paths. Any other
    remote workspace has no host filesystem view, so every path the run touches
    is staged into a private host directory first and the produced artifacts are
    published back afterwards. ``storage/...`` arguments resolve against the
    sandbox Storage mount, which is what lets a run read user Storage without
    materializing it into the conversation.
    """

    def __init__(self, conversation: Any, *, sandbox_execution: bool = False) -> None:
        workspace = getattr(conversation, "workspace", None)
        if workspace is None:
            raise WorkspaceStagingError(
                "df_run_pipeline requires an active conversation."
            )
        self._conversation = conversation
        self._workspace = workspace
        self._sandbox = sandbox_execution
        if sandbox_execution:
            self._stage = None
        elif is_remote_workspace(workspace):
            self._stage: Path | None = Path(
                tempfile.mkdtemp(prefix="pyromind-dataflow-run-")
            ).resolve()
        else:
            self._stage = None
        self._staged_files = 0

    @property
    def remote(self) -> bool:
        """True when the run is staged onto the agent host."""
        return self._stage is not None

    @property
    def sandbox(self) -> bool:
        """True when the run executes inside the workspace's own sandbox."""
        return self._sandbox

    @property
    def workspace(self) -> Any:
        return self._workspace

    @property
    def stage_root(self) -> Path:
        """Host directory mirroring the workspace for a remote run."""
        if self._stage is None:
            return Path(self._workspace.working_dir)
        return self._stage

    def pipeline_path(self, path: str) -> Path:
        self._reject_sandbox("pipeline_path")
        return self._staged(path, label="pipeline")

    def input_path(self, path: str, *, with_siblings: bool = False) -> Path:
        """Stage a readable file or directory, including ``storage/...``.

        ``with_siblings`` keeps the input's containing workspace directory in
        the staged run, so a pipeline that resolves assets relative to its
        input (the image tree a JSONL manifest names) finds them. A Storage
        directory can be arbitrarily large, so a Storage manifest brings only
        the images its records name instead of the whole tree.
        """
        self._reject_sandbox("input_path")
        if not self.remote:
            return _resolve_input_path(self._conversation, path)
        staged = self._staged(path, label="input")
        if with_siblings and staged.is_file():
            self._stage_siblings(path, staged)
        return staged

    def sandbox_pipeline(self, path: str) -> SandboxTarget:
        """Resolve the pipeline script inside the sandbox."""
        return sandbox_target(self._workspace, path)

    def sandbox_input(self, path: str) -> SandboxTarget:
        """Resolve a readable input inside the sandbox, including ``storage/``."""
        return sandbox_target(self._workspace, path)

    def sandbox_support_file(self, path: str) -> SandboxTarget:
        return sandbox_target(self._workspace, path)

    def sandbox_output(self, path: str) -> SandboxTarget:
        """Resolve a writable output inside the sandbox.

        Outputs stay inside ``public_data/``, because ``storage/`` is read-only.
        """
        target = sandbox_target(self._workspace, path)
        if target.from_storage:
            raise WorkspaceStagingError(
                "storage/ is read-only; write outputs inside public_data/."
            )
        if not target.relative.parts or target.relative.parts[0] != "public_data":
            raise WorkspaceStagingError(
                "Write and edit paths must stay within public_data/."
            )
        return target

    def _reject_sandbox(self, method: str) -> None:
        if self._sandbox:
            raise WorkspaceStagingError(
                f"{method} is unavailable when the run executes in the sandbox."
            )

    def _stage_siblings(self, path: str, staged: Path) -> None:
        relative, from_storage = resolve_workspace_path(self._workspace, path)
        assert self._stage is not None
        directory = PurePosixPath(*relative.parts[:-1])
        if from_storage:
            self._stage_storage_images(directory, staged)
            return
        if not directory.parts:
            return
        stage_workspace_path(
            self._workspace,
            directory.as_posix(),
            destination=self._stage,
        )

    def _stage_storage_images(self, directory: PurePosixPath, manifest: Path) -> None:
        """Stage the images a Storage manifest names, not the whole tree."""
        references = _manifest_image_references(manifest)
        if not references:
            return
        assert self._stage is not None
        stage_workspace_members(
            self._workspace,
            PurePosixPath(STORAGE_ALIAS, *directory.parts).as_posix(),
            references,
            destination=self._stage,
        )

    def record_staged_files(self) -> None:
        """Freeze the staged file count before the pipeline adds its own output."""
        if self._stage is None:
            return
        self._staged_files = sum(1 for path in self._stage.rglob("*") if path.is_file())

    def stage_summary(self) -> str | None:
        """Describe what a remote run staged, so a failed run is debuggable."""
        if self._sandbox:
            return f"execution=sandbox workspace={self._workspace.working_dir}"
        if self._stage is None:
            return None
        return f"staged_root={self._stage} staged_files={self._staged_files}"

    def from_storage(self, path: str) -> bool:
        """True when ``path`` addresses Storage rather than the workspace."""
        try:
            _, from_storage = resolve_workspace_path(self._workspace, path)
        except WorkspaceStagingError:
            return False
        return from_storage

    def support_file_path(self, path: str) -> Path:
        self._reject_sandbox("support_file_path")
        return self._staged(path, label="support file")

    def output_path(self, path: str) -> Path:
        """Resolve a writable output path, which must stay in ``public_data/``."""
        self._reject_sandbox("output_path")
        if not self.remote:
            return _resolve_output_path(self._conversation, path)
        relative, from_storage = resolve_workspace_path(self._workspace, path)
        if from_storage:
            raise WorkspaceStagingError(
                "storage/ is read-only; write outputs inside public_data/."
            )
        if not relative.parts or relative.parts[0] != "public_data":
            raise WorkspaceStagingError(
                "Write and edit paths must stay within public_data/."
            )
        assert self._stage is not None
        return self._stage / relative

    def display_path(self, path: Path | None) -> str | None:
        """Express a run path the way the model addresses it in the workspace."""
        if path is None:
            return None
        if not self.remote:
            return str(path)
        assert self._stage is not None
        try:
            relative = path.relative_to(self._stage)
        except ValueError:
            return str(path)
        return relative.as_posix()

    def publish_artifacts(self, paths: Sequence[Path | None]) -> None:
        """Copy produced artifacts back into a remote workspace."""
        if not self.remote:
            return
        assert self._stage is not None
        published: list[Path] = []
        for path in paths:
            if path is None or not path.exists():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(self._stage):
                continue
            if any(
                resolved == parent or resolved.is_relative_to(parent)
                for parent in published
            ):
                continue
            publish_workspace_path(
                self._workspace, path, path.relative_to(self._stage).as_posix()
            )
            published.append(resolved)

    def cleanup(self) -> None:
        if self._stage is not None:
            shutil.rmtree(self._stage, ignore_errors=True)

    def _staged(self, path: str, *, label: str) -> Path:
        if not self.remote:
            return resolve_workspace_file(self._conversation, path)
        assert self._stage is not None
        try:
            return stage_workspace_path(self._workspace, path, destination=self._stage)
        except WorkspaceStagingError as exc:
            # Keep the concrete staging error so callers can classify it.
            raise type(exc)(f"Invalid {label}: {exc}") from exc


def _missing_image_hint(run: _WorkspaceRun, path: str) -> str:
    """Explain how to stage the images a vision pipeline could not resolve."""
    if run.sandbox:
        return (
            "\nThe run executes inside the sandbox, so image paths must resolve "
            "there. Keep the manifest and every image it references inside one "
            "workspace directory or Storage directory and pass the manifest "
            "file. Passing a directory instead makes the runtime scan images "
            "and drop manifest fields such as image_labels."
        )
    if not run.remote:
        return (
            "\nVision pipelines resolve image paths relative to args[0]. Keep "
            "the manifest and every image it references inside one workspace "
            "directory and pass the manifest file."
        )
    if run.from_storage(path):
        return (
            "\nThe images a storage manifest names are staged with it, up to "
            f"{_MANIFEST_RECORD_LIMIT} records and {_MANIFEST_IMAGE_LIMIT} "
            "images. For a larger sample, copy the records you need and their "
            "images into one workspace directory and pass that manifest file. "
            "Passing a directory instead makes the runtime scan images and drop "
            "manifest fields such as image_labels."
        )
    return (
        "\nOnly the input's own directory tree is staged, so the images it "
        "references are absent. Pass the manifest file from a workspace "
        "directory that also holds every image it references. Passing a "
        "directory instead makes the runtime scan images and drop manifest "
        "fields such as image_labels."
    )


def _manifest_image_references(manifest: Path) -> list[str]:
    """Relative image paths the first records of a staged manifest name.

    A Storage manifest travels without its directory, so the images its records
    reference have to be collected from the manifest itself. Both lists stay
    bounded: a sample only reads a few records, and a manifest that names more
    images than the bound falls back to the missing-image hint.
    """
    references: list[str] = []
    seen: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= _MANIFEST_RECORD_LIMIT:
                break
            record = _manifest_record(line)
            if record is None:
                continue
            for value in _record_strings(record):
                if value in seen or not _is_relative_image_path(value):
                    continue
                seen.add(value)
                references.append(value)
                if len(references) >= _MANIFEST_IMAGE_LIMIT:
                    return references
    return references


def _manifest_record(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        record = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def _record_strings(value: Any, depth: int = 0) -> Iterator[str]:
    if depth > 3:
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for nested in value.values():
            yield from _record_strings(nested, depth + 1)
        return
    if isinstance(value, list):
        for nested in value:
            yield from _record_strings(nested, depth + 1)


def _is_relative_image_path(value: str) -> bool:
    candidate = value.strip()
    if not candidate or "://" in candidate or candidate.startswith("/"):
        return False
    path = PurePosixPath(candidate)
    if ".." in path.parts:
        return False
    return path.suffix.lower() in _IMAGE_SUFFIXES


class DfRunPipelineExecutor(ToolExecutor):
    def __init__(self, *, runtime_dir: str | None = None) -> None:
        self._runtime_dir = Path(runtime_dir) if runtime_dir else None
        self._sample_executor = ProcessLocalSampleExecutor()
        self._active_sandbox_run: SandboxSampleExecutor | None = None

    def interrupt(self) -> None:
        self._sample_executor.interrupt()
        active = self._active_sandbox_run
        if active is not None:
            active.interrupt()

    def _stage_runtime_files(self, target_dir: Path) -> Path | None:
        """Copy runtime helpers into a hidden tool-owned directory."""
        if self._runtime_dir is None:
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in RUNTIME_FILENAMES:
            src = self._runtime_dir / filename
            dst = target_dir / filename
            if src.is_file():
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return target_dir

    @staticmethod
    def _add_runtime_pythonpath(env_extra: dict[str, str], runtime_dir: Path) -> None:
        existing = env_extra.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
        paths = [str(runtime_dir)]
        if existing:
            paths.append(existing)
        env_extra["PYTHONPATH"] = os.pathsep.join(paths)

    def _preflight_managed_image_pipeline(self, pipeline: Path) -> None:
        if self._runtime_dir is None:
            raise ValueError("DataFlow pipeline runtime_dir is not configured.")
        image_utils = self._runtime_dir / "image_utils.py"
        public_names = runtime_public_names(image_utils)
        validate_managed_image_pipeline(pipeline, public_names)

    def __call__(
        self, action: DfRunPipelineAction, conversation: Any = None
    ) -> DfRunPipelineObservation:
        try:
            workspace = getattr(conversation, "workspace", None)
            run = _WorkspaceRun(
                conversation, sandbox_execution=supports_sandbox_execution(workspace)
            )
        except WorkspaceStagingError as exc:
            return _df_failure(
                stage="pipeline_resolution",
                code="workspace_unavailable",
                message=str(exc),
            )
        try:
            if run.sandbox:
                return self._run_in_sandbox(action, conversation, run)
            return self._run(action, conversation, run)
        finally:
            run.cleanup()

    def _sandbox_executor(self, run: _WorkspaceRun) -> SandboxSampleExecutor:
        """Build the sandbox executor for one run, pinned to the runtime bundle."""
        assert self._runtime_dir is not None
        return SandboxSampleExecutor(
            run.workspace,
            runtime_dir=self._runtime_dir,
            runtime_filenames=RUNTIME_FILENAMES,
            runtime_fingerprint=runtime_bundle_fingerprint(
                self._runtime_dir, RUNTIME_FILENAMES
            ),
        )

    def _run_in_sandbox(
        self,
        action: DfRunPipelineAction,
        conversation: Any,
        run: _WorkspaceRun,
    ) -> DfRunPipelineObservation:
        """Run the sample inside the sandbox that already holds the data.

        The pipeline script, its input, and the produced artifacts stay in the
        sandbox; only the run's log tails and a bounded result envelope come
        back to the agent host.
        """
        try:
            pipeline = run.sandbox_pipeline(action.pipeline_path)
        except WorkspaceStagingError as exc:
            return _df_failure(
                stage="pipeline_resolution",
                code="workspace_pipeline_not_found",
                message=f"Invalid pipeline path: {exc}",
            )
        process_args = list(action.args)
        standard_input: SandboxTarget | None = None
        output: SandboxTarget | None = None
        support: SandboxTarget | None = None
        image_task = action.model_profile == "vision"
        if action.output_schema is not None:
            if len(process_args) < 2:
                return _df_failure(
                    stage="input_resolution",
                    code="standard_pipeline_args_missing",
                    message=(
                        "output_schema requires standard pipeline arguments: "
                        "args[0]=workspace input and args[1]=workspace output JSONL."
                    ),
                )
            try:
                standard_input = run.sandbox_input(process_args[0])
            except WorkspaceStagingError as exc:
                return _df_failure(
                    stage="input_resolution",
                    code="workspace_input_not_found",
                    message=f"Invalid standard pipeline input: {exc}",
                )
            try:
                output = run.sandbox_output(process_args[1])
            except WorkspaceStagingError as exc:
                return _df_failure(
                    stage="output_resolution",
                    code=(
                        "workspace_output_outside"
                        if "outside" in str(exc).lower()
                        else "workspace_output_not_writable"
                    ),
                    message=f"Invalid standard pipeline output: {exc}",
                )
            process_args[0] = str(standard_input.absolute)
            process_args[1] = str(output.absolute)
        elif len(process_args) >= 2:
            legacy_input, legacy_output = self._sandbox_legacy_paths(
                run, process_args, pipeline
            )
            standard_input = legacy_input
            if legacy_output is not None:
                process_args[1] = str(legacy_output)
                output = sandbox_target_from_absolute(run.workspace, legacy_output)
        if (
            standard_input is not None
            and output is not None
            and sandbox_is_dir(run.workspace, str(standard_input.absolute))
            and output.absolute.is_relative_to(standard_input.absolute)
        ):
            return _df_failure(
                stage="output_resolution",
                code="output_inside_source",
                message="Pipeline output must be outside the source directory.",
            )
        if action.support_file_path is not None:
            if len(process_args) < 2:
                return _df_failure(
                    stage="input_resolution",
                    code="standard_pipeline_args_missing",
                    message="support_file_path requires input and output arguments.",
                )
            if len(process_args) > 2:
                return _df_failure(
                    stage="input_resolution",
                    code="support_file_argument_conflict",
                    message=(
                        "Do not pass a third positional argument when "
                        "support_file_path is set; the tool appends it."
                    ),
                )
            try:
                support = run.sandbox_support_file(action.support_file_path)
            except WorkspaceStagingError as exc:
                return _df_failure(
                    stage="input_resolution",
                    code="workspace_support_file_not_found",
                    message=f"Invalid support file: {exc}",
                )
            if support.relative.suffix.lower() != ".json":
                return _df_failure(
                    stage="input_resolution",
                    code="support_file_not_json",
                    message="support_file_path must point to a JSON file.",
                )
            try:
                json.loads(
                    sandbox_read_text(
                        run.workspace, str(support.absolute), limit=10_000_000
                    )
                )
            except (SandboxExecutionError, json.JSONDecodeError) as exc:
                return _df_failure(
                    stage="input_resolution",
                    code="support_file_invalid_json",
                    message=f"support_file_path is not valid JSON: {exc}",
                )
        if self._runtime_dir is None:
            return _df_failure(
                stage="runtime_dependency",
                code="dataflow_runtime_invalid",
                message="DataFlow pipeline runtime is not configured.",
            )
        try:
            executor = self._sandbox_executor(run)
        except ValueError as exc:
            return _df_failure(
                stage="runtime_dependency",
                code="dataflow_runtime_invalid",
                message=f"Invalid DataFlow runtime: {exc}",
            )
        if action.output_schema in {"vision", "structured"}:
            try:
                with tempfile.TemporaryDirectory(
                    prefix="pyromind-df-preflight-"
                ) as staging:
                    local_pipeline = Path(staging) / "pipeline.py"
                    local_pipeline.write_text(
                        sandbox_read_text(run.workspace, str(pipeline.absolute)),
                        encoding="utf-8",
                    )
                    self._preflight_managed_image_pipeline(local_pipeline)
            except (SandboxExecutionError, OSError, ValueError) as exc:
                return _df_failure(
                    stage="pipeline_resolution",
                    code="managed_image_pipeline_invalid",
                    message=f"Invalid managed image pipeline: {exc}",
                )
        env_extra: dict[str, str] = {}
        if action.model_profile != "none":
            try:
                env_extra = build_dataflow_env(conversation, action.model_profile)
            except ValueError as exc:
                return _df_failure(
                    stage="model_configuration",
                    code="dataflow_model_configuration_invalid",
                    message=f"Invalid DataFlow model configuration: {exc}",
                )
        sandbox_env = dict(env_extra)
        if action.output_schema is not None:
            sandbox_env["DF_OUTPUT_SCHEMA"] = action.output_schema
        if output is not None:
            log_dir = output.absolute.parent
            state_dir = log_dir / f".{output.absolute.stem}.state"
            sandbox_env["DF_LOG_DIR"] = str(log_dir)
            sandbox_env["DF_STATE_DIR"] = str(log_dir)
            sandbox_env["DF_RESUME"] = "0"
            sandbox_env["DF_EXECUTION_REVISION"] = "1"
            sandbox_env["DF_RUNTIME_FINGERPRINT"] = executor.runtime_fingerprint
        else:
            state_dir = pipeline.absolute.parent / f".{pipeline.absolute.stem}.state"
            log_dir = state_dir
        image_root: str | None = None
        if action.output_schema == "vision" and standard_input is not None:
            image_root = str(
                standard_input.absolute
                if sandbox_is_dir(run.workspace, str(standard_input.absolute))
                else standard_input.absolute.parent
            )
        config_summary = (
            "model=none api_key_configured=no"
            if action.model_profile == "none"
            else summarize_dataflow_env(env_extra)
        )
        spec = {
            "python": action.python
            or os.environ.get(ENV_SANDBOX_DATAFLOW_PYTHON)
            or None,
            "venv": sandbox_dataflow_venv(SUPPORTED_DATAFLOW_VERSION),
            "dataflow_version": SUPPORTED_DATAFLOW_VERSION,
            "packages": list(DATAFLOW_RUNTIME_PACKAGES),
            "model_profile": action.model_profile,
            "cwd": str(pipeline.absolute.parent),
            "pipeline": str(pipeline.absolute),
            "args": process_args,
            "support_file": str(support.absolute) if support is not None else None,
            "support_dir": str(state_dir / "support"),
            "output_schema": action.output_schema,
            "output_path": str(output.absolute) if output is not None else None,
            "input_path": (
                str(standard_input.absolute) if standard_input is not None else None
            ),
            "image_root": image_root,
            "log_dir": str(log_dir),
            "state_dir": str(state_dir),
            "env": sandbox_env,
            "timeout": action.timeout,
            "execution_revision": 1,
            "runtime_fingerprint": executor.runtime_fingerprint,
        }
        try:
            self._active_sandbox_run = executor
            try:
                envelope = executor.run(spec, timeout=action.timeout)
            finally:
                self._active_sandbox_run = None
        except SandboxExecutionError as exc:
            message = str(exc)
            timed_out = "timed out" in message
            return _df_failure(
                stage="timeout" if timed_out else "pipeline_execution",
                code=(
                    "dataflow_pipeline_timeout"
                    if timed_out
                    else "sandbox_sample_failed"
                ),
                message=message,
            )
        return self._sandbox_observation(
            envelope,
            run=run,
            action=action,
            config_summary=config_summary,
            report_path=f"{log_dir}/report.json" if output is not None else None,
            output_path=str(output.display) if output is not None else None,
            image_task=image_task,
        )

    @staticmethod
    def _sandbox_legacy_paths(
        run: _WorkspaceRun,
        process_args: list[str],
        pipeline: SandboxTarget,
    ) -> tuple[SandboxTarget | None, PurePosixPath | None]:
        """Resolve legacy arguments that may name a workspace path.

        Legacy arguments were historically interpreted by the child process
        relative to the pipeline script directory. When ``args[0]`` names an
        existing workspace path, resolve it from the workspace root and anchor a
        relative output at the workspace root, matching the host-side behavior.
        Anchoring at the input's own directory instead would write the output
        beside a Storage input, which is read-only.
        """
        try:
            candidate = sandbox_target(run.workspace, process_args[0])
        except WorkspaceStagingError:
            candidate = None
        if candidate is None or not sandbox_exists(
            run.workspace, str(candidate.absolute)
        ):
            second = PurePosixPath(process_args[1])
            output = (
                second if second.is_absolute() else pipeline.absolute.parent / second
            )
            return None, output
        process_args[0] = str(candidate.absolute)
        second = PurePosixPath(process_args[1])
        output = (
            second
            if second.is_absolute()
            else PurePosixPath(str(run.workspace.working_dir)) / second
        )
        return candidate, output

    @staticmethod
    def _sandbox_observation(
        envelope: dict[str, Any],
        *,
        run: _WorkspaceRun,
        action: DfRunPipelineAction,
        config_summary: str,
        report_path: str | None,
        output_path: str | None,
        image_task: bool,
    ) -> DfRunPipelineObservation:
        """Turn the sandbox envelope into the tool's observation."""
        rc_value = envelope.get("rc")
        rc = rc_value if isinstance(rc_value, int) else 1
        failure_stage = envelope.get("failure_stage")
        error_code = envelope.get("error_code")
        error_message = envelope.get("error_message")
        stdout = str(envelope.get("stdout_tail") or "")
        stderr = str(envelope.get("stderr_tail") or "")
        record_count = envelope.get("record_count")
        sample_records = envelope.get("sample_records") or []
        if not isinstance(sample_records, list):
            sample_records = []
        if not isinstance(record_count, int):
            record_count = None
        if (
            image_task
            and isinstance(error_message, str)
            and "missing image" in error_message
            and action.args
        ):
            error_message += _missing_image_hint(run, action.args[0])
        text = f"Pipeline model: {config_summary}\nexit_code={rc}\n"
        stage_summary = run.stage_summary()
        if stage_summary is not None:
            text += f"{stage_summary}\n"
        interpreter = envelope.get("interpreter")
        if isinstance(interpreter, str) and interpreter:
            source = envelope.get("interpreter_source")
            text += f"dataflow_interpreter={interpreter} ({source})\n"
        text += f"--- stdout (tail) ---\n{stdout}\n--- stderr (tail) ---\n{stderr}"
        return DfRunPipelineObservation.from_text(
            text=text,
            is_error=rc != 0,
            exit_code=rc,
            execution="sandbox",
            failure_stage=failure_stage if isinstance(failure_stage, str) else None,
            error_code=error_code if isinstance(error_code, str) else None,
            error_message=error_message if isinstance(error_message, str) else None,
            stdout_tail=stdout,
            stderr_tail=stderr,
            report_path=report_path,
            output_path=output_path,
            record_count=record_count,
            sample_records=sample_records,
        )

    def _run(
        self,
        action: DfRunPipelineAction,
        conversation: Any,
        run: _WorkspaceRun,
    ) -> DfRunPipelineObservation:
        try:
            pipeline = run.pipeline_path(action.pipeline_path)
        except Exception as exc:
            return _df_failure(
                stage="pipeline_resolution",
                code="workspace_pipeline_not_found",
                message=f"Invalid pipeline path: {exc}",
            )
        process_args = list(action.args)
        standard_input_path: Path | None = None
        output_path: Path | None = None
        support_file_path: Path | None = None
        state_dir: Path | None = None
        log_dir: Path | None = None
        # Image pipelines read image files relative to their input, so the
        # staged run needs the input's directory, not just the manifest.
        image_task = action.model_profile == "vision"
        if action.output_schema is not None:
            if len(process_args) < 2:
                return _df_failure(
                    stage="input_resolution",
                    code="standard_pipeline_args_missing",
                    message=(
                        "output_schema requires standard pipeline arguments: "
                        "args[0]=workspace input and args[1]=workspace output JSONL."
                    ),
                )
            try:
                standard_input_path = run.input_path(
                    process_args[0], with_siblings=image_task
                )
            except WorkspaceArchiveTruncatedError as exc:
                return _truncated_input_failure(exc)
            except ValueError as exc:
                return _df_failure(
                    stage="input_resolution",
                    code=(
                        "workspace_input_outside"
                        if "outside" in str(exc).lower()
                        else "workspace_input_not_found"
                    ),
                    message=f"Invalid standard pipeline input: {exc}",
                )
            try:
                output_path = run.output_path(process_args[1])
            except ValueError as exc:
                return _df_failure(
                    stage="output_resolution",
                    code=(
                        "workspace_output_outside"
                        if "outside" in str(exc).lower()
                        else "workspace_output_not_writable"
                    ),
                    message=f"Invalid standard pipeline output: {exc}",
                )
            process_args[0] = str(standard_input_path)
            process_args[1] = str(output_path)
        elif len(process_args) >= 2:
            # Legacy mode: the child process runs with cwd=pipeline.parent, but
            # the rest of this tool family resolves arguments from the workspace
            # root. Resolve args[0] from the workspace root when the input
            # exists there (falling back to the historical pipeline-relative
            # interpretation) and anchor the relative output at the same base so
            # both stay consistent.
            try:
                legacy_paths = _resolve_legacy_input_arg(
                    run,
                    process_args[0],
                    pipeline,
                    with_siblings=image_task,
                )
            except WorkspaceArchiveTruncatedError as exc:
                return _truncated_input_failure(exc)
            if legacy_paths is not None:
                input_path, base = legacy_paths
                standard_input_path = input_path
                process_args[0] = str(input_path)
                output_path = base / process_args[1]
                process_args[1] = str(output_path)
            else:
                output_path = Path(process_args[1])
                if not output_path.is_absolute():
                    output_path = pipeline.parent / output_path

        if action.support_file_path is not None:
            if len(process_args) < 2:
                return _df_failure(
                    stage="input_resolution",
                    code="standard_pipeline_args_missing",
                    message="support_file_path requires input and output arguments.",
                )
            if len(process_args) > 2:
                return _df_failure(
                    stage="input_resolution",
                    code="support_file_argument_conflict",
                    message=(
                        "Do not pass a third positional argument when "
                        "support_file_path is set; the tool appends it."
                    ),
                )
            try:
                support_file_path = run.support_file_path(action.support_file_path)
            except ValueError as exc:
                return _df_failure(
                    stage="input_resolution",
                    code="workspace_support_file_not_found",
                    message=f"Invalid support file: {exc}",
                )
            if support_file_path.suffix.lower() != ".json":
                return _df_failure(
                    stage="input_resolution",
                    code="support_file_not_json",
                    message="support_file_path must point to a JSON file.",
                )
            try:
                json.loads(support_file_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                return _df_failure(
                    stage="input_resolution",
                    code="support_file_invalid_json",
                    message=f"support_file_path is not valid JSON: {exc}",
                )
            try:
                json.loads(support_file_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                return _df_failure(
                    stage="input_resolution",
                    code="support_file_invalid_json",
                    message=f"support_file_path is not valid JSON: {exc}",
                )

        if (
            standard_input_path is not None
            and standard_input_path.is_dir()
            and output_path is not None
            and output_path.resolve().is_relative_to(standard_input_path.resolve())
        ):
            return _df_failure(
                stage="output_resolution",
                code="output_inside_source",
                message="Pipeline output must be outside the source directory.",
            )

        if action.output_schema in {"vision", "structured"}:
            try:
                self._preflight_managed_image_pipeline(pipeline)
            except ValueError as exc:
                return _df_failure(
                    stage="pipeline_resolution",
                    code="managed_image_pipeline_invalid",
                    message=f"Invalid managed image pipeline: {exc}",
                )

        python = resolve_dataflow_python(action.python)
        if action.model_profile != "none":
            ok, detail = check_dataflow_installed(python)
            if not ok:
                return _df_failure(
                    stage="runtime_dependency",
                    code="dataflow_not_installed",
                    message=(
                        f"DataFlow is not installed for interpreter `{python}`.\n"
                        "Install it with `uv pip install open-dataflow` (preferably "
                        "in a dedicated venv) or set DATAFLOW_PYTHON to an "
                        f"interpreter that has it.\nImport check: {detail or 'failed'}"
                    ),
                )
        if action.output_schema in {"vision", "structured"}:
            version_ok, version_detail = check_dataflow_version(python)
            if not version_ok:
                return _df_failure(
                    stage="runtime_dependency",
                    code="dataflow_version_mismatch",
                    message=(
                        f"Local DataFlow version must match Pyromind: {version_detail}"
                    ),
                )
        env_extra: dict[str, str] = {}
        if action.model_profile != "none":
            try:
                env_extra = build_dataflow_env(
                    conversation,
                    action.model_profile,
                    gateway=action.labeling_gateway,
                )
            except ValueError as exc:
                return _df_failure(
                    stage="model_configuration",
                    code="dataflow_model_configuration_invalid",
                    message=f"Invalid DataFlow model configuration: {exc}",
                )
            try:
                preflight_dataflow_llm(env_extra)
            except ValueError as exc:
                return _df_failure(
                    stage="model_configuration",
                    code="dataflow_llm_preflight_failed",
                    message=f"DataFlow LLM preflight failed: {exc}",
                )

        run.record_staged_files()
        if action.output_schema is not None:
            env_extra["DF_OUTPUT_SCHEMA"] = action.output_schema
        if output_path is not None:
            state_dir = output_path.parent / f".{output_path.stem}.state"
            state_dir.mkdir(parents=True, exist_ok=True)
            log_dir = output_path.parent
            env_extra["DF_LOG_DIR"] = str(log_dir)
            env_extra["DF_STATE_DIR"] = str(log_dir)
            env_extra["DF_RESUME"] = "0"
            env_extra["DF_EXECUTION_REVISION"] = "1"
            if self._runtime_dir is not None:
                try:
                    env_extra["DF_RUNTIME_FINGERPRINT"] = runtime_bundle_fingerprint(
                        self._runtime_dir,
                        RUNTIME_FILENAMES,
                    )
                except ValueError as exc:
                    return _df_failure(
                        stage="runtime_dependency",
                        code="dataflow_runtime_invalid",
                        message=f"Invalid DataFlow runtime: {exc}",
                    )
        elif self._runtime_dir is not None:
            state_dir = pipeline.parent / f".{pipeline.stem}.state"
            state_dir.mkdir(parents=True, exist_ok=True)
            log_dir = state_dir

        if self._runtime_dir is not None:
            assert state_dir is not None
            runtime_stage_dir = self._stage_runtime_files(state_dir / "runtime")
            if runtime_stage_dir is not None:
                self._add_runtime_pythonpath(env_extra, runtime_stage_dir)
        if support_file_path is not None:
            assert state_dir is not None
            frozen_support = state_dir / "support" / support_file_path.name
            frozen_support.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(support_file_path, frozen_support)
            process_args.append(str(frozen_support))
        source_fingerprint_before = (
            _source_fingerprint(standard_input_path)
            if standard_input_path is not None
            else None
        )
        config_summary = (
            "model=none api_key_configured=no"
            if action.model_profile == "none"
            else summarize_dataflow_env(env_extra)
        )
        validation_rc: int | None = None
        report_rc: int | None = None
        rc, stdout, stderr = self._sample_executor.run(
            python,
            [str(pipeline), *process_args],
            cwd=str(pipeline.parent),
            env_extra=env_extra,
            timeout=action.timeout,
        )
        pipeline_rc = rc
        source_fingerprint_error: str | None = None
        try:
            source_fingerprint_after = (
                _source_fingerprint(standard_input_path)
                if standard_input_path is not None
                else None
            )
        except OSError as exc:
            # Deleting or otherwise making the source unreadable is itself an
            # integrity violation. Keep returning a structured tool failure.
            source_fingerprint_after = None
            source_fingerprint_error = str(exc)
        source_integrity_rc = 0
        if (
            source_fingerprint_before is not None
            and source_fingerprint_before != source_fingerprint_after
        ):
            source_integrity_rc = 90
            if rc == 0:
                rc = source_integrity_rc
            stderr = (
                f"{stderr}\nSource data changed during pipeline execution; "
                "the source must remain read-only."
            )
        if log_dir is not None and source_fingerprint_before is not None:
            (log_dir / "source_integrity.json").write_text(
                json.dumps(
                    {
                        "before": source_fingerprint_before,
                        "after": source_fingerprint_after,
                        "unchanged": source_integrity_rc == 0,
                        "error": source_fingerprint_error,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if rc == 0 and action.output_schema not in {None, "structured"}:
            assert output_path is not None
            assert state_dir is not None
            validator = (
                state_dir / "runtime" / "validate_prepared_data.py"
                if self._runtime_dir is not None
                else pipeline.parent / "validate_prepared_data.py"
            )
            validation_args = [
                str(validator),
                str(output_path),
                "--schema",
                action.output_schema,
                "--report",
                str((log_dir or state_dir) / "validation.json"),
            ]
            if action.output_schema == "vision":
                assert standard_input_path is not None
                input_path = standard_input_path
                image_root = input_path if input_path.is_dir() else input_path.parent
                validation_args.extend(["--image-root", str(image_root)])
            validation_rc, validation_stdout, validation_stderr = (
                self._sample_executor.run(
                    python,
                    validation_args,
                    cwd=str(pipeline.parent),
                    env_extra=env_extra,
                    timeout=min(action.timeout, 600),
                )
            )
            stdout = f"{stdout}\n--- validation ---\n{validation_stdout}"
            stderr = f"{stderr}\n--- validation ---\n{validation_stderr}"
            rc = validation_rc
        report_path: Path | None = None
        if output_path is not None and state_dir is not None:
            report_path = (log_dir or state_dir) / "report.json"
            report_script = (
                state_dir / "runtime" / "generate_report.py"
                if self._runtime_dir is not None
                else pipeline.parent / "generate_report.py"
            )
            report_args = [
                str(report_script),
                "--log-dir",
                str(log_dir or state_dir),
                "--pipeline-exit-code",
                str(pipeline_rc),
                "--execution-revision",
                "1",
                "--resumed",
                "false",
                "--output-file",
                str(output_path),
            ]
            report_rc, report_stdout, report_stderr = self._sample_executor.run(
                python,
                report_args,
                cwd=str(pipeline.parent),
                env_extra=env_extra,
                timeout=min(action.timeout, 600),
            )
            stdout = f"{stdout}\n--- report ---\n{report_stdout}"
            stderr = f"{stderr}\n--- report ---\n{report_stderr}"
            if rc == 0 and report_rc != 0:
                rc = report_rc
        sample_records: list[dict[str, Any]] = []
        record_count: int | None = None
        if rc == 0 and output_path is not None and output_path.is_file():
            sample_records, record_count = _read_output_records(output_path)
        failure_stage: str | None = None
        error_code: str | None = None
        error_message: str | None = None
        report_failure = _read_report_failure(report_path) if rc != 0 else {}
        reported_stage = report_failure.get("stage")
        reported_error = report_failure.get("error")
        if rc != 0:
            if rc == 124:
                failure_stage = "timeout"
                timed_out_step = (
                    "pipeline"
                    if pipeline_rc == 124
                    else "validation"
                    if validation_rc == 124
                    else "report"
                )
                error_code = f"dataflow_{timed_out_step}_timeout"
                error_message = (
                    f"DataFlow {timed_out_step} timed out after {action.timeout}s."
                )
            elif rc < 0:
                failure_stage = "cancelled"
                error_code = "dataflow_pipeline_cancelled"
                error_message = "DataFlow local Sample execution was cancelled."
            elif pipeline_rc != 0:
                failure_stage = (
                    reported_stage
                    if isinstance(reported_stage, str) and reported_stage.strip()
                    else "pipeline_execution"
                )
                error_code = "dataflow_pipeline_failed"
                error_message = (
                    reported_error
                    if isinstance(reported_error, str) and reported_error.strip()
                    else f"DataFlow pipeline exited with code {pipeline_rc}."
                )
            elif source_integrity_rc != 0:
                failure_stage = "source_integrity"
                error_code = "source_data_modified"
                error_message = "Pipeline modified its source data."
            elif validation_rc is not None and validation_rc != 0:
                failure_stage = "schema_validation"
                error_code = "dataflow_schema_validation_failed"
                error_message = (
                    "DataFlow output schema validation exited with code "
                    f"{validation_rc}."
                )
            elif report_rc is not None and report_rc != 0:
                failure_stage = "report_generation"
                error_code = "dataflow_report_generation_failed"
                error_message = (
                    f"DataFlow report generation exited with code {report_rc}."
                )
            else:
                failure_stage = "pipeline_execution"
                error_code = "dataflow_pipeline_failed"
                error_message = f"DataFlow pipeline exited with code {rc}."
        if (
            image_task
            and error_message is not None
            and "missing image" in error_message
            and action.args
        ):
            error_message += _missing_image_hint(run, action.args[0])
        try:
            run.publish_artifacts((output_path, log_dir, state_dir))
            publish_error = None
        except WorkspaceStagingError as exc:
            publish_error = str(exc)
        text = f"Pipeline model: {config_summary}\nexit_code={rc}\n"
        stage_summary = run.stage_summary()
        if stage_summary is not None:
            text += f"{stage_summary}\n"
        text += (
            f"--- stdout (tail) ---\n{stdout[-_LOG_TAIL_CHARS:]}\n"
            f"--- stderr (tail) ---\n{stderr[-_LOG_TAIL_CHARS:]}"
        )
        if publish_error is not None:
            text += f"\n--- publish error ---\n{publish_error}"
        return DfRunPipelineObservation.from_text(
            text=text,
            is_error=rc != 0,
            exit_code=rc,
            execution="host",
            failure_stage=failure_stage,
            error_code=error_code,
            error_message=error_message,
            stdout_tail=stdout[-_LOG_TAIL_CHARS:],
            stderr_tail=stderr[-_LOG_TAIL_CHARS:],
            report_path=run.display_path(report_path),
            output_path=run.display_path(output_path),
            record_count=record_count,
            sample_records=sample_records,
        )


class DfRunPipelineTool(ToolDefinition[DfRunPipelineAction, DfRunPipelineObservation]):
    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        runtime_dir_value = params.pop("runtime_dir", None)
        runtime_dir = str(runtime_dir_value) if runtime_dir_value is not None else None
        if params:
            names = ", ".join(sorted(params))
            raise ValueError(f"DfRunPipelineTool got unknown params: {names}")
        return [
            cls(
                description=(
                    "Run an agent-authored DataFlow-compatible Python pipeline in "
                    "an isolated subprocess. The exact input is processed; this "
                    "tool never samples or truncates it. model_profile=none "
                    "injects no model credentials and does not require DataFlow. "
                    "text/vision inject the conversation or managed vision model "
                    "configuration; a user-supplied labeling_gateway replaces the "
                    "managed vision model for this run. Scripts must never hardcode "
                    "secrets. Logging, retry, checkpoint, report, and canonical "
                    "JSONL validation helpers are provided to the run automatically "
                    "— the agent must NOT create or copy them manually. For standard "
                    "runs with output_schema, input/output arguments are resolved "
                    "from the workspace root, and read-only inputs may also be "
                    "addressed as 'storage/...' against the mounted Storage; legacy "
                    "arguments remain unchanged. When the conversation runs on a "
                    "platform sandbox, the sample executes inside that sandbox and "
                    "the same paths resolve there. The tool returns textual status "
                    "only; image content is read by the DataFlow VLM."
                ),
                action_type=DfRunPipelineAction,
                observation_type=DfRunPipelineObservation,
                executor=DfRunPipelineExecutor(runtime_dir=runtime_dir),
            )
        ]

    @classmethod
    def is_usable(cls) -> bool:
        ok, _ = check_dataflow_installed(resolve_dataflow_python())
        return ok


register_tool("df_run_pipeline", DfRunPipelineTool)
