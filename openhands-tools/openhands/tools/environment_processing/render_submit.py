"""Submit a template-driven manifest render task to Pyromind Studio.

The render step ("fill the template") runs as its own CustomCommandCPUNode so
the agent and the conversation stay free of the full dataset: the agent only
writes/confirms a render template and points at the Storage data source. The
node segments the parquet (bounded memory), writes each shard manifest to
Storage, and emits ``shards.json``. The agent then rolls out per-shard
``edp_submit`` validation from that index.

Render does not need LLM credentials (no API calls), only pandas+pyarrow to
read the mount.
"""

from __future__ import annotations

import io
import json
import shlex
import struct
import tempfile
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Self, cast

from pydantic import Field
from rich.text import Text

from openhands.sdk.conversation.state import ActiveLongTask
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.environment_processing.platform_env import (
    resolve_platform_env,
)
from openhands.tools.pyromind_dataset.definition import (
    _default_storage_base_url,
    _resolve_conversation_headers,
    _resolve_secret_headers,
    download_tail_from_pyromind,
    upload_local_file_to_pyromind,
)
from openhands.tools.workflow.task_submission import (
    PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET,
    create_workflow_api_client,
    submit_workflow_task,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


DATAFLOW_NODE_TYPE = "CustomCommandCPUNode"
RENDER_FILENAME = "render_manifest.py"
RENDER_TEMPLATE_FILENAME = "render_template.json"
RENDER_REQUIREMENTS = "render_requirements.txt"

PREFLIGHT_TAIL_BYTES = 64 * 1024
PREFLIGHT_MAX_FOOTER_BYTES = 8 * 1024 * 1024


class EdpRenderAction(Action):
    """Submit a template-driven manifest render to Pyromind Studio."""

    template_path: str = Field(
        description=(
            "Local render template JSON (field mapping + shard_size). Pass "
            "the same workspace-relative path you gave file_editor/apply_patch "
            "when writing the template (e.g. 'public_data/render_template.json'); "
            "absolute paths work too. The tool stages it into the task and the "
            "node renders against it."
        )
    )
    data_source: str = Field(
        description=(
            "Storage path of the source data (e.g. "
            "'datasets/tmax/data/train-*.parquet'). The node reads it from "
            "its Storage mount."
        )
    )
    output_root: str | None = Field(
        default=None,
        description=(
            "Storage root for batch-XXX/ shards and shards.json. Defaults to "
            "~/.pyromind-agent/<conversation>/edp_render/<run_id>."
        ),
    )
    shard_size: int = Field(
        default=500, ge=1, description="Records per shard manifest."
    )
    limit: int | None = Field(
        default=None, ge=1, description="Optional caps on rendered records."
    )
    cpu: int = Field(default=4, ge=1, le=64)
    memory: int = Field(default=16, ge=1, le=256)

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Submit render task: ", style="bold blue")
        content.append(self.data_source)
        return content


class EdpRenderObservation(Observation):
    """Result of a render task submission."""

    status: str = Field(default="Unknown")
    task_id: str | None = Field(default=None)
    run_id: str | None = Field(default=None)
    output_dir: str | None = Field(default=None)

    @property
    def visualize(self) -> Text:
        text = Text()
        style = "green" if self.status != "Failed" else "red"
        text.append(f"Render submit: {self.status}\n", style=style)
        if self.run_id:
            text.append(f"run_id={self.run_id}\n")
        if self.output_dir:
            text.append(f"output_dir={self.output_dir}\n")
        text.append(self.text)
        return text


def _pod_path(storage_path: str) -> str:
    return f"/target-workspace{storage_path}"


def _spec_column_refs(name: str, spec: Any) -> list[tuple[str, str]]:
    """Collect (field, column) references a template spec reads from the row."""
    if isinstance(spec, str):
        return [(name, spec)]
    if not isinstance(spec, dict):
        return []
    if "field" in spec:
        return [(name, str(spec["field"]))]
    kind = spec.get("kind")
    if kind == "message":
        return [(name, str(spec.get("source_field", "messages")))]
    if kind == "pytest_wrapper":
        source = spec.get("source", spec.get("source_field"))
        if source is not None:
            return _spec_column_refs(name, source)
    return []


def _collect_column_refs(fields: dict[str, Any]) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for name, spec in fields.items():
        refs.extend(_spec_column_refs(name, spec))
    return refs


def _check_column_refs(schema: Any, refs: list[tuple[str, str]]) -> list[str]:
    import pyarrow as pa

    errors: list[str] = []
    top_names = list(schema.names)
    for field_name, column in refs:
        parts = column.split(".")
        if parts[0] not in top_names:
            errors.append(
                f"field {field_name!r} references top-level column "
                f"{parts[0]!r} which does not exist (available columns: "
                f"{top_names})"
            )
            continue
        arrow_field = schema.field(parts[0])
        for depth, part in enumerate(parts[1:], start=1):
            field_type = arrow_field.type
            if not pa.types.is_struct(field_type):
                errors.append(
                    f"field {field_name!r}: column {column!r} descends into "
                    f"{part!r} but {parts[depth - 1]!r} is {field_type}, "
                    "not a struct"
                )
                break
            if part not in field_type.names:
                errors.append(
                    f"field {field_name!r}: column {column!r} references "
                    f"missing nested field {part!r} (struct fields: "
                    f"{list(field_type.names)})"
                )
                break
            arrow_field = field_type.field(part)
    return errors


class _ParquetTailFile(io.RawIOBase):
    """Read-only file-like view over [virtual "PAR1" head][gap][tail bytes].

    pyarrow only seeks/reads the footer region at the end of a parquet
    file, so a suffix-range download plus this view is enough to read the
    schema without fetching the data blocks.
    """

    def __init__(self, total_size: int, tail: bytes) -> None:
        self._size = total_size
        self._tail = tail
        self._tail_start = total_size - len(tail)
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence!r}")
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        end = (
            self._size
            if size is None or size < 0
            else min(self._pos + size, self._size)
        )
        chunks = bytearray()
        pos = self._pos
        while pos < end:
            if pos < 4 and pos < self._tail_start:
                span = min(end - pos, 4 - pos)
                chunks += b"PAR1"[:span]
                pos += span
            elif pos >= self._tail_start:
                offset = pos - self._tail_start
                span = min(end - pos, len(self._tail) - offset)
                chunks += self._tail[offset : offset + span]
                pos += span
            else:
                span = min(end - pos, self._tail_start - pos)
                chunks += b"\x00" * span
                pos += span
        self._pos = end
        return bytes(chunks)


def _preflight_template(
    template: dict[str, Any],
    data_source: str,
    *,
    storage_base_url: str,
    headers: dict[str, str],
    timeout: float,
) -> str | None:
    """Best-effort template-vs-schema check before submitting the render.

    Returns a failure message when the template references parquet columns
    that do not exist, or None when the check passes or cannot run (glob
    sources, oversized footers, storage errors never block submission —
    the node-side render still fails fast on the same error).
    """
    if any(ch in data_source for ch in "*?["):
        return None
    refs = _collect_column_refs(template.get("fields") or {})
    if not refs:
        return None
    try:
        import pyarrow.parquet as pq

        tail, total = download_tail_from_pyromind(
            storage_path=data_source,
            storage_base_url=storage_base_url,
            headers=headers,
            timeout=timeout,
            tail_bytes=PREFLIGHT_TAIL_BYTES,
        )
        if len(tail) < 8 or tail[-4:] != b"PAR1":
            return None
        footer_len = struct.unpack("<I", tail[-8:-4])[0]
        if footer_len + 8 > len(tail):
            if footer_len + 8 > PREFLIGHT_MAX_FOOTER_BYTES:
                return None
            tail, total = download_tail_from_pyromind(
                storage_path=data_source,
                storage_base_url=storage_base_url,
                headers=headers,
                timeout=timeout,
                tail_bytes=footer_len + 8,
            )
        schema = pq.ParquetFile(_ParquetTailFile(total, tail)).schema_arrow
    except Exception:  # noqa: BLE001
        return None
    errors = _check_column_refs(schema, refs)
    if errors:
        return "Render template preflight failed: " + "; ".join(errors)
    return None


def build_render_command(
    *,
    output_dir: str,
    data_source: str,
    shard_size: int,
    limit: int | None,
) -> str:
    """Assemble the render command: install minimal deps, run render_manifest."""
    pod_output = _pod_path(str(PurePosixPath("/" + output_dir.lstrip("/"))))
    venv = "/tmp/edp-venv"
    data_pod = _pod_path(str(PurePosixPath("/" + data_source.lstrip("/"))))
    out_pod = pod_output

    steps = [
        f"python3 -m venv {venv}",
        (
            f"{venv}/bin/pip install --quiet --use-deprecated=legacy-resolver "
            f"-r {pod_output}/{RENDER_REQUIREMENTS}"
            " > /tmp/edp-render-install.log 2>&1"
            " || { echo 'render dependencies install failed; "
            "last install log lines:'; tail -n 100 "
            "/tmp/edp-render-install.log; exit 1; }"
        ),
        f"test -f {pod_output}/{RENDER_FILENAME}",
        f"test -f {pod_output}/{RENDER_TEMPLATE_FILENAME}",
        (
            f"{venv}/bin/python {pod_output}/{RENDER_FILENAME} "
            f"--template {pod_output}/{RENDER_TEMPLATE_FILENAME} "
            f"--data-source {shlex.quote(data_pod)} "
            f"--output-root {shlex.quote(out_pod)} "
            f"--shard-size {int(shard_size)}"
        )
        + (f" --limit {int(limit)}" if limit is not None else ""),
    ]
    return " && ".join(steps)


class EdpRenderExecutor(ToolExecutor[EdpRenderAction, EdpRenderObservation]):
    """Stage template/render script and submit the render task."""

    def __init__(
        self,
        *,
        env: str | None,
        cluster: str | None,
        headers: dict[str, str] | None,
        runtime_dir: str | None,
        output_root: str | None,
        storage_base_url: str | None,
        storage_headers: dict[str, str] | None,
        storage_secret_headers: dict[str, str] | None,
        timeout: int = 30,
    ) -> None:
        self._env = env
        self._cluster = cluster
        self._headers = headers or {}
        self._runtime_dir = runtime_dir
        self._output_root = output_root
        self._storage_base_url = storage_base_url
        self._storage_headers = storage_headers or {}
        self._storage_secret_headers = storage_secret_headers or {}
        self._timeout = timeout

    def _storage_headers_for(self, conversation: BaseConversation) -> dict[str, str]:
        headers = {**self._storage_headers}
        headers.update(_resolve_conversation_headers(conversation))
        headers.update(
            _resolve_secret_headers(conversation, self._storage_secret_headers)
        )
        return headers

    def _stage(
        self,
        conversation: BaseConversation,
        local_path: Path,
        output_dir: str,
        filename: str,
    ) -> None:
        local = local_path
        if local.name != filename:
            with tempfile.TemporaryDirectory(prefix="edp-render-") as tmp:
                target = Path(tmp) / filename
                target.write_bytes(local.read_bytes())
                local = target
        upload_local_file_to_pyromind(
            local_path=local,
            target_dir=output_dir,
            storage_base_url=self._storage_base_url or _default_storage_base_url(),
            headers=self._storage_headers_for(conversation),
            timeout=float(self._timeout),
        )

    def __call__(
        self,
        action: EdpRenderAction,
        conversation: BaseConversation | None = None,
    ) -> EdpRenderObservation:
        if conversation is None:
            return EdpRenderObservation.from_text(
                text="edp_render requires an active conversation.",
                status="Failed",
                is_error=True,
            )
        template_path = Path(action.template_path)
        if not template_path.is_file():
            # file_editor/apply_patch write agent-relative paths into the
            # conversation workspace, so retry against its root before
            # giving up (the executor cwd is the server, not the workspace).
            workspace_dir = Path(
                cast("Any", conversation).workspace.working_dir
            ).resolve()
            resolved = (
                template_path
                if template_path.is_absolute()
                else workspace_dir / template_path
            )
            if not resolved.is_file():
                return EdpRenderObservation.from_text(
                    text=(
                        f"render template not found: tried {template_path} and "
                        f"{resolved}. Pass the same path you gave "
                        "file_editor/apply_patch (workspace-relative) or an "
                        "absolute local path."
                    ),
                    status="Failed",
                    is_error=True,
                )
            template_path = resolved

        try:
            template = json.loads(template_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return EdpRenderObservation.from_text(
                text=f"render template is unreadable or invalid JSON: {exc}",
                status="Failed",
                is_error=True,
            )
        if not isinstance(template, dict):
            return EdpRenderObservation.from_text(
                text="render template must be a JSON object.",
                status="Failed",
                is_error=True,
            )

        preflight = _preflight_template(
            template,
            action.data_source,
            storage_base_url=self._storage_base_url or _default_storage_base_url(),
            headers=self._storage_headers_for(conversation),
            timeout=float(self._timeout),
        )
        if preflight is not None:
            return EdpRenderObservation.from_text(
                text=(
                    f"{preflight}. Fix the template and resubmit — nothing "
                    "was submitted to the platform."
                ),
                status="Failed",
                is_error=True,
            )

        run_id = uuid.uuid4()
        output_root = self._output_root or (
            f"/.pyromind-agent/{conversation.id}/edp_render/{run_id}"
        )
        # Nest under the requested root so concurrent runs never collide.
        root = str(PurePosixPath("/" + output_root.lstrip("/")) / str(run_id))
        output_dir = root

        try:
            state = cast("ConversationState", conversation.state)
            auth_token = state.secret_registry.get_secret_value(
                PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET
            )
            if not auth_token:
                raise ValueError(
                    "Platform auth_token missing from conversation secrets."
                )
            platform_env = resolve_platform_env(self._env)
            cluster = self._cluster or ""
            if not cluster:
                raise ValueError(
                    "cluster not wired into the edp_render tool; the platform "
                    "request should provide an x-cluster header."
                )

            if self._runtime_dir is None:
                raise ValueError("render runtime_dir not configured.")
            runtime = Path(self._runtime_dir)
            render_script = runtime / RENDER_FILENAME
            if not render_script.is_file():
                raise ValueError(f"render script not found: {render_script}")
            self._stage(conversation, render_script, output_dir, RENDER_FILENAME)
            self._stage(
                conversation, template_path, output_dir, RENDER_TEMPLATE_FILENAME
            )
            req_path = runtime / "pod_runtime" / RENDER_REQUIREMENTS
            if not req_path.is_file():
                raise ValueError(f"render requirements not found: {req_path}")
            self._stage(conversation, req_path, output_dir, RENDER_REQUIREMENTS)

            command = build_render_command(
                output_dir=output_dir,
                data_source=str(PurePosixPath("/" + action.data_source.lstrip("/"))),
                shard_size=action.shard_size,
                limit=action.limit,
            )
            workflow = {
                "id": str(run_id),
                "name": f"agent-edp-render-{str(run_id)[:8]}",
                "nodes": [
                    {
                        "id": "1",
                        "type": "default",
                        "position": {"x": 0, "y": 0},
                        "data": {
                            "display_name": "Manifest Render",
                            "nodeType": DATAFLOW_NODE_TYPE,
                            "config": {
                                "command": command,
                                "cpu": action.cpu,
                                "memory": action.memory,
                            },
                        },
                    }
                ],
                "edges": [],
                "viewport": {"x": 0, "y": 0, "zoom": 1},
                "timestamp": datetime.now(UTC).isoformat(),
            }

            client = create_workflow_api_client(
                env=platform_env,
                cluster=cluster,
                auth_token=auth_token,
                headers=self._headers,
                timeout=self._timeout,
            )
            response = submit_workflow_task(
                client=client,
                workflow=workflow,
                name=str(workflow["name"]),
                conversation_id=str(conversation.id),
            )
            task_id = response.task_id
        except Exception as exc:  # noqa: BLE001
            return EdpRenderObservation.from_text(
                text=f"Failed to submit render task: {exc}",
                status="Failed",
                is_error=True,
            )

        conversation.register_active_long_task(
            ActiveLongTask(
                task_id=task_id,
                kind="environment_processing_render",
                status=response.status,
            )
        )
        return EdpRenderObservation.from_text(
            text=(
                "Render task submitted. "
                f"task_id={task_id}, run_id={run_id}, output_dir={output_dir}. "
                "The platform runs it asynchronously and the terminal "
                "callback will resume this conversation automatically. NOW "
                "reply to the user with a short status (what was submitted, "
                "task_id, that you will report back on completion) and then "
                "END your turn — do not poll the output dir or sleep-wait. "
                "After the terminal callback, read "
                f"{output_dir}/shards.json for the shard list, then roll out "
                "edp_submit per shard."
            ),
            status=response.status,
            task_id=task_id,
            run_id=str(run_id),
            output_dir=output_dir,
        )


class EdpRenderTool(ToolDefinition[EdpRenderAction, EdpRenderObservation]):
    """Submit a template-driven manifest render to Pyromind Studio."""

    @classmethod
    def create(
        cls,
        conv_state: Any = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        env_value = params.pop("env", None)
        env = str(env_value) if env_value is not None else None
        cluster_value = params.pop("cluster", None)
        cluster = str(cluster_value) if cluster_value is not None else None
        params.pop("current_user", None)
        headers = params.pop("headers", None) or {}
        runtime_dir_value = params.pop("runtime_dir", None)
        runtime_dir = str(runtime_dir_value) if runtime_dir_value is not None else None
        output_root_value = params.pop("output_root", None)
        output_root = str(output_root_value) if output_root_value is not None else None
        storage_base_url_value = params.pop("storage_base_url", None)
        storage_base_url = (
            str(storage_base_url_value) if storage_base_url_value is not None else None
        )
        storage_headers = params.pop("storage_headers", None) or None
        storage_secret_headers = params.pop("storage_secret_headers", None) or None
        timeout = int(params.pop("timeout", 30))
        if params:
            names = ", ".join(sorted(params))
            raise ValueError(f"EdpRenderTool got unknown params: {names}")
        return [
            cls(
                description=(
                    "Submit a template-driven manifest render to the platform. "
                    "The node segments the Storage parquet, writes per-shard "
                    "manifests plus shards.json, so the agent never holds the "
                    "full dataset or manifest."
                ),
                action_type=EdpRenderAction,
                observation_type=EdpRenderObservation,
                executor=EdpRenderExecutor(
                    env=env,
                    cluster=cluster,
                    headers=headers,
                    runtime_dir=runtime_dir,
                    output_root=output_root,
                    storage_base_url=storage_base_url,
                    storage_headers=storage_headers,
                    storage_secret_headers=storage_secret_headers,
                    timeout=timeout,
                ),
                annotations=ToolAnnotations(
                    title="edp_render",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


register_tool(EdpRenderTool.name, EdpRenderTool)
