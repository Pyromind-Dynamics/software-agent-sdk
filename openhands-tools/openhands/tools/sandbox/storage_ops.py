"""Sandbox storage operations: upload (sandbox -> storage)
and download (storage -> sandbox).

These tools extend the sandbox toolset with the ability to transfer files
between a running custom sandbox and Pyromind Storage, filling the gap
left by the sandbox_create/delete/read_file/terminal/write_file/delete_file
tools that only operate inside the sandbox or read from it.
"""

from __future__ import annotations

import base64
import shlex
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import httpx
from pydantic import Field

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.pyromind_archive.definition import (
    _normalize_storage_path,
)
from openhands.tools.pyromind_dataset.definition import (
    PYROMIND_AGENT_STORAGE_ROOT,
    _decode_json_response,
    _default_storage_base_url,
    _extract_api_data,
    _resolve_conversation_headers,
    _resolve_secret_headers,
    upload_local_file_to_pyromind,
)
from openhands.tools.sandbox.definition import (
    _SandboxExecutorMixin,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation


# ---------------------------------------------------------------------------
# sandbox_upload
# ---------------------------------------------------------------------------


class SandboxUploadAction(Action):
    """Upload a file from a custom sandbox to Pyromind Storage."""

    sandbox_id: str = Field(description="The id of the custom sandbox.")
    sandbox_path: str = Field(
        description="Absolute path of the file inside the container to upload."
    )
    storage_path: str | None = Field(
        default=None,
        description=(
            "Target storage directory. Defaults to "
            "'/.pyromind-agent/<conversation_id>/uploads'."
        ),
    )


class SandboxUploadObservation(Observation):
    """Result of an upload operation."""

    sandbox_id: str | None = Field(
        default=None, description="The sandbox the file was uploaded from."
    )
    storage_path: str | None = Field(
        default=None, description="Target storage path of the uploaded file."
    )


class SandboxUploadExecutor(
    _SandboxExecutorMixin,
    ToolExecutor[SandboxUploadAction, SandboxUploadObservation],
):
    """Upload a file from a custom sandbox to Pyromind Storage."""

    def __init__(
        self,
        *,
        storage_base_url: str | None = None,
        storage_headers: dict[str, str] | None = None,
        storage_secret_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._storage_base_url = (
            storage_base_url or _default_storage_base_url()
        ).rstrip("/")
        self._storage_headers = dict(storage_headers or {})
        self._storage_secret_headers = dict(storage_secret_headers or {})

    def _resolved_storage_headers(
        self,
        conversation: BaseConversation,
    ) -> dict[str, str]:
        headers = {"accept": "*/*", **self._storage_headers}
        headers.update(_resolve_conversation_headers(conversation))
        headers.update(
            _resolve_secret_headers(conversation, self._storage_secret_headers)
        )
        return headers

    def __call__(
        self,
        action: SandboxUploadAction,
        conversation: BaseConversation | None = None,
    ) -> SandboxUploadObservation:
        if conversation is None:
            return SandboxUploadObservation.from_text(
                text="Upload requires an active conversation.",
                is_error=True,
            )
        try:
            client = self._sandbox_client(conversation)
            storage_path = _normalize_storage_path(
                action.storage_path
                or f"{PYROMIND_AGENT_STORAGE_ROOT}/{conversation.id}/uploads",
                "storage_path",
            )

            # 1. Prefer the workspace mount when the sandbox exposes it.
            probe = client.exec_command(
                action.sandbox_id,
                "ls -d /target-workspace 2>/dev/null || echo __NO_MOUNT__",
                timeout=60,
            )
            if "__NO_MOUNT__" not in probe.output:
                from openhands.tools.pyromind_archive.definition import _pod_path

                pod_target = _pod_path(storage_path)
                copy_cmd = (
                    f"mkdir -p {shlex.quote(pod_target)} && "
                    f"cp -r {shlex.quote(action.sandbox_path)} "
                    f"{shlex.quote(pod_target)}/"
                )
                result = client.exec_command(
                    action.sandbox_id,
                    copy_cmd,
                    timeout=120,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Failed to copy into workspace mount: "
                        f"{result.stderr or result.output}"
                    )
                return SandboxUploadObservation.from_text(
                    text=(
                        f"Uploaded {action.sandbox_path} to storage path "
                        f"{storage_path} (workspace mount)."
                    ),
                    sandbox_id=action.sandbox_id,
                    storage_path=storage_path,
                )

            # 2. Fallback: stream the file back as base64.
            read_cmd = (
                f"base64 < {shlex.quote(action.sandbox_path)} 2>/dev/null "
                f"|| base64 {shlex.quote(action.sandbox_path)}"
            )
            result = client.exec_command(
                action.sandbox_id,
                read_cmd,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to read sandbox file: {result.stderr or result.output}"
                )
            data = base64.b64decode(result.output.strip() or "")
            headers = self._resolved_storage_headers(conversation)
            with tempfile.NamedTemporaryFile(
                prefix="pyromind_upload_", suffix=".bin", delete=False
            ) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)
            try:
                final_path = upload_local_file_to_pyromind(
                    local_path=tmp_path,
                    target_dir=storage_path,
                    storage_base_url=self._storage_base_url,
                    headers=headers,
                    timeout=30.0,
                )
            finally:
                tmp_path.unlink(missing_ok=True)
            return SandboxUploadObservation.from_text(
                text=(f"Uploaded {action.sandbox_path} to storage path {final_path}."),
                sandbox_id=action.sandbox_id,
                storage_path=final_path,
            )
        except Exception as exc:  # noqa: BLE001
            return SandboxUploadObservation.from_text(
                text=f"Upload failed: {exc}",
                is_error=True,
            )


# ---------------------------------------------------------------------------
# sandbox_download
# ---------------------------------------------------------------------------


class SandboxDownloadAction(Action):
    """Download a file from Pyromind Storage into a custom sandbox."""

    sandbox_id: str = Field(description="The id of the custom sandbox.")
    storage_path: str = Field(
        description="Storage path of the file to download (source)."
    )
    sandbox_path: str = Field(
        description="Absolute destination path inside the container."
    )


class SandboxDownloadObservation(Observation):
    """Result of a download operation."""

    sandbox_id: str | None = Field(
        default=None, description="The sandbox the file was downloaded to."
    )
    storage_path: str | None = Field(
        default=None, description="Source storage path of the downloaded file."
    )
    sandbox_path: str | None = Field(
        default=None, description="Destination path inside the container."
    )
    size: int | None = Field(default=None, description="Number of bytes downloaded.")


class SandboxDownloadExecutor(
    _SandboxExecutorMixin,
    ToolExecutor[SandboxDownloadAction, SandboxDownloadObservation],
):
    """Download a file from storage into a custom sandbox via pre-signed URL."""

    def __init__(
        self,
        *,
        storage_base_url: str | None = None,
        storage_headers: dict[str, str] | None = None,
        storage_secret_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._storage_base_url = (
            storage_base_url or _default_storage_base_url()
        ).rstrip("/")
        self._storage_headers = dict(storage_headers or {})
        self._storage_secret_headers = dict(storage_secret_headers or {})

    def _resolved_storage_headers(
        self,
        conversation: BaseConversation,
    ) -> dict[str, str]:
        headers = {"accept": "*/*", **self._storage_headers}
        headers.update(_resolve_conversation_headers(conversation))
        headers.update(
            _resolve_secret_headers(conversation, self._storage_secret_headers)
        )
        return headers

    def _storage_download_url(
        self,
        storage_path: str,
        headers: dict[str, str],
    ) -> str:
        """Pre-signed storage URL that the sandbox can curl inside itself."""

        path = _normalize_storage_path(storage_path, "storage_path")
        response = httpx.post(
            f"{self._storage_base_url}/get_url",
            headers=headers,
            json={"path": path},
            timeout=30,
        )
        payload = _decode_json_response(response, "Pyromind storage get_url API")
        if isinstance(payload, str):
            raise ValueError(payload)
        data = _extract_api_data("get_url", payload)
        if isinstance(data, str):
            raise ValueError(data)
        url = data.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError(
                "Pyromind storage get_url API response is missing url data."
            )
        return url

    def __call__(
        self,
        action: SandboxDownloadAction,
        conversation: BaseConversation | None = None,
    ) -> SandboxDownloadObservation:
        if conversation is None:
            return SandboxDownloadObservation.from_text(
                text="Download requires an active conversation.",
                is_error=True,
            )
        try:
            client = self._sandbox_client(conversation)
            url = self._storage_download_url(
                action.storage_path,
                self._resolved_storage_headers(conversation),
            )
            parent = str(Path(action.sandbox_path).parent or ".")
            command = (
                f"mkdir -p {shlex.quote(parent)} && "
                f"curl -fsSL {shlex.quote(url)} -o {shlex.quote(action.sandbox_path)} "
                f"&& wc -c < {shlex.quote(action.sandbox_path)}"
            )
            result = client.exec_command(
                action.sandbox_id,
                command,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to download into sandbox: {result.stderr or result.output}"
                )
            size_str = (
                result.output.strip().splitlines()[-1] if result.output.strip() else "0"
            )
            size = int(size_str) if size_str.isdigit() else 0
            return SandboxDownloadObservation.from_text(
                text=(
                    f"Downloaded {action.storage_path} to "
                    f"{action.sandbox_path} ({size} bytes)."
                ),
                sandbox_id=action.sandbox_id,
                storage_path=action.storage_path,
                sandbox_path=action.sandbox_path,
                size=size,
            )
        except Exception as exc:  # noqa: BLE001
            return SandboxDownloadObservation.from_text(
                text=f"Download failed: {exc}",
                is_error=True,
            )


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


class SandboxUploadTool(ToolDefinition[SandboxUploadAction, SandboxUploadObservation]):
    """Upload a file from a custom sandbox to Pyromind Storage."""

    @classmethod
    def create(
        cls,
        conv_state: Any = None,  # noqa: ARG003
        **params: Any,
    ) -> list[Self]:
        return [
            cls(
                description=(
                    "Upload a file from a custom (headless) Pyromind sandbox "
                    "to Pyromind Storage. Uses the workspace mount when "
                    "available, otherwise streams the file back as base64. "
                    "Output: the storage path of the uploaded file."
                ),
                action_type=SandboxUploadAction,
                observation_type=SandboxUploadObservation,
                executor=SandboxUploadExecutor(
                    cluster=params.get("cluster"),
                    env=params.get("env"),
                    current_user=params.get("current_user"),
                    headers=params.get("headers", {}),
                    storage_base_url=params.get("storage_base_url"),
                    storage_headers=params.get("storage_headers"),
                    storage_secret_headers=params.get("storage_secret_headers"),
                ),
                annotations=ToolAnnotations(
                    title="Sandbox upload file",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


class SandboxDownloadTool(
    ToolDefinition[SandboxDownloadAction, SandboxDownloadObservation]
):
    """Download a file from Pyromind Storage into a custom sandbox."""

    @classmethod
    def create(
        cls,
        conv_state: Any = None,  # noqa: ARG003
        **params: Any,
    ) -> list[Self]:
        return [
            cls(
                description=(
                    "Download a file from Pyromind Storage into a custom "
                    "(headless) sandbox. Fetches a pre-signed URL and curls "
                    "it inside the sandbox. Output: the destination path and "
                    "file size."
                ),
                action_type=SandboxDownloadAction,
                observation_type=SandboxDownloadObservation,
                executor=SandboxDownloadExecutor(
                    cluster=params.get("cluster"),
                    env=params.get("env"),
                    current_user=params.get("current_user"),
                    headers=params.get("headers", {}),
                    storage_base_url=params.get("storage_base_url"),
                    storage_headers=params.get("storage_headers"),
                    storage_secret_headers=params.get("storage_secret_headers"),
                ),
                annotations=ToolAnnotations(
                    title="Sandbox download file",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


register_tool(SandboxUploadTool.name, SandboxUploadTool)
register_tool(SandboxDownloadTool.name, SandboxDownloadTool)
