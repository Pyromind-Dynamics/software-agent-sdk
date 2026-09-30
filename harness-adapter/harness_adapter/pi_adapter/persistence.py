from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from harness_adapter.pi_adapter.tool_output import (
    build_business_tool_output_filename,
    build_business_tool_output_relative_path,
    business_tool_output_id,
    normalize_utf8_text,
)


@dataclass(frozen=True, slots=True)
class SavedToolOutput:
    sha256: str
    relative_path: str
    path: Path


class PiSessionFiles:
    def __init__(self, conversation_dir: Path) -> None:
        self.directory = conversation_dir / "pi"
        self.session_path = self.directory / "session.json"
        self.session_log_path = self.directory / "session.jsonl"
        self.inflight_path = self.directory / "inflight.json"
        self.business_state_path = self.directory / "business-state.json"
        self.checkpoint_index_path = self.directory / "fork-index.json"
        self.completions_path = self.directory / "run-completions.json"
        self.sandbox_path = self.directory / "sandbox.json"
        # Stages what the next sandbox creation has to replay: an optional fork
        # copy plus the workflow DSL. The file name predates workflow staging.
        self.sandbox_staging_path = self.directory / "sandbox-fork.json"
        self.terminal_output_directory = self.directory / "terminal-output"

    def initialize(self, session: dict[str, Any]) -> None:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _atomic_json(self.session_path, session)
        self.ensure_session_log()

    def ensure_session_log(self) -> None:
        """Create the explicit Pi session file before the Node manager opens it."""
        try:
            descriptor = os.open(
                self.session_log_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            return
        os.close(descriptor)

    def load_session(self) -> dict[str, Any]:
        return _load_object(self.session_path)

    def load_inflight(self) -> dict[str, Any] | None:
        if not self.inflight_path.is_file():
            return None
        return _load_object(self.inflight_path)

    def save_inflight(self, value: dict[str, Any]) -> None:
        _atomic_json(self.inflight_path, value)

    def clear_inflight(self) -> None:
        self.inflight_path.unlink(missing_ok=True)

    def load_pending_completions(self) -> dict[str, Any]:
        if not self.completions_path.is_file():
            return {}
        return _load_object(self.completions_path)

    def save_pending_completion(self, run_id: str, value: dict[str, Any]) -> None:
        pending = self.load_pending_completions()
        pending[run_id] = value
        _atomic_json(self.completions_path, pending)

    def clear_pending_completion(self, run_id: str) -> None:
        pending = self.load_pending_completions()
        if run_id in pending:
            del pending[run_id]
            _atomic_json(self.completions_path, pending)

    def load_business_state(self) -> dict[str, Any]:
        if not self.business_state_path.is_file():
            return {}
        return _load_object(self.business_state_path)

    def save_business_state(self, value: dict[str, Any]) -> None:
        _atomic_json(self.business_state_path, value)

    def load_checkpoint_index(self) -> dict[str, str]:
        if not self.checkpoint_index_path.is_file():
            return {}
        value = _load_object(self.checkpoint_index_path)
        return {
            str(key): str(item)
            for key, item in value.items()
            if isinstance(key, str) and isinstance(item, str)
        }

    def save_checkpoint_index(self, value: dict[str, str]) -> None:
        _atomic_json(self.checkpoint_index_path, value)

    def load_sandbox(self) -> dict[str, Any] | None:
        if not self.sandbox_path.is_file():
            return None
        return _load_object(self.sandbox_path)

    def save_sandbox(self, value: dict[str, Any]) -> None:
        _atomic_json(self.sandbox_path, value)

    def clear_sandbox(self) -> None:
        self.sandbox_path.unlink(missing_ok=True)

    def load_pending_sandbox_staging(self) -> dict[str, Any] | None:
        if not self.sandbox_staging_path.is_file():
            return None
        return _load_object(self.sandbox_staging_path)

    def save_pending_sandbox_staging(
        self, *, source_conversation_id: str | None, workflow_dsl: str
    ) -> None:
        _atomic_json(
            self.sandbox_staging_path,
            {
                "source_conversation_id": source_conversation_id,
                "workflow_dsl": workflow_dsl,
            },
        )

    def clear_pending_sandbox_staging(self) -> None:
        self.sandbox_staging_path.unlink(missing_ok=True)

    def save_business_tool_output(self, text: str) -> SavedToolOutput:
        normalized_text = normalize_utf8_text(text)
        encoded = normalized_text.encode("utf-8")
        output_id = business_tool_output_id(normalized_text)
        relative_path = build_business_tool_output_relative_path(output_id)
        output_path = self.terminal_output_directory / (
            build_business_tool_output_filename(output_id)
        )

        _ensure_private_directory(self.directory)
        _ensure_private_directory(self.terminal_output_directory)
        if _reuse_existing_output(output_path, output_id):
            return SavedToolOutput(output_id, relative_path, output_path)

        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            dir=self.terminal_output_directory,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            if output_path.is_symlink():
                raise RuntimeError(
                    "PI_WORKSPACE_INVALID: business tool output must not be a "
                    "symbolic link"
                )
            os.replace(temporary, output_path)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise
        return SavedToolOutput(output_id, relative_path, output_path)

    def load_business_tool_output(self, output_id: str) -> str:
        output_path = self.terminal_output_directory / (
            build_business_tool_output_filename(output_id)
        )
        with _open_verified_output(output_path, output_id) as stream:
            encoded = stream.read()
        try:
            return encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                "PI_WORKSPACE_INVALID: business tool output must be valid UTF-8"
            ) from exc


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _ensure_private_directory(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError(
            f"PI_WORKSPACE_INVALID: {path.name} must not be a symbolic link"
        )
    try:
        path.mkdir(mode=0o700, exist_ok=True)
    except FileExistsError as exc:
        raise RuntimeError(
            f"PI_WORKSPACE_INVALID: {path.name} must be a directory"
        ) from exc
    if not path.is_dir():
        raise RuntimeError(f"PI_WORKSPACE_INVALID: {path.name} must be a directory")
    path.chmod(0o700)


def _reuse_existing_output(path: Path, output_id: str) -> bool:
    try:
        stream = _open_verified_output(path, output_id)
    except FileNotFoundError:
        return False
    with stream:
        os.fchmod(stream.fileno(), 0o600)
    return True


def _open_verified_output(path: Path, output_id: str) -> BinaryIO:
    file_stat = os.lstat(path)
    if stat.S_ISLNK(file_stat.st_mode):
        raise RuntimeError(
            "PI_WORKSPACE_INVALID: business tool output must not be a symbolic link"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise RuntimeError(
                "PI_WORKSPACE_INVALID: business tool output must be a regular file"
            )
    except Exception:
        os.close(descriptor)
        raise
    stream = os.fdopen(descriptor, "rb")
    try:
        if _hash_stream(stream) != output_id:
            raise RuntimeError(
                "PI_WORKSPACE_INVALID: existing business tool output does not match "
                "its content hash"
            )
        stream.seek(0)
        return stream
    except Exception:
        stream.close()
        raise


def _hash_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()
