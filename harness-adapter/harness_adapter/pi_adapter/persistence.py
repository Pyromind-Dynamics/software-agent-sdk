from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class PiSessionFiles:
    def __init__(self, conversation_dir: Path) -> None:
        self.directory = conversation_dir / "pi"
        self.session_path = self.directory / "session.json"
        self.session_log_path = self.directory / "session.jsonl"
        self.inflight_path = self.directory / "inflight.json"
        self.business_state_path = self.directory / "business-state.json"
        self.checkpoint_index_path = self.directory / "fork-index.json"

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
