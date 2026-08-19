from __future__ import annotations

import fcntl
import hashlib
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import Field, ValidationError

from pyromind_runtime.application.snapshot_projector import SnapshotProjector
from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.commands import CommandReceipt, ProductCommand
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import ConversationSnapshot


_DIR_MODE = 0o700
_FILE_MODE = 0o600


class ProductStoreError(RuntimeError):
    pass


class ProductStoreCorruptionError(ProductStoreError):
    pass


class CommandConflictError(ProductStoreError):
    pass


class _Metadata(ContractModel):
    conversation_id: str
    user_id: str
    harness_id: str = Field(default="openhands", min_length=1)
    capabilities: HarnessCapabilities
    last_sequence: int = Field(default=0, ge=0)


class _StoredCommand(ContractModel):
    command_id: str
    fingerprint: str
    receipt: CommandReceipt


class FileProductStore:
    """Conversation-owned ProductEvent store under ``<conversation>/product``."""

    def __init__(self, conversation_dir: Path | str) -> None:
        self.conversation_dir = Path(conversation_dir)
        self.directory = self.conversation_dir / "product"
        self.metadata_path = self.directory / "meta.json"
        self.events_path = self.directory / "events.jsonl"
        self.snapshot_path = self.directory / "snapshot.json"
        self.commands_path = self.directory / "commands.jsonl"
        self.lock_path = self.directory / ".lock"
        self._thread_lock = threading.RLock()
        self._projector = SnapshotProjector()

    def create(
        self,
        snapshot: ConversationSnapshot,
        *,
        user_id: str,
        harness_id: str = "openhands",
    ) -> None:
        if not self.conversation_dir.is_dir():
            raise ProductStoreError("conversation directory does not exist")
        try:
            self.directory.mkdir(mode=_DIR_MODE)
        except FileExistsError:
            metadata = self._load_metadata()
            if metadata.conversation_id != snapshot.conversation_id:
                raise ProductStoreCorruptionError(
                    "product store belongs to another conversation"
                )
            if metadata.user_id != user_id:
                raise PermissionError("conversation does not belong to current user")
            return
        metadata = _Metadata(
            conversation_id=snapshot.conversation_id,
            user_id=user_id,
            harness_id=harness_id,
            capabilities=snapshot.capabilities,
        )
        try:
            self._atomic_write(self.metadata_path, metadata.model_dump_json())
            self._atomic_write(self.snapshot_path, snapshot.model_dump_json())
            self._create_log(self.events_path)
            self._create_log(self.commands_path)
        except Exception:
            for path in (
                self.metadata_path,
                self.snapshot_path,
                self.events_path,
                self.commands_path,
                self.lock_path,
            ):
                path.unlink(missing_ok=True)
            self.directory.rmdir()
            raise

    def authorize(self, user_id: str) -> None:
        metadata = self._load_metadata()
        if metadata.user_id != user_id:
            raise PermissionError("conversation does not belong to current user")

    def harness_id(self) -> str:
        """Return persisted ownership; pre-version-two records are OpenHands."""
        if not self.metadata_path.is_file():
            return "openhands"
        return self._load_metadata().harness_id

    def load_snapshot(self) -> ConversationSnapshot:
        with self._lock():
            metadata = self._load_metadata()
            events = self._load_events(repair_tail=True)
            metadata = self._reconcile(metadata, events)
            return self._recover_snapshot(metadata, events)

    def replay(self, after_seq: int = 0) -> tuple[ProductEvent, ...]:
        if after_seq < 0:
            raise ValueError("after_seq must be non-negative")
        with self._lock():
            return tuple(
                event
                for event in self._load_events(repair_tail=True)
                if event.seq > after_seq
            )

    def append(self, event: ProductEvent) -> tuple[ProductEvent, ConversationSnapshot]:
        if event.seq != 0:
            raise ValueError("event seq must be zero before persistence")
        with self._lock():
            metadata = self._load_metadata()
            if event.conversation_id != metadata.conversation_id:
                raise ValueError("event belongs to another conversation")
            events = self._load_events(repair_tail=True)
            metadata = self._reconcile(metadata, events)
            snapshot = self._recover_snapshot(metadata, events)
            source_identity = self._source_identity(event)
            for persisted in events:
                if persisted.event_id == event.event_id:
                    return persisted, snapshot
                if (
                    source_identity is not None
                    and self._source_identity(persisted) == source_identity
                ):
                    return persisted, snapshot
            persisted = event.model_copy(update={"seq": len(events) + 1})
            updated = self._projector.reduce(snapshot, persisted)
            self._append_line(self.events_path, persisted.model_dump_json())
            self._atomic_write(self.snapshot_path, updated.model_dump_json())
            self._atomic_write(
                self.metadata_path,
                metadata.model_copy(
                    update={"last_sequence": persisted.seq}
                ).model_dump_json(),
            )
            return persisted, updated

    def claim_command(self, command: ProductCommand) -> tuple[CommandReceipt, bool]:
        fingerprint = self._fingerprint(command)
        with self._lock():
            commands = self._load_commands(repair_tail=True)
            existing = commands.get(command.command_id)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise CommandConflictError(
                        "command_id reused with different payload: "
                        f"{command.command_id}"
                    )
                return existing.receipt, False
            receipt = CommandReceipt(command_id=command.command_id, status="accepted")
            self._append_line(
                self.commands_path,
                _StoredCommand(
                    command_id=command.command_id,
                    fingerprint=fingerprint,
                    receipt=receipt,
                ).model_dump_json(),
            )
            return receipt, True

    def complete_command(self, receipt: CommandReceipt) -> CommandReceipt:
        if receipt.status == "accepted":
            raise ValueError("terminal receipt required")
        with self._lock():
            commands = self._load_commands(repair_tail=True)
            existing = commands.get(receipt.command_id)
            if existing is None:
                raise CommandConflictError("command was not claimed")
            if existing.receipt.status != "accepted":
                if existing.receipt == receipt:
                    return receipt
                raise CommandConflictError("command already completed")
            self._append_line(
                self.commands_path,
                existing.model_copy(update={"receipt": receipt}).model_dump_json(),
            )
            return receipt

    @contextmanager
    def _lock(self) -> Iterator[None]:
        if not self.directory.is_dir():
            raise ProductStoreError("product store does not exist")
        with self._thread_lock:
            fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, _FILE_MODE)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    def _load_metadata(self) -> _Metadata:
        try:
            return _Metadata.model_validate_json(self.metadata_path.read_bytes())
        except (OSError, ValidationError, ValueError) as exc:
            raise ProductStoreCorruptionError("product meta.json is invalid") from exc

    def _load_events(self, *, repair_tail: bool) -> list[ProductEvent]:
        return self._load_jsonl(
            self.events_path,
            ProductEvent.model_validate_json,
            repair_tail=repair_tail,
        )

    def _load_commands(self, *, repair_tail: bool) -> dict[str, _StoredCommand]:
        records = self._load_jsonl(
            self.commands_path,
            _StoredCommand.model_validate_json,
            repair_tail=repair_tail,
        )
        return {record.command_id: record for record in records}

    def _load_jsonl(self, path: Path, validator, *, repair_tail: bool):
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ProductStoreCorruptionError(f"{path.name} is missing") from exc
        lines = raw.splitlines(keepends=True)
        records = []
        valid_bytes = 0
        for index, line in enumerate(lines):
            if not line.strip():
                valid_bytes += len(line)
                continue
            try:
                records.append(validator(line))
            except (ValidationError, ValueError) as exc:
                if any(remaining.strip() for remaining in lines[index + 1 :]):
                    raise ProductStoreCorruptionError(
                        f"invalid record in {path.name} at line {index + 1}"
                    ) from exc
                if repair_tail:
                    with path.open("r+b") as stream:
                        stream.truncate(valid_bytes)
                break
            valid_bytes += len(line)
        return records

    def _reconcile(self, metadata: _Metadata, events: list[ProductEvent]) -> _Metadata:
        for index, event in enumerate(events, start=1):
            if event.seq != index:
                raise ProductStoreCorruptionError(
                    f"expected ProductEvent seq {index}, found {event.seq}"
                )
            if event.conversation_id != metadata.conversation_id:
                raise ProductStoreCorruptionError("ProductEvent conversation mismatch")
        if metadata.last_sequence == len(events):
            return metadata
        reconciled = metadata.model_copy(update={"last_sequence": len(events)})
        self._atomic_write(self.metadata_path, reconciled.model_dump_json())
        return reconciled

    def _recover_snapshot(
        self, metadata: _Metadata, events: list[ProductEvent]
    ) -> ConversationSnapshot:
        duplicate_sources = self._has_duplicate_sources(events)
        try:
            snapshot = ConversationSnapshot.model_validate_json(
                self.snapshot_path.read_bytes()
            )
        except (OSError, ValidationError, ValueError):
            snapshot = ConversationSnapshot(
                conversation_id=metadata.conversation_id,
                capabilities=metadata.capabilities,
            )
        if snapshot.conversation_id != metadata.conversation_id:
            raise ProductStoreCorruptionError("snapshot conversation mismatch")
        if snapshot.through_seq > len(events) or duplicate_sources:
            snapshot = ConversationSnapshot(
                conversation_id=metadata.conversation_id,
                capabilities=metadata.capabilities,
            )
        seen_sources = {
            identity
            for event in events[: snapshot.through_seq]
            if (identity := self._source_identity(event)) is not None
        }
        for event in events[snapshot.through_seq :]:
            identity = self._source_identity(event)
            if identity is not None and identity in seen_sources:
                snapshot = snapshot.model_copy(update={"through_seq": event.seq})
                continue
            snapshot = self._projector.reduce(snapshot, event)
            if identity is not None:
                seen_sources.add(identity)
        if snapshot.through_seq != metadata.last_sequence:
            raise ProductStoreCorruptionError("snapshot recovery did not reach tail")
        self._atomic_write(self.snapshot_path, snapshot.model_dump_json())
        return snapshot

    @classmethod
    def _has_duplicate_sources(cls, events: list[ProductEvent]) -> bool:
        seen: set[tuple[str, str]] = set()
        for event in events:
            identity = cls._source_identity(event)
            if identity is None:
                continue
            if identity in seen:
                return True
            seen.add(identity)
        return False

    @staticmethod
    def _source_identity(event: ProductEvent) -> tuple[str, str] | None:
        if event.source_event_id is None:
            return None
        return event.type, event.source_event_id

    @staticmethod
    def _fingerprint(command: ProductCommand) -> str:
        payload = command.model_dump_json(exclude={"command_id"})
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _create_log(path: Path) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _FILE_MODE)
        os.close(fd)

    @staticmethod
    def _append_line(path: Path, payload: str) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _atomic_write(path: Path, payload: str) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _FILE_MODE)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
