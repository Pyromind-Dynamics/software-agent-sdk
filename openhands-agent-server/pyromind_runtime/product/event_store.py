from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from pyromind_runtime.contracts.base import ContractModel, utc_now
from pyromind_runtime.contracts.events import ProductEvent
from pyromind_runtime.product.models import (
    CommandReceipt,
    ConversationMetadata,
    ConversationSnapshot,
)


type SnapshotReducer = Callable[
    [ConversationSnapshot, ProductEvent], ConversationSnapshot
]

_CONVERSATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DIR_MODE = stat.S_IRWXU
_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR


class EventStoreError(RuntimeError):
    pass


class ConversationAlreadyExistsError(EventStoreError):
    pass


class ConversationNotFoundError(EventStoreError):
    pass


class EventStoreCorruptionError(EventStoreError):
    pass


class CommandReceiptConflictError(EventStoreError):
    pass


class FileConversationEventStore:
    def __init__(self, root: Path | str, conversation_id: str) -> None:
        if not _CONVERSATION_ID.fullmatch(conversation_id):
            raise ValueError("conversation_id contains unsafe path characters")
        self.root = Path(root)
        self.conversation_id = conversation_id
        self.directory = self.root / conversation_id
        self.metadata_path = self.directory / "metadata.json"
        self.events_path = self.directory / "events.jsonl"
        self.snapshot_path = self.directory / "snapshot.json"
        self.lock_path = self.directory / ".store.lock"
        self._thread_lock = threading.RLock()

    def create(
        self,
        metadata: ConversationMetadata,
        snapshot: ConversationSnapshot,
    ) -> None:
        self._validate_initial_state(metadata, snapshot)
        self.root.mkdir(mode=_DIR_MODE, parents=True, exist_ok=True)
        try:
            self.directory.mkdir(mode=_DIR_MODE)
        except FileExistsError as exc:
            raise ConversationAlreadyExistsError(self.conversation_id) from exc

        try:
            self._atomic_write_model(self.metadata_path, metadata)
            self._create_empty_event_log()
            self._atomic_write_model(self.snapshot_path, snapshot)
        except Exception:
            for path in (
                self.snapshot_path,
                self.events_path,
                self.metadata_path,
                self.lock_path,
            ):
                path.unlink(missing_ok=True)
            self.directory.rmdir()
            raise

    def load_metadata(self) -> ConversationMetadata:
        with self._lock():
            metadata = self._load_metadata_locked()
            events = self._load_events_locked(repair_tail=True)
            return self._reconcile_metadata_locked(metadata, events)

    def load_snapshot(self, reducer: SnapshotReducer) -> ConversationSnapshot:
        with self._lock():
            metadata = self._load_metadata_locked()
            events = self._load_events_locked(repair_tail=True)
            metadata = self._reconcile_metadata_locked(metadata, events)
            return self._recover_snapshot_locked(metadata, events, reducer)

    def replay(self, after_seq: int = 0) -> tuple[ProductEvent, ...]:
        if after_seq < 0:
            raise ValueError("after_seq must be non-negative")
        with self._lock():
            events = self._load_events_locked(repair_tail=True)
            return tuple(event for event in events if event.seq > after_seq)

    def append(
        self,
        event: ProductEvent,
        reducer: SnapshotReducer,
    ) -> tuple[ProductEvent, ConversationSnapshot]:
        if event.conversation_id != self.conversation_id:
            raise ValueError("event conversation_id does not match store")
        if event.seq != 0:
            raise ValueError("event seq must be zero before persistence")

        with self._lock():
            metadata = self._load_metadata_locked()
            events = self._load_events_locked(repair_tail=True)
            metadata = self._reconcile_metadata_locked(metadata, events)
            snapshot = self._recover_snapshot_locked(metadata, events, reducer)
            persisted = event.model_copy(update={"seq": metadata.last_sequence + 1})

            self._append_event_locked(persisted)
            updated_snapshot = reducer(snapshot, persisted)
            self._validate_reduced_snapshot(updated_snapshot, persisted)
            self._atomic_write_model(self.snapshot_path, updated_snapshot)

            updated_metadata = metadata.model_copy(
                update={
                    "last_sequence": persisted.seq,
                    "updated_at": utc_now(),
                }
            )
            self._atomic_write_model(self.metadata_path, updated_metadata)
            return persisted, updated_snapshot

    def claim_command(self, command_id: str) -> tuple[CommandReceipt, bool]:
        if not command_id:
            raise ValueError("command_id must not be empty")
        with self._lock():
            metadata = self._load_metadata_locked()
            existing = metadata.command_receipts.get(command_id)
            if existing is not None:
                return existing, False
            receipt = CommandReceipt(command_id=command_id, status="accepted")
            receipts = {**metadata.command_receipts, command_id: receipt}
            self._atomic_write_model(
                self.metadata_path,
                metadata.model_copy(
                    update={"command_receipts": receipts, "updated_at": utc_now()}
                ),
            )
            return receipt, True

    def complete_command(self, receipt: CommandReceipt) -> CommandReceipt:
        if receipt.status == "accepted":
            raise ValueError("completed command receipt must be terminal")
        with self._lock():
            metadata = self._load_metadata_locked()
            existing = metadata.command_receipts.get(receipt.command_id)
            if existing is None:
                raise CommandReceiptConflictError(
                    f"command was not claimed: {receipt.command_id}"
                )
            if existing.status != "accepted":
                if existing == receipt:
                    return existing
                raise CommandReceiptConflictError(
                    f"command already completed: {receipt.command_id}"
                )
            receipts = {**metadata.command_receipts, receipt.command_id: receipt}
            self._atomic_write_model(
                self.metadata_path,
                metadata.model_copy(
                    update={"command_receipts": receipts, "updated_at": utc_now()}
                ),
            )
            return receipt

    @contextmanager
    def _lock(self) -> Iterator[None]:
        if not self.directory.is_dir():
            raise ConversationNotFoundError(self.conversation_id)
        with self._thread_lock:
            fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, _FILE_MODE)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    def _load_metadata_locked(self) -> ConversationMetadata:
        return self._load_model(self.metadata_path, ConversationMetadata)

    def _load_events_locked(self, *, repair_tail: bool) -> list[ProductEvent]:
        if not self.events_path.exists():
            raise EventStoreCorruptionError("events.jsonl is missing")
        raw = self.events_path.read_bytes()
        lines = raw.splitlines(keepends=True)
        events: list[ProductEvent] = []
        valid_bytes = 0
        for index, line in enumerate(lines):
            if not line.strip():
                valid_bytes += len(line)
                continue
            try:
                event = ProductEvent.model_validate_json(line)
            except (ValidationError, ValueError) as exc:
                is_tail = all(not remaining.strip() for remaining in lines[index + 1 :])
                if not is_tail:
                    raise EventStoreCorruptionError(
                        f"invalid event at line {index + 1}"
                    ) from exc
                if repair_tail:
                    self._truncate_events_locked(valid_bytes)
                break
            expected_seq = len(events) + 1
            if event.seq != expected_seq:
                raise EventStoreCorruptionError(
                    f"expected event seq {expected_seq}, found {event.seq}"
                )
            if event.conversation_id != self.conversation_id:
                raise EventStoreCorruptionError(
                    f"event {event.seq} belongs to a different conversation"
                )
            events.append(event)
            valid_bytes += len(line)
        return events

    def _recover_snapshot_locked(
        self,
        metadata: ConversationMetadata,
        events: list[ProductEvent],
        reducer: SnapshotReducer,
    ) -> ConversationSnapshot:
        try:
            snapshot = self._load_model(self.snapshot_path, ConversationSnapshot)
        except EventStoreCorruptionError:
            snapshot = ConversationSnapshot(
                conversation_id=self.conversation_id,
                capabilities=metadata.capabilities,
            )
        if snapshot.conversation_id != self.conversation_id:
            raise EventStoreCorruptionError(
                "snapshot belongs to a different conversation"
            )
        if snapshot.through_seq > len(events):
            raise EventStoreCorruptionError("snapshot is ahead of the event log")

        updated = snapshot
        for event in events[snapshot.through_seq :]:
            updated = reducer(updated, event)
            self._validate_reduced_snapshot(updated, event)
        if updated != snapshot:
            self._atomic_write_model(self.snapshot_path, updated)
        return updated

    def _reconcile_metadata_locked(
        self,
        metadata: ConversationMetadata,
        events: list[ProductEvent],
    ) -> ConversationMetadata:
        if metadata.conversation_id != self.conversation_id:
            raise EventStoreCorruptionError(
                "metadata belongs to a different conversation"
            )
        actual_last_sequence = len(events)
        if metadata.last_sequence > actual_last_sequence:
            raise EventStoreCorruptionError("metadata is ahead of the event log")
        if metadata.last_sequence == actual_last_sequence:
            return metadata
        updated = metadata.model_copy(
            update={
                "last_sequence": actual_last_sequence,
                "updated_at": utc_now(),
            }
        )
        self._atomic_write_model(self.metadata_path, updated)
        return updated

    def _append_event_locked(self, event: ProductEvent) -> None:
        payload = event.model_dump_json().encode() + b"\n"
        fd = os.open(
            self.events_path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            _FILE_MODE,
        )
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(fd, payload[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)

    def _truncate_events_locked(self, length: int) -> None:
        fd = os.open(self.events_path, os.O_WRONLY)
        try:
            os.ftruncate(fd, length)
            os.fsync(fd)
        finally:
            os.close(fd)

    def _create_empty_event_log(self) -> None:
        fd = os.open(
            self.events_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            _FILE_MODE,
        )
        os.close(fd)

    @staticmethod
    def _load_model[ModelT: ContractModel](
        path: Path,
        model_type: type[ModelT],
    ) -> ModelT:
        try:
            return model_type.model_validate_json(path.read_bytes())
        except (OSError, ValidationError, ValueError) as exc:
            raise EventStoreCorruptionError(f"failed to load {path.name}") from exc

    @staticmethod
    def _atomic_write_model(path: Path, model: ContractModel) -> None:
        payload = json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ).encode()
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid4().hex[:8]}")
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            _FILE_MODE,
        )
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(fd, payload[offset:])
            os.fsync(fd)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            os.close(fd)
        try:
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _validate_initial_state(
        self,
        metadata: ConversationMetadata,
        snapshot: ConversationSnapshot,
    ) -> None:
        if metadata.conversation_id != self.conversation_id:
            raise ValueError("metadata conversation_id does not match store")
        if snapshot.conversation_id != self.conversation_id:
            raise ValueError("snapshot conversation_id does not match store")
        if metadata.last_sequence != 0 or snapshot.through_seq != 0:
            raise ValueError("new conversations must start at sequence zero")
        if metadata.capabilities != snapshot.capabilities:
            raise ValueError("metadata and snapshot capabilities do not match")

    def _validate_reduced_snapshot(
        self,
        snapshot: ConversationSnapshot,
        event: ProductEvent,
    ) -> None:
        if snapshot.conversation_id != self.conversation_id:
            raise EventStoreCorruptionError(
                "snapshot reducer changed the conversation_id"
            )
        if snapshot.through_seq != event.seq:
            raise EventStoreCorruptionError(
                "snapshot reducer did not advance through_seq to event seq"
            )
