from __future__ import annotations

import fcntl
import hashlib
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import Field, TypeAdapter, ValidationError, model_serializer

from pyromind_runtime.application.snapshot_projector import SnapshotProjector
from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.commands import (
    CommandReceipt,
    InputResponseCommand,
    ProductCommand,
)
from pyromind_runtime.domain.evaluation import EvaluationCommitState
from pyromind_runtime.domain.events import ProductEvent, WorkflowRunState
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.domain.user_input import UserInputResolution
from pyromind_runtime.domain.workflow_execution import WorkflowExecutionRequest


_DIR_MODE = 0o700
_FILE_MODE = 0o600

# Appending an event only needs the trailing window of the log: the tail doubles
# as the idempotency guard for redelivered records. Anything longer is read
# forwards instead, which stays reserved for recovery rather than the hot path.
# 1024 records is roughly six minutes of streaming traffic, far wider than the
# redelivery window a reconnect can produce, since reconnects resync via
# ``history.synced`` rather than replaying deltas through the append path.
_TAIL_WINDOW = 1024
_TAIL_CHUNK_BYTES = 65536


class ProductStoreError(RuntimeError):
    pass


class ProductStoreCorruptionError(ProductStoreError):
    pass


class CommandConflictError(ProductStoreError):
    pass


class _Metadata(ContractModel):
    conversation_id: str
    user_id: str
    x_cluster: str | None = None
    harness_id: str = Field(default="openhands", min_length=1)
    profile_id: str | None = None
    tool_allowlist: tuple[str, ...] | None = None
    command_allowlist: tuple[str, ...] | None = None
    capabilities: HarnessCapabilities
    last_sequence: int = Field(default=0, ge=0)


class CommandIntent(ContractModel):
    """Recoverable description of a claimed command.

    The ledger records fingerprints and receipts only; the intent carries the
    minimum needed to reconcile input-response receipts after a crash.
    """

    type: str = Field(min_length=1)
    request_id: str | None = None


class _StoredCommand(ContractModel):
    command_id: str
    fingerprint: str
    receipt: CommandReceipt
    intent: CommandIntent | None = None

    @model_serializer(mode="wrap")
    def _omit_empty_intent(self, handler):
        data = handler(self)
        if self.intent is None:
            data.pop("intent", None)
        return data


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
        self._lock_depth = threading.local()
        self._projector = SnapshotProjector()

    def create(
        self,
        snapshot: ConversationSnapshot,
        *,
        user_id: str,
        harness_id: str = "openhands",
        profile_id: str | None = None,
        tool_allowlist: tuple[str, ...] | None = None,
        command_allowlist: tuple[str, ...] | None = None,
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
            if metadata.user_id != user_id or metadata.profile_id != profile_id:
                raise PermissionError("conversation does not belong to current user")
            return
        metadata = _Metadata(
            conversation_id=snapshot.conversation_id,
            user_id=user_id,
            harness_id=harness_id,
            profile_id=profile_id,
            tool_allowlist=tool_allowlist,
            command_allowlist=command_allowlist,
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

    def authorize(self, user_id: str, profile_id: str | None = None) -> None:
        metadata = self._load_metadata()
        if metadata.user_id != user_id or metadata.profile_id != profile_id:
            raise PermissionError("conversation does not belong to current user")

    def permits_tool(self, name: str) -> bool:
        allowed = self._load_metadata().tool_allowlist
        return allowed is not None and name in allowed

    def access_policy(self) -> dict:
        metadata = self._load_metadata()
        return {
            "profile_id": metadata.profile_id,
            "tool_allowlist": metadata.tool_allowlist,
            "command_allowlist": metadata.command_allowlist,
        }

    def require_command(self, command_type: str) -> None:
        allowed = self._load_metadata().command_allowlist
        if allowed is not None and command_type not in allowed:
            raise PermissionError("command is not enabled for this conversation")

    def load_x_cluster(self) -> str | None:
        return self._load_metadata().x_cluster

    def save_x_cluster(self, x_cluster: str | None) -> None:
        if not x_cluster or not x_cluster.strip():
            return
        with self._lock():
            metadata = self._load_metadata()
            if metadata.x_cluster == x_cluster:
                return
            metadata = metadata.model_copy(update={"x_cluster": x_cluster})
            self._atomic_write(self.metadata_path, metadata.model_dump_json())

    def load_workflow_execution_requests(self) -> dict[str, WorkflowExecutionRequest]:
        with self._lock():
            return self._load_workflow_execution_requests()

    def _load_workflow_execution_requests(self) -> dict[str, WorkflowExecutionRequest]:
        path = self.directory / "workflow-execution-requests.json"
        if not path.exists():
            return {}
        return TypeAdapter(dict[str, WorkflowExecutionRequest]).validate_json(
            path.read_text(encoding="utf-8")
        )

    def save_workflow_execution_request(
        self, state: WorkflowExecutionRequest, *, expected_revision: int | None
    ) -> None:
        with self._lock():
            requests = self._load_workflow_execution_requests()
            previous = requests.get(state.request_id)
            if (previous.revision if previous else None) != expected_revision:
                raise ValueError("workflow execution request changed concurrently")
            if self._load_metadata().conversation_id != state.conversation_id:
                raise ValueError(
                    "workflow execution request belongs to another conversation"
                )
            requests[state.request_id] = state
            self._atomic_write(
                self.directory / "workflow-execution-requests.json",
                TypeAdapter(dict[str, WorkflowExecutionRequest])
                .dump_json(requests)
                .decode(),
            )

    def load_pipeline_runs(self) -> dict[str, PipelineRun]:
        with self._lock():
            return self._load_pipeline_runs()

    def _load_pipeline_runs(self) -> dict[str, PipelineRun]:
        path = self.directory / "pipeline-runs.json"
        if not path.exists():
            return {}
        return TypeAdapter(dict[str, PipelineRun]).validate_json(path.read_text())

    def save_pipeline_run(
        self, state: PipelineRun, *, expected_revision: int | None
    ) -> None:
        with self._lock():
            runs = self._load_pipeline_runs()
            previous = runs.get(str(state.run_id))
            revision = previous.revision if previous else None
            if revision != expected_revision:
                raise ValueError("pipeline run changed concurrently; reload its state")
            if self._load_metadata().conversation_id != state.conversation_id:
                raise ValueError("pipeline run belongs to another conversation")
            runs[str(state.run_id)] = state
            self._atomic_write(
                self.directory / "pipeline-runs.json",
                TypeAdapter(dict[str, PipelineRun]).dump_json(runs).decode(),
            )

    def load_workflow_runs(self) -> dict[str, WorkflowRunState]:
        with self._lock():
            return self._load_workflow_runs()

    def save_workflow_run(self, state: WorkflowRunState) -> None:
        with self._lock():
            runs = self._load_workflow_runs()
            runs[state.run_id] = state
            self._atomic_write(
                self.directory / "workflow-runs.json",
                TypeAdapter(dict[str, WorkflowRunState]).dump_json(runs).decode(),
            )

    def _load_workflow_runs(self) -> dict[str, WorkflowRunState]:
        path = self.directory / "workflow-runs.json"
        if not path.exists():
            return {}
        return TypeAdapter(dict[str, WorkflowRunState]).validate_json(
            path.read_text(encoding="utf-8")
        )

    def load_evaluation_commit(self, run_id: str) -> EvaluationCommitState | None:
        with self._lock():
            return self._load_evaluation_commits().get(run_id)

    def save_evaluation_commit(self, state: EvaluationCommitState) -> None:
        with self._lock():
            commits = self._load_evaluation_commits()
            commits[state.run_id] = state
            self._atomic_write(
                self.directory / "evaluation-commits.json",
                TypeAdapter(dict[str, EvaluationCommitState])
                .dump_json(commits)
                .decode(),
            )

    def _load_evaluation_commits(self) -> dict[str, EvaluationCommitState]:
        path = self.directory / "evaluation-commits.json"
        if not path.exists():
            return {}
        return TypeAdapter(dict[str, EvaluationCommitState]).validate_json(
            path.read_text(encoding="utf-8")
        )

    def harness_id(self) -> str:
        """Return persisted ownership; pre-version-two records are OpenHands."""
        if not self.metadata_path.is_file():
            return "openhands"
        return self._load_metadata().harness_id

    def load_snapshot(self) -> ConversationSnapshot:
        with self._lock():
            metadata = self._load_metadata()
            current = self._current_snapshot(metadata)
            if current is not None:
                return current
            events = self._load_events(repair_tail=True)
            metadata = self._reconcile(metadata, events)
            return self._recover_snapshot(metadata, events)

    def replay(self, after_seq: int = 0) -> tuple[ProductEvent, ...]:
        if after_seq < 0:
            raise ValueError("after_seq must be non-negative")
        with self._lock():
            marker = self._tail_marker()
            if marker is not None and marker.seq <= after_seq:
                # The log ends at or before the caller's cursor, so there is
                # nothing left to send. Proving that from the tail alone keeps
                # a reconnect from re-parsing a multi-megabyte log.
                return ()
            return tuple(
                event
                for event in self._load_events(repair_tail=True)
                if event.seq > after_seq
            )

    def _current_snapshot(self, metadata: _Metadata) -> ConversationSnapshot | None:
        """Return the on-disk snapshot when replaying the log could not change it.

        Reads only the tail marker and the snapshot file, so listing and opening
        conversations stay proportional to the snapshot rather than to the whole
        event log. This is the read-side counterpart of ``_append_within_window``
        and trusts the same invariant: once a record's seq matches the watermark
        and the snapshot already covers that watermark, no record outside the
        window can influence the projection.

        Returns ``None`` when the tail cannot be trusted or the snapshot is
        missing, stale, or half-written, handing the caller the full
        self-healing read instead of guessing.
        """
        marker = self._tail_marker()
        if marker is None or marker.seq != metadata.last_sequence:
            return None
        snapshot = self._read_snapshot(metadata)
        if (
            snapshot is None
            or snapshot.through_seq != metadata.last_sequence
            or snapshot.updated_at is None
        ):
            return None
        return snapshot

    def _tail_marker(self) -> ProductEvent | None:
        """Return the last persisted event by reading only the log's tail."""
        records = self._load_events_tail(1)
        return records[-1] if records else None

    def append(self, event: ProductEvent) -> tuple[ProductEvent, ConversationSnapshot]:
        if event.seq != 0:
            raise ValueError("event seq must be zero before persistence")
        with self._lock():
            metadata = self._load_metadata()
            if event.conversation_id != metadata.conversation_id:
                raise ValueError("event belongs to another conversation")
            window = self._load_events_tail(_TAIL_WINDOW)
            if window:
                snapshot = self._read_snapshot(metadata)
                if (
                    snapshot is not None
                    and window[-1].seq == metadata.last_sequence
                    and snapshot.through_seq == metadata.last_sequence
                ):
                    return self._append_within_window(metadata, snapshot, window, event)
            return self._append_from_full_log(metadata, event)

    def _append_within_window(
        self,
        metadata: _Metadata,
        snapshot: ConversationSnapshot,
        window: list[ProductEvent],
        event: ProductEvent,
    ) -> tuple[ProductEvent, ConversationSnapshot]:
        """Admit an event using only the trailing window of the log.

        Sound because the caller verified that the window ends exactly at
        ``metadata.last_sequence`` and that the snapshot already covers it, so
        no record outside the window can influence admission or projection.
        """
        duplicate = self._find_duplicate(window, event)
        if duplicate is not None:
            return duplicate, snapshot
        return self._commit(metadata, snapshot, event, metadata.last_sequence + 1)

    def _append_from_full_log(
        self, metadata: _Metadata, event: ProductEvent
    ) -> tuple[ProductEvent, ConversationSnapshot]:
        events = self._load_events(repair_tail=True)
        metadata = self._reconcile(metadata, events)
        snapshot = self._recover_snapshot(metadata, events)
        duplicate = self._find_duplicate(events, event)
        if duplicate is not None:
            return duplicate, snapshot
        return self._commit(metadata, snapshot, event, len(events) + 1)

    def _find_duplicate(
        self, events: list[ProductEvent], event: ProductEvent
    ) -> ProductEvent | None:
        source_identity = self._source_identity(event)
        for persisted in events:
            if persisted.event_id == event.event_id:
                return persisted
            if (
                source_identity is not None
                and self._source_identity(persisted) == source_identity
            ):
                return persisted
        return None

    def _commit(
        self,
        metadata: _Metadata,
        snapshot: ConversationSnapshot,
        event: ProductEvent,
        seq: int,
    ) -> tuple[ProductEvent, ConversationSnapshot]:
        persisted = event.model_copy(update={"seq": seq})
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
        intent = None
        if isinstance(command, InputResponseCommand):
            intent = CommandIntent(type="input_response", request_id=command.request_id)
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
                    intent=intent,
                ).model_dump_json(),
            )
            return receipt, True

    def load_input_command_receipts(
        self,
    ) -> dict[str, tuple[CommandReceipt, str | None]]:
        """command_id -> (receipt, request_id) for every input-response command."""
        with self._lock():
            commands = self._load_commands(repair_tail=True)
            return {
                record.command_id: (
                    record.receipt,
                    record.intent.request_id if record.intent is not None else None,
                )
                for record in commands.values()
                if record.intent is not None and record.intent.type == "input_response"
            }

    def resolve_pending_input(
        self, resolution: UserInputResolution
    ) -> tuple[ProductEvent | None, ConversationSnapshot]:
        """Append one input resolution if its request is still pending.

        Runs under the store lock so the pending check and the append are one
        step: exactly one resolution ever wins for a request. Returns the
        persisted event and updated snapshot, or ``(None, snapshot)`` when the
        request was already resolved.
        """
        with self._lock():
            metadata = self._load_metadata()
            snapshot = self.load_snapshot()
            pending = next(
                (
                    item
                    for item in snapshot.pending_inputs
                    if item.request_id == resolution.request_id
                ),
                None,
            )
            if pending is None:
                return None, snapshot
            payload = dict(resolution.model_dump(mode="json"))
            if pending.operation_id is not None:
                payload.setdefault("operation_id", pending.operation_id)
            event = ProductEvent(
                conversation_id=metadata.conversation_id,
                event_id=f"input:{resolution.request_id}:{resolution.outcome}",
                type="input.resolved",
                run_id=pending.run_id,
                payload=payload,
                # All resolutions for one request share this identity, so the
                # command path, the event pump and recovery dedupe to one record.
                source_event_id=f"input:{resolution.request_id}:resolution",
            )
            persisted, updated = self.append(event)
            return persisted, updated

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
        """Serialize store access per thread and across processes.

        ``flock`` guards one open file description, so a nested acquisition on
        the same thread must reuse the outer lock instead of opening a second
        descriptor — that would block against itself.
        """
        if not self.directory.is_dir():
            raise ProductStoreError("product store does not exist")
        with self._thread_lock:
            depth = getattr(self._lock_depth, "value", 0)
            fd = None
            if depth == 0:
                fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, _FILE_MODE)
                fcntl.flock(fd, fcntl.LOCK_EX)
            self._lock_depth.value = depth + 1
            try:
                yield
            finally:
                self._lock_depth.value = depth
                if fd is not None:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    finally:
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

    def _load_events_tail(self, limit: int) -> list[ProductEvent] | None:
        """Parse at most ``limit`` trailing records by reading backwards.

        Returns ``None`` when the tail cannot be trusted — an unparseable
        record, or an empty log — leaving the caller to take the full,
        self-healing read instead of guessing.
        """
        records: list[ProductEvent] = []
        for line in self._read_trailing_lines(self.events_path, limit):
            if not line.strip():
                continue
            try:
                records.append(ProductEvent.model_validate_json(line))
            except (ValidationError, ValueError):
                return None
        return records or None

    @staticmethod
    def _read_trailing_lines(path: Path, limit: int) -> list[bytes]:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            position = stream.tell()
            buffer = b""
            while position > 0 and buffer.count(b"\n") <= limit:
                chunk = min(_TAIL_CHUNK_BYTES, position)
                position -= chunk
                stream.seek(position)
                buffer = stream.read(chunk) + buffer
            lines = buffer.splitlines(keepends=True)
            if position > 0 and lines:
                # The oldest line sits on a chunk boundary and may be partial.
                lines = lines[1:]
        return lines[-limit:]

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
        # The log is already as long as the persisted watermark, so the per-event
        # sweep below has nothing left to learn. Verify the tail only instead: this
        # runs on every appended event, and a full sweep there is O(n) each time.
        if metadata.last_sequence == len(events):
            if events and (
                events[-1].seq != len(events)
                or events[-1].conversation_id != metadata.conversation_id
            ):
                raise ProductStoreCorruptionError("ProductEvent tail mismatch")
            return metadata
        for index, event in enumerate(events, start=1):
            if event.seq != index:
                raise ProductStoreCorruptionError(
                    f"expected ProductEvent seq {index}, found {event.seq}"
                )
            if event.conversation_id != metadata.conversation_id:
                raise ProductStoreCorruptionError("ProductEvent conversation mismatch")
        reconciled = metadata.model_copy(update={"last_sequence": len(events)})
        self._atomic_write(self.metadata_path, reconciled.model_dump_json())
        return reconciled

    def _read_snapshot(self, metadata: _Metadata) -> ConversationSnapshot | None:
        try:
            snapshot = ConversationSnapshot.model_validate_json(
                self.snapshot_path.read_bytes()
            )
        except (OSError, ValidationError, ValueError):
            return None
        if snapshot.conversation_id != metadata.conversation_id:
            raise ProductStoreCorruptionError("snapshot conversation mismatch")
        return snapshot

    @staticmethod
    def _empty_snapshot(metadata: _Metadata) -> ConversationSnapshot:
        return ConversationSnapshot(
            conversation_id=metadata.conversation_id,
            capabilities=metadata.capabilities,
        )

    def _recover_snapshot(
        self, metadata: _Metadata, events: list[ProductEvent]
    ) -> ConversationSnapshot:
        duplicate_sources = self._has_duplicate_sources(events)
        snapshot = self._read_snapshot(metadata)
        if snapshot is None:
            snapshot = self._empty_snapshot(metadata)
            dirty = True
        else:
            dirty = False
        if (
            snapshot.through_seq > len(events)
            or duplicate_sources
            or snapshot.updated_at is None
        ):
            snapshot = self._empty_snapshot(metadata)
            dirty = True
        # An empty tail means the on-disk snapshot already reflects the log. Skip the
        # source set and the 1 MB rewrite on that path: they only pay for themselves
        # when there is something left to fold in.
        if snapshot.through_seq < len(events):
            seen_sources = {
                identity
                for event in events[: snapshot.through_seq]
                if (identity := self._source_identity(event)) is not None
            }
            for event in events[snapshot.through_seq :]:
                identity = self._source_identity(event)
                if identity is not None and identity in seen_sources:
                    snapshot = snapshot.model_copy(update={"through_seq": event.seq})
                    dirty = True
                    continue
                snapshot = self._projector.reduce(snapshot, event)
                if identity is not None:
                    seen_sources.add(identity)
                dirty = True
        if snapshot.through_seq != metadata.last_sequence:
            raise ProductStoreCorruptionError("snapshot recovery did not reach tail")
        if dirty:
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
