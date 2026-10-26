from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

from pyromind_runtime.domain.commands import CommandReceipt, InputResponseCommand
from pyromind_runtime.domain.errors import (
    ProductRuntimeError,
    UserInputValidationError,
)
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import (
    ConversationSnapshot,
    TimelineOperation,
)
from pyromind_runtime.domain.user_input import PendingUserInput, UserInputResolution
from pyromind_runtime.infrastructure.file_product_store import (
    CommandConflictError,
    FileProductStore,
)
from pyromind_runtime.ports.harness import (
    AttachRecovery,
    InterruptedInputRecord,
)


logger = logging.getLogger(__name__)

# One request has at most one winning resolution: interrupted recoveries
# reference the same identity so the command path, the event pump and this
# coordinator converge on a single durable record.
_RECOVERY_INTERRUPT_REASON = (
    "The execution that asked this question is no longer running."
)


def validate_answers(pending: PendingUserInput, command: InputResponseCommand) -> None:
    """Semantic answer validation; the request stays pending on failure."""
    expected = {question.id: question for question in pending.questions}
    provided = {answer.question_id for answer in command.answers}
    missing = sorted(expected.keys() - provided)
    extra = sorted(provided - expected.keys())
    if missing or extra:
        raise UserInputValidationError(
            "Answers must address exactly the questions asked"
            f" (missing: {missing}, unexpected: {extra})."
        )
    for answer in command.answers:
        question = expected[answer.question_id]
        option_ids = {option.id for option in question.options}
        if answer.option_id is not None and answer.option_id not in option_ids:
            raise UserInputValidationError(
                f"Option {answer.option_id!r} is not one of question "
                f"{answer.question_id!r}."
            )


class UserInputCoordinator:
    """Coordinates pending user-input requests for the conversation runtime.

    Owns the durable product state: resolutions, answer-command receipts, the
    automatic-turns suspension policy, and leftover coordination after an
    execution died. The runtime stays the only caller; the adapter keeps only
    execution-scoped wait state.
    """

    def __init__(self, publish) -> None:
        self._publish = publish
        self._locks: dict[str, asyncio.Lock] = {}
        # Conversations whose leftovers were coordinated in this process. New
        # input requests or a release invalidate the entry.
        self._coordinated: set[str] = set()
        # Per-conversation records the execution side needs at attach: ask_user
        # calls whose native tool result must be repaired.
        self._attach_records: dict[str, tuple[InterruptedInputRecord, ...]] = {}

    # -- live command path -------------------------------------------------

    @staticmethod
    def require_no_pending_input(snapshot: ConversationSnapshot) -> None:
        if snapshot.pending_inputs:
            raise ProductRuntimeError(
                "input_pending",
                "Answer or cancel the pending question before sending a message.",
            )

    async def submit_response(
        self,
        store: FileProductStore,
        conversation_id: str,
        command: InputResponseCommand,
        receipt: CommandReceipt,
    ) -> CommandReceipt:
        """Validate, persist the resolution, then the receipt.

        Fixed order (plan §6.2): validation, resolution with ``command_id``,
        completed receipt — delivery happens afterwards in the runtime and is
        never promised by the receipt.
        """
        try:
            resolution = UserInputResolution(
                request_id=command.request_id,
                outcome=command.status,
                answers=command.answers if command.status == "answered" else (),
                command_id=command.command_id,
            )
            async with self._lock(conversation_id):
                self._validate(store, command)
                if command.status == "cancelled":
                    # Persist suspension before resolution. Readers then see a pending
                    # question under suspension and reject messages, rather than briefly
                    # accepting a message before the suspension is written.
                    await self._suspend(store, conversation_id, "input_cancelled")
                persisted, _ = await asyncio.to_thread(
                    store.resolve_pending_input, resolution
                )
                if persisted is None:
                    raise ProductRuntimeError(
                        "input_not_pending",
                        "The input request is no longer pending.",
                    )
                self._publish(persisted)
                completed = receipt.model_copy(
                    update={
                        "status": "completed",
                        "response": {
                            "request_id": command.request_id,
                            "status": command.status,
                        },
                    }
                )
                store.complete_command(completed)
        except (ProductRuntimeError, UserInputValidationError) as error:
            self._fail_receipt(store, receipt, error)
            raise
        self._coordinated.discard(conversation_id)
        return completed

    async def resolve_event(self, store: FileProductStore, event: ProductEvent) -> None:
        """Serialize execution-side resolutions with answer commands."""
        resolution = UserInputResolution.model_validate(event.payload)
        async with self._lock(event.conversation_id):
            snapshot = store.load_snapshot()
            pending = next(
                (
                    p
                    for p in snapshot.pending_inputs
                    if p.request_id == resolution.request_id
                ),
                None,
            )
            if pending is None or (
                event.run_id is not None and event.run_id != pending.run_id
            ):
                return
            if resolution.outcome != "answered":
                # As in submit_response, persist suspension first to avoid an
                # interrupted-but-unsuspended window.
                await self._suspend(store, event.conversation_id, "input_interrupted")
            persisted, _ = await asyncio.to_thread(
                store.resolve_pending_input, resolution
            )
            if persisted is not None:
                self._publish(persisted)
        self.invalidate(event.conversation_id)

    async def interrupt_pending(
        self, store: FileProductStore, conversation_id: str, reason: str
    ) -> None:
        async with self._lock(conversation_id):
            # Shutdown always suspends execution. Persist it before resolution so
            # readers never see an interrupted-but-unsuspended window, including
            # failures where input.requested was not persisted or no input is pending.
            await self._suspend(store, conversation_id, reason)
            for pending in store.load_snapshot().pending_inputs:
                persisted, _ = await asyncio.to_thread(
                    store.resolve_pending_input,
                    UserInputResolution(
                        request_id=pending.request_id,
                        outcome="interrupted",
                        reason=reason,
                    ),
                )
                if persisted is not None:
                    self._publish(persisted)
        self.invalidate(conversation_id)

    @staticmethod
    def _validate(store: FileProductStore, command: InputResponseCommand) -> None:
        snapshot = store.load_snapshot()
        if not snapshot.capabilities.user_input_reply:
            raise ProductRuntimeError(
                "capability_not_supported",
                "This conversation cannot receive input responses.",
            )
        pending = next(
            (
                item
                for item in snapshot.pending_inputs
                if item.request_id == command.request_id
            ),
            None,
        )
        if pending is None:
            raise ProductRuntimeError(
                "input_not_pending",
                f"No pending input request {command.request_id!r}.",
            )
        if command.status == "answered":
            validate_answers(pending, command)

    @staticmethod
    def _fail_receipt(
        store: FileProductStore,
        receipt: CommandReceipt,
        error: ProductRuntimeError | UserInputValidationError,
    ) -> None:
        code = error.code if isinstance(error, ProductRuntimeError) else "invalid_input"
        try:
            store.complete_command(
                receipt.model_copy(
                    update={
                        "status": "failed",
                        "response": {"code": code, "message": str(error)},
                    }
                )
            )
        except CommandConflictError:
            logger.warning(
                "Could not record failed input receipt %s", receipt.command_id
            )

    # -- suspension policy --------------------------------------------------

    async def suspend(
        self, store: FileProductStore, conversation_id: str, reason: str
    ) -> None:
        async with self._lock(conversation_id):
            await self._suspend(store, conversation_id, reason)

    async def _suspend(
        self, store: FileProductStore, conversation_id: str, reason: str
    ) -> None:
        event = ProductEvent(
            event_id=f"execution-policy:suspend:{uuid4().hex}",
            conversation_id=conversation_id,
            type="execution.policy.updated",
            payload={"automatic_turns_suspended": True, "reason": reason},
        )
        persisted, _ = await asyncio.to_thread(store.append, event)
        self._publish(persisted)

    async def release_after_message(
        self,
        store: FileProductStore,
        conversation_id: str,
        *,
        after_seq: int | None = None,
    ) -> None:
        """Lift the suspension after one user message was accepted."""
        event = ProductEvent(
            event_id=f"execution-policy:resume:{uuid4().hex}",
            conversation_id=conversation_id,
            type="execution.policy.updated",
            payload={"automatic_turns_suspended": False},
        )
        async with self._lock(conversation_id):
            if after_seq is not None and any(
                (
                    item.type == "execution.policy.updated"
                    and item.payload.get("automatic_turns_suspended") is True
                )
                or (
                    item.type == "input.resolved"
                    and item.payload.get("outcome") in {"cancelled", "interrupted"}
                )
                for item in store.replay(after_seq)
            ):
                return
            persisted, _ = await asyncio.to_thread(store.append, event)
            self._publish(persisted)

    async def record_delivery_failure(
        self,
        store: FileProductStore,
        conversation_id: str,
        command: InputResponseCommand,
        error: Exception,
    ) -> None:
        """Keep the saved answer, record the failure, and stop auto-resume."""
        detail = (
            f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
        )
        notice = ProductEvent(
            event_id=f"input:{command.request_id}:delivery-failed",
            conversation_id=conversation_id,
            type="notice.raised",
            payload={
                "severity": "warning",
                "code": "input_delivery_failed",
                "message": (
                    "The answer was saved but could not be delivered to the "
                    f"agent; automatic execution stays suspended ({detail})."
                ),
            },
        )
        persisted, _ = await asyncio.to_thread(store.append, notice)
        self._publish(persisted)
        await self.suspend(store, conversation_id, "input_delivery_failed")

    # -- recovery ------------------------------------------------------------

    def invalidate(self, conversation_id: str) -> None:
        """Drop the coordinated memo: new input activity needs a fresh look."""
        self._coordinated.discard(conversation_id)

    def attach_recovery(self, store: FileProductStore) -> AttachRecovery:
        """Build the attach-time execution policy for a coordinated store."""
        snapshot = store.load_snapshot()
        return AttachRecovery(
            automatic_turns_suspended=snapshot.automatic_turns_suspended,
            interrupted_inputs=self._attach_records.get(snapshot.conversation_id, ()),
        )

    async def coordinate_recovery(
        self, store: FileProductStore, conversation_id: str
    ) -> None:
        async with self._lock(conversation_id):
            await self._coordinate_recovery(store, conversation_id)

    async def _coordinate_recovery(
        self, store: FileProductStore, conversation_id: str
    ) -> None:
        """Reconcile leftover inputs and receipts once no execution is live.

        Requests without a resolution become ``interrupted``; claimed receipts
        are completed or failed; answered-but-unconsumed requests enable the
        automatic-turns suspension. Idempotent: deterministic identities and
        receipt equality make repeated runs converge.
        """
        if conversation_id in self._coordinated:
            return
        snapshot = store.load_snapshot()
        receipts = store.load_input_command_receipts()
        if not snapshot.pending_inputs and not receipts:
            self._coordinated.add(conversation_id)
            self._attach_records.setdefault(conversation_id, ())
            return
        # Persist suspension before any resolution so recovery itself cannot expose
        # an interrupted-but-unsuspended window.
        _resolved, history = _scan_input_history(store)
        if snapshot.pending_inputs or _has_unconsumed_answers(history, snapshot):
            await self._suspend(store, conversation_id, "input_recovery")
        interrupted_requests: dict[str, str | None] = {}
        for pending in snapshot.pending_inputs:
            persisted, _ = await asyncio.to_thread(
                store.resolve_pending_input,
                UserInputResolution(
                    request_id=pending.request_id,
                    outcome="interrupted",
                    reason=_RECOVERY_INTERRUPT_REASON,
                ),
            )
            if persisted is None:
                continue
            self._publish(persisted)
            interrupted_requests[pending.request_id] = pending.operation_id
        resolved_commands, requests = _scan_input_history(store)
        self._reconcile_receipts(store, receipts, resolved_commands)
        records = [
            InterruptedInputRecord(
                request_id=request_id,
                operation_id=(
                    record.get("operation_id")
                    if isinstance(record.get("operation_id"), str)
                    else None
                ),
            )
            for request_id, record in requests.items()
            if _needs_native_repair(record, snapshot)
        ]
        records.extend(
            InterruptedInputRecord(request_id=request_id, operation_id=operation_id)
            for request_id, operation_id in interrupted_requests.items()
        )
        self._attach_records[conversation_id] = tuple(records)
        self._coordinated.add(conversation_id)

    @staticmethod
    def _reconcile_receipts(
        store: FileProductStore,
        receipts: dict[str, tuple[CommandReceipt, str | None]],
        resolved_commands: dict[str, dict[str, Any]],
    ) -> None:
        for command_id, (receipt, _request_id) in receipts.items():
            if receipt.status != "accepted":
                continue
            payload = resolved_commands.get(command_id)
            if payload is None:
                # Claimed, no resolution: the execution died before deciding.
                recovery = receipt.model_copy(
                    update={
                        "status": "failed",
                        "response": {
                            "code": "input_not_pending",
                            "message": (
                                "The execution was interrupted before this "
                                "command was processed."
                            ),
                        },
                    }
                )
            else:
                recovery = receipt.model_copy(
                    update={
                        "status": "completed",
                        "response": {
                            "request_id": payload.get("request_id"),
                            "status": payload.get("outcome"),
                        },
                    }
                )
            try:
                store.complete_command(recovery)
            except CommandConflictError:
                logger.warning(
                    "Input receipt reconciliation conflicted for %s", command_id
                )

    def _lock(self, conversation_id: str) -> asyncio.Lock:
        return self._locks.setdefault(conversation_id, asyncio.Lock())


def _scan_input_history(
    store: FileProductStore,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Map command_id -> resolution payload and request_id -> history."""
    resolved_commands: dict[str, dict[str, Any]] = {}
    requests: dict[str, dict[str, Any]] = {}
    for event in store.replay():
        if event.type == "input.requested":
            request_id = event.payload.get("request_id")
            operation_id = event.payload.get("operation_id")
            if isinstance(request_id, str) and isinstance(operation_id, str):
                requests[request_id] = {"operation_id": operation_id}
        elif event.type == "input.resolved":
            payload = event.payload
            request_id = payload.get("request_id")
            if isinstance(request_id, str):
                record = requests.setdefault(request_id, {})
                record["outcome"] = payload.get("outcome")
                record["resolved_seq"] = event.seq
            command_id = payload.get("command_id")
            if isinstance(command_id, str):
                resolved_commands[command_id] = dict(payload)
        elif (
            event.type == "execution.policy.updated"
            and event.payload.get("automatic_turns_suspended") is False
        ):
            for record in requests.values():
                if "outcome" in record:
                    record["resumed"] = True
    return resolved_commands, requests


def _consumed_operation_ids(snapshot: ConversationSnapshot) -> set[str]:
    return {
        item.item_id
        for item in snapshot.timeline
        if isinstance(item, TimelineOperation) and item.status == "completed"
    }


def _needs_native_repair(
    record: dict[str, Any], snapshot: ConversationSnapshot
) -> bool:
    """A resolved ask_user whose native tool result never landed.

    Interrupted and answered-but-unconsumed requests both need an interrupted
    tool result appended to the native session log; a consumed answer or a
    request without an operation link does not.
    """
    outcome = record.get("outcome")
    if outcome is None:
        return False
    operation_id = record.get("operation_id")
    if not isinstance(operation_id, str):
        return outcome == "interrupted"
    return operation_id not in _consumed_operation_ids(snapshot)


def _has_unconsumed_answers(
    requests: dict[str, dict[str, Any]], snapshot: ConversationSnapshot
) -> bool:
    """An answered request whose ask_user tool never completed.

    The timeline records the tool operation, so a completed operation with the
    request's ``operation_id`` proves the answer reached the agent; anything
    else is an execution that died before consuming it.
    """
    consumed = _consumed_operation_ids(snapshot)
    for request_id, record in requests.items():
        if record.get("outcome") != "answered" or record.get("resumed"):
            continue
        operation_id = record.get("operation_id")
        if not isinstance(operation_id, str) or operation_id not in consumed:
            logger.info(
                "Recovered answered input without a consumed tool result: %s",
                request_id,
            )
            return True
    return False
