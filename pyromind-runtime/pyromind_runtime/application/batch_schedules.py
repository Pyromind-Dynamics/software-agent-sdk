from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from pyromind_runtime.domain.batch_schedule import (
    BatchReceipt,
    BatchRunState,
    BatchScheduleConfig,
    BatchScheduleState,
)
from pyromind_runtime.ports.batch_schedule import (
    BatchScheduleBlocked,
    BatchScheduleRuntime,
    BatchScheduleStore,
)


class BatchSchedules:
    def __init__(
        self,
        store: BatchScheduleStore,
        runtime_factory: Callable[[str, str], BatchScheduleRuntime],
    ) -> None:
        self.store = store
        self.runtime_factory = runtime_factory

    def get(self, project_id: str, owner_user_id: str) -> BatchScheduleState:
        state = self.store.load(project_id)
        if state is not None:
            if state.owner_user_id != owner_user_id:
                raise PermissionError("Project batch schedule belongs to another user")
            return state
        return BatchScheduleState(project_id=project_id, owner_user_id=owner_user_id)

    def configure(
        self, project_id: str, owner_user_id: str, config: BatchScheduleConfig
    ) -> BatchScheduleState:
        with self.store.lease(project_id):
            current = self.get(project_id, owner_user_id)
            if any(item.status != "released" for item in current.receipts) and (
                config.source_manifest_uri != current.config.source_manifest_uri
                or config.validation_dataset_uri
                != current.config.validation_dataset_uri
            ):
                raise ValueError(
                    "Finish the current batch before changing data sources"
                )
            state = current.model_copy(
                update={"config": config, "last_checked_at": None, "last_error": None}
            )
            state = state.model_copy(
                update={"status": "idle" if config.enabled else "disabled"}
            )
            if config.enabled:
                try:
                    self._preflight(
                        self.runtime_factory(project_id, owner_user_id), config
                    )
                except Exception as exc:
                    state = self._blocked(state, exc)
            self.store.save(state)
            return state

    def tick(
        self, project_id: str, owner_user_id: str, *, force: bool = False
    ) -> BatchScheduleState:
        with self.store.lease(project_id):
            state = self.get(project_id, owner_user_id)
            now = datetime.now(UTC)
            if (
                not force
                and state.last_checked_at is not None
                and (now - state.last_checked_at).total_seconds()
                < state.config.poll_interval_seconds
            ):
                return state
            state = state.model_copy(
                update={"last_checked_at": now, "last_error": None}
            )
            unfinished = next(
                (item for item in state.receipts if item.status != "released"), None
            )
            if not state.config.enabled and unfinished is None:
                state = state.model_copy(update={"status": "disabled"})
                self.store.save(state)
                return state
            try:
                runtime = self.runtime_factory(project_id, owner_user_id)
                if unfinished is not None:
                    state = self._advance(state, unfinished, runtime)
                elif state.config.enabled:
                    self._preflight(runtime, state.config)
                    known = {
                        (item.tracer_resource_id, item.batch_id)
                        for item in state.receipts
                    }
                    pending = sorted(runtime.list_ready(), key=lambda item: item.index)
                    batch = next(
                        (
                            item
                            for item in pending
                            if (runtime.tracer_resource_id, item.batch_id) not in known
                        ),
                        None,
                    )
                    if batch is not None:
                        receipt = BatchReceipt(
                            batch_id=batch.batch_id,
                            batch_index=batch.index,
                            tracer_resource_id=runtime.tracer_resource_id,
                            release_id=runtime.release_id,
                            created_at=now,
                        )
                        state = state.model_copy(
                            update={"receipts": (*state.receipts, receipt)}
                        )
                        self.store.save(state)
                        state = self._advance(state, receipt, runtime)
                    else:
                        state = state.model_copy(update={"status": "idle"})
            except Exception as exc:
                state = self._blocked(self.get(project_id, owner_user_id), exc)
                state = state.model_copy(update={"last_checked_at": now})
            self.store.save(state)
            return state

    @staticmethod
    def _preflight(runtime: BatchScheduleRuntime, config: BatchScheduleConfig) -> None:
        if not config.source_manifest_uri or not config.validation_dataset_uri:
            raise BatchScheduleBlocked(
                "Configure confirmed ground truth and a fixed validation dataset first."
            )
        if config.source_manifest_uri == config.validation_dataset_uri:
            raise BatchScheduleBlocked(
                "Training ground truth and fixed validation data must be separate."
            )
        runtime.preflight(config)

    def _advance(
        self,
        state: BatchScheduleState,
        receipt: BatchReceipt,
        runtime: BatchScheduleRuntime,
    ) -> BatchScheduleState:
        if receipt.tracer_resource_id != runtime.tracer_resource_id:
            raise BatchScheduleBlocked(
                "Tracer changed while a batch still needs completion."
            )
        observed = runtime.find_run(receipt)
        if observed is None:
            if not state.config.enabled:
                raise BatchScheduleBlocked(
                    "An unsubmitted batch is retained while scheduling is disabled."
                )
            if receipt.release_id != runtime.release_id:
                raise BatchScheduleBlocked(
                    "Published graph changed before this batch was submitted."
                )
            self._preflight(runtime, state.config)
            if not receipt.claimed:
                if receipt.status in {"claiming", "claim_uncertain"} and not any(
                    item.batch_id == receipt.batch_id for item in runtime.list_ready()
                ):
                    receipt = receipt.model_copy(update={"status": "claim_uncertain"})
                    self.store.save(self._replace(state, receipt))
                    raise BatchScheduleBlocked(
                        "The previous Tracer claim result is uncertain. "
                        "The batch is retained; verify its ownership before retrying."
                    )
                receipt = receipt.model_copy(update={"status": "claiming"})
                state = self._replace(state, receipt)
                self.store.save(state)
                runtime.claim(receipt)
                receipt = receipt.model_copy(
                    update={"claimed": True, "status": "pending"}
                )
                state = self._replace(state, receipt)
                self.store.save(state)
            inputs = runtime.materialize(receipt, state.config)
            observed = runtime.start_run(receipt, inputs)
        state, receipt = self._record_run(state, receipt, observed)
        self.store.save(state)
        if observed.status in {"failed", "canceled"}:
            raise BatchScheduleBlocked(
                "The batch run did not succeed; its data is retained for repair."
            )
        if observed.status != "succeeded":
            return state
        if not observed.report_archived:
            raise BatchScheduleBlocked(
                "Waiting for the candidate model and evaluation report to be archived."
            )
        runtime.release(receipt)
        receipt = receipt.model_copy(
            update={"status": "released", "released_at": datetime.now(UTC)}
        )
        return self._replace(state, receipt).model_copy(
            update={"status": "idle" if state.config.enabled else "disabled"}
        )

    def _record_run(
        self, state: BatchScheduleState, receipt: BatchReceipt, run: BatchRunState
    ) -> tuple[BatchScheduleState, BatchReceipt]:
        receipt = receipt.model_copy(
            update={
                "run_id": run.run_id,
                "status": (
                    "releasing"
                    if run.status == "succeeded"
                    else "failed"
                    if run.status in {"failed", "canceled"}
                    else "running"
                ),
            }
        )
        state = self._replace(state, receipt).model_copy(update={"status": "running"})
        return state, receipt

    @staticmethod
    def _replace(
        state: BatchScheduleState, receipt: BatchReceipt
    ) -> BatchScheduleState:
        return state.model_copy(
            update={
                "receipts": tuple(
                    receipt
                    if (item.tracer_resource_id, item.batch_id)
                    == (receipt.tracer_resource_id, receipt.batch_id)
                    else item
                    for item in state.receipts
                )
            }
        )

    @staticmethod
    def _blocked(state: BatchScheduleState, error: Exception) -> BatchScheduleState:
        message = (
            str(error)
            if isinstance(error, BatchScheduleBlocked)
            else (
                "Batch scheduling is unavailable; "
                "check owner credentials and runtime resources."
            )
        )
        return state.model_copy(update={"status": "blocked", "last_error": message})
