from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.ports.external_tasks import ExternalTaskRegistry
from pyromind_runtime.ports.product_store import ProductStore


class PipelineRuns:
    """Reserve immutable pipeline runs and coordinate production submissions."""

    def __init__(
        self,
        conversation_id: str,
        store: ProductStore,
        registry: ExternalTaskRegistry,
    ) -> None:
        self.conversation_id = conversation_id
        self._store = store
        self._registry = registry

    def resolve(self, run_id: str) -> dict[str, Any] | None:
        run = self._store.load_pipeline_runs().get(str(UUID(run_id)))
        if run is None:
            return None
        if run.conversation_id != self.conversation_id:
            raise ValueError("pipeline run belongs to another conversation")
        return run.model_dump(mode="json")

    def reserve(
        self, definition: dict[str, Any], resume_run_id: str | None
    ) -> dict[str, Any]:
        fingerprint = hashlib.sha256(
            json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        previous = self.resolve(resume_run_id) if resume_run_id else None
        if resume_run_id and previous is None:
            raise ValueError("unknown inference run in this conversation")
        expected_revision = None
        if previous is not None:
            prior = PipelineRun.model_validate(previous)
            if prior.fingerprint != fingerprint:
                raise ValueError("evaluation configuration changed; use a new full run")
            if prior.submission_status in {"preparing", "uncertain"}:
                raise ValueError(
                    "submission is in progress or uncertain; reconcile it first"
                )
            if prior.task_id:
                task = self._registry.resolve(self.conversation_id, prior.task_id)
                projected = next(
                    (
                        item
                        for item in self._store.load_snapshot().external_tasks
                        if item.task_id == prior.task_id
                    ),
                    None,
                )
                status = (
                    projected.status
                    if projected is not None
                    else (task.get("status") if task else None)
                )
                if status not in {
                    "succeeded",
                    "failed",
                    "terminated",
                    "stopped",
                }:
                    raise ValueError(
                        "pipeline task is still active; stop it before resume"
                    )
            expected_revision = prior.revision
            run = prior.model_copy(
                update={
                    "revision": prior.revision + 1,
                    "execution_revision": prior.execution_revision + 1,
                    "task_id": None,
                    "submission_status": "preparing",
                }
            )
        else:
            run_id = uuid4()
            run = PipelineRun(
                run_id=run_id,
                conversation_id=self.conversation_id,
                definition=definition,
                fingerprint=fingerprint,
                output_dir=(
                    f"/.pyromind-agent/{self.conversation_id}/data_preparation/{run_id}"
                ),
            )
        self._store.save_pipeline_run(run, expected_revision=expected_revision)
        return run.model_dump(mode="json")

    def finish_submission(
        self,
        run_id: str,
        revision: int,
        task_id: str | None,
        status: Literal["submitted", "failed", "uncertain"],
    ) -> None:
        value = self.resolve(run_id)
        if value is None:
            raise ValueError("unknown pipeline run")
        run = PipelineRun.model_validate(value)
        if run.revision != revision or run.submission_status != "preparing":
            raise ValueError("pipeline submission has already changed")
        run = run.model_copy(
            update={
                "revision": revision + 1,
                "task_id": task_id,
                "submission_status": status,
            }
        )
        self._store.save_pipeline_run(run, expected_revision=revision)
        if task_id:
            now = datetime.now(UTC).isoformat()
            payload: JsonObject = {
                "task_id": task_id,
                "run_id": run_id,
                "kind": "data_preparation",
                "status": "pending",
                "output_dir": run.output_dir,
                "submitted_at": now,
                "updated_at": now,
                "resume_pending": False,
            }
            self._registry.register(self.conversation_id, payload)

    def task_for(
        self, task_id: str | None, run_id: str | None, output_dir: str | None
    ) -> str | None:
        for run in self._store.load_pipeline_runs().values():
            if run.conversation_id != self.conversation_id:
                continue
            if (
                (task_id and run.task_id == task_id)
                or (run_id and str(run.run_id) == run_id)
                or (output_dir and run.output_dir == output_dir.rstrip("/"))
            ):
                return run.task_id
        return None
