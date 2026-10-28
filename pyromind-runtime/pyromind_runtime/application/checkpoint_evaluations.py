"""Persisted, harness-independent polling and checkpoint selection."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid5

from pyromind_runtime.application.checkpoint_selection import select_checkpoint
from pyromind_runtime.domain.base import utc_now
from pyromind_runtime.domain.checkpoint_evaluation import (
    FINISHED_CANDIDATES,
    TERMINAL_TASK_STATUSES,
    CheckpointCandidate,
    CheckpointEvaluationRequest,
    CheckpointEvaluationState,
    TrainingObservation,
)
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.training_evaluation import CheckpointSelectionPolicy
from pyromind_runtime.domain.versions import check_persistable
from pyromind_runtime.ports.checkpoint_evaluation import (
    CheckpointEvaluationPlatform,
    EvaluationCredentialsRequired,
)
from pyromind_runtime.ports.external_tasks import ExternalTaskRegistry
from pyromind_runtime.ports.product_store import ProductStore


KIND = "checkpoint_evaluation"


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def job_state(run: PipelineRun) -> CheckpointEvaluationState:
    if run.definition.get("kind") != KIND:
        raise ValueError("not a checkpoint evaluation job")
    return CheckpointEvaluationState.model_validate(run.result["checkpoint_evaluation"])


class CheckpointEvaluationCoordinator:
    def __init__(
        self,
        store: ProductStore,
        registry: ExternalTaskRegistry,
        platform: CheckpointEvaluationPlatform,
        *,
        poll_seconds: int = 60,
    ):
        self.store, self.registry, self.platform = store, registry, platform
        self.poll_seconds = poll_seconds

    async def register(
        self,
        run_id: UUID,
        conversation_id: str,
        request: CheckpointEvaluationRequest,
        context: RequestContext,
    ) -> PipelineRun:
        raw = request.model_dump(mode="json")
        check_persistable(raw)
        request_hash = fingerprint(raw)
        existing = self.store.load_pipeline_runs().get(str(run_id))
        if existing:
            if existing.definition.get("request_sha256") != request_hash:
                raise ValueError("request_id already belongs to a different selection")
            return existing
        output = f"/.pyromind-agent/{conversation_id}/data_preparation/{run_id}"
        frozen = await self.platform.freeze(request, output, context)
        definition = {
            **frozen,
            "kind": KIND,
            "request_sha256": request_hash,
            "selection": request.selection.model_dump(mode="json"),
        }
        check_persistable(definition)
        state = CheckpointEvaluationState(next_poll_at=utc_now())
        run = PipelineRun(
            run_id=run_id,
            conversation_id=conversation_id,
            definition=definition,
            fingerprint=fingerprint(definition),
            output_dir=output,
            submission_status="submitted",
            result={"checkpoint_evaluation": state.model_dump(mode="json")},
        )
        self.store.save_pipeline_run(run, expected_revision=None)
        return run

    def save(self, run: PipelineRun, state: CheckpointEvaluationState) -> PipelineRun:
        result = state.selection.model_dump(mode="json") if state.selection else {}
        result["checkpoint_evaluation"] = state.model_dump(mode="json")
        updated = run.model_copy(
            update={"result": result, "revision": run.revision + 1}
        )
        self.store.save_pipeline_run(updated, expected_revision=run.revision)
        return updated

    def cancel(self, run: PipelineRun) -> PipelineRun:
        state = job_state(run)
        if state.status in {"completed", "cancelled"}:
            return run
        state.cancel_requested = True
        state.status = "cancelling"
        state.next_poll_at = utc_now()
        return self.save(run, state)

    async def reconcile(
        self, run: PipelineRun, step: int, task_id: str, context: RequestContext
    ) -> PipelineRun:
        state = job_state(run)
        candidate = state.candidates.get(str(step))
        if candidate is None or candidate.status not in {"submitting", "uncertain"}:
            raise ValueError("candidate has no uncertain submission")
        child = self.store.load_pipeline_runs()[str(candidate.run_id)]
        owner = self.registry.owner(task_id)
        if owner is not None and owner != run.conversation_id:
            raise ValueError("task belongs to a different conversation")
        await self.platform.verify_submission(child, task_id, context)
        if child.task_id is not None and child.task_id != task_id:
            raise ValueError("candidate already has another platform task")
        candidate.task_id = task_id
        candidate.status = "running"
        child = child.model_copy(
            update={
                "task_id": task_id,
                "submission_status": "submitted",
                "revision": child.revision + 1,
            }
        )
        self.store.save_pipeline_run(child, expected_revision=child.revision - 1)
        state.error = None
        state.status = "cancelling" if state.cancel_requested else "watching"
        state.next_poll_at = utc_now()
        run = self.save(run, state)
        self._register_task(run, candidate)
        return run

    async def tick(
        self,
        run: PipelineRun,
        context: RequestContext | None,
        *,
        now: datetime | None = None,
    ) -> PipelineRun:
        now = now or utc_now()
        state = job_state(run)
        if state.status in {"completed", "cancelled"} or (
            state.next_poll_at > now
            and not (state.status == "waiting_credentials" and context is not None)
        ):
            return run
        state.next_poll_at = now + timedelta(seconds=self.poll_seconds)
        if context is None:
            state.status = "waiting_credentials"
            state.error = "Fresh platform credentials are required to continue."
            return self.save(run, state)
        state.error = None
        state.status = (
            "cancelling"
            if state.cancel_requested
            else ("draining" if state.terminal_since else "watching")
        )
        try:
            await self._collect(run, state, context, now)
            if not state.cancel_requested and not state.discovery_closed:
                observation = await self.platform.observe(run.definition, context)
                self._discover(state, observation, now)
            if state.cancel_requested:
                state.discovery_closed = True
                for candidate in state.candidates.values():
                    if candidate.status in {"discovered", "ready"}:
                        candidate.status = "cancelled"
                    elif candidate.status == "running" and candidate.task_id:
                        await self.platform.stop(candidate.task_id, context)
            self._select(run, state)
            run = self.save(run, state)
            busy = any(
                c.status in {"submitting", "running", "uncertain"}
                for c in state.candidates.values()
            )
            if not state.cancel_requested and not busy:
                ready = sorted(
                    (c for c in state.candidates.values() if c.status == "ready"),
                    key=lambda c: c.artifact.step,
                )
                if ready:
                    run = await self._submit(run, state, ready[0], context, now)
            if any(c.status == "uncertain" for c in state.candidates.values()):
                state.status = "needs_attention"
                state.error = (
                    "Submission outcome is uncertain; reconcile the "
                    "platform task before retrying."
                )
            elif state.discovery_closed and all(
                c.status in FINISHED_CANDIDATES for c in state.candidates.values()
            ):
                state.status = "cancelled" if state.cancel_requested else "completed"
            self._select(run, state)
            run = self.save(run, state)
            await self.platform.publish(run, context)
            return run
        except EvaluationCredentialsRequired:
            state.status = "waiting_credentials"
            state.error = (
                "Platform authentication expired; refresh credentials to continue."
            )
        except Exception as error:
            # Exception text can contain platform credentials or response bodies.
            state.error = f"Polling pending: {type(error).__name__}"
        # Reload: a durable write may have succeeded immediately before an error.
        run = self.store.load_pipeline_runs()[str(run.run_id)]
        if state.status in {"completed", "cancelled"}:
            state.status = "cancelling" if state.cancel_requested else "draining"
        return self.save(run, state)

    def _discover(
        self,
        state: CheckpointEvaluationState,
        observation: TrainingObservation,
        now: datetime,
    ) -> None:
        state.training_status = observation.status
        state.telemetry = observation.telemetry
        state.warnings = sorted(set(state.warnings + observation.warnings))
        if (
            observation.status in TERMINAL_TASK_STATUSES
            and state.terminal_since is None
        ):
            state.terminal_since = now
        # Rapid callbacks/manual refreshes must not count as independent polls.
        count_observation = (
            state.last_observed_at is None
            or (now - state.last_observed_at).total_seconds() >= self.poll_seconds
        )
        seen = set()
        for item in observation.checkpoints:
            key = str(item.artifact.step)
            seen.add(key)
            candidate = state.candidates.setdefault(
                key, CheckpointCandidate(artifact=item.artifact)
            )
            if candidate.artifact.path != item.artifact.path:
                raise ValueError(
                    "checkpoint path changed for an existing training step"
                )
            if candidate.status == "succeeded" and candidate.evaluation is not None:
                if (
                    item.fingerprint != candidate.fingerprint
                    or not item.artifact.complete
                ):
                    candidate.evaluation = candidate.evaluation.model_copy(
                        update={
                            "eligible": False,
                            "issues": ["checkpoint_changed_after_evaluation"],
                        }
                    )
                    candidate.error = "Checkpoint changed after evaluation"
            if candidate.status not in {"discovered", "ready"}:
                continue
            candidate.artifact = item.artifact
            if not item.artifact.complete or not item.fingerprint:
                candidate.status = "discovered"
                candidate.observations = 0
                candidate.stable_since = None
                candidate.fingerprint = None
                candidate.error = (
                    "; ".join(item.artifact.issues) or "File metadata unavailable"
                )
                continue
            if candidate.fingerprint != item.fingerprint:
                candidate.fingerprint = item.fingerprint
                candidate.observations = 1
                candidate.stable_since = now
                candidate.status = "discovered"
            elif count_observation:
                candidate.observations += 1
            elapsed = (now - (candidate.stable_since or now)).total_seconds()
            progressed = (
                observation.training_step is not None
                and observation.training_step > item.artifact.step
            )
            if candidate.observations >= 3 and (
                progressed
                or observation.status in TERMINAL_TASK_STATUSES
                or elapsed >= 300
            ):
                candidate.status = "ready"
                candidate.error = None
        for key, candidate in state.candidates.items():
            if key not in seen and candidate.evaluation is not None:
                candidate.evaluation = candidate.evaluation.model_copy(
                    update={
                        "eligible": False,
                        "issues": ["checkpoint_deleted_after_evaluation"],
                    }
                )
                candidate.error = "Checkpoint deleted after evaluation"
            if key not in seen and candidate.status in {"discovered", "ready"}:
                candidate.status = "discovered"
                candidate.observations = 0
                candidate.stable_since = None
                candidate.error = "Checkpoint disappeared from training output"
        if count_observation:
            state.last_observed_at = now
        if state.terminal_since:
            state.status = "draining"
            if (now - state.terminal_since).total_seconds() >= 600:
                state.discovery_closed = True
                for candidate in state.candidates.values():
                    if candidate.status == "discovered":
                        candidate.status = "unavailable"
                        candidate.error = (
                            candidate.error
                            or "Checkpoint did not stabilize before discovery closed"
                        )

    async def _collect(
        self,
        run: PipelineRun,
        state: CheckpointEvaluationState,
        context: RequestContext,
        now: datetime,
    ) -> None:
        children = self.store.load_pipeline_runs()
        for candidate in state.candidates.values():
            if candidate.status not in {"submitting", "uncertain", "running"}:
                continue
            child = children.get(str(candidate.run_id))
            if not candidate.task_id and child and child.task_id:
                candidate.task_id = child.task_id
                candidate.status = "running"
            if not candidate.task_id:
                candidate.status = "uncertain"
                continue
            self._register_task(run, candidate)
            execution = await self.platform.inspect(run, candidate, context)
            if execution.status not in TERMINAL_TASK_STATUSES:
                continue
            if execution.status == "succeeded" and execution.evaluation is None:
                candidate.result_wait_since = candidate.result_wait_since or now
                if (now - candidate.result_wait_since).total_seconds() < 600:
                    candidate.error = "Waiting for completed evaluation artifacts"
                    continue
            self.registry.update_status(
                run.conversation_id, candidate.task_id, execution.status
            )
            if execution.status == "succeeded" and execution.evaluation is not None:
                row = execution.evaluation
                if (
                    row.run_id != str(candidate.run_id)
                    or row.step != candidate.artifact.step
                    or row.checkpoint != candidate.artifact
                    or row.dataset_role != "validation"
                    or row.comparison_group != run.definition["comparison_group"]
                ):
                    candidate.status = "failed"
                    candidate.error = (
                        "Evaluation provenance does not match its candidate"
                    )
                    continue
                candidate.evaluation = row
                candidate.status = "succeeded" if row.eligible else "failed"
                candidate.error = "; ".join(row.issues) or None
            elif (
                execution.weights_not_ready
                and candidate.attempt < 2
                and not state.cancel_requested
                and not state.discovery_closed
            ):
                candidate.status = "discovered"
                candidate.observations = 0
                candidate.stable_since = None
                candidate.fingerprint = None
                candidate.task_id = None
                candidate.error = (
                    "Weights not ready; waiting for a fresh stability window"
                )
            else:
                candidate.status = "cancelled" if state.cancel_requested else "failed"
                candidate.error = (
                    execution.error or "Evaluation did not produce a complete result"
                )

    async def _submit(
        self,
        run: PipelineRun,
        state: CheckpointEvaluationState,
        candidate: CheckpointCandidate,
        context: RequestContext,
        now: datetime,
    ) -> PipelineRun:
        latest = await self.platform.observe(run.definition, context)
        current = next(
            (
                item
                for item in latest.checkpoints
                if item.artifact.path == candidate.artifact.path
            ),
            None,
        )
        if (
            not current
            or not current.artifact.complete
            or current.fingerprint != candidate.fingerprint
            or current.artifact != candidate.artifact
        ):
            candidate.status = "unavailable" if state.discovery_closed else "discovered"
            candidate.observations = 0
            candidate.stable_since = None
            candidate.error = "Checkpoint changed before submission"
            return self.save(run, state)
        candidate.task_id = None
        candidate.result_wait_since = None
        candidate.attempt += 1
        candidate.run_id = uuid5(
            run.run_id, f"{candidate.artifact.path}:{candidate.attempt}"
        )
        template = run.definition["template"]
        assert isinstance(template, dict)
        inference = template["inference"]
        evaluation = template["evaluation_config"]
        assert isinstance(inference, dict) and isinstance(evaluation, dict)
        source = {
            "type": "checkpoint",
            "kind": candidate.artifact.kind,
            "base_model": candidate.artifact.base_model,
        }
        definition = {
            **template,
            "inference": {**inference, "model_path": candidate.artifact.path},
            "evaluation_config": {**evaluation, "model_source": source},
            "dataset_role": "validation",
            "parent_run_id": str(run.run_id),
            "checkpoint": candidate.artifact.model_dump(mode="json"),
            "comparison_group": run.definition["comparison_group"],
        }
        child = PipelineRun(
            run_id=candidate.run_id,
            conversation_id=run.conversation_id,
            definition=definition,
            fingerprint=fingerprint(definition),
            output_dir=f"/.pyromind-agent/{run.conversation_id}/data_preparation/{candidate.run_id}",
        )
        existing_child = self.store.load_pipeline_runs().get(str(child.run_id))
        if existing_child is not None:
            if (
                existing_child.fingerprint != child.fingerprint
                or existing_child.task_id
            ):
                raise ValueError("candidate already has a different execution")
            child = existing_child
        else:
            self.store.save_pipeline_run(child, expected_revision=None)
        candidate.status = "submitting"
        run = self.save(run, state)
        try:
            workflow = await self.platform.prepare(run, child, context)
        except EvaluationCredentialsRequired:
            candidate.status = "ready"
            candidate.attempt -= 1
            candidate.error = "Waiting for credentials to prepare this candidate"
            raise
        except Exception as error:
            candidate.status = "failed"
            candidate.error = f"Preparation failed: {type(error).__name__}"
            return self.save(run, state)
        try:
            candidate.task_id = await self.platform.submit(child, workflow, context)
        except Exception:
            candidate.status = "uncertain"
            self.store.save_pipeline_run(
                child.model_copy(
                    update={
                        "submission_status": "uncertain",
                        "revision": child.revision + 1,
                    }
                ),
                expected_revision=child.revision,
            )
            return self.save(run, state)
        candidate.status = "running"
        child = child.model_copy(
            update={
                "task_id": candidate.task_id,
                "submission_status": "submitted",
                "revision": child.revision + 1,
            }
        )
        self.store.save_pipeline_run(child, expected_revision=child.revision - 1)
        run = self.save(run, state)
        self._register_task(run, candidate, now=now)
        return run

    def _register_task(
        self,
        run: PipelineRun,
        candidate: CheckpointCandidate,
        *,
        now: datetime | None = None,
    ) -> None:
        assert candidate.task_id
        if self.registry.resolve(run.conversation_id, candidate.task_id):
            return
        timestamp = (now or utc_now()).isoformat()
        self.registry.register(
            run.conversation_id,
            {
                "task_id": candidate.task_id,
                "run_id": str(candidate.run_id),
                "kind": "data_preparation",
                "status": "pending",
                "submitted_at": timestamp,
                "updated_at": timestamp,
                "resume_auto_run": False,
            },
        )

    @staticmethod
    def _select(run: PipelineRun, state: CheckpointEvaluationState) -> None:
        policy = CheckpointSelectionPolicy.model_validate(run.definition["selection"])
        policy = policy.model_copy(
            update={"comparison_group": run.definition["comparison_group"]}
        )
        selection = select_checkpoint(
            [
                c.evaluation
                for c in state.candidates.values()
                if c.evaluation is not None
            ],
            policy,
        )
        state.selection = selection.model_copy(update={"telemetry": state.telemetry})
