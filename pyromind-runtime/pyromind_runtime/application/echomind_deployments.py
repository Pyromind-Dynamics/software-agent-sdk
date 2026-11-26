from __future__ import annotations

from uuid import UUID, uuid4

from pyromind_runtime.domain.base import utc_now
from pyromind_runtime.domain.echomind_deployments import (
    TERMINAL_DEPLOYMENT_STATUSES,
    DeploymentConflictError,
    DeploymentRecord,
    DeploymentRequest,
    DeploymentStatus,
    DeploymentTarget,
)
from pyromind_runtime.ports.echomind_deployments import (
    DeploymentProvider,
    DeploymentStore,
)


class ManualDeploymentService:
    def __init__(self, store: DeploymentStore, provider: DeploymentProvider) -> None:
        self.store = store
        self.provider = provider

    def list(self, project_id: str) -> list[DeploymentRecord]:
        self.provider.authorize(project_id)
        return self.store.list(project_id)

    def start(self, project_id: str, request: DeploymentRequest) -> DeploymentRecord:
        self.provider.authorize(project_id)
        project_id = str(UUID(project_id))
        with self.store.lease(project_id):
            records = self.store.list(project_id)
            for existing in records:
                if existing.request.request_id == request.request_id:
                    if existing.request != request:
                        raise DeploymentConflictError(
                            "Request ID already belongs to another deployment"
                        )
                    return existing
            if any(r.status not in TERMINAL_DEPLOYMENT_STATUSES for r in records):
                raise DeploymentConflictError("Another deployment is still in progress")
            previous = self.provider.current()
            target = None
            candidate_id = request.candidate_id
            if request.rollback_deployment_id:
                selected = next(
                    (
                        r
                        for r in records
                        if r.deployment_id == request.rollback_deployment_id
                    ),
                    None,
                )
                if selected is None or selected.status != "succeeded":
                    raise ValueError("Only a successful deployment can be rolled back")
                if (
                    selected.target is None
                    or selected.target.resource_id != previous.resource_id
                    or selected.target.model_uri != previous.model_uri
                ):
                    raise DeploymentConflictError(
                        "Rollback must refer to the current deployment"
                    )
                target = selected.previous
                model_uri = target.model_uri
                candidate_id = next(
                    (
                        item.candidate_id
                        for item in reversed(records)
                        if item.status == "succeeded"
                        and item.target is not None
                        and item.target.resource_id == target.resource_id
                        and item.target.model_uri == target.model_uri
                        and item.candidate_id is not None
                    ),
                    None,
                )
            else:
                assert request.candidate_id is not None
                candidate = self.provider.candidate(request.candidate_id)
                model_uri = candidate.model_uri
            if model_uri == previous.model_uri:
                raise DeploymentConflictError("The selected model is already serving")
            record = DeploymentRecord(
                deployment_id=uuid4().hex,
                project_id=project_id,
                request=request,
                candidate_id=candidate_id,
                model_uri=model_uri,
                previous=previous,
                target=target,
                status="waiting_inference" if target else "submitting",
            )
            self.store.save(record)
            if target is not None:
                return record
            try:
                target = self.provider.find(self._name(record), model_uri)
                if target is None:
                    target = self.provider.create(self._name(record), model_uri)
            except ValueError:
                return self._save(
                    record, "failed", error="Platform rejected the deployment request."
                )
            except Exception:
                return self._save(
                    record,
                    "submission_unknown",
                    error="Submission uncertain; reconciling by deployment ID.",
                )
            return self._save(record, "waiting_inference", target=target)

    def advance(self, project_id: str, deployment_id: str) -> DeploymentRecord:
        self.provider.authorize(project_id)
        with self.store.lease(project_id):
            record = next(
                (
                    r
                    for r in self.store.list(project_id)
                    if r.deployment_id == deployment_id
                ),
                None,
            )
            if record is None:
                raise FileNotFoundError("Deployment not found")
            if record.status in TERMINAL_DEPLOYMENT_STATUSES:
                return self._cleanup(record)
            try:
                return self._advance(record)
            except Exception:
                if record.status in {"switching", "verifying"}:
                    return self._save(
                        record,
                        "rolling_back",
                        error="Switch failed; restoring previous serving model.",
                    )
                if record.status in {"rolling_back", "rollback_verifying"}:
                    return self._save(
                        record,
                        record.status,
                        error="Previous routing unverified; recovery will retry.",
                    )
                return self._save(
                    record,
                    record.status,
                    error="Platform unavailable; retrying without duplicate jobs.",
                )

    def _advance(self, record: DeploymentRecord) -> DeploymentRecord:
        if record.status in {"submitting", "submission_unknown"}:
            target = self.provider.find(self._name(record), record.model_uri)
            if target is None:
                return self._save(
                    record,
                    "submission_unknown",
                    error="No matching job yet; duplicate submission is disabled.",
                )
            return self._save(record, "waiting_inference", target=target)
        assert record.target is not None
        if record.status == "waiting_inference":
            if self._expired(record, 1800):
                return self._save(
                    record,
                    "failed",
                    error="Candidate timed out; previous model retained.",
                )
            target = self.provider.ready(record.target)
            if target is None:
                return record
            if not self.provider.routes_to(record.previous):
                return self._save(
                    record,
                    "failed",
                    error="Serving route changed externally; switch aborted.",
                )
            return self._save(record, "switching", target=target)
        if record.status == "switching":
            if not self.provider.routes_to(record.target):
                if not self.provider.routes_to(record.previous):
                    return self._save(
                        record, "failed", error="Serving route changed; switch aborted."
                    )
                self.provider.switch(record.target)
            return self._save(record, "verifying")
        if record.status == "verifying":
            if self.provider.route_ready(record.target):
                self.provider.commit(record.target)
                return self._save(record, "succeeded")
            if self._expired(record, 300):
                return self._save(
                    record,
                    "rolling_back",
                    error="New route did not become healthy; restoring previous model.",
                )
            return record
        if record.status == "rolling_back":
            if not self.provider.routes_to(record.previous):
                self.provider.switch(record.previous)
            return self._save(record, "rollback_verifying", error=record.error)
        if self.provider.route_ready(record.previous):
            self.provider.commit(record.previous)
            return self._save(
                record,
                "failed",
                error="Failed; previous serving model restored and verified.",
            )
        if self._expired(record, 300):
            return self._save(
                record,
                "rolling_back",
                error="Previous route is still unhealthy; continuing recovery.",
            )
        return record

    def _cleanup(self, record: DeploymentRecord) -> DeploymentRecord:
        if record.cleanup_status != "pending" or record.target is None:
            return record
        released = False
        try:
            released = self.provider.release_candidate(
                record.target, name=self._name(record), previous=record.previous
            )
        except Exception:
            pass
        updated = record.model_copy(
            update={
                "cleanup_status": "released" if released else "pending",
                "cleanup_error": None
                if released
                else "Candidate resource release is unverified; cleanup will retry.",
                "updated_at": utc_now(),
            }
        )
        self.store.save(updated)
        return updated

    def _save(
        self,
        record: DeploymentRecord,
        status: DeploymentStatus,
        *,
        target: DeploymentTarget | None = None,
        error: str | None = None,
    ) -> DeploymentRecord:
        now = utc_now()
        updated = record.model_copy(
            update={
                "status": status,
                "target": target or record.target,
                "error": error,
                "updated_at": now,
                "phase_started_at": now
                if status != record.status
                else record.phase_started_at,
                "cleanup_status": "pending"
                if status == "failed"
                and record.request.candidate_id is not None
                and record.target is not None
                else record.cleanup_status,
            }
        )
        self.store.save(updated)
        return updated

    @staticmethod
    def _name(record: DeploymentRecord) -> str:
        return f"em-model-{record.deployment_id}"

    @staticmethod
    def _expired(record: DeploymentRecord, seconds: int) -> bool:
        return (utc_now() - record.phase_started_at).total_seconds() > seconds
