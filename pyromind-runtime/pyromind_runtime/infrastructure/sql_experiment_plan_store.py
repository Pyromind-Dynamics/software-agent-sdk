from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Column,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    and_,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.engine import Connection, Engine, RowMapping

from pyromind_runtime.domain.experiment_plans import (
    ExperimentContext,
    ExperimentContextView,
    ExperimentPlanView,
    PlanConflictError,
    PlanOwner,
    PlanProposal,
    PlanRevision,
    PlanRevisionSummary,
    UpdateExperimentContext,
)
from pyromind_runtime.infrastructure.sql_database import (
    configure_engine,
    connection_transaction,
)


metadata = MetaData()
projects = Table(
    "experiment_projects",
    metadata,
    Column("project_id", String(64), primary_key=True),
    Column("owner_key", String(64), nullable=False, unique=True),
    Column("user_id", String(255), nullable=False),
    Column("conversation_id", String(128), nullable=False),
    Column("profile_id", String(128)),
    Column("current_revision", Integer, nullable=False),
    Column("context", JSON),
    Column("context_sequence", Integer, nullable=False, server_default="0"),
)
documents = Table(
    "experiment_plan_documents",
    metadata,
    Column("kind", String(16), primary_key=True),
    Column("document_id", String(64), primary_key=True),
    Column("owner_key", String(64), nullable=False, index=True),
    Column("project_id", String(64), ForeignKey(projects.c.project_id)),
    Column("revision", Integer),
    Column("status", String(16)),
    Column("base_revision", Integer),
    Column("attributes", JSON, nullable=False),
    Column("confirmation_command_id", String(128)),
    Column("confirmed_revision", Integer),
    UniqueConstraint(
        "owner_key", "confirmation_command_id", name="uq_plan_document_confirmation"
    ),
    UniqueConstraint("project_id", "revision", name="uq_plan_document_revision"),
)
# Read projections retain the existing domain mapping, not physical legacy tables.
proposals = (
    select(
        documents.c.document_id.label("proposal_id"),
        documents.c.owner_key,
        documents.c.status,
        documents.c.base_revision,
        documents.c.attributes.label("payload"),
        documents.c.confirmation_command_id,
        documents.c.confirmed_revision,
    )
    .where(documents.c.kind == "proposal")
    .subquery("plan_proposals")
)
revisions = (
    select(
        documents.c.project_id,
        documents.c.revision,
        documents.c.attributes.label("payload"),
    )
    .where(documents.c.kind == "revision")
    .subquery("plan_revisions")
)


def _owner_key(owner: PlanOwner) -> str:
    return hashlib.sha256(owner.model_dump_json().encode()).hexdigest()


class SQLExperimentPlanStore:
    """Transactional plan storage shared by MariaDB and local SQLite.

    A connection-scoped advisory lock also serializes the first proposal and
    first confirmation, before any project row exists to SELECT FOR UPDATE.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = configure_engine(engine)
        self._local_lock = threading.RLock()

    def close(self) -> None:
        self.engine.dispose()

    def context(
        self, owner: PlanOwner, *, create: bool = False
    ) -> ExperimentContextView:
        with self._transaction(owner) as connection:
            key = _owner_key(owner)
            row = self._project(connection, key)
            if row is None:
                if not create:
                    return ExperimentContextView()
                identity = uuid4().hex
                connection.execute(
                    projects.insert().values(
                        project_id=identity,
                        owner_key=key,
                        user_id=owner.user_id,
                        conversation_id=owner.conversation_id,
                        profile_id=owner.profile_id,
                        current_revision=0,
                        context_sequence=0,
                        context={},
                    )
                )
                return ExperimentContextView(project_id=identity)
            body = row["context"]
            if body is None and row["current_revision"]:
                saved = connection.scalar(
                    select(revisions.c.payload).where(
                        revisions.c.project_id == row["project_id"],
                        revisions.c.revision == row["current_revision"],
                    )
                )["plan"]
                body = {
                    key: saved[key]
                    for key in ("goal", "resources", "stages", "edges")
                    if key in saved
                }
            return ExperimentContextView(
                project_id=row["project_id"],
                sequence=row["context_sequence"],
                context=ExperimentContext.model_validate(body or {}),
            )

    def update_context(
        self, owner: PlanOwner, request: UpdateExperimentContext
    ) -> ExperimentContextView:
        import json

        from pyromind_runtime.domain.persisted_payload import validate_persisted_payload

        payload = request.context.model_dump(mode="json")
        validate_persisted_payload(payload)
        if len(json.dumps(payload).encode()) > 262144:
            raise ValueError("experiment context exceeds 256 KiB")
        current = self.context(owner, create=True)
        with self._transaction(owner) as connection:
            result = connection.execute(
                update(projects)
                .where(
                    projects.c.project_id == current.project_id,
                    projects.c.owner_key == _owner_key(owner),
                    projects.c.context_sequence == request.expected_sequence,
                )
                .values(context=payload, context_sequence=request.expected_sequence + 1)
            )
            if result.rowcount != 1:
                raise PlanConflictError("context changed; reread and merge")
        return self.context(owner)

    @contextmanager
    def _transaction(self, owner: PlanOwner) -> Iterator[Connection]:
        key = _owner_key(owner)
        with self._local_lock, self.engine.connect() as connection:
            named_lock = connection.dialect.name in {"mysql", "mariadb"}
            if named_lock:
                acquired = connection.scalar(
                    text("SELECT GET_LOCK(:name, 10)"), {"name": key}
                )
                connection.commit()
                if acquired != 1:
                    raise PlanConflictError("plan is busy; retry shortly")
            try:
                with connection_transaction(connection):
                    yield connection
            finally:
                if named_lock:
                    connection.execute(
                        text("SELECT RELEASE_LOCK(:name)"), {"name": key}
                    )
                    connection.commit()

    @staticmethod
    def _project(connection: Connection, key: str) -> RowMapping | None:
        return (
            connection.execute(
                select(projects).where(projects.c.owner_key == key).with_for_update()
            )
            .mappings()
            .one_or_none()
        )

    @staticmethod
    def _pending(connection: Connection, key: str) -> RowMapping | None:
        return (
            connection.execute(
                select(proposals).where(
                    proposals.c.owner_key == key, proposals.c.status == "pending"
                )
            )
            .mappings()
            .one_or_none()
        )

    @staticmethod
    def _proposal(row: RowMapping) -> PlanProposal:
        return PlanProposal.model_validate({**row["payload"], "status": row["status"]})

    def view(self, owner: PlanOwner) -> ExperimentPlanView:
        key = _owner_key(owner)
        with self._transaction(owner) as connection:
            project = self._project(connection, key)
            pending = self._pending(connection, key)
            saved: list[PlanRevision] = []
            if project:
                saved = [
                    PlanRevision.model_validate(row[0])
                    for row in connection.execute(
                        select(revisions.c.payload)
                        .where(revisions.c.project_id == project["project_id"])
                        .order_by(revisions.c.revision.desc())
                    )
                ]
            return ExperimentPlanView(
                project_id=project["project_id"] if project else None,
                current_revision=project["current_revision"] if project else 0,
                current=saved[0] if saved else None,
                pending_proposal=self._proposal(pending) if pending else None,
                history=tuple(
                    PlanRevisionSummary(
                        revision=item.revision,
                        name=item.plan.name,
                        confirmed_at=item.confirmed_at,
                    )
                    for item in saved
                ),
            )

    def revision(self, owner: PlanOwner, revision: int) -> PlanRevision:
        with self._transaction(owner) as connection:
            project = self._project(connection, _owner_key(owner))
            payload = (
                connection.scalar(
                    select(revisions.c.payload).where(
                        revisions.c.project_id == project["project_id"],
                        revisions.c.revision == revision,
                    )
                )
                if project
                else None
            )
            if payload is None:
                raise PlanConflictError("plan revision does not exist")
            return PlanRevision.model_validate(payload)

    def propose(
        self,
        owner: PlanOwner,
        proposal: PlanProposal,
        *,
        source_proposal_id: str | None,
        initial: bool,
    ) -> PlanProposal:
        key = _owner_key(owner)
        with self._transaction(owner) as connection:
            project = self._project(connection, key)
            current = project["current_revision"] if project else 0
            pending = self._pending(connection, key)
            if proposal.base_revision != current:
                raise PlanConflictError("plan revision changed; reload the plan")
            if initial and (project or pending):
                raise PlanConflictError(
                    "plan already exists; use revise_experiment_plan"
                )
            if source_proposal_id and (
                not pending or pending["proposal_id"] != source_proposal_id
            ):
                raise PlanConflictError("source proposal is no longer pending")
            if pending and not source_proposal_id:
                raise PlanConflictError(
                    "include source_proposal_id to revise the draft"
                )
            connection.execute(
                update(documents)
                .where(documents.c.kind == "proposal")
                .where(documents.c.owner_key == key, documents.c.status == "pending")
                .values(status="superseded")
            )
            connection.execute(
                insert(documents).values(
                    kind="proposal",
                    document_id=proposal.proposal_id,
                    owner_key=key,
                    status="pending",
                    base_revision=current,
                    attributes=proposal.model_dump(mode="json"),
                )
            )
        return proposal

    def confirm(
        self,
        owner: PlanOwner,
        proposal_id: str,
        expected_revision: int,
        command_id: str,
    ) -> PlanRevision:
        key = _owner_key(owner)
        with self._transaction(owner) as connection:
            row = (
                connection.execute(
                    select(proposals).where(
                        proposals.c.owner_key == key,
                        proposals.c.proposal_id == proposal_id,
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise PlanConflictError("proposal does not exist")
            project = self._project(connection, key)
            if row["status"] == "confirmed":
                if row["base_revision"] != expected_revision or project is None:
                    raise PlanConflictError("confirmation does not match the proposal")
                payload = connection.scalar(
                    select(revisions.c.payload).where(
                        revisions.c.project_id == project["project_id"],
                        revisions.c.revision == row["confirmed_revision"],
                    )
                )
                return PlanRevision.model_validate(payload)
            if row["status"] != "pending":
                raise PlanConflictError("proposal is no longer pending")
            current = project["current_revision"] if project else 0
            if current != expected_revision or row["base_revision"] != current:
                raise PlanConflictError("plan revision changed; reload the plan")
            collision = connection.scalar(
                select(proposals.c.proposal_id).where(
                    proposals.c.owner_key == key,
                    proposals.c.confirmation_command_id == command_id,
                )
            )
            if collision:
                raise PlanConflictError(
                    "confirmation command belongs to another proposal"
                )
            project_id = project["project_id"] if project else uuid4().hex
            revision = PlanRevision(
                project_id=project_id,
                revision=current + 1,
                plan=self._proposal(row).plan,
                confirmed_by=owner.user_id,
            )
            if project is None:
                connection.execute(
                    insert(projects).values(
                        project_id=project_id,
                        owner_key=key,
                        **owner.model_dump(),
                        current_revision=revision.revision,
                    )
                )
            else:
                connection.execute(
                    update(projects)
                    .where(projects.c.project_id == project_id)
                    .values(current_revision=revision.revision)
                )
            connection.execute(
                insert(documents).values(
                    kind="revision",
                    document_id=hashlib.sha256(
                        f"{project_id}:{revision.revision}".encode()
                    ).hexdigest(),
                    owner_key=key,
                    project_id=project_id,
                    revision=revision.revision,
                    attributes=revision.model_dump(mode="json"),
                )
            )
            connection.execute(
                update(documents)
                .where(documents.c.kind == "proposal")
                .where(documents.c.document_id == proposal_id)
                .values(
                    status="confirmed",
                    confirmation_command_id=command_id,
                    confirmed_revision=revision.revision,
                )
            )
            return revision

    def reject(self, owner: PlanOwner, proposal_id: str) -> None:
        with self._transaction(owner) as connection:
            row = connection.execute(
                select(proposals.c.status).where(
                    proposals.c.owner_key == _owner_key(owner),
                    proposals.c.proposal_id == proposal_id,
                )
            ).one_or_none()
            if row is None or row[0] not in {"pending", "rejected"}:
                raise PlanConflictError("proposal is no longer pending")
            connection.execute(
                update(documents)
                .where(documents.c.kind == "proposal")
                .where(
                    and_(
                        documents.c.owner_key == _owner_key(owner),
                        documents.c.document_id == proposal_id,
                    )
                )
                .values(status="rejected")
            )
