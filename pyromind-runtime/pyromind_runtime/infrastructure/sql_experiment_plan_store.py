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
    ExperimentPlanView,
    PlanConflictError,
    PlanOwner,
    PlanProposal,
    PlanRevision,
    PlanRevisionSummary,
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
)
proposals = Table(
    "experiment_plan_proposals",
    metadata,
    Column("proposal_id", String(64), primary_key=True),
    Column("owner_key", String(64), nullable=False, index=True),
    Column("status", String(16), nullable=False),
    Column("base_revision", Integer, nullable=False),
    Column("payload", JSON, nullable=False),
    Column("confirmation_command_id", String(128)),
    Column("confirmed_revision", Integer),
    UniqueConstraint("owner_key", "confirmation_command_id"),
)
revisions = Table(
    "experiment_plan_revisions",
    metadata,
    Column(
        "project_id", String(64), ForeignKey(projects.c.project_id), primary_key=True
    ),
    Column("revision", Integer, primary_key=True),
    Column("payload", JSON, nullable=False),
)


def _owner_key(owner: PlanOwner) -> str:
    return hashlib.sha256(owner.model_dump_json().encode()).hexdigest()


class SQLExperimentPlanStore:
    """Transactional plan storage; MariaDB in production, SQLite in unit tests.

    A connection-scoped advisory lock also serializes the first proposal and
    first confirmation, before any project row exists to SELECT FOR UPDATE.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._local_lock = threading.RLock()

    def close(self) -> None:
        self.engine.dispose()

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
                with connection.begin():
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
                update(proposals)
                .where(proposals.c.owner_key == key, proposals.c.status == "pending")
                .values(status="superseded")
            )
            connection.execute(
                insert(proposals).values(
                    proposal_id=proposal.proposal_id,
                    owner_key=key,
                    status="pending",
                    base_revision=current,
                    payload=proposal.model_dump(mode="json"),
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
                insert(revisions).values(
                    project_id=project_id,
                    revision=revision.revision,
                    payload=revision.model_dump(mode="json"),
                )
            )
            connection.execute(
                update(proposals)
                .where(proposals.c.proposal_id == proposal_id)
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
                update(proposals)
                .where(
                    and_(
                        proposals.c.owner_key == _owner_key(owner),
                        proposals.c.proposal_id == proposal_id,
                    )
                )
                .values(status="rejected")
            )
