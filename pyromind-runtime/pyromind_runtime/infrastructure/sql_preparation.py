from __future__ import annotations

import json
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from pyromind_runtime.domain.experiment_plans import (
    PlanConflictError,
    PlanOwner,
)
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload
from pyromind_runtime.domain.preparation import (
    PreparationState,
    ProjectView,
)
from pyromind_runtime.infrastructure.sql_database import (
    configure_engine,
    insert_if_absent,
    write_transaction,
)
from pyromind_runtime.infrastructure.sql_project_store import (
    metadata,
    owner_key,
    projects,
)


preparations = sa.Table(
    "experiment_preparations",
    metadata,
    sa.Column("owner_key", sa.String(64), primary_key=True),
    sa.Column("revision", sa.Integer, nullable=False),
    sa.Column("document", sa.JSON, nullable=False),
    sa.Column("project_id", sa.String(64)),
    sa.Column("last_message_id", sa.String(128)),
)


class SQLPreparationStore:
    def __init__(self, engine: Engine):
        self.engine = configure_engine(engine)

    def initialize(self, owner: PlanOwner) -> None:
        view = PreparationState()
        with write_transaction(self.engine) as conn:
            insert_if_absent(
                conn,
                preparations,
                {
                    "owner_key": owner_key(owner),
                    "revision": 0,
                    "document": view.model_dump(mode="json"),
                },
                "owner_key",
            )

    def _row(self, conn, owner):
        row = (
            conn.execute(
                sa.select(preparations)
                .where(preparations.c.owner_key == owner_key(owner))
                .with_for_update()
            )
            .mappings()
            .first()
        )
        if row is None:
            raise PermissionError("preparation not found")
        return row

    def get(self, owner: PlanOwner) -> PreparationState:
        with self.engine.connect() as conn:
            return PreparationState.model_validate(self._row(conn, owner)["document"])

    def save(
        self, owner: PlanOwner, view: PreparationState, expected_revision: int
    ) -> PreparationState:
        payload = view.model_copy(
            update={"revision": expected_revision + 1}
        ).model_dump(mode="json")
        with write_transaction(self.engine) as conn:
            row = self._row(conn, owner)
            if row["revision"] != expected_revision or row["project_id"]:
                raise PlanConflictError("preparation changed; reread and merge")
            validate_persisted_payload(payload, path="")
            if len(json.dumps(payload).encode()) > 262144:
                raise ValueError("preparation exceeds 256 KiB")
            conn.execute(
                preparations.update()
                .where(preparations.c.owner_key == owner_key(owner))
                .values(revision=expected_revision + 1, document=payload)
            )
        return PreparationState.model_validate(payload)

    def invalidate(self, owner: PlanOwner, command_id: str) -> int | None:
        with write_transaction(self.engine) as conn:
            row = self._row(conn, owner)
            if row["project_id"]:
                return None
            if row["last_message_id"] == command_id:
                return row["revision"]
            view = PreparationState.model_validate(row["document"])
            view = view.model_copy(
                update={
                    "revision": row["revision"] + 1,
                    "review_pending": True,
                    "phase": "planning" if view.plan else "collecting",
                }
            )
            conn.execute(
                preparations.update()
                .where(preparations.c.owner_key == owner_key(owner))
                .values(
                    revision=view.revision,
                    document=view.model_dump(mode="json"),
                    last_message_id=command_id,
                )
            )
            return view.revision

    def _project(self, row, state: PreparationState) -> ProjectView:
        if state.plan is None:
            raise ValueError("accepted project plan is missing")
        return ProjectView(
            project_id=row["project_id"],
            conversation_id=row["conversation_id"],
            cluster=row["cluster"],
            plan_sequence=state.revision,
            plan=state.plan,
            creation_command_id=row["creation_command_id"],
        )

    def project(self, owner: PlanOwner) -> ProjectView | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    sa.select(projects, preparations.c.document)
                    .join(
                        preparations, projects.c.owner_key == preparations.c.owner_key
                    )
                    .where(projects.c.owner_key == owner_key(owner))
                )
                .mappings()
                .first()
            )
            return (
                self._project(row, PreparationState.model_validate(row["document"]))
                if row and row["creation_command_id"]
                else None
            )

    def activate(
        self,
        owner: PlanOwner,
        expected_revision: int,
        command_id: str,
        *,
        cluster: str | None = None,
    ) -> ProjectView:
        with write_transaction(self.engine) as conn:
            row = self._row(conn, owner)
            view = PreparationState.model_validate(row["document"])
            if row["project_id"]:
                return self._project(
                    conn.execute(
                        sa.select(projects).where(
                            projects.c.project_id == row["project_id"]
                        )
                    )
                    .mappings()
                    .one(),
                    view,
                )
            if row["revision"] != expected_revision:
                raise PlanConflictError("preparation changed; refresh the current plan")
            if (
                view.phase != "ready"
                or view.review_pending
                or view.plan is None
                or view.entry_task is None
            ):
                raise ValueError("preparation is not ready")
            identity = uuid4().hex
            conn.execute(
                projects.insert().values(
                    project_id=identity,
                    owner_key=owner_key(owner),
                    user_id=owner.user_id,
                    conversation_id=owner.conversation_id,
                    cluster=cluster,
                    profile_id=owner.profile_id,
                    creation_command_id=command_id,
                    workflow_plan_path="public_data/workflowPLAN.json",
                )
            )
            activated = view.model_copy(
                update={"phase": "active", "project_id": identity}
            )
            conn.execute(
                preparations.update()
                .where(preparations.c.owner_key == owner_key(owner))
                .values(project_id=identity, document=activated.model_dump(mode="json"))
            )
            return ProjectView(
                project_id=identity,
                conversation_id=owner.conversation_id,
                cluster=cluster,
                plan_sequence=expected_revision,
                plan=view.plan,
                creation_command_id=command_id,
            )
