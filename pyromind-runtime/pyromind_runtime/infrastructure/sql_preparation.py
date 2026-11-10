from __future__ import annotations

import json
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from pyromind_runtime.domain.experiment_plans import (
    ExperimentContext,
    PlanConflictError,
    PlanOwner,
)
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload
from pyromind_runtime.domain.preparation import (
    PreparationPolicy,
    PreparationView,
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

    def initialize(self, owner: PlanOwner, policy: PreparationPolicy) -> None:
        view = PreparationView(
            policy=policy,
            missing=tuple(field.key for field in policy.fields if field.required),
        )
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

    def get(self, owner: PlanOwner) -> PreparationView:
        with self.engine.connect() as conn:
            return PreparationView.model_validate(self._row(conn, owner)["document"])

    def save(
        self, owner: PlanOwner, view: PreparationView, expected_revision: int
    ) -> PreparationView:
        payload = view.model_copy(
            update={"revision": expected_revision + 1}
        ).model_dump(mode="json")
        with write_transaction(self.engine) as conn:
            row = self._row(conn, owner)
            if row["revision"] != expected_revision or row["project_id"]:
                raise PlanConflictError("preparation changed; reread and merge")
            # Evidence may have been appended without changing the revision.
            payload["evidence"] = {
                **view.evidence,
                **row["document"].get("evidence", {}),
            }
            validate_persisted_payload(payload, path="")
            if len(json.dumps(payload).encode()) > 262144:
                raise ValueError("preparation exceeds 256 KiB")
            conn.execute(
                preparations.update()
                .where(preparations.c.owner_key == owner_key(owner))
                .values(revision=expected_revision + 1, document=payload)
            )
        return PreparationView.model_validate(payload)

    def invalidate(self, owner: PlanOwner, command_id: str) -> int | None:
        with write_transaction(self.engine) as conn:
            row = self._row(conn, owner)
            if row["project_id"]:
                return None
            if row["last_message_id"] == command_id:
                return row["revision"]
            view = PreparationView.model_validate(row["document"])
            view = view.model_copy(
                update={
                    "revision": row["revision"] + 1,
                    "review_pending": True,
                    "phase": "collecting" if view.missing else "planning",
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

    def add_evidence(self, owner: PlanOwner, reference: str, description: str) -> int:
        with write_transaction(self.engine) as conn:
            row = self._row(conn, owner)
            if row["project_id"]:
                raise ValueError("preparation is already activated")
            view = PreparationView.model_validate(row["document"])
            evidence = {**view.evidence, reference: description}
            view = view.model_copy(update={"evidence": evidence})
            conn.execute(
                preparations.update()
                .where(preparations.c.owner_key == owner_key(owner))
                .values(document=view.model_dump(mode="json"))
            )
            return view.revision

    def _project(self, row) -> ProjectView:
        creation = row["creation_metadata"]
        return ProjectView(
            project_id=row["project_id"],
            conversation_id=row["conversation_id"],
            cluster=row["cluster"],
            plan_sequence=row["context_sequence"],
            plan=ExperimentContext.model_validate(row["context"]),
            creation_command_id=creation["command_id"],
        )

    def project(self, owner: PlanOwner) -> ProjectView | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    sa.select(projects).where(projects.c.owner_key == owner_key(owner))
                )
                .mappings()
                .first()
            )
            return self._project(row) if row and row["creation_metadata"] else None

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
            if row["project_id"]:
                return self._project(
                    conn.execute(
                        sa.select(projects).where(
                            projects.c.project_id == row["project_id"]
                        )
                    )
                    .mappings()
                    .one()
                )
            view = PreparationView.model_validate(row["document"])
            if row["revision"] != expected_revision:
                raise PlanConflictError("preparation changed; refresh the current plan")
            if (
                view.phase != "ready"
                or view.review_pending
                or view.missing
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
                    current_revision=0,
                    context_sequence=1,
                    context=view.plan.model_dump(mode="json"),
                )
            )
            creation = {
                "command_id": command_id,
                "preparation_revision": expected_revision,
                "preparation": view.model_dump(mode="json"),
            }
            conn.execute(
                projects.update()
                .where(projects.c.project_id == identity)
                .values(creation_metadata=creation)
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
                plan_sequence=1,
                plan=view.plan,
                creation_command_id=command_id,
            )
