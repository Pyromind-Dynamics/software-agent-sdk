"""Consolidate ten experiment tables to seven, preserving legacy identities.

Copy/verify precedes every source drop. All steps tolerate completed prior DDL
because MariaDB DDL commits independently of the Alembic version transaction.
Workers must be stopped during this startup migration.
"""

import hashlib
import json

import sqlalchemy as sa
from alembic import op


revision = "0004_compact_storage"
down_revision = "0003_experiment_tasks"
branch_labels = depends_on = None
OPTIONS = dict(
    mysql_engine="InnoDB", mysql_charset="utf8mb4", mysql_collate="utf8mb4_bin"
)


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def _table(name):
    return sa.Table(name, sa.MetaData(), autoload_with=op.get_bind())


def _copy(target, keys, values):
    connection = op.get_bind()
    condition = sa.and_(*(target.c[key] == values[key] for key in keys))
    existing = connection.execute(sa.select(target).where(condition)).mappings().first()
    if existing is None:
        # Reflection can expose MariaDB JSON aliases as LONGTEXT; serialize them
        # explicitly while retaining JSON bind processing on SQLite/JSON types.
        bound = {
            key: json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list))
            and not isinstance(target.c[key].type, sa.JSON)
            else value
            for key, value in values.items()
        }
        connection.execute(target.insert().values(**bound))
        existing = (
            connection.execute(sa.select(target).where(condition)).mappings().one()
        )
    for key, value in values.items():
        actual = (
            _json(existing[key]) if isinstance(value, (dict, list)) else existing[key]
        )
        if actual != value:
            raise RuntimeError(f"migration verification failed: {target.name}.{key}")


def upgrade():
    connection = op.get_bind()
    names = set(sa.inspect(connection).get_table_names())
    if "experiment_plan_documents" not in names:
        op.create_table(
            "experiment_plan_documents",
            sa.Column("kind", sa.String(16), primary_key=True),
            sa.Column("document_id", sa.String(64), primary_key=True),
            sa.Column("owner_key", sa.String(64), nullable=False),
            sa.Column(
                "project_id",
                sa.String(64),
                sa.ForeignKey("experiment_projects.project_id"),
            ),
            sa.Column("revision", sa.Integer),
            sa.Column("status", sa.String(16)),
            sa.Column("base_revision", sa.Integer),
            sa.Column("attributes", sa.JSON, nullable=False),
            sa.Column("confirmation_command_id", sa.String(128)),
            sa.Column("confirmed_revision", sa.Integer),
            sa.UniqueConstraint(
                "owner_key",
                "confirmation_command_id",
                name="uq_plan_document_confirmation",
            ),
            sa.UniqueConstraint(
                "project_id", "revision", name="uq_plan_document_revision"
            ),
            **OPTIONS,
        )
    indexes = {
        item["name"]
        for item in sa.inspect(connection).get_indexes("experiment_plan_documents")
    }
    if "ix_experiment_plan_documents_owner_key" not in indexes:
        op.create_index(
            "ix_experiment_plan_documents_owner_key",
            "experiment_plan_documents",
            ["owner_key"],
        )
    if "runtime_messages" not in names:
        op.create_table(
            "runtime_messages",
            sa.Column("direction", sa.String(8), primary_key=True),
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("owner", sa.String(64)),
            sa.Column("conversation_id", sa.String(128)),
            sa.Column("task_id", sa.String(128)),
            sa.Column("source", sa.String(32)),
            sa.Column("attributes", sa.JSON, nullable=False),
            sa.Column("state", sa.String(32), nullable=False),
            sa.Column("created_at", sa.DateTime),
            **OPTIONS,
        )
    indexes = {
        item["name"] for item in sa.inspect(connection).get_indexes("runtime_messages")
    }
    for name, columns in (
        ("ix_runtime_messages_pending", ["direction", "state", "created_at"]),
        ("ix_runtime_messages_task", ["direction", "task_id", "state"]),
    ):
        if name not in indexes:
            op.create_index(name, "runtime_messages", columns)
    if "attributes" not in {
        item["name"]
        for item in sa.inspect(connection).get_columns("experiment_task_runs")
    }:
        op.add_column("experiment_task_runs", sa.Column("attributes", sa.JSON))

    documents, messages = (
        _table("experiment_plan_documents"),
        _table("runtime_messages"),
    )
    projects, runs = _table("experiment_projects"), _table("experiment_task_runs")
    owners = dict(
        connection.execute(sa.select(projects.c.project_id, projects.c.owner_key)).all()
    )
    if "experiment_plan_proposals" in names:
        for row in connection.execute(
            sa.select(_table("experiment_plan_proposals"))
        ).mappings():
            _copy(
                documents,
                ("kind", "document_id"),
                {
                    "kind": "proposal",
                    "document_id": row["proposal_id"],
                    "owner_key": row["owner_key"],
                    "project_id": None,
                    "revision": None,
                    "status": row["status"],
                    "base_revision": row["base_revision"],
                    "attributes": _json(row["payload"]),
                    "confirmation_command_id": row["confirmation_command_id"],
                    "confirmed_revision": row["confirmed_revision"],
                },
            )
    if "experiment_plan_revisions" in names:
        for row in connection.execute(
            sa.select(_table("experiment_plan_revisions"))
        ).mappings():
            _copy(
                documents,
                ("kind", "document_id"),
                {
                    "kind": "revision",
                    "document_id": hashlib.sha256(
                        f"{row['project_id']}:{row['revision']}".encode()
                    ).hexdigest(),
                    "owner_key": owners[row["project_id"]],
                    "project_id": row["project_id"],
                    "revision": row["revision"],
                    "attributes": _json(row["payload"]),
                    "status": None,
                    "base_revision": None,
                    "confirmation_command_id": None,
                    "confirmed_revision": None,
                },
            )
    for source, direction in (("runtime_inbox", "in"), ("runtime_outbox", "out")):
        if source not in names:
            continue
        for row in connection.execute(sa.select(_table(source))).mappings():
            _copy(
                messages,
                ("direction", "id"),
                {
                    "direction": direction,
                    "id": row["id"],
                    "owner": row.get("owner"),
                    "conversation_id": row.get("conversation_id"),
                    "task_id": row.get("task_id"),
                    "source": row.get("source"),
                    "attributes": _json(row["payload"]),
                    "state": row["state"],
                    "created_at": row.get("created_at"),
                },
            )
    if "experiment_artifacts" in names:
        for row in connection.execute(
            sa.select(_table("experiment_artifacts"))
        ).mappings():
            current = (
                connection.execute(sa.select(runs).where(runs.c.id == row["run_id"]))
                .mappings()
                .one()
            )
            if current["owner"] != row["owner"]:
                raise RuntimeError("artifact owner verification failed")
            attributes = dict(_json(current["attributes"]) or {})
            registered = dict(attributes.get("artifacts", {}))
            value = {
                "id": row["id"],
                "uri": row["uri"],
                "sha256": row["sha256"],
                "metadata": _json(row["payload"]),
            }
            if row["name"] in registered and registered[row["name"]] != value:
                raise RuntimeError("artifact migration content conflict")
            registered[row["name"]] = value
            attributes["artifacts"] = registered
            bound = (
                attributes
                if isinstance(runs.c.attributes.type, sa.JSON)
                else json.dumps(attributes, ensure_ascii=False)
            )
            connection.execute(
                runs.update().where(runs.c.id == row["run_id"]).values(attributes=bound)
            )
            saved = _json(
                connection.scalar(
                    sa.select(runs.c.attributes).where(runs.c.id == row["run_id"])
                )
            )
            if saved != attributes:
                raise RuntimeError("artifact migration verification failed")
    # Durable copy before independently committed MariaDB DROP operations.
    # Let Alembic own the commit boundary so its schema-version update is
    # committed too. A raw Connection.commit() would detach its transaction.
    with op.get_context().autocommit_block():
        for name in (
            "experiment_plan_revisions",
            "experiment_plan_proposals",
            "experiment_artifacts",
            "runtime_inbox",
            "runtime_outbox",
        ):
            if name in sa.inspect(connection).get_table_names():
                op.drop_table(name)


def downgrade():
    raise RuntimeError("Restore a database backup to downgrade compact storage.")
