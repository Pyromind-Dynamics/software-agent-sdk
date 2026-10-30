"""Durable execution, role queues and delivery ledgers. Frozen schema revision."""

from typing import Any

import sqlalchemy as sa
from alembic import op


revision = "0002_execution_queue"
down_revision = "0001_experiment_plans"
branch_labels = None
depends_on = None
metadata = sa.MetaData()
_TABLE_OPTIONS: dict[str, Any] = dict(
    mysql_engine="InnoDB", mysql_charset="utf8mb4", mysql_collate="utf8mb4_bin"
)
roles = sa.Table(
    "agent_role_sessions",
    metadata,
    sa.Column("lane", sa.String(64), primary_key=True),
    sa.Column("owner", sa.String(64), nullable=False),
    sa.Column("active_id", sa.String(64)),
    sa.Column("generation", sa.Integer, nullable=False, default=0),
    sa.Column("lease_until", sa.DateTime),
    sa.Column("session_ref", sa.JSON),
    **_TABLE_OPTIONS,
)
work = sa.Table(
    "runtime_work_items",
    metadata,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("owner", sa.String(64), nullable=False),
    sa.Column("lane", sa.String(64), nullable=False),
    sa.Column("dedup_key", sa.String(64), nullable=False),
    sa.Column("kind", sa.String(64), nullable=False),
    sa.Column("payload", sa.JSON, nullable=False),
    sa.Column("state", sa.String(32), nullable=False),
    sa.Column("priority", sa.Integer, nullable=False),
    sa.Column("created_at", sa.DateTime, nullable=False),
    sa.Column("available_at", sa.DateTime, nullable=False),
    sa.Column("lease_until", sa.DateTime),
    sa.Column("generation", sa.Integer, nullable=False),
    sa.Column("attempts", sa.Integer, nullable=False),
    sa.Column("checkpoint", sa.JSON, nullable=False),
    sa.UniqueConstraint("owner", "dedup_key"),
    sa.Index("ix_runtime_work_ready", "state", "available_at", "priority"),
    **_TABLE_OPTIONS,
)

runs = sa.Table(
    "experiment_task_runs",
    metadata,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("owner", sa.String(64), nullable=False),
    sa.Column("project_id", sa.String(64), nullable=False),
    sa.Column("revision", sa.Integer, nullable=False),
    sa.Column("task_id", sa.String(128), nullable=False),
    sa.Column("role_id", sa.String(64), nullable=False),
    sa.Column("active_key", sa.String(64), unique=True),
    sa.Column("request_key", sa.String(64), nullable=False, unique=True),
    sa.Column("state", sa.String(32), nullable=False),
    sa.Column("report_state", sa.String(32), nullable=False),
    sa.Column("proposal", sa.JSON),
    sa.Column("fingerprint", sa.String(64)),
    sa.Column("decision_id", sa.String(128)),
    sa.Column("decision", sa.String(16)),
    sa.Column("metrics", sa.JSON),
    sa.Column("created_at", sa.DateTime, nullable=False),
    **_TABLE_OPTIONS,
)
submissions = sa.Table(
    "external_task_submissions",
    metadata,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("run_id", sa.String(64), nullable=False, unique=True),
    sa.Column("owner", sa.String(64), nullable=False),
    sa.Column("state", sa.String(32), nullable=False),
    sa.Column("task_id", sa.String(128), unique=True),
    sa.Column("request_fingerprint", sa.String(64), nullable=False),
    sa.Column("evidence", sa.JSON, nullable=False),
    sa.Column("updated_at", sa.DateTime, nullable=False),
    **_TABLE_OPTIONS,
)
inbox = sa.Table(
    "runtime_inbox",
    metadata,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("task_id", sa.String(128), nullable=False),
    sa.Column("source", sa.String(32), nullable=False),
    sa.Column("payload", sa.JSON, nullable=False),
    sa.Column("state", sa.String(32), nullable=False),
    sa.Column("created_at", sa.DateTime, nullable=False),
    **_TABLE_OPTIONS,
)
artifacts = sa.Table(
    "experiment_artifacts",
    metadata,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("run_id", sa.String(64), nullable=False),
    sa.Column("owner", sa.String(64), nullable=False),
    sa.Column("name", sa.String(255), nullable=False),
    sa.Column("uri", sa.Text, nullable=False),
    sa.Column("sha256", sa.String(64), nullable=False),
    sa.Column("payload", sa.JSON, nullable=False),
    sa.UniqueConstraint("run_id", "name"),
    **_TABLE_OPTIONS,
)
outbox = sa.Table(
    "runtime_outbox",
    metadata,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("owner", sa.String(64), nullable=False),
    sa.Column("conversation_id", sa.String(128), nullable=False),
    sa.Column("payload", sa.JSON, nullable=False),
    sa.Column("state", sa.String(32), nullable=False),
    **_TABLE_OPTIONS,
)


def upgrade() -> None:
    metadata.create_all(op.get_bind())


def downgrade() -> None:
    metadata.drop_all(op.get_bind())
