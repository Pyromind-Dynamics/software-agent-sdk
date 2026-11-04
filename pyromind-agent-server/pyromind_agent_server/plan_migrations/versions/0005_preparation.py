"""Conversation preparation before explicit Project creation."""

import sqlalchemy as sa
from alembic import op


revision = "0005_preparation"
down_revision = "0004_compact_storage"
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    if "creation_metadata" not in {
        column["name"]
        for column in sa.inspect(connection).get_columns("experiment_projects")
    }:
        op.add_column("experiment_projects", sa.Column("creation_metadata", sa.JSON))
    if not sa.inspect(connection).has_table("experiment_preparations"):
        op.create_table(
            "experiment_preparations",
            sa.Column("owner_key", sa.String(64), primary_key=True),
            sa.Column("revision", sa.Integer, nullable=False),
            sa.Column("document", sa.JSON, nullable=False),
            sa.Column("project_id", sa.String(64)),
            sa.Column("last_message_id", sa.String(128)),
            mysql_charset="utf8mb4",
            mysql_engine="InnoDB",
        )


def downgrade():
    op.drop_table("experiment_preparations")
    op.drop_column("experiment_projects", "creation_metadata")
