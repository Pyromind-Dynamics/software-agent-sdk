"""Projects, proposals and immutable plan revisions."""

import sqlalchemy as sa
from alembic import op


revision = "0001_experiment_plans"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "experiment_projects",
        sa.Column("project_id", sa.String(64), primary_key=True),
        sa.Column("owner_key", sa.String(64), nullable=False, unique=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("conversation_id", sa.String(128), nullable=False),
        sa.Column("profile_id", sa.String(128)),
        sa.Column("current_revision", sa.Integer, nullable=False),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_bin",
    )
    op.create_table(
        "experiment_plan_proposals",
        sa.Column("proposal_id", sa.String(64), primary_key=True),
        sa.Column("owner_key", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("base_revision", sa.Integer, nullable=False),
        sa.Column("payload", sa.JSON, nullable=False),
        sa.Column("confirmation_command_id", sa.String(128)),
        sa.Column("confirmed_revision", sa.Integer),
        sa.UniqueConstraint("owner_key", "confirmation_command_id"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_bin",
    )
    op.create_index(
        "ix_experiment_plan_proposals_owner_key",
        "experiment_plan_proposals",
        ["owner_key"],
    )
    op.create_table(
        "experiment_plan_revisions",
        sa.Column(
            "project_id",
            sa.String(64),
            sa.ForeignKey("experiment_projects.project_id"),
            primary_key=True,
        ),
        sa.Column("revision", sa.Integer, primary_key=True),
        sa.Column("payload", sa.JSON, nullable=False),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_bin",
    )


def downgrade() -> None:
    op.drop_table("experiment_plan_revisions")
    op.drop_table("experiment_plan_proposals")
    op.drop_table("experiment_projects")
