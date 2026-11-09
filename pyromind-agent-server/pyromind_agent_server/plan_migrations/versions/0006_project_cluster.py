"""Add the Storage cluster route to Project identity."""

import sqlalchemy as sa
from alembic import op


revision = "0006_project_cluster"
down_revision = "0005_preparation"
branch_labels = None
depends_on = None


def upgrade():
    columns = {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("experiment_projects")
    }
    if "cluster" not in columns:
        op.add_column(
            "experiment_projects",
            sa.Column("cluster", sa.String(255), nullable=True),
        )


def downgrade():
    op.drop_column("experiment_projects", "cluster")
