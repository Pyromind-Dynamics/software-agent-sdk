"""Record the conversation-relative workflow plan path on Project."""

import sqlalchemy as sa
from alembic import op


revision = "0007_project_plan_path"
down_revision = "0006_project_cluster"
branch_labels = depends_on = None


def upgrade() -> None:
    columns = {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("experiment_projects")
    }
    if "workflow_plan_path" not in columns:
        op.add_column(
            "experiment_projects",
            sa.Column("workflow_plan_path", sa.String(4096), nullable=True),
        )


def downgrade() -> None:
    op.drop_column("experiment_projects", "workflow_plan_path")
