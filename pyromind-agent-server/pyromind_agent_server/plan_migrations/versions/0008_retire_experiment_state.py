"""Remove retired execution storage and duplicated Project plan fields."""

import json

import sqlalchemy as sa
from alembic import op


revision = "0008_retire_experiment_state"
down_revision = "0007_project_plan_path"
branch_labels = depends_on = None

RETIRED_TABLES = (
    "experiment_plan_documents",
    "runtime_messages",
    "external_task_submissions",
    "runtime_work_items",
    "agent_role_sessions",
    "experiment_task_runs",
)
RETIRED_COLUMNS = (
    "current_revision",
    "context",
    "context_sequence",
    "creation_metadata",
)


def upgrade() -> None:
    connection = op.get_bind()
    columns = {
        column["name"]
        for column in sa.inspect(connection).get_columns("experiment_projects")
    }
    if "creation_command_id" not in columns:
        op.add_column(
            "experiment_projects",
            sa.Column("creation_command_id", sa.String(128), nullable=True),
        )
    projects = sa.Table("experiment_projects", sa.MetaData(), autoload_with=connection)
    if "creation_metadata" in columns:
        for row in connection.execute(sa.select(projects)).mappings():
            creation = row["creation_metadata"]
            if isinstance(creation, str):
                creation = json.loads(creation)
            if creation and not row["creation_command_id"]:
                connection.execute(
                    projects.update()
                    .where(projects.c.project_id == row["project_id"])
                    .values(creation_command_id=creation["command_id"])
                )
    connection.execute(
        projects.update()
        .where(projects.c.workflow_plan_path.is_(None))
        .values(workflow_plan_path="public_data/workflowPLAN.json")
    )
    # MariaDB commits DDL separately, so retry only operations still outstanding.
    for name in RETIRED_TABLES:
        if sa.inspect(connection).has_table(name):
            op.drop_table(name)
    remaining = [name for name in RETIRED_COLUMNS if name in columns]
    if remaining:
        with op.batch_alter_table("experiment_projects") as batch:
            for name in remaining:
                batch.drop_column(name)


def downgrade() -> None:
    raise NotImplementedError("Retired experiment data cannot be reconstructed")
