"""Link execution task runs to node-level stage runs."""

import sqlalchemy as sa
from alembic import op


revision = "0006_stage_task_links"
down_revision = "0005_preparation"
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    columns = {
        column["name"]
        for column in sa.inspect(connection).get_columns("experiment_task_runs")
    }
    if "stage_run_id" not in columns:
        op.add_column(
            "experiment_task_runs",
            sa.Column("stage_run_id", sa.String(64)),
        )
        op.create_index(
            "ix_experiment_task_runs_stage_run",
            "experiment_task_runs",
            ["stage_run_id"],
        )


def downgrade():
    connection = op.get_bind()
    indexes = {
        index["name"]
        for index in sa.inspect(connection).get_indexes("experiment_task_runs")
    }
    if "ix_experiment_task_runs_stage_run" in indexes:
        op.drop_index(
            "ix_experiment_task_runs_stage_run",
            table_name="experiment_task_runs",
        )
    columns = {
        column["name"]
        for column in sa.inspect(connection).get_columns("experiment_task_runs")
    }
    if "stage_run_id" in columns:
        op.drop_column("experiment_task_runs", "stage_run_id")
