"""Advisory context and standalone tasks; existing queue/ledgers stay in place."""

import sqlalchemy as sa
from alembic import op


revision = "0003_experiment_tasks"
down_revision = "0002_execution_queue"
branch_labels = depends_on = None


def upgrade():
    op.add_column("experiment_projects", sa.Column("context", sa.JSON))
    op.add_column(
        "experiment_projects",
        sa.Column("context_sequence", sa.Integer, nullable=False, server_default="0"),
    )
    with op.batch_alter_table("experiment_task_runs") as batch:
        batch.alter_column("revision", existing_type=sa.Integer, nullable=True)
        batch.add_column(sa.Column("task_spec", sa.JSON))
        batch.add_column(sa.Column("result", sa.JSON))
    constraints = sa.inspect(op.get_bind()).get_unique_constraints(
        "external_task_submissions"
    )
    with op.batch_alter_table(
        "external_task_submissions",
        naming_convention={"uq": "uq_%(table_name)s_%(column_0_name)s"},
    ) as batch:
        for constraint in constraints:
            if constraint["column_names"] == ["run_id"]:
                batch.drop_constraint(
                    constraint["name"] or "uq_external_task_submissions_run_id",
                    type_="unique",
                )
        batch.add_column(
            sa.Column(
                "operation_key", sa.String(128), nullable=False, server_default="legacy"
            )
        )
        batch.create_unique_constraint(
            "uq_execution_operation", ["run_id", "operation_key"]
        )


def downgrade():
    raise RuntimeError("Preserve execution history; downgrade using a database backup.")
