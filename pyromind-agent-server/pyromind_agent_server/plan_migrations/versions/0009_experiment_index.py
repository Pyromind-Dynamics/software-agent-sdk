"""Add the derived experiment and evaluation metric indexes."""

from alembic import op
from pyromind_runtime.infrastructure.sql_experiment_index import (
    experiment_metric_rows,
    experiment_versions,
)


revision = "0009_experiment_index"
down_revision = "0008_retire_experiment_state"
branch_labels = depends_on = None


def upgrade():
    bind = op.get_bind()
    experiment_versions.create(bind, checkfirst=True)
    experiment_metric_rows.create(bind, checkfirst=True)


def downgrade():
    bind = op.get_bind()
    experiment_metric_rows.drop(bind, checkfirst=True)
    experiment_versions.drop(bind, checkfirst=True)
