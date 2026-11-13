"""Transactional experiment and metric projections for SQLite and MySQL."""

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from pyromind_runtime.application.experiments import experiment_metrics
from pyromind_runtime.domain.experiment_plans import PlanOwner
from pyromind_runtime.domain.experiments import ExperimentCatalog, ExperimentVersion
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload
from pyromind_runtime.domain.versions import fingerprint
from pyromind_runtime.infrastructure.sql_database import (
    insert_if_absent,
    write_transaction,
)
from pyromind_runtime.infrastructure.sql_project_store import owner_key


metadata = sa.MetaData()
options = dict(mysql_engine="InnoDB", mysql_charset="utf8mb4")

experiment_versions = sa.Table(
    "experiment_version_index",
    metadata,
    sa.Column("index_id", sa.String(64), primary_key=True),
    sa.Column("owner_key", sa.String(64), nullable=False),
    sa.Column("conversation_id", sa.String(128), nullable=False),
    sa.Column("project_id", sa.String(64)),
    sa.Column("experiment_id", sa.String(128), nullable=False),
    sa.Column("display_version", sa.String(32), nullable=False),
    sa.Column("status", sa.String(32), nullable=False),
    sa.Column("created_at", sa.BigInteger, nullable=False),
    sa.Column("source_sequence", sa.BigInteger, nullable=False),
    sa.Column("observed_at", sa.BigInteger, nullable=False),
    sa.Column("manifest_digest", sa.String(64), nullable=False),
    sa.Column("archive_ref", sa.Text, nullable=False),
    sa.Column("document", sa.JSON, nullable=False),
    sa.Index("ix_experiment_versions_owner", "owner_key", "project_id"),
    **options,
)

experiment_metric_rows = sa.Table(
    "experiment_metric_index",
    metadata,
    sa.Column("metric_id", sa.String(64), primary_key=True),
    sa.Column(
        "index_id",
        sa.String(64),
        sa.ForeignKey("experiment_version_index.index_id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("evaluation_run_id", sa.String(128), nullable=False),
    sa.Column("step_run_id", sa.String(128), nullable=False),
    sa.Column("task_id", sa.String(128)),
    sa.Column("step", sa.BigInteger, nullable=False),
    sa.Column("dataset_role", sa.String(16), nullable=False),
    sa.Column("role", sa.String(16), nullable=False),
    sa.Column("comparison_group", sa.Text, nullable=False),
    sa.Column("metric", sa.Text, nullable=False),
    sa.Column("value", sa.Double, nullable=False),
    sa.Column("selected", sa.Boolean, nullable=False),
    sa.Index("ix_experiment_metrics_version", "index_id"),
    **options,
)


class SQLExperimentIndex:
    def __init__(self, engine: Engine):
        self.engine = engine

    def save(
        self,
        owner: PlanOwner,
        experiment: ExperimentVersion,
        *,
        archive_ref: str,
        manifest_digest: str,
        source_sequence: int,
        observed_at: int,
    ) -> None:
        if owner.conversation_id != experiment.conversation_id:
            raise ValueError("experiment belongs to a different conversation")
        document = experiment.model_dump(mode="json")
        validate_persisted_payload(document)
        validate_persisted_payload(archive_ref)
        scope = owner_key(owner)
        index_id = fingerprint(
            [scope, experiment.artifact_project_id, experiment.experiment_id]
        )
        version = dict(
            index_id=index_id,
            owner_key=scope,
            conversation_id=experiment.conversation_id,
            project_id=experiment.project_id,
            experiment_id=experiment.experiment_id,
            display_version=experiment.version,
            status=experiment.execution.status,
            created_at=experiment.created_at,
            source_sequence=source_sequence,
            observed_at=observed_at,
            manifest_digest=manifest_digest,
            archive_ref=archive_ref,
            document=document,
        )
        tasks = {
            evaluation.run_id: evaluation.task_id
            for evaluation in experiment.evaluations
        }
        metrics = experiment_metrics(
            ExperimentCatalog(
                conversation_id=experiment.conversation_id, versions=(experiment,)
            )
        )
        rows = [
            dict(
                metric_id=fingerprint(
                    [index_id, row.evaluation_run_id, row.step_run_id, row.metric]
                ),
                index_id=index_id,
                evaluation_run_id=row.evaluation_run_id,
                step_run_id=row.step_run_id,
                task_id=tasks.get(row.evaluation_run_id),
                step=row.step,
                dataset_role=row.dataset_role,
                role=row.role,
                comparison_group=row.comparison_group,
                metric=row.metric,
                value=row.value,
                selected=row.selected,
            )
            for row in metrics
        ]
        with write_transaction(self.engine) as conn:
            insert_if_absent(conn, experiment_versions, version, "index_id")
            current = conn.execute(
                sa.select(
                    experiment_versions.c.source_sequence,
                    experiment_versions.c.observed_at,
                )
                .where(experiment_versions.c.index_id == index_id)
                .with_for_update()
            ).one()
            if (source_sequence, observed_at) < (
                current.source_sequence,
                current.observed_at,
            ):
                return
            conn.execute(
                experiment_versions.update()
                .where(experiment_versions.c.index_id == index_id)
                .values(**version)
            )
            conn.execute(
                experiment_metric_rows.delete().where(
                    experiment_metric_rows.c.index_id == index_id
                )
            )
            if rows:
                conn.execute(experiment_metric_rows.insert(), rows)
