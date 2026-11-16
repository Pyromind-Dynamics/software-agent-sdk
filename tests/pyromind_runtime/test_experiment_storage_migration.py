import importlib
import os
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from pyromind_runtime.domain.experiment_plans import PlanOwner
from pyromind_runtime.domain.preparation import PreparationState
from pyromind_runtime.infrastructure.sql_database import create_database_engine
from pyromind_runtime.infrastructure.sql_preparation import SQLPreparationStore
from pyromind_runtime.infrastructure.sql_project_store import owner_key

from .test_preparation import ready_request as ready_request


RETIRED = importlib.import_module(
    "pyromind_agent_server.plan_migrations.versions.0008_retire_experiment_state"
)


@pytest.fixture(params=["sqlite", "mariadb"])
def migration_database(request, tmp_path):
    url = f"sqlite:///{tmp_path / 'experiments.sqlite3'}"
    if request.param == "mariadb":
        url = os.environ.get("EXPERIMENT_MIGRATION_TEST_URL")
        if not url:
            pytest.skip("No independent MariaDB migration test database configured")
    engine = create_database_engine(url)
    assert sa.inspect(engine).get_table_names() == [], "Use an empty test database"
    config = Config()
    config.set_main_option(
        "script_location",
        str(
            Path(__file__).resolve().parents[2]
            / "pyromind-agent-server/pyromind_agent_server/plan_migrations"
        ),
    )
    config.attributes["database_url"] = url
    try:
        yield engine, config
    finally:
        metadata = sa.MetaData()
        metadata.reflect(bind=engine)
        metadata.drop_all(engine)
        engine.dispose()


def assert_compact_schema(engine):
    inspector = sa.inspect(engine)
    assert set(inspector.get_table_names()) == {
        "experiment_projects",
        "experiment_preparations",
        "experiment_plan_schema_version",
    }
    assert not set(RETIRED.RETIRED_COLUMNS) & {
        column["name"] for column in inspector.get_columns("experiment_projects")
    }


def test_fresh_database_has_only_current_storage(migration_database):
    engine, config = migration_database
    command.upgrade(config, "head")
    command.upgrade(config, "head")
    assert_compact_schema(engine)


def test_retirement_preserves_projects_preparation_and_supports_retry(
    migration_database, ready_request
):
    engine, config = migration_database
    command.upgrade(config, "0007_project_plan_path")
    metadata = sa.MetaData()
    metadata.reflect(bind=engine)
    for table, column in (
        ("experiment_projects", "context"),
        ("experiment_projects", "creation_metadata"),
        ("experiment_preparations", "document"),
        ("experiment_plan_documents", "attributes"),
        ("runtime_messages", "attributes"),
    ):
        metadata.tables[table].c[column].type = sa.JSON()
    owner = PlanOwner(user_id="42", profile_id="prepared", conversation_id="chat")
    state = PreparationState(
        revision=7, phase="active", plan=ready_request.plan, project_id="project"
    )
    document = state.model_dump(mode="json")
    with engine.begin() as conn:
        conn.execute(
            metadata.tables["experiment_projects"]
            .insert()
            .values(
                project_id="project",
                owner_key=owner_key(owner),
                user_id=owner.user_id,
                profile_id=owner.profile_id,
                conversation_id=owner.conversation_id,
                cluster="cluster",
                current_revision=1,
                context=ready_request.plan.model_dump(mode="json"),
                context_sequence=3,
                creation_metadata={"command_id": "accepted", "preparation": document},
            )
        )
        conn.execute(
            metadata.tables["experiment_preparations"]
            .insert()
            .values(
                owner_key=owner_key(owner),
                revision=7,
                document=document,
                project_id="project",
            )
        )
        conn.execute(
            metadata.tables["experiment_plan_documents"]
            .insert()
            .values(
                kind="revision",
                document_id="old",
                owner_key=owner_key(owner),
                project_id="project",
                revision=1,
                attributes={"plan": ready_request.plan.model_dump(mode="json")},
            )
        )
        conn.execute(
            metadata.tables["runtime_messages"]
            .insert()
            .values(
                direction="out",
                id="old",
                attributes={},
                state="pending",
            )
        )
    command.upgrade(config, "head")
    assert_compact_schema(engine)
    store = SQLPreparationStore(engine)
    project = store.project(owner)
    assert project is not None
    assert project.creation_command_id == "accepted"
    assert project.plan == ready_request.plan
    assert project.plan_sequence == 7
    assert store.get(owner).model_dump(mode="json") == document
    assert store.project(owner.model_copy(update={"profile_id": "foreign"})) is None
    with engine.begin() as conn:
        assert (
            conn.scalar(sa.text("SELECT workflow_plan_path FROM experiment_projects"))
            == "public_data/workflowPLAN.json"
        )
        # Rerunning the body also covers recovery after MariaDB committed DDL
        # but the Alembic version update was interrupted.
        context = MigrationContext.configure(conn)
        with Operations.context(context):
            RETIRED.upgrade()
    assert_compact_schema(engine)
