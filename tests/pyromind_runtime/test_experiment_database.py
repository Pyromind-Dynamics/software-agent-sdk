"""Shared SQL storage contracts, legacy migration and offline configuration."""

import hashlib
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from pyromind_agent_server import plan_migrations
from pyromind_agent_server.plan_database import PROJECT_ROOT, resolve_database_url
from pyromind_runtime.domain.experiment_plans import (
    PlanOwner,
    PlanProposal,
    PlanRevision,
    WorkflowPlan,
)
from pyromind_runtime.infrastructure.sql_database import (
    create_database_engine,
    write_transaction,
)
from pyromind_runtime.infrastructure.sql_experiment_execution import (
    SQLExperimentExecutionStore,
    owner_key,
)
from pyromind_runtime.infrastructure.sql_experiment_plan_store import (
    SQLExperimentPlanStore,
)
from pyromind_runtime.infrastructure.sql_work_queue import now


TABLES = {
    "experiment_projects",
    "experiment_plan_documents",
    "experiment_task_runs",
    "agent_role_sessions",
    "runtime_work_items",
    "external_task_submissions",
    "runtime_messages",
}


def migrate(url, target="head"):
    config = Config()
    config.set_main_option("script_location", str(next(iter(plan_migrations.__path__))))
    config.attributes["database_url"] = url
    command.upgrade(config, target)
    engine = create_database_engine(url)
    try:
        with engine.connect() as conn:
            assert conn.scalar(
                sa.text("SELECT version_num FROM experiment_plan_schema_version")
            ) == ("0004_compact_storage" if target == "head" else target)
    finally:
        engine.dispose()


@pytest.fixture(params=["sqlite", "mariadb"])
def database_url(request, tmp_path):
    if request.param == "sqlite":
        url = f"sqlite:///{tmp_path / 'experiments.sqlite3'}"
    else:
        url = os.getenv("EXPERIMENT_DATABASE_TEST_URL")
        if not url:
            pytest.skip("requires a dedicated MariaDB database")
    migrate(url)
    return url


def test_selector_ignores_remote_url_and_resolves_paths_independent_of_cwd(
    tmp_path, monkeypatch
):
    environment = {
        "EXPERIMENT_DB_BACKEND": "sqlite",
        "EXPERIMENT_PLAN_DATABASE_URL": "mysql+pymysql://unused@unreachable/example",
    }
    first = resolve_database_url(environment)
    monkeypatch.chdir(tmp_path)
    assert resolve_database_url(environment) == first
    assert (
        resolve_database_url(
            {**environment, "EXPERIMENT_PLAN_DATABASE_URL": "invalid inactive URL"}
        )
        == first
    )
    assert first.database == str(
        PROJECT_ROOT / "workspace/experiments/experiments.sqlite3"
    )
    assert resolve_database_url({}) is None
    assert resolve_database_url(
        {"EXPERIMENT_PLAN_DATABASE_URL": "sqlite:///relative.db"}
    ).database == str(PROJECT_ROOT / "relative.db")
    remote = "mysql+pymysql://user@host:13306/database"
    assert (
        resolve_database_url({"EXPERIMENT_PLAN_DATABASE_URL": remote}).drivername
        == "mysql+pymysql"
    )
    with pytest.raises(ValueError, match="must be"):
        resolve_database_url({"EXPERIMENT_DB_BACKEND": "unsupported"})
    with pytest.raises(ValueError, match="requires"):
        resolve_database_url({"EXPERIMENT_DB_BACKEND": "mariadb"})


def test_startup_sqlite_does_not_source_credentials(tmp_path):
    forbidden = tmp_path / "credentials.env"
    forbidden.write_text("exit 88\n")
    prefix = (
        (PROJECT_ROOT / "start_inference.sh")
        .read_text()
        .split("export PYROMIND_HARNESS_BACKEND")[0]
    )
    environment = {**os.environ, "EXPERIMENT_PLAN_ENV_FILE": str(forbidden)}
    environment.pop("EXPERIMENT_DB_BACKEND", None)
    result = subprocess.run(
        ["bash", "-c", prefix + '\nprintf "%s" "$EXPERIMENT_DB_BACKEND"'],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and result.stdout == "sqlite"


def test_sqlite_engine_settings_and_persistent_file(tmp_path):
    path = tmp_path / "nested" / "experiments.db"
    engine = create_database_engine(f"sqlite:///{path}")
    with engine.connect() as conn:
        assert conn.scalar(sa.text("PRAGMA journal_mode")) == "wal"
        assert conn.scalar(sa.text("PRAGMA foreign_keys")) == 1
        assert conn.scalar(sa.text("PRAGMA busy_timeout")) == 5000
    engine.dispose()
    assert path.is_file()


def test_shared_concurrency_and_json_contract(database_url):
    engines = [create_database_engine(database_url) for _ in range(2)]
    stores = [SQLExperimentPlanStore(engine) for engine in engines]
    owner = PlanOwner(
        user_id="db-contract", conversation_id=uuid4().hex, profile_id="profile"
    )
    plan = WorkflowPlan.model_validate(
        {
            "name": "共享测试",
            "goal": "keep JSON intact",
            "stages": [{"id": "init", "name": "Init"}],
        }
    )
    draft = PlanProposal(
        proposal_id=uuid4().hex, base_revision=0, plan=plan, summary="初始"
    )
    try:
        stores[0].propose(owner, draft, source_proposal_id=None, initial=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            saved = list(
                pool.map(
                    lambda store: store.confirm(owner, draft.proposal_id, 0, "confirm"),
                    stores,
                )
            )
        assert saved[0] == saved[1]
        ledgers = [SQLExperimentExecutionStore(engine) for engine in engines]
        run = ledgers[0].request(
            owner, saved[0].project_id, 1, "task", "eval", "request"
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            claimed = [
                item
                for item in pool.map(lambda store: store.queue.claim(), ledgers)
                if item
            ]
        # The test database is dedicated; old completed work cannot be claimed.
        assert len(claimed) == 1
        item = claimed[0]
        ledgers[0].queue.finish(item["id"], item["generation"], {})
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(
                pool.map(
                    lambda store: store.ingest(
                        owner.conversation_id, "unbound", "kafka", {"status": "running"}
                    ),
                    ledgers,
                )
            )
        from pyromind_runtime.infrastructure.sql_experiment_execution import messages

        with engines[1].connect() as conn:
            assert (
                conn.scalar(
                    sa.select(sa.func.count())
                    .select_from(messages)
                    .where(
                        messages.c.direction == "in",
                        messages.c.id
                        == hashlib.sha256(
                            ("kafka:" + owner.conversation_id).encode()
                        ).hexdigest(),
                    )
                )
                == 1
            )

        # Different writers append metadata without replacing one another.
        def append(pair):
            engine, name = pair
            store = SQLExperimentExecutionStore(engine)
            with write_transaction(engine) as conn:
                row = store._run(conn, owner, run["id"])
                store._register_artifact(
                    conn,
                    row,
                    {
                        "name": name,
                        "uri": "/" + name,
                        "sha256": "a" * 64,
                        "metadata": {"中文": [1, None, True]},
                    },
                )

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(append, zip(engines, ["first", "second"])))
        restored = SQLExperimentExecutionStore(engines[1]).list(owner)[0]
        assert {item["name"] for item in restored["artifacts"]} == {"first", "second"}
        assert restored["artifacts"][0]["payload"] == {"中文": [1, None, True]}
        with pytest.raises(ValueError, match="changed"):
            with write_transaction(engines[1]) as conn:
                row = ledgers[1]._run(conn, owner, run["id"])
                ledgers[1]._register_artifact(
                    conn, row, {"name": "first", "uri": "/first", "sha256": "b" * 64}
                )
        assert set(sa.inspect(engines[0]).get_table_names()) == TABLES | {
            "experiment_plan_schema_version"
        }
    finally:
        for engine in engines:
            engine.dispose()


def seed_legacy(url):
    migrate(url, "0003_experiment_tasks")
    engine = create_database_engine(url)
    metadata = sa.MetaData()
    metadata.reflect(engine)
    # MariaDB reflects its JSON alias as LONGTEXT on some server versions.
    for table in metadata.tables.values():
        for column in table.columns:
            if column.name in {
                "payload",
                "evidence",
                "context",
                "task_spec",
                "result",
                "metrics",
                "proposal",
                "checkpoint",
                "session_ref",
            }:
                column.type = sa.JSON()
    owner = PlanOwner(
        user_id="legacy", conversation_id=uuid4().hex, profile_id="profile"
    )
    project_id = uuid4().hex
    plan = WorkflowPlan.model_validate(
        {
            "name": "历史方案",
            "goal": "基线",
            "stages": [{"id": "init", "name": "Init"}],
            "tasks": [{"id": "task", "stage_id": "init", "name": "baseline"}],
        }
    )
    first = PlanRevision(
        project_id=project_id, revision=1, plan=plan, confirmed_by=owner.user_id
    )
    draft = PlanProposal(
        proposal_id="pending", base_revision=1, plan=plan, summary="待确认"
    )
    confirmed = PlanProposal(
        proposal_id="confirmed",
        base_revision=0,
        plan=plan,
        summary="旧版",
        status="confirmed",
    )
    identity = owner_key(owner)
    with write_transaction(engine) as conn:
        conn.execute(
            metadata.tables["experiment_projects"]
            .insert()
            .values(
                project_id=project_id,
                owner_key=identity,
                **owner.model_dump(),
                current_revision=1,
                context_sequence=0,
            )
        )
        conn.execute(
            metadata.tables["experiment_plan_revisions"]
            .insert()
            .values(
                project_id=project_id, revision=1, payload=first.model_dump(mode="json")
            )
        )
        for proposal in (draft, confirmed):
            conn.execute(
                metadata.tables["experiment_plan_proposals"]
                .insert()
                .values(
                    proposal_id=proposal.proposal_id,
                    owner_key=identity,
                    status=proposal.status,
                    base_revision=proposal.base_revision,
                    payload=proposal.model_dump(mode="json"),
                    confirmation_command_id="original-confirm"
                    if proposal.status == "confirmed"
                    else None,
                    confirmed_revision=1 if proposal.status == "confirmed" else None,
                )
            )
        conn.execute(
            metadata.tables["experiment_task_runs"]
            .insert()
            .values(
                id="legacy-run",
                owner=identity,
                project_id=project_id,
                revision=1,
                task_id="task",
                role_id="eval",
                request_key=identity,
                state="waiting_platform",
                report_state="not_started",
                created_at=now(),
            )
        )
        conn.execute(
            metadata.tables["external_task_submissions"]
            .insert()
            .values(
                id="submission",
                run_id="legacy-run",
                owner=identity,
                operation_key="legacy",
                state="running",
                task_id="platform-old",
                request_fingerprint="a" * 64,
                evidence={},
                updated_at=now(),
            )
        )
        conn.execute(
            metadata.tables["experiment_artifacts"]
            .insert()
            .values(
                id="original-artifact",
                run_id="legacy-run",
                owner=identity,
                name="metrics",
                uri="/old/metrics.json",
                sha256="a" * 64,
                payload={"中文": [1, None]},
            )
        )
        conn.execute(
            metadata.tables["runtime_inbox"]
            .insert()
            .values(
                id="same-id",
                task_id="platform-old",
                source="kafka",
                state="pending",
                payload={"status": "running"},
                created_at=now(),
            )
        )
        conn.execute(
            metadata.tables["runtime_outbox"]
            .insert()
            .values(
                id="same-id",
                owner=identity,
                conversation_id=owner.conversation_id,
                state="pending",
                payload={"run_id": "legacy-run"},
            )
        )
    engine.dispose()
    return owner, first, draft


def assert_legacy_upgrade(url, expected):
    owner, first, draft = expected
    migrate(url)
    migrate(url)
    engine = create_database_engine(url)
    try:
        plans = SQLExperimentPlanStore(engine)
        view = plans.view(owner)
        assert view.current == first and view.pending_proposal == draft
        assert plans.confirm(owner, "confirmed", 0, "original-confirm") == first
        saved = plans.confirm(owner, "pending", 1, "next-confirm")
        assert saved.revision == 2 and plans.revision(owner, 1) == first
        store = SQLExperimentExecutionStore(engine)
        row = store.list(owner)[0]
        assert row["artifacts"][0] == {
            "id": "original-artifact",
            "owner": owner_key(owner),
            "run_id": "legacy-run",
            "name": "metrics",
            "uri": "/old/metrics.json",
            "sha256": "a" * 64,
            "payload": {"中文": [1, None]},
        }
        assert store.apply_observations() == 1
        assert any(item["id"] == "same-id" for item in store.pending_deliveries())
        store.acknowledge_delivery("same-id")
        assert not store.pending_deliveries()
        from pyromind_runtime.infrastructure.sql_experiment_execution import messages

        with engine.connect() as conn:
            assert (
                conn.scalar(
                    sa.select(sa.func.count())
                    .select_from(messages)
                    .where(messages.c.id == "same-id")
                )
                == 2
            )
        assert set(sa.inspect(engine).get_table_names()) == TABLES | {
            "experiment_plan_schema_version"
        }
    finally:
        engine.dispose()


@pytest.mark.parametrize("interrupted", [False, True])
def test_legacy_copy_verify_and_restartable_drop(tmp_path, monkeypatch, interrupted):
    url = f"sqlite:///{tmp_path / 'legacy.db'}"
    expected = seed_legacy(url)
    if interrupted:
        from alembic import op

        original = op.drop_table

        def fail_drop(name, *args, **kwargs):
            if name == "experiment_artifacts":
                raise RuntimeError("injected migration interruption")
            return original(name, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(op, "drop_table", fail_drop)
            with pytest.raises(RuntimeError, match="interruption"):
                migrate(url)
    assert_legacy_upgrade(url, expected)


def test_mariadb_legacy_migration():
    url = os.getenv("EXPERIMENT_MIGRATION_TEST_URL")
    if not url:
        pytest.skip("requires a separate empty MariaDB migration database")
    assert_legacy_upgrade(url, seed_legacy(url))


def test_copy_mismatch_preserves_legacy_sources(tmp_path, monkeypatch):
    from alembic import op
    from pyromind_runtime.infrastructure.sql_experiment_plan_store import documents

    url = f"sqlite:///{tmp_path / 'mismatch.db'}"
    seed_legacy(url)

    def fail_before_drop(*_args, **_kwargs):
        raise RuntimeError("interrupted before drop")

    with monkeypatch.context() as patch:
        patch.setattr(op, "drop_table", fail_before_drop)
        with pytest.raises(RuntimeError, match="before drop"):
            migrate(url)
    engine = create_database_engine(url)
    try:
        with write_transaction(engine) as conn:
            conn.execute(
                documents.update()
                .where(
                    documents.c.kind == "proposal", documents.c.document_id == "pending"
                )
                .values(attributes={"corrupted": True})
            )
        with pytest.raises(RuntimeError, match="verification failed"):
            migrate(url)
        assert {
            "experiment_plan_proposals",
            "experiment_plan_revisions",
            "experiment_artifacts",
            "runtime_inbox",
            "runtime_outbox",
        }.issubset(sa.inspect(engine).get_table_names())
    finally:
        engine.dispose()
