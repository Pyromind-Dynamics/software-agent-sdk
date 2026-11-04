import runpy
import sqlite3

from pyromind_agent_server.plan_database import PROJECT_ROOT


def test_restore_deleted_workspace_and_preserve_existing_data(tmp_path):
    restore = runpy.run_path(
        str(PROJECT_ROOT / "scripts/restore_experiment_sqlite.py")
    )["restore"]
    target = tmp_path / "workspace/experiments/experiments.sqlite3"
    path, backup, tables = restore(str(target))
    assert path == target
    assert backup is None
    assert "experiment_preparations" in tables
    with sqlite3.connect(target) as connection:
        connection.execute(
            "INSERT INTO experiment_preparations (owner_key, revision, document) "
            "VALUES (?, ?, ?)",
            ("test-owner", 1, "{}"),
        )

    _, backup, repeated_tables = restore(str(target))
    assert repeated_tables == tables
    assert backup is not None and backup.is_file()
    for database in (target, backup):
        with sqlite3.connect(database) as connection:
            assert connection.execute(
                "SELECT revision FROM experiment_preparations WHERE owner_key = ?",
                ("test-owner",),
            ).fetchone() == (1,)
