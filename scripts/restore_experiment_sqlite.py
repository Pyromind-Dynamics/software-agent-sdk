"""Recreate/upgrade the local experiment database without deleting existing data.

Run from the repository: uv run python scripts/restore_experiment_sqlite.py
Stop the server before repairing a deleted workspace, then restart it afterwards.
Deleted conversation history and database rows require a separate backup.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from pyromind_agent_server import plan_migrations
from pyromind_agent_server.plan_database import resolve_database_url
from pyromind_runtime.infrastructure.sql_database import create_database_engine
from pyromind_runtime.infrastructure.sql_experiment_execution import (
    metadata as execution_metadata,
)
from pyromind_runtime.infrastructure.sql_experiment_plan_store import (
    metadata as plan_metadata,
)
from pyromind_runtime.infrastructure.sql_preparation import (
    metadata as preparation_metadata,
)
from sqlalchemy import inspect


def restore(path: str | None = None) -> tuple[Path, Path | None, list[str]]:
    environment = {**os.environ, "EXPERIMENT_DB_BACKEND": "sqlite"}
    if path is not None:
        environment["EXPERIMENT_SQLITE_PATH"] = path
    url = resolve_database_url(environment)
    assert url is not None and url.database is not None
    target = Path(url.database)
    backup = None
    if target.is_file():
        with sqlite3.connect(target) as source:
            if source.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise RuntimeError("SQLite integrity check failed; preserve the file")
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            backup = target.with_name(f"{target.name}.{stamp}.bak")
            with sqlite3.connect(backup) as destination:
                source.backup(destination)

    config = Config()
    config.set_main_option("script_location", str(next(iter(plan_migrations.__path__))))
    config.attributes["database_url"] = url
    command.upgrade(config, "head")
    engine = create_database_engine(url)
    try:
        inspector = inspect(engine)
        for metadata in (plan_metadata, execution_metadata, preparation_metadata):
            for table in metadata.tables.values():
                actual = {
                    column["name"] for column in inspector.get_columns(table.name)
                }
                missing = set(table.columns.keys()) - actual
                if missing:
                    raise RuntimeError(
                        f"{table.name}: missing columns {sorted(missing)}"
                    )
        with engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA integrity_check").all() == [
                ("ok",)
            ]
            if connection.exec_driver_sql("PRAGMA foreign_key_check").first():
                raise RuntimeError("SQLite foreign key check failed")
            version = connection.exec_driver_sql(
                "SELECT version_num FROM experiment_plan_schema_version"
            ).scalar_one()
            if version != ScriptDirectory.from_config(config).get_current_head():
                raise RuntimeError("Database migration version does not match head")
        return target, backup, sorted(inspector.get_table_names())
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path", help="SQLite path; relative paths use repository root"
    )
    args = parser.parse_args()
    target, backup, tables = restore(args.path)
    print(f"Database: {target}")
    print(f"Backup: {backup or '(new database)'}")
    print(f"Verified tables: {', '.join(tables)}")
    print("Restart the server if the workspace was deleted while it was running.")


if __name__ == "__main__":
    main()
