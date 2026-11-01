from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pyromind_runtime.infrastructure.sql_database import create_database_engine
from pyromind_runtime.infrastructure.sql_experiment_plan_store import (
    SQLExperimentPlanStore,
)
from sqlalchemy.engine import URL, make_url


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_database_url(environment: Mapping[str, str] | None = None) -> URL | None:
    env = os.environ if environment is None else environment
    backend = env.get("EXPERIMENT_DB_BACKEND", "").strip().lower()
    configured = env.get("EXPERIMENT_PLAN_DATABASE_URL", "").strip()
    if not backend:
        if not configured:
            return None  # Legacy callers may leave experiments disabled.
        url = make_url(configured)
        backend = "sqlite" if url.drivername == "sqlite" else "mariadb"
    if backend == "sqlite":
        configured_url = (
            make_url(configured) if configured.startswith("sqlite:") else None
        )
        path = (
            env.get("EXPERIMENT_SQLITE_PATH")
            or (
                configured_url.database
                if configured_url and configured_url.drivername == "sqlite"
                else None
            )
            or "workspace/experiments/experiments.sqlite3"
        )
        if path == ":memory:":
            raise ValueError("experiment database requires a persistent SQLite file")
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = PROJECT_ROOT / target
        return URL.create("sqlite", database=str(target.resolve()))
    if backend != "mariadb":
        raise ValueError("EXPERIMENT_DB_BACKEND must be sqlite or mariadb")
    if not configured:
        raise ValueError("MariaDB requires EXPERIMENT_PLAN_DATABASE_URL")
    url = make_url(configured)
    if url.drivername not in {"mysql+pymysql", "mariadb+pymysql"}:
        raise ValueError("MariaDB requires a PyMySQL database URL")
    return url


def build_experiment_plan_store() -> SQLExperimentPlanStore | None:
    url = resolve_database_url()
    return SQLExperimentPlanStore(create_database_engine(url)) if url else None
