from __future__ import annotations

import os

from pyromind_runtime.infrastructure.sql_experiment_plan_store import (
    SQLExperimentPlanStore,
)
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url


def build_experiment_plan_store() -> SQLExperimentPlanStore | None:
    configured = os.getenv("EXPERIMENT_PLAN_DATABASE_URL")
    if not configured:
        return None
    url = make_url(configured)
    if url.drivername != "mysql+pymysql":
        raise ValueError("EXPERIMENT_PLAN_DATABASE_URL requires mysql+pymysql")
    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=1800,
        hide_parameters=True,
        connect_args={"connect_timeout": 10, "read_timeout": 30, "write_timeout": 30},
    )
    return SQLExperimentPlanStore(engine)
