from __future__ import annotations

import hashlib

from sqlalchemy import Column, MetaData, String, Table

from pyromind_runtime.domain.experiment_plans import PlanOwner


metadata = MetaData()
projects = Table(
    "experiment_projects",
    metadata,
    Column("project_id", String(64), primary_key=True),
    Column("owner_key", String(64), nullable=False, unique=True),
    Column("user_id", String(255), nullable=False),
    Column("conversation_id", String(128), nullable=False),
    Column("cluster", String(255)),
    Column("profile_id", String(128)),
    Column("creation_command_id", String(128)),
    Column("workflow_plan_path", String(4096)),
)


def owner_key(owner: PlanOwner) -> str:
    return hashlib.sha256(owner.model_dump_json().encode()).hexdigest()
