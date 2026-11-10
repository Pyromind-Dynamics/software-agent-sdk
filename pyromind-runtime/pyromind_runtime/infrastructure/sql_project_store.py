from __future__ import annotations

import hashlib

from sqlalchemy import JSON, Column, Integer, MetaData, String, Table

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
    Column("current_revision", Integer, nullable=False),
    Column("context", JSON),
    Column("creation_metadata", JSON),
    Column("context_sequence", Integer, nullable=False, server_default="0"),
)


def owner_key(owner: PlanOwner) -> str:
    return hashlib.sha256(owner.model_dump_json().encode()).hexdigest()
