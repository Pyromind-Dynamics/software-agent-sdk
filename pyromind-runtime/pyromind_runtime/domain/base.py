from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict


def utc_now() -> datetime:
    return datetime.now(UTC)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
