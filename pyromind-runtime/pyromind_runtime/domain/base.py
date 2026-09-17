from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict


def utc_now() -> datetime:
    return datetime.now(UTC)


def as_utc(value: datetime) -> datetime:
    """Return ``value`` as an aware UTC datetime.

    Releases predating the ``updated_at`` backfill persisted some timestamps
    without an offset. Reading those naive values back and comparing them with
    aware instants raises ``TypeError``, so normalize every timestamp that
    enters a domain model.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


type AwareDatetime = Annotated[datetime, AfterValidator(as_utc)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
