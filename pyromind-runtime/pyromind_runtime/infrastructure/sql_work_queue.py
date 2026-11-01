"""Durable work admission, role serialization and fenced completion."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from pyromind_runtime.domain.persisted_payload import validate_persisted_payload
from pyromind_runtime.infrastructure.sql_database import (
    configure_engine,
    insert_if_absent,
    write_transaction,
)


metadata = sa.MetaData()
_TABLE_OPTIONS: dict[str, Any] = dict(
    mysql_engine="InnoDB", mysql_charset="utf8mb4", mysql_collate="utf8mb4_bin"
)
roles = sa.Table(
    "agent_role_sessions",
    metadata,
    sa.Column("lane", sa.String(64), primary_key=True),
    sa.Column("owner", sa.String(64), nullable=False),
    sa.Column("active_id", sa.String(64)),
    sa.Column("generation", sa.Integer, nullable=False, default=0),
    sa.Column("lease_until", sa.DateTime),
    sa.Column("session_ref", sa.JSON),
    **_TABLE_OPTIONS,
)
work = sa.Table(
    "runtime_work_items",
    metadata,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("owner", sa.String(64), nullable=False),
    sa.Column("lane", sa.String(64), nullable=False),
    sa.Column("dedup_key", sa.String(64), nullable=False),
    sa.Column("kind", sa.String(64), nullable=False),
    sa.Column("payload", sa.JSON, nullable=False),
    sa.Column("state", sa.String(32), nullable=False),
    sa.Column("priority", sa.Integer, nullable=False),
    sa.Column("created_at", sa.DateTime, nullable=False),
    sa.Column("available_at", sa.DateTime, nullable=False),
    sa.Column("lease_until", sa.DateTime),
    sa.Column("generation", sa.Integer, nullable=False),
    sa.Column("attempts", sa.Integer, nullable=False),
    sa.Column("checkpoint", sa.JSON, nullable=False),
    sa.UniqueConstraint("owner", "dedup_key"),
    sa.Index("ix_runtime_work_ready", "state", "available_at", "priority"),
    **_TABLE_OPTIONS,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class StaleLeaseError(ValueError):
    pass


class SQLWorkQueue:
    def __init__(self, engine: Engine):
        self.engine = configure_engine(engine)

    def enqueue(
        self,
        owner: str,
        lane: str,
        key: str,
        kind: str,
        payload: dict,
        priority: int = 100,
    ) -> str:
        with write_transaction(self.engine) as conn:
            return self.put(conn, owner, lane, key, kind, payload, priority)

    def put(
        self,
        conn: Connection,
        owner: str,
        lane: str,
        key: str,
        kind: str,
        payload: dict,
        priority: int = 100,
    ) -> str:
        validate_persisted_payload(payload)
        lane = digest(owner + ":" + lane)
        key = digest(key)
        # A deterministic identity makes callers recover an enqueue response loss.
        identity = digest(owner + ":" + key)
        # Ensure and lock the lane before checking admission. Concurrent writers
        # use database-native insert conflict handling rather than a read/insert race.
        values = {"lane": lane, "owner": owner, "generation": 0}
        insert_if_absent(conn, roles, values, "lane")
        conn.execute(
            sa.select(roles.c.lane).where(roles.c.lane == lane).with_for_update()
        ).one()
        found = (
            conn.execute(sa.select(work).where(work.c.id == identity).with_for_update())
            .mappings()
            .first()
        )
        if found:
            if (
                found["kind"] != kind
                or found["payload"] != payload
                or found["lane"] != lane
            ):
                raise ValueError("deduplication key reused with different work")
            return identity
        conn.execute(
            work.insert().values(
                id=identity,
                owner=owner,
                lane=lane,
                dedup_key=key,
                kind=kind,
                payload=payload,
                state="queued",
                priority=priority,
                created_at=now(),
                available_at=now(),
                generation=0,
                attempts=0,
                checkpoint={},
            )
        )
        return identity

    def claim(
        self, lease_seconds: int = 90, kinds: tuple[str, ...] | None = None
    ) -> dict | None:
        with write_transaction(self.engine) as conn:
            candidates = conn.execute(
                sa.select(work)
                .where(
                    work.c.state == "queued",
                    work.c.available_at <= now(),
                    work.c.kind.in_(kinds) if kinds is not None else sa.true(),
                )
                .order_by(work.c.priority, work.c.created_at, work.c.id)
                .with_for_update(skip_locked=True)
            ).mappings()
            for candidate in candidates:
                lane = (
                    conn.execute(
                        sa.select(roles)
                        .where(roles.c.lane == candidate["lane"])
                        .with_for_update(skip_locked=True)
                    )
                    .mappings()
                    .first()
                )
                # Expired work must be reconciled, never automatically redispatched.
                if not lane or lane["active_id"] not in {None, candidate["id"]}:
                    continue
                generation = lane["generation"] + 1
                lease = now() + timedelta(seconds=lease_seconds)
                claimed = conn.execute(
                    roles.update()
                    .where(
                        roles.c.lane == lane["lane"],
                        roles.c.generation == lane["generation"],
                        sa.or_(
                            roles.c.active_id.is_(None),
                            roles.c.active_id == candidate["id"],
                        ),
                    )
                    .values(
                        active_id=candidate["id"],
                        generation=generation,
                        lease_until=lease,
                    )
                )
                if claimed.rowcount != 1:
                    continue
                admitted = conn.execute(
                    work.update()
                    .where(work.c.id == candidate["id"], work.c.state == "queued")
                    .values(
                        state="running",
                        generation=generation,
                        lease_until=lease,
                        attempts=candidate["attempts"] + 1,
                    )
                )
                if admitted.rowcount != 1:
                    return None
                return {**candidate, "generation": generation, "state": "running"}
        return None

    def fenced(self, conn: Connection, identity: str, generation: int) -> dict:
        row = (
            conn.execute(sa.select(work).where(work.c.id == identity).with_for_update())
            .mappings()
            .one()
        )
        if (
            row["state"] != "running"
            or row["generation"] != generation
            or row["lease_until"] <= now()
        ):
            raise StaleLeaseError("work lease is no longer owned")
        return dict(row)

    def heartbeat(self, identity: str, generation: int, seconds: int = 90) -> None:
        with write_transaction(self.engine) as conn:
            row = self.fenced(conn, identity, generation)
            lease = now() + timedelta(seconds=seconds)
            conn.execute(
                work.update().where(work.c.id == identity).values(lease_until=lease)
            )
            conn.execute(
                roles.update()
                .where(roles.c.lane == row["lane"])
                .values(lease_until=lease)
            )

    def finish(
        self, identity: str, generation: int, checkpoint: dict, state: str = "completed"
    ) -> None:
        validate_persisted_payload(checkpoint)
        if state not in {
            "completed",
            "failed",
            "waiting_authorization",
            "reconciliation_required",
        }:
            raise ValueError("invalid completion state")
        with write_transaction(self.engine) as conn:
            row = self.fenced(conn, identity, generation)
            conn.execute(
                work.update()
                .where(work.c.id == identity)
                .values(
                    state=state,
                    checkpoint={**row["checkpoint"], **checkpoint},
                    lease_until=None,
                )
            )
            conn.execute(
                roles.update()
                .where(roles.c.lane == row["lane"])
                .values(
                    active_id=identity if state == "reconciliation_required" else None,
                    lease_until=None,
                )
            )

    def expired(self) -> list[dict]:
        with self.engine.connect() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    sa.select(work).where(
                        work.c.state == "running", work.c.lease_until < now()
                    )
                ).mappings()
            ]

    def reconcile_expired(self, identity: str, safe_to_retry: bool) -> None:
        with write_transaction(self.engine) as conn:
            row = (
                conn.execute(
                    sa.select(work).where(work.c.id == identity).with_for_update()
                )
                .mappings()
                .one()
            )
            if row["state"] != "running" or row["lease_until"] > now():
                raise ValueError("work has not expired")
            conn.execute(
                work.update()
                .where(work.c.id == identity)
                .values(
                    state="queued" if safe_to_retry else "reconciliation_required",
                    generation=row["generation"] + 1,
                    lease_until=None,
                )
            )
            conn.execute(
                roles.update()
                .where(roles.c.lane == row["lane"])
                .values(active_id=identity, lease_until=None)
            )

    def list(self, owner: str) -> list[dict]:
        with self.engine.connect() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    sa.select(work)
                    .where(work.c.owner == owner)
                    .order_by(work.c.created_at)
                ).mappings()
            ]

    def retry(
        self, owner: str, identity: str, *, previous_execution_stopped: bool
    ) -> None:
        if not previous_execution_stopped:
            raise ValueError("previous execution must be proved stopped before retry")
        with write_transaction(self.engine) as conn:
            row = (
                conn.execute(
                    sa.select(work)
                    .where(work.c.id == identity, work.c.owner == owner)
                    .with_for_update()
                )
                .mappings()
                .first()
            )
            if not row:
                raise PermissionError("work item not found")
            if row["kind"] == "submit_execution":
                raise ValueError(
                    "external submissions cannot be retried through the work queue"
                )
            if row["state"] not in {
                "failed",
                "reconciliation_required",
                "waiting_authorization",
            }:
                raise ValueError("work item is not retryable")
            if row["attempts"] >= 3:
                raise ValueError("work item retry limit reached")
            lane = (
                conn.execute(
                    sa.select(roles)
                    .where(roles.c.lane == row["lane"])
                    .with_for_update()
                )
                .mappings()
                .one()
            )
            if lane["active_id"] not in {None, identity}:
                raise ValueError("role is processing another work item")
            delay = (5, 20, 60)[min(max(row["attempts"] - 1, 0), 2)]
            conn.execute(
                work.update()
                .where(work.c.id == identity)
                .values(
                    state="queued",
                    available_at=now() + timedelta(seconds=delay),
                    generation=row["generation"] + 1,
                    lease_until=None,
                )
            )
            conn.execute(
                roles.update()
                .where(roles.c.lane == row["lane"])
                .values(active_id=None, lease_until=None)
            )
