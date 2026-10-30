"""Transactional execution ledger; no platform or model calls occur inside SQL."""

from __future__ import annotations

import json
from contextlib import contextmanager
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from pyromind_runtime.domain.experiment_plans import PlanOwner
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload
from pyromind_runtime.infrastructure.sql_experiment_plan_store import projects
from pyromind_runtime.infrastructure.sql_work_queue import (
    _TABLE_OPTIONS,
    SQLWorkQueue,
    digest,
    metadata,
    now,
)


runs = sa.Table(
    "experiment_task_runs",
    metadata,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("owner", sa.String(64), nullable=False),
    sa.Column("project_id", sa.String(64), nullable=False),
    sa.Column("revision", sa.Integer, nullable=False),
    sa.Column("task_id", sa.String(128), nullable=False),
    sa.Column("role_id", sa.String(64), nullable=False),
    sa.Column("active_key", sa.String(64), unique=True),
    sa.Column("request_key", sa.String(64), nullable=False, unique=True),
    sa.Column("state", sa.String(32), nullable=False),
    sa.Column("report_state", sa.String(32), nullable=False),
    sa.Column("proposal", sa.JSON),
    sa.Column("fingerprint", sa.String(64)),
    sa.Column("decision_id", sa.String(128)),
    sa.Column("decision", sa.String(16)),
    sa.Column("metrics", sa.JSON),
    sa.Column("created_at", sa.DateTime, nullable=False),
    **_TABLE_OPTIONS,
)
submissions = sa.Table(
    "external_task_submissions",
    metadata,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("run_id", sa.String(64), nullable=False, unique=True),
    sa.Column("owner", sa.String(64), nullable=False),
    sa.Column("state", sa.String(32), nullable=False),
    sa.Column("task_id", sa.String(128), unique=True),
    sa.Column("request_fingerprint", sa.String(64), nullable=False),
    sa.Column("evidence", sa.JSON, nullable=False),
    sa.Column("updated_at", sa.DateTime, nullable=False),
    **_TABLE_OPTIONS,
)
inbox = sa.Table(
    "runtime_inbox",
    metadata,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("task_id", sa.String(128), nullable=False),
    sa.Column("source", sa.String(32), nullable=False),
    sa.Column("payload", sa.JSON, nullable=False),
    sa.Column("state", sa.String(32), nullable=False),
    sa.Column("created_at", sa.DateTime, nullable=False),
    **_TABLE_OPTIONS,
)
artifacts = sa.Table(
    "experiment_artifacts",
    metadata,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("run_id", sa.String(64), nullable=False),
    sa.Column("owner", sa.String(64), nullable=False),
    sa.Column("name", sa.String(255), nullable=False),
    sa.Column("uri", sa.Text, nullable=False),
    sa.Column("sha256", sa.String(64), nullable=False),
    sa.Column("payload", sa.JSON, nullable=False),
    sa.UniqueConstraint("run_id", "name"),
    **_TABLE_OPTIONS,
)
outbox = sa.Table(
    "runtime_outbox",
    metadata,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("owner", sa.String(64), nullable=False),
    sa.Column("conversation_id", sa.String(128), nullable=False),
    sa.Column("payload", sa.JSON, nullable=False),
    sa.Column("state", sa.String(32), nullable=False),
    **_TABLE_OPTIONS,
)


def owner_key(owner: PlanOwner) -> str:
    return digest(owner.model_dump_json())


def fingerprint(value: dict) -> str:
    return digest(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    )


class SQLExperimentExecutionStore:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.queue = SQLWorkQueue(engine)

    @contextmanager
    def _admission(self):
        try:
            with self.engine.begin() as conn:
                yield conn
        except IntegrityError as exc:
            raise ValueError(
                "execution request conflicts with an active task or request"
            ) from exc

    def _run(self, conn, owner: PlanOwner, run_id: str) -> dict:
        row = (
            conn.execute(
                sa.select(runs)
                .where(runs.c.id == run_id, runs.c.owner == owner_key(owner))
                .with_for_update()
            )
            .mappings()
            .first()
        )
        if not row:
            raise PermissionError("execution not found")
        return dict(row)

    def changed(self, conn, owner: PlanOwner, run_id: str, suffix: str) -> None:
        identity = digest(run_id + ":" + suffix)
        if not conn.execute(
            sa.select(outbox.c.id).where(outbox.c.id == identity)
        ).first():
            conn.execute(
                outbox.insert().values(
                    id=identity,
                    owner=owner_key(owner),
                    conversation_id=owner.conversation_id,
                    state="pending",
                    payload={"run_id": run_id},
                )
            )

    def request(
        self,
        owner: PlanOwner,
        project: str,
        revision: int,
        task: str,
        role: str,
        request_id: str,
    ) -> dict:
        key = digest(owner_key(owner) + ":" + request_id)
        active = digest(owner_key(owner) + ":" + project + ":" + task)
        with self._admission() as conn:
            previous = (
                conn.execute(sa.select(runs).where(runs.c.request_key == key))
                .mappings()
                .first()
            )
            if previous:
                if (previous["task_id"], previous["revision"], previous["role_id"]) != (
                    task,
                    revision,
                    role,
                ):
                    raise ValueError("request ID reused with different execution")
                return dict(previous)
            identity = uuid4().hex
            conn.execute(
                runs.insert().values(
                    id=identity,
                    owner=owner_key(owner),
                    project_id=project,
                    revision=revision,
                    task_id=task,
                    role_id=role,
                    active_key=active,
                    request_key=key,
                    state="queued",
                    report_state="not_started",
                    created_at=now(),
                )
            )
            self.queue.put(
                conn,
                owner_key(owner),
                "role:" + role,
                "prepare:" + identity,
                "prepare_execution",
                {"run_id": identity, "owner": owner.model_dump(mode="json")},
            )
            self.changed(conn, owner, identity, "created")
            return self._run(conn, owner, identity)

    def list(self, owner: PlanOwner) -> list[dict]:
        with self.engine.connect() as conn:
            result = []
            for row in conn.execute(
                sa.select(runs)
                .where(runs.c.owner == owner_key(owner))
                .order_by(runs.c.created_at)
            ).mappings():
                value = dict(row)
                value["artifacts"] = [
                    dict(a)
                    for a in conn.execute(
                        sa.select(artifacts).where(artifacts.c.run_id == row["id"])
                    ).mappings()
                ]
                value["submission"] = (
                    conn.execute(
                        sa.select(submissions).where(submissions.c.run_id == row["id"])
                    )
                    .mappings()
                    .first()
                )
                if value["submission"]:
                    value["submission"] = dict(value["submission"])
                result.append(value)
            return result

    def propose(self, owner: PlanOwner, run_id: str, proposal: dict) -> dict:
        validate_persisted_payload(proposal)
        with self.engine.begin() as conn:
            row = self._run(conn, owner, run_id)
            if row["state"] not in {"queued", "preparing", "awaiting_confirmation"}:
                raise ValueError("execution no longer accepts proposals")
            fp = fingerprint(proposal)
            conn.execute(
                runs.update()
                .where(runs.c.id == run_id)
                .values(
                    proposal=proposal, fingerprint=fp, state="awaiting_confirmation"
                )
            )
            self.changed(conn, owner, run_id, fp)
            return self._run(conn, owner, run_id)

    def decide(
        self,
        owner: PlanOwner,
        run_id: str,
        command_id: str,
        revision: int,
        fingerprint: str,
        decision: str,
    ) -> dict:
        with self.engine.begin() as conn:
            row = self._run(conn, owner, run_id)
            if row["revision"] != revision or row["fingerprint"] != fingerprint:
                raise ValueError("execution proposal changed")
            if row["decision_id"] == command_id:
                if row["decision"] != decision:
                    raise ValueError(
                        "confirmation command reused with a different decision"
                    )
                return row
            existing = (
                conn.execute(
                    sa.select(submissions).where(submissions.c.run_id == run_id)
                )
                .mappings()
                .first()
            )
            if existing and existing["evidence"].get("command_id") == command_id:
                return row
            if decision == "allow":
                current = conn.execute(
                    sa.select(projects.c.current_revision)
                    .where(
                        projects.c.project_id == row["project_id"],
                        projects.c.owner_key == owner_key(owner),
                    )
                    .with_for_update()
                ).scalar_one_or_none()
                if current != revision:
                    raise ValueError("plan changed; prepare a new execution proposal")
            if row["state"] != "awaiting_confirmation":
                raise ValueError("execution is not awaiting confirmation")
            if decision == "deny":
                conn.execute(
                    runs.update()
                    .where(runs.c.id == run_id)
                    .values(state="rejected", active_key=None)
                )
            elif decision == "allow":
                conn.execute(
                    submissions.insert().values(
                        id=uuid4().hex,
                        run_id=run_id,
                        owner=owner_key(owner),
                        state="prepared",
                        request_fingerprint=fingerprint,
                        evidence={
                            "command_id": command_id,
                            "confirmed_by": owner.user_id,
                        },
                        updated_at=now(),
                    )
                )
                conn.execute(
                    runs.update()
                    .where(runs.c.id == run_id)
                    .values(state="queued_submission")
                )
                self.queue.put(
                    conn,
                    owner_key(owner),
                    "submission:" + run_id,
                    "submit:" + run_id,
                    "submit_execution",
                    {"run_id": run_id, "owner": owner.model_dump(mode="json")},
                )
            else:
                raise ValueError("invalid decision")
            conn.execute(
                runs.update()
                .where(runs.c.id == run_id)
                .values(decision_id=command_id, decision=decision)
            )
            self.changed(conn, owner, run_id, command_id)
            return self._run(conn, owner, run_id)

    def begin_submission(
        self, owner: PlanOwner, run_id: str, work_id: str, generation: int
    ) -> dict:
        with self.engine.begin() as conn:
            self.queue.fenced(conn, work_id, generation)
            row = self._run(conn, owner, run_id)
            sub = (
                conn.execute(
                    sa.select(submissions)
                    .where(submissions.c.run_id == run_id)
                    .with_for_update()
                )
                .mappings()
                .one()
            )
            if sub["state"] != "prepared":
                raise ValueError("submission cannot be replayed")
            if row["state"] != "queued_submission":
                raise ValueError("execution is no longer eligible for submission")
            conn.execute(
                submissions.update()
                .where(submissions.c.run_id == run_id)
                .values(state="submitting", updated_at=now())
            )
            conn.execute(
                runs.update().where(runs.c.id == run_id).values(state="submitting")
            )
            return row

    def submission_result(
        self, owner: PlanOwner, run_id: str, task_id: str | None
    ) -> None:
        with self.engine.begin() as conn:
            self._run(conn, owner, run_id)
            state = "waiting_platform" if task_id else "submission_uncertain"
            conn.execute(
                submissions.update()
                .where(submissions.c.run_id == run_id)
                .values(task_id=task_id, state=state, updated_at=now())
            )
            conn.execute(runs.update().where(runs.c.id == run_id).values(state=state))
            self.changed(conn, owner, run_id, state)

    def ingest(self, identity: str, task_id: str, source: str, payload: dict) -> None:
        validate_persisted_payload(payload)
        with self.engine.begin() as conn:
            identity = digest(source + ":" + identity)
            if conn.execute(
                sa.select(inbox.c.id).where(inbox.c.id == identity)
            ).first():
                return
            conn.execute(
                inbox.insert().values(
                    id=identity,
                    task_id=task_id,
                    source=source,
                    payload=payload,
                    state="pending",
                    created_at=now(),
                )
            )

    def recover_submissions(self) -> int:
        # No evidence of an API response means acceptance is uncertain, not failed.
        with self.engine.begin() as conn:
            ids = list(
                conn.execute(
                    sa.select(submissions.c.run_id).where(
                        submissions.c.state == "submitting"
                    )
                ).scalars()
            )
            if ids:
                conn.execute(
                    submissions.update()
                    .where(submissions.c.run_id.in_(ids))
                    .values(state="submission_uncertain")
                )
                conn.execute(
                    runs.update()
                    .where(runs.c.id.in_(ids))
                    .values(state="submission_uncertain")
                )
            return len(ids)

    def apply_observations(self) -> int:
        terminal = {"succeeded", "failed", "terminated", "stopped"}
        count = 0
        with self.engine.begin() as conn:
            events = conn.execute(
                sa.select(inbox)
                .where(inbox.c.state == "pending")
                .with_for_update(skip_locked=True)
            ).mappings()
            for event in events:
                sub = (
                    conn.execute(
                        sa.select(submissions)
                        .where(submissions.c.task_id == event["task_id"])
                        .with_for_update()
                    )
                    .mappings()
                    .first()
                )
                if not sub:
                    continue
                status = str(event["payload"].get("status", "")).lower()
                if status not in terminal | {"running", "pending"}:
                    conn.execute(
                        inbox.update()
                        .where(inbox.c.id == event["id"])
                        .values(state="invalid")
                    )
                    continue
                if sub["state"] in terminal and status != sub["state"]:
                    if status in terminal:
                        conn.execute(
                            inbox.update()
                            .where(inbox.c.id == event["id"])
                            .values(state="conflict")
                        )
                        continue
                else:
                    conn.execute(
                        submissions.update()
                        .where(submissions.c.id == sub["id"])
                        .values(state=status, updated_at=now())
                    )
                    if status in terminal:
                        conn.execute(
                            runs.update()
                            .where(runs.c.id == sub["run_id"])
                            .values(
                                state="waiting_artifacts"
                                if status == "succeeded"
                                else "failed",
                                active_key=None
                                if status != "succeeded"
                                else runs.c.active_key,
                            )
                        )
                conn.execute(
                    inbox.update()
                    .where(inbox.c.id == event["id"])
                    .values(state="applied")
                )
                count += 1
        return count

    def register_artifacts(
        self, owner: PlanOwner, run_id: str, verified: list[dict], metrics: dict
    ) -> None:
        """Accept only results verified by an application-owned artifact reader."""
        validate_persisted_payload(verified)
        validate_persisted_payload(metrics)
        with self.engine.begin() as conn:
            row = self._run(conn, owner, run_id)
            if row["state"] != "waiting_artifacts":
                raise ValueError("execution is not awaiting verified artifacts")
            expected = set(row["proposal"].get("expected_artifacts", []))
            by_name = {item["name"]: item for item in verified}
            if set(by_name) != expected or len(by_name) != len(verified):
                raise ValueError("verified artifacts do not match frozen manifest")
            for item in verified:
                checksum = item["sha256"]
                if len(checksum) != 64 or any(
                    c not in "0123456789abcdef" for c in checksum
                ):
                    raise ValueError("invalid artifact digest")
                conn.execute(
                    artifacts.insert().values(
                        id=uuid4().hex,
                        owner=owner_key(owner),
                        run_id=run_id,
                        name=item["name"],
                        uri=item["uri"],
                        sha256=checksum,
                        payload=item.get("metadata", {}),
                    )
                )
            conn.execute(
                runs.update()
                .where(runs.c.id == run_id)
                .values(
                    state="succeeded",
                    report_state="queued",
                    metrics=metrics,
                    active_key=None,
                )
            )
            self.queue.put(
                conn,
                owner_key(owner),
                "role:" + row["role_id"],
                "report:" + run_id,
                "generate_report",
                {"run_id": run_id, "owner": owner.model_dump(mode="json")},
            )
            self.changed(conn, owner, run_id, "artifacts")

    def pending_deliveries(self) -> list[dict]:
        with self.engine.connect() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    sa.select(outbox).where(outbox.c.state == "pending")
                ).mappings()
            ]

    def acknowledge_delivery(self, identity: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                outbox.update().where(outbox.c.id == identity).values(state="delivered")
            )

    def cancel(self, owner: PlanOwner, run_id: str) -> dict:
        from pyromind_runtime.infrastructure.sql_work_queue import work

        with self.engine.begin() as conn:
            row = self._run(conn, owner, run_id)
            if row["state"] in {
                "cancelled",
                "succeeded",
                "failed",
                "rejected",
                "abandoned",
            }:
                return row
            sub = (
                conn.execute(
                    sa.select(submissions)
                    .where(submissions.c.run_id == run_id)
                    .with_for_update()
                )
                .mappings()
                .first()
            )
            # Unknown remote acceptance is never proof of termination.
            if sub and sub["state"] in {"submitting", "submission_uncertain"}:
                raise ValueError(
                    "submission acceptance is uncertain; reconcile or abandon it"
                )
            conn.execute(
                work.update()
                .where(
                    work.c.owner == owner_key(owner),
                    work.c.payload["run_id"].as_string() == run_id,
                    work.c.state == "queued",
                )
                .values(state="cancelled")
            )
            active_work = conn.execute(
                sa.select(work.c.id).where(
                    work.c.owner == owner_key(owner),
                    work.c.payload["run_id"].as_string() == run_id,
                    work.c.state == "running",
                )
            ).first()
            remote = sub and sub["task_id"]
            state = "cancel_requested" if remote or active_work else "cancelled"
            conn.execute(
                runs.update()
                .where(runs.c.id == run_id)
                .values(
                    state=state,
                    active_key=runs.c.active_key
                    if state == "cancel_requested"
                    else None,
                )
            )
            if state == "cancel_requested":
                self.queue.put(
                    conn,
                    owner_key(owner),
                    "control:" + run_id,
                    "cancel:" + run_id,
                    "cancel_execution",
                    {"run_id": run_id, "owner": owner.model_dump(mode="json")},
                    0,
                )
            self.changed(conn, owner, run_id, state)
            return self._run(conn, owner, run_id)

    def abandon(self, owner: PlanOwner, run_id: str, command_id: str) -> dict:
        with self.engine.begin() as conn:
            row = self._run(conn, owner, run_id)
            if row["state"] == "abandoned":
                return row
            if row["state"] != "submission_uncertain":
                raise ValueError("only uncertain submissions can be abandoned")
            sub = (
                conn.execute(
                    sa.select(submissions)
                    .where(submissions.c.run_id == run_id)
                    .with_for_update()
                )
                .mappings()
                .one()
            )
            evidence = {
                **sub["evidence"],
                "resolution": {
                    "command_id": command_id,
                    "action": "abandon",
                    "user_id": owner.user_id,
                    "resolved_at": now().isoformat(),
                    "remote_may_be_running": True,
                },
            }
            conn.execute(
                submissions.update()
                .where(submissions.c.id == sub["id"])
                .values(evidence=evidence)
            )
            conn.execute(
                runs.update()
                .where(runs.c.id == run_id)
                .values(state="abandoned", active_key=None)
            )
            self.changed(conn, owner, run_id, "abandoned")
            return self._run(conn, owner, run_id)

    def resolve(
        self,
        owner: PlanOwner,
        run_id: str,
        task_id: str,
        command_id: str,
        verified_evidence: dict,
    ) -> dict:
        validate_persisted_payload(verified_evidence)
        with self.engine.begin() as conn:
            row = self._run(conn, owner, run_id)
            sub = (
                conn.execute(
                    sa.select(submissions)
                    .where(submissions.c.run_id == run_id)
                    .with_for_update()
                )
                .mappings()
                .one()
            )
            resolution = sub["evidence"].get("resolution", {})
            if resolution.get("command_id") == command_id:
                if sub["task_id"] != task_id:
                    raise ValueError(
                        "resolution command cannot bind a different platform task"
                    )
                return row
            if row["state"] != "submission_uncertain":
                raise ValueError("execution is not awaiting submission reconciliation")
            evidence = {
                **sub["evidence"],
                "resolution": {
                    "command_id": command_id,
                    "action": "bind",
                    "user_id": owner.user_id,
                    "resolved_at": now().isoformat(),
                    "verification": verified_evidence,
                },
            }
            conn.execute(
                submissions.update()
                .where(submissions.c.id == sub["id"])
                .values(
                    task_id=task_id,
                    state="waiting_platform",
                    evidence=evidence,
                    updated_at=now(),
                )
            )
            conn.execute(
                runs.update()
                .where(runs.c.id == run_id)
                .values(state="waiting_platform")
            )
            self.changed(conn, owner, run_id, "resolved")
            return self._run(conn, owner, run_id)

    def role_scope(self, owner: PlanOwner, run_id: str, role: str, scope: dict) -> dict:
        with self.engine.begin() as conn:
            item = self.queue.fenced(conn, scope["work_id"], scope["generation"])
            row = self._run(conn, owner, run_id)
            if (
                row["role_id"] != role
                or item["owner"] != owner_key(owner)
                or item["payload"].get("run_id") != run_id
            ):
                raise PermissionError("role work belongs to a different execution")
            return row

    def propose_from_role(
        self, owner: PlanOwner, run_id: str, role: str, scope: dict, proposal: dict
    ) -> dict:
        validate_persisted_payload(proposal)
        with self.engine.begin() as conn:
            item = self.queue.fenced(conn, scope["work_id"], scope["generation"])
            row = self._run(conn, owner, run_id)
            if (
                row["role_id"] != role
                or item["owner"] != owner_key(owner)
                or item["payload"].get("run_id") != run_id
                or item["kind"] != "prepare_execution"
            ):
                raise PermissionError("role cannot change this execution")
            if row["state"] not in {"queued", "preparing", "awaiting_confirmation"}:
                raise ValueError("execution no longer accepts proposals")
            fp = fingerprint(proposal)
            conn.execute(
                runs.update()
                .where(runs.c.id == run_id)
                .values(
                    proposal=proposal, fingerprint=fp, state="awaiting_confirmation"
                )
            )
            self.changed(conn, owner, run_id, fp)
            return self._run(conn, owner, run_id)

    def save_work_checkpoint(self, identity: str, generation: int, value: dict) -> None:
        from pyromind_runtime.infrastructure.sql_work_queue import work

        validate_persisted_payload(value)
        with self.engine.begin() as conn:
            self.queue.fenced(conn, identity, generation)
            conn.execute(
                work.update().where(work.c.id == identity).values(checkpoint=value)
            )

    def resume_authorized(self, owner: PlanOwner) -> None:
        from pyromind_runtime.infrastructure.sql_work_queue import work

        with self.engine.begin() as conn:
            conn.execute(
                work.update()
                .where(
                    work.c.owner == owner_key(owner),
                    work.c.state == "waiting_authorization",
                )
                .values(state="queued", available_at=now())
            )

    def preparation_needs_input(
        self, owner: PlanOwner, run_id: str, scope: dict, reason: str
    ) -> None:
        with self.engine.begin() as conn:
            item = self.queue.fenced(conn, scope["work_id"], scope["generation"])
            row = self._run(conn, owner, run_id)
            if (
                item["owner"] != owner_key(owner)
                or item["payload"].get("run_id") != run_id
            ):
                raise PermissionError("work belongs to a different execution")
            if row["state"] not in {"queued", "preparing"}:
                return
            conn.execute(
                runs.update()
                .where(runs.c.id == run_id)
                .values(
                    state="needs_input",
                    active_key=None,
                    proposal={"unresolved": [reason[:16000]]},
                )
            )
            self.changed(conn, owner, run_id, "needs-input")
