"""Transactional execution ledger; no platform or model calls occur inside SQL."""

from __future__ import annotations

import json
from contextlib import contextmanager
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from pyromind_runtime.domain.commands import UserMessageCommand
from pyromind_runtime.domain.experiment_plans import PlanOwner
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload
from pyromind_runtime.domain.task_correlation import TaskCorrelation
from pyromind_runtime.infrastructure.sql_database import (
    configure_engine,
    insert_if_absent,
    write_transaction,
)
from pyromind_runtime.infrastructure.sql_experiment_plan_store import projects
from pyromind_runtime.infrastructure.sql_work_queue import (
    _TABLE_OPTIONS,
    SQLWorkQueue,
    digest,
    metadata,
    now,
    work,
)


runs = sa.Table(
    "experiment_task_runs",
    metadata,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("owner", sa.String(64), nullable=False),
    sa.Column("project_id", sa.String(64), nullable=False),
    sa.Column("revision", sa.Integer),
    sa.Column("task_spec", sa.JSON),
    sa.Column("result", sa.JSON),
    sa.Column("attributes", sa.JSON, default=dict),
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
    sa.Column("run_id", sa.String(64), nullable=False),
    sa.Column("operation_key", sa.String(128), nullable=False, server_default="legacy"),
    sa.Column("owner", sa.String(64), nullable=False),
    sa.Column("state", sa.String(32), nullable=False),
    sa.Column("task_id", sa.String(128), unique=True),
    sa.Column("request_fingerprint", sa.String(64), nullable=False),
    sa.Column("evidence", sa.JSON, nullable=False),
    sa.Column("updated_at", sa.DateTime, nullable=False),
    sa.UniqueConstraint("run_id", "operation_key", name="uq_execution_operation"),
    **_TABLE_OPTIONS,
)
messages = sa.Table(
    "runtime_messages",
    metadata,
    sa.Column("direction", sa.String(8), primary_key=True),
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("owner", sa.String(64)),
    sa.Column("conversation_id", sa.String(128)),
    sa.Column("task_id", sa.String(128)),
    sa.Column("source", sa.String(32)),
    sa.Column("attributes", sa.JSON, nullable=False),
    sa.Column("state", sa.String(32), nullable=False),
    sa.Column("created_at", sa.DateTime),
    sa.Index("ix_runtime_messages_pending", "direction", "state", "created_at"),
    sa.Index("ix_runtime_messages_task", "direction", "task_id", "state"),
    **_TABLE_OPTIONS,
)
inbox = (
    sa.select(
        messages.c.id,
        messages.c.task_id,
        messages.c.source,
        messages.c.attributes.label("payload"),
        messages.c.state,
        messages.c.created_at,
    )
    .where(messages.c.direction == "in")
    .subquery("inbox_messages")
)
outbox = (
    sa.select(
        messages.c.id,
        messages.c.owner,
        messages.c.conversation_id,
        messages.c.attributes.label("payload"),
        messages.c.state,
    )
    .where(messages.c.direction == "out")
    .subquery("outbox_messages")
)


def owner_key(owner: PlanOwner) -> str:
    return digest(owner.model_dump_json())


def _submission_key(operation: str, scope: dict) -> str:
    call_id = scope.get("tool_call_id")
    if call_id:
        return "call:" + digest(scope["work_id"] + ":" + call_id)
    return operation


def _current_submission(row: dict, entries: list[dict]) -> dict | None:
    attributes = row.get("attributes") or {}
    submission_id = attributes.get("current_submission_id")
    workflow_id = attributes.get("workflow_request_id")
    return next(
        (
            entry
            for entry in entries
            if (not submission_id or entry["id"] == submission_id)
            and (
                not workflow_id
                or entry["evidence"].get("workflow_request_id") == workflow_id
            )
        ),
        None,
    )


def _current_platform_task(conn, row: dict) -> str | None:
    statement = sa.select(submissions.c.task_id).where(
        submissions.c.run_id == row["id"]
    )
    attributes = row["attributes"] or {}
    if submission_id := attributes.get("current_submission_id"):
        statement = statement.where(submissions.c.id == submission_id)
    elif request_id := attributes.get("workflow_request_id"):
        statement = statement.where(
            submissions.c.evidence["workflow_request_id"].as_string() == request_id
        )
    return conn.execute(statement.order_by(submissions.c.updated_at.desc())).scalar()


def fingerprint(value: dict) -> str:
    return digest(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    )


class SQLExperimentExecutionStore:
    def __init__(self, engine: Engine):
        self.engine = configure_engine(engine)
        self.queue = SQLWorkQueue(engine)

    @contextmanager
    def _admission(self):
        try:
            with write_transaction(self.engine) as conn:
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
                messages.insert().values(
                    direction="out",
                    id=identity,
                    owner=owner_key(owner),
                    conversation_id=owner.conversation_id,
                    state="pending",
                    attributes={
                        "run_id": run_id,
                        "owner": owner.model_dump(mode="json"),
                    },
                )
            )

    def request(
        self,
        owner: PlanOwner,
        project: str,
        revision: int | None,
        task: str,
        role: str,
        request_id: str,
        *,
        task_spec: dict | None = None,
    ) -> dict:
        if task_spec:
            validate_persisted_payload(task_spec)
        with self._admission() as conn:
            return self.request_in_transaction(
                conn,
                owner,
                project,
                revision,
                task,
                role,
                request_id,
                task_spec=task_spec,
            )

    def request_in_transaction(
        self,
        conn,
        owner: PlanOwner,
        project: str,
        revision: int | None,
        task: str,
        role: str,
        request_id: str,
        *,
        task_spec: dict | None = None,
    ) -> dict:
        key = digest(owner_key(owner) + ":" + request_id)
        active = digest(owner_key(owner) + ":" + project + ":" + task)
        previous = (
            conn.execute(sa.select(runs).where(runs.c.request_key == key))
            .mappings()
            .first()
        )
        if previous:
            binding_keys = {"plan_sequence", "stage_snapshot"}
            saved_spec = previous["task_spec"]
            if {
                key: value
                for key, value in (saved_spec or {}).items()
                if key not in binding_keys
            } != {
                key: value
                for key, value in (task_spec or {}).items()
                if key not in binding_keys
            }:
                raise ValueError("request ID reused with different task content")
            if (previous["task_id"], previous["revision"], previous["role_id"]) != (
                task,
                revision,
                role,
            ):
                raise ValueError("request ID reused with different execution")
            return dict(previous)
        if task_spec and task_spec.get("plan_sequence") is not None:
            project_row = (
                conn.execute(
                    sa.select(projects)
                    .where(
                        projects.c.project_id == project,
                        projects.c.owner_key == owner_key(owner),
                    )
                    .with_for_update()
                )
                .mappings()
                .first()
            )
            if (
                project_row is None
                or project_row["context_sequence"] != task_spec["plan_sequence"]
            ):
                raise ValueError("workflow plan changed; reread before delegation")
            stage = next(
                (
                    stage
                    for stage in project_row["context"].get("stages", [])
                    if stage["id"] == task_spec.get("stage_id")
                ),
                None,
            )
            if (
                stage is None
                or not stage.get("enabled", True)
                or stage.get("placeholder", False)
            ):
                raise ValueError("task requires an enabled Project stage")
            validate_persisted_payload(task_spec)
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
                task_spec=task_spec,
            )
        )
        self.queue.put(
            conn,
            owner_key(owner),
            "role:" + role,
            "prepare:" + identity,
            "execute_task" if task_spec else "prepare_execution",
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
                    {
                        "id": item.get("id"),
                        "owner": row["owner"],
                        "run_id": row["id"],
                        "name": name,
                        "uri": item["uri"],
                        "sha256": item["sha256"],
                        "payload": item.get("metadata", {}),
                    }
                    for name, item in (row["attributes"] or {})
                    .get("artifacts", {})
                    .items()
                ]
                value["submissions"] = [
                    dict(sub)
                    for sub in conn.execute(
                        sa.select(submissions)
                        .where(submissions.c.run_id == row["id"])
                        .order_by(submissions.c.updated_at.desc())
                    ).mappings()
                ]
                value["submission"] = _current_submission(value, value["submissions"])
                result.append(value)
            return result

    def managed_submission(
        self, owner: PlanOwner, run_id: str, scope: dict, arguments: dict
    ) -> dict:
        """Reserve an existing evaluation tool invocation, not a new executor."""
        from uuid import NAMESPACE_URL, uuid5

        validate_persisted_payload(arguments)
        with write_transaction(self.engine) as conn:
            item = self.queue.fenced(conn, scope["work_id"], scope["generation"])
            row = self._run(conn, owner, run_id)
            if (
                item["owner"] != owner_key(owner)
                or item["payload"].get("run_id") != run_id
            ):
                raise PermissionError("work belongs to a different execution")
            if not row["task_spec"] or row["task_spec"]["intent"] != "execute":
                raise PermissionError("this task only authorizes preparation")
            if row["state"] in {
                "cancel_requested",
                "cancelled",
                "submission_uncertain",
                "abandoned",
            }:
                raise ValueError("execution cannot submit work")
            policy = (row["task_spec"] or {}).get("execution_policy") or {}
            if policy.get("require_review"):
                checkpoint = (row.get("attributes") or {}).get("checkpoint") or {}
                if checkpoint.get("status") != "approved":
                    raise PermissionError("submission requires current sample approval")
            operation = str(arguments.get("operation", "inference"))
            operation_key = _submission_key(operation, scope)
            per_call = operation_key != operation
            previous = (
                conn.execute(
                    sa.select(submissions)
                    .where(
                        submissions.c.run_id == run_id,
                        submissions.c.operation_key == operation_key,
                    )
                    .with_for_update()
                )
                .mappings()
                .first()
            )
            attempt = 0
            if previous:
                if not per_call and previous["state"] == "not_submitted":
                    attempt = previous["evidence"].get("attempt", 0) + 1
                elif previous["request_fingerprint"] != fingerprint(arguments):
                    raise ValueError(
                        "submission identity reused with different arguments"
                    )
                elif previous["evidence"].get("receipt"):
                    return {
                        "receipt": previous["evidence"]["receipt"],
                        "operation_id": previous["evidence"]["operation_id"],
                    }
                else:
                    raise ValueError(
                        "submission needs reconciliation; do not repeat it"
                    )
            if per_call and row["state"] == "submitting":
                raise ValueError("another submission is awaiting a platform receipt")
            identity = previous["id"] if previous else uuid4().hex
            operation_id = identity + ":" + str(attempt)
            evidence = {
                "operation_id": operation_id,
                "operation": operation,
                "attempt": attempt,
                "work_id": item["id"],
                "pipeline_run_id": str(
                    uuid5(NAMESPACE_URL, f"{owner.conversation_id}:{operation_id}")
                ),
            }
            values = dict(
                state="submitting",
                request_fingerprint=fingerprint(arguments),
                evidence=evidence,
                updated_at=now(),
            )
            if previous:
                conn.execute(
                    submissions.update()
                    .where(submissions.c.id == identity)
                    .values(**values)
                )
            else:
                conn.execute(
                    submissions.insert().values(
                        id=identity,
                        owner=owner_key(owner),
                        run_id=run_id,
                        operation_key=operation_key,
                        **values,
                    )
                )
            run_updates = {}
            if per_call:
                attributes = dict(row["attributes"] or {})
                attributes["current_submission_id"] = identity
                run_updates["attributes"] = attributes
                if operation in {"inference", "resume", "pipeline"}:
                    attributes.pop("artifacts", None)
                    run_updates.update(
                        result=None,
                        metrics=None,
                        active_key=digest(
                            owner_key(owner)
                            + ":"
                            + row["project_id"]
                            + ":"
                            + row["task_id"]
                        ),
                    )
            conn.execute(
                runs.update()
                .where(runs.c.id == run_id)
                .values(state="submitting", report_state="not_started", **run_updates)
            )
            return {"operation_id": operation_id}

    def managed_receipt(
        self, owner: PlanOwner, run_id: str, scope: dict, operation: str, receipt: dict
    ) -> None:
        validate_persisted_payload(receipt)
        with write_transaction(self.engine) as conn:
            self.queue.fenced(conn, scope["work_id"], scope["generation"])
            self._run(conn, owner, run_id)
            sub = (
                conn.execute(
                    sa.select(submissions)
                    .where(
                        submissions.c.run_id == run_id,
                        submissions.c.operation_key
                        == _submission_key(operation, scope),
                    )
                    .with_for_update()
                )
                .mappings()
                .one()
            )
            state = (
                "waiting_platform"
                if receipt.get("task_id")
                else "submission_uncertain"
                if receipt.get("uncertain")
                else "not_submitted"
            )
            conn.execute(
                submissions.update()
                .where(submissions.c.id == sub["id"])
                .values(
                    state=state,
                    task_id=receipt.get("task_id"),
                    evidence={**sub["evidence"], "receipt": receipt},
                    updated_at=now(),
                )
            )
            conn.execute(
                runs.update()
                .where(runs.c.id == run_id)
                .values(state="preparing" if state == "not_submitted" else state)
            )
            self.changed(conn, owner, run_id, sub["id"] + ":" + state)

    def complete_agent_turn(
        self,
        owner: PlanOwner,
        run_id: str,
        scope: dict,
        text: str,
        report: bool = False,
        *,
        completed: bool = False,
    ) -> dict:
        """Apply a child turn outcome to the execution ledger."""
        with write_transaction(self.engine) as conn:
            item = self.queue.fenced(conn, scope["work_id"], scope["generation"])
            row = self._run(conn, owner, run_id)
            if item["owner"] != owner_key(owner) or item["payload"]["run_id"] != run_id:
                raise PermissionError("work belongs to a different execution")
            followup = item["payload"].get("platform_followup")
            if followup and followup != _current_platform_task(conn, row):
                return row
            if row["state"] == "submitting":
                conn.execute(
                    submissions.update()
                    .where(
                        submissions.c.run_id == run_id,
                        submissions.c.state == "submitting",
                    )
                    .values(state="submission_uncertain")
                )
                conn.execute(
                    runs.update()
                    .where(runs.c.id == run_id)
                    .values(state="submission_uncertain")
                )
                self.changed(conn, owner, run_id, "submission-uncertain")
            if report and row["state"] in {
                "waiting_confirmation",
                "submitting",
                "submission_uncertain",
                "waiting_platform",
            }:
                return self._run(conn, owner, run_id)
            state = row["state"]
            task = row.get("task_spec") or {}
            policy = task.get("execution_policy") or {}
            contract = policy.get("artifact_contract")
            checkpoint = (row.get("attributes") or {}).get("checkpoint") or {}
            review_satisfied = (
                not policy.get("require_review")
                or checkpoint.get("status") == "approved"
            )
            local_delivery = state in {"queued", "preparing"} or (
                report and state == "needs_input"
            )
            handoff_state = None
            if completed and row["report_state"] != "ready":
                if report and state in {"succeeded", "failed", "cancelled"}:
                    handoff_state = state
                elif (
                    report
                    and state == "waiting_artifacts"
                    and contract in {"data", "model"}
                    and review_satisfied
                ):
                    handoff_state = "succeeded"
                elif (
                    local_delivery
                    and contract == "data"
                    and task.get("intent") == "execute"
                    and review_satisfied
                ):
                    # A local turn has no platform outcome to verify.
                    handoff_state = "reported"
            if handoff_state:
                result = dict(row.get("result") or {})
                summary = text[:16000]
                validate_persisted_payload({"summary": summary})
                result.pop("waiting_reason", None)
                result["summary"] = summary
                conn.execute(
                    runs.update()
                    .where(runs.c.id == run_id)
                    .values(
                        state=handoff_state,
                        report_state="ready",
                        result=result,
                        active_key=None,
                    )
                )
                self.changed(conn, owner, run_id, "report:" + fingerprint(result))
            elif report and row["report_state"] != "ready":
                result = dict(row.get("result") or {})
                reason = text[:16000]
                validate_persisted_payload({"summary": reason})
                result.update(summary=reason, waiting_reason=reason)
                conn.execute(
                    runs.update()
                    .where(runs.c.id == run_id)
                    .values(report_state="failed", result=result)
                )
                self.changed(conn, owner, run_id, "report-failed")
            elif row["state"] in {"queued", "preparing"}:
                prepared = row["task_spec"]["intent"] == "prepare"
                result = dict(row.get("result") or {})
                reason = text[:16000]
                validate_persisted_payload({"summary": reason})
                result.update(summary=reason, blockers=[] if prepared else [reason])
                delivery_failed = contract == "data" and not completed
                conn.execute(
                    runs.update()
                    .where(runs.c.id == run_id)
                    .values(
                        state="prepared" if prepared else "needs_input",
                        active_key=None,
                        result=result,
                        report_state="failed"
                        if delivery_failed and not prepared
                        else "ready",
                    )
                )
                self.changed(conn, owner, run_id, "needs-input")
            return self._run(conn, owner, run_id)

    def record_checkpoint(
        self, owner: PlanOwner, run_id: str, scope: dict, checkpoint: dict
    ) -> None:
        validate_persisted_payload(checkpoint)
        with write_transaction(self.engine) as conn:
            self.queue.fenced(conn, scope["work_id"], scope["generation"])
            row = self._run(conn, owner, run_id)
            policy = (row.get("task_spec") or {}).get("execution_policy") or {}
            if not policy.get("require_review") or row["state"] not in {
                "queued",
                "preparing",
                "waiting_user",
            }:
                raise ValueError("execution cannot register a sample checkpoint")
            attributes = dict(row.get("attributes") or {})
            attributes["checkpoint"] = checkpoint
            conn.execute(
                runs.update()
                .where(runs.c.id == run_id)
                .values(
                    attributes=attributes,
                    state="waiting_user",
                    report_state="ready",
                    result={
                        "summary": checkpoint["document"]["summary"],
                        "checkpoint": checkpoint,
                    },
                )
            )
            self.changed(
                conn, owner, run_id, "checkpoint:" + checkpoint["checkpoint_id"]
            )

    def resolve_checkpoint(
        self,
        owner: PlanOwner,
        run_id: str,
        checkpoint_id: str,
        command_id: str,
        approved: bool,
    ) -> None:
        with write_transaction(self.engine) as conn:
            row = self._run(conn, owner, run_id)
            attributes = dict(row.get("attributes") or {})
            checkpoint = dict(attributes.get("checkpoint") or {})
            if checkpoint.get("checkpoint_id") != checkpoint_id:
                raise ValueError("sample changed; review the current sample")
            if checkpoint.get("decision_id") == command_id:
                return
            if checkpoint.get("status") != "pending" or row["state"] != "waiting_user":
                raise ValueError("sample is no longer waiting for confirmation")
            checkpoint.update(
                status="approved" if approved else "revise", decision_id=command_id
            )
            attributes["checkpoint"] = checkpoint
            result = {**(row.get("result") or {}), "checkpoint": checkpoint}
            conn.execute(
                runs.update()
                .where(runs.c.id == run_id)
                .values(attributes=attributes, result=result)
            )
            self.changed(conn, owner, run_id, "checkpoint-answer:" + command_id)

    def queue_artifact_report(
        self, owner: PlanOwner, run_id: str, *, platform_task_id: str | None = None
    ) -> None:
        with write_transaction(self.engine) as conn:
            row = self._run(conn, owner, run_id)
            task_id = _current_platform_task(conn, row)
            if platform_task_id and task_id != platform_task_id:
                return
            if row["state"] != "waiting_artifacts":
                raise ValueError("platform artifacts are not ready for inspection")
            if row["report_state"] != "not_started":
                return
            attributes = row["attributes"] or {}
            followup = bool(
                attributes.get("current_submission_id")
                or attributes.get("workflow_request_id")
            )
            self.queue.put(
                conn,
                owner_key(owner),
                "role:" + row["role_id"],
                f"verify:{run_id}:{task_id}",
                "execute_task" if followup else "generate_report",
                {
                    "run_id": run_id,
                    "owner": owner.model_dump(mode="json"),
                    "message": "Inspect platform outputs and report the outcome.",
                    **({"platform_followup": task_id} if followup else {}),
                },
            )
            conn.execute(
                runs.update().where(runs.c.id == run_id).values(report_state="queued")
            )

    def stage_workflow(
        self, owner: PlanOwner, run_id: str, scope: dict, request_id: str
    ) -> None:
        with write_transaction(self.engine) as conn:
            item = self.queue.fenced(conn, scope["work_id"], scope["generation"])
            row = self._run(conn, owner, run_id)
            if (
                item["owner"] != owner_key(owner)
                or item["payload"].get("run_id") != run_id
            ):
                raise PermissionError("work belongs to a different execution")
            policy = (row.get("task_spec") or {}).get("execution_policy") or {}
            if (
                policy.get("submission_tool") != "submit_workflow"
                or row["task_spec"]["intent"] != "execute"
            ):
                raise PermissionError("execution does not allow workflow submission")
            attributes = dict(row.get("attributes") or {})
            previous = attributes.get("workflow_request_id")
            if row["state"] in {
                "cancel_requested",
                "cancelled",
                "submitting",
                "submission_uncertain",
                "abandoned",
            }:
                raise ValueError("task cannot stage another workflow")
            if previous != request_id and row["state"] == "waiting_confirmation":
                raise ValueError("resolve the pending workflow confirmation first")
            prior = next(
                (
                    sub
                    for sub in conn.execute(
                        sa.select(submissions)
                        .where(submissions.c.run_id == run_id)
                        .with_for_update()
                    ).mappings()
                    if sub["evidence"].get("workflow_request_id") == previous
                ),
                None,
            )
            if previous != request_id:
                if prior is not None:
                    conn.execute(
                        submissions.update()
                        .where(submissions.c.id == prior["id"])
                        .values(
                            evidence={
                                **prior["evidence"],
                                "result": row.get("result"),
                                "artifacts": attributes.get("artifacts", {}),
                            }
                        )
                    )
                attributes.pop("artifacts", None)
            attributes["workflow_request_id"] = request_id
            conn.execute(
                runs.update()
                .where(runs.c.id == run_id)
                .values(
                    attributes=attributes,
                    state="waiting_confirmation",
                    report_state="not_started",
                    result=None,
                    metrics=None,
                    active_key=digest(
                        owner_key(owner)
                        + ":"
                        + row["project_id"]
                        + ":"
                        + row["task_id"]
                    ),
                )
            )
            self.changed(conn, owner, run_id, "workflow:" + request_id)

    def begin_workflow(
        self,
        owner: PlanOwner,
        run_id: str,
        request_id: str,
        definition_fingerprint: str,
        allow: bool,
    ) -> str | None:
        with write_transaction(self.engine) as conn:
            row = self._run(conn, owner, run_id)
            if (row.get("attributes") or {}).get("workflow_request_id") != request_id:
                raise PermissionError("workflow belongs to another execution")
            existing = next(
                (
                    sub
                    for sub in conn.execute(
                        sa.select(submissions).where(
                            submissions.c.run_id == run_id,
                        )
                    ).mappings()
                    if sub["evidence"].get("workflow_request_id") == request_id
                ),
                None,
            )
            if existing:
                if existing["request_fingerprint"] != definition_fingerprint:
                    raise ValueError("workflow definition changed")
                return existing["id"]
            if row["state"] != "waiting_confirmation":
                raise ValueError("execution is not awaiting workflow confirmation")
            if not allow:
                conn.execute(
                    runs.update()
                    .where(runs.c.id == run_id)
                    .values(state="cancelled", active_key=None)
                )
                self.changed(conn, owner, run_id, "workflow-denied:" + request_id)
                return None
            identity = uuid4().hex
            conn.execute(
                submissions.insert().values(
                    id=identity,
                    run_id=run_id,
                    operation_key="workflow:" + digest(request_id),
                    owner=owner_key(owner),
                    state="submitting",
                    request_fingerprint=definition_fingerprint,
                    evidence={"workflow_request_id": request_id},
                    updated_at=now(),
                )
            )
            conn.execute(
                runs.update().where(runs.c.id == run_id).values(state="submitting")
            )
            return identity

    def pending_runs(self) -> list[dict]:
        with self.engine.connect() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    sa.select(
                        runs,
                        projects.c.user_id,
                        projects.c.conversation_id,
                        projects.c.profile_id,
                    )
                    .join(projects, runs.c.project_id == projects.c.project_id)
                    .where(runs.c.task_spec.is_not(None))
                ).mappings()
            ]

    def mark_submission_uncertain(
        self, owner: PlanOwner, run_id: str, submission_id: str
    ) -> None:
        with write_transaction(self.engine) as conn:
            self._run(conn, owner, run_id)
            updated = conn.execute(
                submissions.update()
                .where(
                    submissions.c.id == submission_id,
                    submissions.c.run_id == run_id,
                    submissions.c.state == "submitting",
                )
                .values(state="submission_uncertain", updated_at=now())
            )
            if updated.rowcount:
                conn.execute(
                    runs.update()
                    .where(runs.c.id == run_id)
                    .values(state="submission_uncertain")
                )
                self.changed(conn, owner, run_id, "uncertain:" + submission_id)

    def recover_managed_receipt(
        self, owner: PlanOwner, run_id: str, submission_id: str, receipt: dict
    ) -> None:
        """Repair from the existing tool ledger, never repeat external creation."""
        validate_persisted_payload(receipt)
        with write_transaction(self.engine) as conn:
            row = self._run(conn, owner, run_id)
            sub = (
                conn.execute(
                    sa.select(submissions)
                    .where(
                        submissions.c.id == submission_id,
                        submissions.c.run_id == run_id,
                    )
                    .with_for_update()
                )
                .mappings()
                .one()
            )
            if sub["state"] not in {"submitting", "submission_uncertain"}:
                return
            task_id = receipt.get("task_id")
            if not task_id:
                if (
                    receipt.get("submission_not_sent") is not True
                    or not receipt.get("operation_id")
                    or receipt["operation_id"] != sub["evidence"].get("operation_id")
                    or sub["task_id"] is not None
                    or row["state"] != "submission_uncertain"
                ):
                    raise ValueError(
                        "recovery requires a recorded task or proof "
                        "the operation was not sent"
                    )
                submission_state, run_state = "not_submitted", "needs_input"
            else:
                submission_state = run_state = "waiting_platform"
            conn.execute(
                submissions.update()
                .where(submissions.c.id == submission_id)
                .values(
                    task_id=task_id,
                    state=submission_state,
                    evidence={**sub["evidence"], "receipt": receipt},
                    updated_at=now(),
                )
            )
            conn.execute(
                runs.update().where(runs.c.id == run_id).values(state=run_state)
            )
            self.changed(conn, owner, run_id, "recovered:" + submission_id)

    def note_wait(
        self,
        owner: PlanOwner,
        run_id: str,
        reason: str,
        *,
        artifact_failure: bool = False,
    ) -> None:
        with write_transaction(self.engine) as conn:
            row = self._run(conn, owner, run_id)
            result = {**(row.get("result") or {}), "waiting_reason": reason}
            conn.execute(
                runs.update()
                .where(runs.c.id == run_id)
                .values(
                    result=result,
                    **(
                        {"state": "failed", "active_key": None}
                        if artifact_failure
                        else {}
                    ),
                )
            )
            if result != row.get("result"):
                self.changed(conn, owner, run_id, "wait:" + fingerprint(result))

    def conflicting_tasks(self) -> set[str]:
        with self.engine.connect() as conn:
            return set(
                conn.execute(
                    sa.select(inbox.c.task_id).where(inbox.c.state == "conflict")
                ).scalars()
            )

    def native_reference(self, owner: PlanOwner, run_id: str) -> str | None:
        matches = [
            item
            for item in self.queue.list(owner_key(owner))
            if item["payload"].get("run_id") == run_id
            and item["checkpoint"].get("execution_ref")
        ]
        return matches[-1]["checkpoint"]["execution_ref"] if matches else None

    def work_items(self, owner: PlanOwner) -> tuple[dict, ...]:
        return tuple(
            {
                "id": item["id"],
                "kind": item["kind"],
                "state": item["state"],
                "run_id": item["payload"].get("run_id"),
                "attempts": item["attempts"],
                "waiting_reason": item["checkpoint"].get("reason"),
            }
            for item in self.queue.list(owner_key(owner))
        )

    def enqueue_manager_message(
        self, owner: PlanOwner, command: UserMessageCommand
    ) -> str:
        return self.queue.enqueue(
            owner_key(owner),
            "role:manager",
            "user:" + command.command_id,
            "manager_turn",
            {
                "owner": owner.model_dump(mode="json"),
                "command": command.model_dump(mode="json"),
            },
            10,
        )

    def continue_task(
        self, owner: PlanOwner, run_id: str, command_id: str, message: str
    ) -> str:
        """Deliver guidance to the child context independently of business state."""
        with write_transaction(self.engine) as conn:
            row = self._run(conn, owner, run_id)
            prior = (
                conn.execute(
                    sa.select(work).where(
                        work.c.owner == owner_key(owner),
                        work.c.dedup_key == digest("continue:" + command_id),
                    )
                )
                .mappings()
                .first()
            )
            if prior:
                if (
                    prior["payload"].get("run_id") != run_id
                    or prior["payload"].get("message") != message
                ):
                    raise ValueError("continuation command reused with different work")
                return prior["id"]
            active = next(
                (
                    item
                    for item in conn.execute(
                        sa.select(work)
                        .where(
                            work.c.owner == owner_key(owner),
                            work.c.state == "running",
                            work.c.kind.in_(
                                ("execute_task", "prepare_execution", "generate_report")
                            ),
                        )
                        .with_for_update()
                    ).mappings()
                    if item["payload"].get("run_id") == run_id
                ),
                None,
            )
            payload = {
                "run_id": run_id,
                "owner": owner.model_dump(mode="json"),
                "message": message,
                "continuation": True,
            }
            if active:
                payload["target_work_id"] = active["id"]
            return self.queue.put(
                conn,
                owner_key(owner),
                "guidance:" + row["role_id"] if active else "role:" + row["role_id"],
                "continue:" + command_id,
                "steer_task" if active else "execute_task",
                payload,
            )

    def guidance_target(self, owner: PlanOwner, run_id: str, work_id: str) -> dict:
        with self.engine.connect() as conn:
            self._run(conn, owner, run_id)
            item = (
                conn.execute(
                    sa.select(work).where(
                        work.c.id == work_id, work.c.owner == owner_key(owner)
                    )
                )
                .mappings()
                .first()
            )
            if item is None or item["payload"].get("run_id") != run_id:
                raise PermissionError("guidance belongs to another execution")
            return dict(item)

    def enqueue_continuation(
        self, owner: PlanOwner, run_id: str, guidance_id: str, message: str
    ) -> str:
        """The child finished before live guidance arrived; resume its session next."""
        with write_transaction(self.engine) as conn:
            row = self._run(conn, owner, run_id)
            return self.queue.put(
                conn,
                owner_key(owner),
                "role:" + row["role_id"],
                "guidance-resume:" + guidance_id,
                "execute_task",
                {
                    "run_id": run_id,
                    "owner": owner.model_dump(mode="json"),
                    "message": message,
                    "continuation": True,
                },
            )

    def platform_run(self, task_id: str) -> dict | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    sa.select(
                        runs,
                        projects.c.user_id,
                        projects.c.conversation_id,
                        projects.c.profile_id,
                    )
                    .join(projects, runs.c.project_id == projects.c.project_id)
                    .join(submissions, submissions.c.run_id == runs.c.id)
                    .where(
                        submissions.c.task_id == task_id, runs.c.task_spec.is_not(None)
                    )
                )
                .mappings()
                .first()
            )
            return dict(row) if row else None

    def cancellation_result(
        self,
        owner: PlanOwner,
        run_id: str,
        scope: dict,
        native_stopped: bool,
        platform_stopped: bool,
    ) -> dict:
        with write_transaction(self.engine) as conn:
            self.queue.fenced(conn, scope["work_id"], scope["generation"])
            row = self._run(conn, owner, run_id)
            state = (
                "cancelled"
                if native_stopped and platform_stopped
                else "cancel_requested"
            )
            if row["state"] in {"succeeded", "waiting_artifacts", "failed", "reported"}:
                return row
            conn.execute(
                runs.update()
                .where(runs.c.id == run_id)
                .values(
                    state=state,
                    active_key=None if state == "cancelled" else row["active_key"],
                )
            )
            self.changed(conn, owner, run_id, state)
            return self._run(conn, owner, run_id)

    def enqueue_report(
        self,
        owner: PlanOwner,
        run_id: str,
        message: str = "",
        command_id: str = "",
        *,
        platform_task_id: str | None = None,
    ) -> str | None:
        with write_transaction(self.engine) as conn:
            row = self._run(conn, owner, run_id)
            attributes = row["attributes"] or {}
            current = attributes.get("current_submission_id")
            sub = _current_platform_task(conn, row)
            if platform_task_id and sub != platform_task_id:
                return None
            followup = bool(
                (current or attributes.get("workflow_request_id")) and not command_id
            )
            identity = self.queue.put(
                conn,
                owner_key(owner),
                "role:" + row["role_id"],
                f"report:{run_id}:{sub}:{command_id}",
                "execute_task" if followup else "generate_report",
                {
                    "run_id": run_id,
                    "owner": owner.model_dump(mode="json"),
                    "message": message,
                    **({"platform_followup": sub} if followup else {}),
                },
            )
            conn.execute(
                runs.update().where(runs.c.id == run_id).values(report_state="queued")
            )
            return identity

    def propose(self, owner: PlanOwner, run_id: str, proposal: dict) -> dict:
        validate_persisted_payload(proposal)
        with write_transaction(self.engine) as conn:
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
        with write_transaction(self.engine) as conn:
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
        with write_transaction(self.engine) as conn:
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
        with write_transaction(self.engine) as conn:
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
        with write_transaction(self.engine) as conn:
            identity = digest(source + ":" + identity)
            insert_if_absent(
                conn,
                messages,
                {
                    "direction": "in",
                    "id": identity,
                    "task_id": task_id,
                    "source": source,
                    "attributes": payload,
                    "state": "pending",
                    "created_at": now(),
                },
                "id",
            )
            saved = (
                conn.execute(
                    sa.select(messages)
                    .where(messages.c.direction == "in", messages.c.id == identity)
                    .with_for_update()
                )
                .mappings()
                .one()
            )
            if saved["task_id"] != task_id or saved["attributes"] != payload:
                raise ValueError("observation identity reused with different content")

    def recover_submissions(self) -> int:
        # No evidence of an API response means acceptance is uncertain, not failed.
        with write_transaction(self.engine) as conn:
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
                    .where(submissions.c.state == "submitting")
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
        with write_transaction(self.engine) as conn:
            events = conn.execute(
                sa.select(messages)
                .where(messages.c.direction == "in", messages.c.state == "pending")
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
                supplied = event["attributes"].get("correlation")
                if supplied is not None:
                    correlation = TaskCorrelation.model_validate(supplied)
                    conversation = conn.execute(
                        sa.select(projects.c.conversation_id)
                        .join(runs, runs.c.project_id == projects.c.project_id)
                        .where(runs.c.id == sub["run_id"])
                    ).scalar_one()
                    expected_conversation = TaskCorrelation(
                        conversation_id=conversation, task_kind=correlation.task_kind
                    ).conversation_id
                    receipt = (sub["evidence"] or {}).get("receipt", {})
                    expected = receipt.get("correlation")
                    if (
                        correlation.conversation_id != expected_conversation
                        or expected is not None
                        and TaskCorrelation.model_validate(expected) != correlation
                    ):
                        conn.execute(
                            messages.update()
                            .where(
                                messages.c.direction == "in",
                                messages.c.id == event["id"],
                            )
                            .values(state="invalid")
                        )
                        continue
                status = str(event["attributes"].get("status", "")).lower()
                if status not in terminal | {"running", "pending"}:
                    conn.execute(
                        messages.update()
                        .where(messages.c.direction == "in")
                        .where(messages.c.id == event["id"])
                        .values(state="invalid")
                    )
                    continue
                if sub["state"] in terminal and status != sub["state"]:
                    if status in terminal and event["source"] != "query":
                        conn.execute(
                            messages.update()
                            .where(messages.c.direction == "in")
                            .where(messages.c.id == event["id"])
                            .values(state="conflict")
                        )
                        continue
                    if status not in terminal:
                        status = sub["state"]
                if event["source"] == "query":
                    conn.execute(
                        messages.update()
                        .where(messages.c.direction == "in")
                        .where(
                            messages.c.task_id == event["task_id"],
                            messages.c.state == "conflict",
                        )
                        .values(state="applied")
                    )
                if (sub["state"] not in terminal or event["source"] == "query") and sub[
                    "state"
                ] != status:
                    conn.execute(
                        submissions.update()
                        .where(submissions.c.id == sub["id"])
                        .values(state=status, updated_at=now())
                    )
                    project = (
                        conn.execute(
                            sa.select(projects)
                            .join(runs, runs.c.project_id == projects.c.project_id)
                            .where(runs.c.id == sub["run_id"])
                        )
                        .mappings()
                        .one()
                    )
                    owner = PlanOwner(
                        user_id=project["user_id"],
                        conversation_id=project["conversation_id"],
                        profile_id=project["profile_id"],
                    )
                    self.changed(
                        conn, owner, sub["run_id"], "observation:" + event["id"]
                    )
                    if status in terminal:
                        run = self._run(conn, owner, sub["run_id"])
                        attributes = run["attributes"] or {}
                        current_submission = attributes.get("current_submission_id")
                        current_workflow = attributes.get("workflow_request_id")
                        if (current_submission and current_submission != sub["id"]) or (
                            current_workflow
                            and sub["evidence"].get("workflow_request_id")
                            and current_workflow
                            != sub["evidence"]["workflow_request_id"]
                        ):
                            conn.execute(
                                messages.update()
                                .where(
                                    messages.c.direction == "in",
                                    messages.c.id == event["id"],
                                )
                                .values(state="applied")
                            )
                            count += 1
                            continue
                        run_state = run["state"]
                        conn.execute(
                            runs.update()
                            .where(runs.c.id == sub["run_id"])
                            .values(
                                state="waiting_artifacts"
                                if status == "succeeded"
                                else "cancelled"
                                if run_state == "cancel_requested"
                                and status in {"stopped", "terminated"}
                                else "failed",
                                active_key=None
                                if status != "succeeded"
                                else runs.c.active_key,
                            )
                        )
                conn.execute(
                    messages.update()
                    .where(messages.c.direction == "in")
                    .where(messages.c.id == event["id"])
                    .values(state="applied")
                )
                count += 1
        return count

    @staticmethod
    def _register_artifact(conn, row: dict, item: dict) -> None:
        validate_persisted_payload(item)
        checksum = item["sha256"]
        if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError(
                f"invalid artifact digest for {item['name']!r}: {item['uri']}"
            )
        attributes = dict(row.get("attributes") or {})
        registered = dict(attributes.get("artifacts", {}))
        previous = registered.get(item["name"])
        if previous:
            if previous["sha256"] != item["sha256"] or previous["uri"] != item["uri"]:
                raise ValueError(
                    f"registered artifact changed: {item['name']!r} "
                    f"({previous['uri']} -> {item['uri']})"
                )
            return
        registered[item["name"]] = {
            "id": item.get("id") or uuid4().hex,
            "uri": item["uri"],
            "sha256": item["sha256"],
            "metadata": item.get("metadata", {}),
        }
        attributes["artifacts"] = registered
        conn.execute(
            runs.update().where(runs.c.id == row["id"]).values(attributes=attributes)
        )
        row["attributes"] = attributes

    def register_artifacts(
        self,
        owner: PlanOwner,
        run_id: str,
        verified: list[dict],
        metrics: dict,
        *,
        platform_task_id: str | None = None,
    ) -> None:
        """Accept only results verified by an application-owned artifact reader."""
        validate_persisted_payload(verified)
        validate_persisted_payload(metrics)
        with write_transaction(self.engine) as conn:
            row = self._run(conn, owner, run_id)
            current = (row["attributes"] or {}).get("current_submission_id")
            task_id = _current_platform_task(conn, row)
            if platform_task_id and task_id != platform_task_id:
                return
            if row["state"] != "waiting_artifacts":
                raise ValueError("execution is not awaiting verified artifacts")
            policy = (row.get("task_spec") or {}).get("execution_policy") or {}
            contract = policy.get("artifact_contract", "evaluation")
            if contract != "evaluation":
                raise ValueError("only platform evaluation artifacts are registered")
            expected = (
                {"metrics.json", "report.json", "predictions.jsonl"}
                if row["task_spec"]
                else set(row["proposal"].get("expected_artifacts", []))
            )
            by_name = {item["name"]: item for item in verified}
            if set(by_name) != expected or len(by_name) != len(verified):
                raise ValueError("verified artifacts do not match frozen manifest")
            for item in verified:
                checksum = item["sha256"]
                if len(checksum) != 64 or any(
                    c not in "0123456789abcdef" for c in checksum
                ):
                    raise ValueError("invalid artifact digest")
                self._register_artifact(conn, row, item)
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
                "report:" + run_id
                if not row["task_spec"]
                else "report:" + run_id + ":" + str(task_id) + ":",
                "execute_task" if current else "generate_report",
                {
                    "run_id": run_id,
                    "owner": owner.model_dump(mode="json"),
                    **({"message": ""} if row["task_spec"] else {}),
                    **({"platform_followup": task_id} if current else {}),
                },
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
        with write_transaction(self.engine) as conn:
            conn.execute(
                messages.update()
                .where(messages.c.direction == "out")
                .where(messages.c.id == identity)
                .values(state="delivered")
            )

    def cancel(self, owner: PlanOwner, run_id: str) -> dict:
        from pyromind_runtime.infrastructure.sql_work_queue import work

        with write_transaction(self.engine) as conn:
            row = self._run(conn, owner, run_id)
            if row["state"] in {
                "cancelled",
                "succeeded",
                "reported",
                "failed",
                "rejected",
                "abandoned",
            }:
                return row
            submitted = list(
                conn.execute(
                    sa.select(submissions)
                    .where(submissions.c.run_id == run_id)
                    .with_for_update()
                )
                .mappings()
                .all()
            )
            # Unknown remote acceptance is never proof of termination.
            if any(
                sub["state"] in {"submitting", "submission_uncertain"}
                for sub in submitted
            ):
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
            remote = any(sub["task_id"] for sub in submitted)
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

    @staticmethod
    def _resolution_submission(conn, run_id: str, command_id: str) -> dict:
        candidates = list(
            conn.execute(
                sa.select(submissions)
                .where(submissions.c.run_id == run_id)
                .with_for_update()
            ).mappings()
        )
        repeated = [
            sub
            for sub in candidates
            if sub["evidence"].get("resolution", {}).get("command_id") == command_id
        ]
        unresolved = repeated or [
            sub
            for sub in candidates
            if sub["state"] in {"submitting", "submission_uncertain"}
        ]
        if len(unresolved) != 1:
            raise ValueError("exactly one uncertain submission must be reconciled")
        return dict(unresolved[0])

    def abandon(self, owner: PlanOwner, run_id: str, command_id: str) -> dict:
        with write_transaction(self.engine) as conn:
            row = self._run(conn, owner, run_id)
            if row["state"] == "abandoned":
                return row
            if row["state"] != "submission_uncertain":
                raise ValueError("only uncertain submissions can be abandoned")
            sub = self._resolution_submission(conn, run_id, command_id)
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
        with write_transaction(self.engine) as conn:
            row = self._run(conn, owner, run_id)
            sub = self._resolution_submission(conn, run_id, command_id)
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
        with write_transaction(self.engine) as conn:
            item = self.queue.fenced(conn, scope["work_id"], scope["generation"])
            row = self._run(conn, owner, run_id)
            if (
                row["role_id"] != role
                or item["owner"] != owner_key(owner)
                or item["payload"].get("run_id") != run_id
            ):
                raise PermissionError("role work belongs to a different execution")
            row["submissions"] = [
                dict(sub)
                for sub in conn.execute(
                    sa.select(submissions)
                    .where(submissions.c.run_id == run_id)
                    .order_by(submissions.c.updated_at.desc())
                ).mappings()
            ]
            row["submission"] = _current_submission(row, row["submissions"])
            if row["state"] == "queued" and item["kind"] in {
                "execute_task",
                "prepare_execution",
            }:
                conn.execute(
                    runs.update().where(runs.c.id == run_id).values(state="preparing")
                )
                row["state"] = "preparing"
                self.changed(conn, owner, run_id, "preparing:" + item["id"])
            return row

    def propose_from_role(
        self, owner: PlanOwner, run_id: str, role: str, scope: dict, proposal: dict
    ) -> dict:
        validate_persisted_payload(proposal)
        with write_transaction(self.engine) as conn:
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
        with write_transaction(self.engine) as conn:
            self.queue.fenced(conn, identity, generation)
            conn.execute(
                work.update().where(work.c.id == identity).values(checkpoint=value)
            )

    def resume_authorized(self, owner: PlanOwner) -> None:
        from pyromind_runtime.infrastructure.sql_work_queue import work

        with write_transaction(self.engine) as conn:
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
        with write_transaction(self.engine) as conn:
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
