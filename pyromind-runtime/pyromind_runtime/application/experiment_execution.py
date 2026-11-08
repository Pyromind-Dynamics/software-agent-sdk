"""Harness-neutral admission and confirmation of experiment executions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from typing import cast

from pyromind_runtime.application.execution_checkpoints import (
    require_approved_checkpoint,
    verify_checkpoint,
)
from pyromind_runtime.domain.execution_policy import CheckpointView
from pyromind_runtime.domain.experiment_execution import (
    ExperimentExecutionView,
    RequestExperimentTask,
)
from pyromind_runtime.domain.experiment_plans import PlanOwner
from pyromind_runtime.ports.experiment_execution import (
    DelegatedExecutionStore,
    ExperimentExecutionStore,
)
from pyromind_runtime.ports.experiment_plans import ExperimentPlanStore


class ExperimentExecutions:
    def __init__(self, plans: ExperimentPlanStore, store: ExperimentExecutionStore):
        self.plans = plans
        self.store = store
        self._checks: dict[str, tuple[float, int]] = {}

    def request(
        self,
        owner: PlanOwner,
        request: RequestExperimentTask,
        allowed_roles: tuple[str, ...],
        require_stage: bool = False,
        execution_policy: dict | None = None,
    ) -> dict:
        if request.role_id not in allowed_roles:
            raise PermissionError("execution role is not enabled")
        if request.task is not None:
            context = self.plans.context(owner, create=not require_stage)
            if context.project_id is None:
                raise PermissionError("Project creation is required")
            frozen = request.model_dump(mode="json")
            if execution_policy is not None:
                frozen["execution_policy"] = execution_policy
            if request.task.artifact_refs:
                rows = {row["id"]: row for row in self.store.list(owner)}
                inputs = dict(request.task.inputs)
                resolved = {}
                for reference in request.task.artifact_refs:
                    source = rows.get(reference.task_run_id)
                    if (
                        not source
                        or source["project_id"] != context.project_id
                        or source["state"] != "succeeded"
                        or source["report_state"] != "ready"
                    ):
                        raise ValueError(
                            "upstream task must have verified outputs "
                            "and a ready report"
                        )
                    artifact = next(
                        (
                            item
                            for item in source.get("artifacts", [])
                            if item["name"] == reference.artifact_name
                        ),
                        None,
                    )
                    if artifact is None or reference.input_name in resolved:
                        raise ValueError("unknown artifact or duplicate input binding")
                    if (
                        reference.input_name in inputs
                        and inputs[reference.input_name] != artifact["uri"]
                    ):
                        raise ValueError(
                            "input differs from its verified artifact reference"
                        )
                    inputs[reference.input_name] = artifact["uri"]
                    resolved[reference.input_name] = {
                        "task_run_id": source["id"],
                        **{key: artifact[key] for key in ("name", "uri", "sha256")},
                    }
                frozen["task"]["inputs"] = inputs
                frozen["input_artifacts"] = resolved
            if require_stage:
                stage = next(
                    (
                        stage
                        for stage in context.context.stages
                        if stage.id == request.stage_id
                    ),
                    None,
                )
                if stage is None or not stage.enabled or stage.placeholder:
                    raise ValueError("task requires an enabled stage in this Project")
                frozen.update(
                    plan_sequence=context.sequence,
                    stage_snapshot=stage.model_dump(mode="json"),
                )
            return self.store.request(
                owner,
                context.project_id,
                None,
                request.request_id,
                request.role_id,
                request.request_id,
                task_spec=frozen,
            )
        view = self.plans.view(owner)
        if not view.current or view.current_revision != request.expected_revision:
            raise ValueError("execution requires the current confirmed plan")
        task = next(
            (t for t in view.current.plan.tasks if t.id == request.task_id), None
        )
        if task is None:
            raise ValueError("task is absent from confirmed plan")
        stage = next(s for s in view.current.plan.stages if s.id == task.stage_id)
        if not stage.enabled or stage.placeholder:
            raise ValueError("task stage is disabled")
        assert view.project_id is not None
        assert request.task_id is not None
        return self.store.request(
            owner,
            view.project_id,
            request.expected_revision,
            request.task_id,
            request.role_id,
            request.request_id,
        )

    def decide(
        self,
        owner: PlanOwner,
        run_id: str,
        command_id: str,
        revision: int,
        fingerprint: str,
        decision: str,
    ) -> dict:
        view = self.plans.view(owner)
        if decision == "allow" and view.current_revision != revision:
            raise ValueError("plan changed; prepare a new execution proposal")
        return self.store.decide(
            owner, run_id, command_id, revision, fingerprint, decision
        )

    def list(self, owner: PlanOwner) -> tuple[ExperimentExecutionView, ...]:
        versions = {}
        result = []
        for row in self.store.list(owner):
            revision = row["revision"]
            if row.get("task_spec"):
                spec, details = row["task_spec"], row.get("result") or {}
                submission = row.get("submission") or {}
                result_view = ExperimentExecutionView(
                    run_id=row["id"],
                    project_id=row["project_id"],
                    task_id=row["task_id"],
                    stage_id=spec.get("stage_id"),
                    plan_sequence=spec.get("plan_sequence"),
                    stage_snapshot=spec.get("stage_snapshot"),
                    task_name=spec["task"]["name"],
                    role_id=row["role_id"],
                    state=row["state"],
                    report_state=row["report_state"],
                    metrics=row.get("metrics"),
                    platform_task_id=submission.get("task_id"),
                    intent=spec["intent"],
                    summary=details.get("summary"),
                    report_uri=details.get("report_uri"),
                    blockers=tuple(details.get("blockers", [])),
                    waiting_reason=details.get("waiting_reason"),
                    checkpoint=CheckpointView.model_validate(
                        {
                            k: v
                            for k, v in row["attributes"]["checkpoint"].items()
                            if k != "decision_id"
                        }
                    )
                    if (row.get("attributes") or {}).get("checkpoint")
                    else None,
                    artifacts=tuple(
                        {key: a[key] for key in ("name", "uri", "sha256")}
                        for a in row.get("artifacts", [])
                    ),
                )
                result.append(result_view)
                continue
            if revision not in versions:
                versions[revision] = self.plans.revision(owner, revision)
            plan = versions[revision].plan
            task = next(t for t in plan.tasks if t.id == row["task_id"])
            submission = row.get("submission") or {}
            result.append(
                ExperimentExecutionView(
                    run_id=row["id"],
                    project_id=row["project_id"],
                    plan_revision=revision,
                    task_id=task.id,
                    stage_id=task.stage_id,
                    task_name=task.name,
                    role_id=row["role_id"],
                    state=row["state"],
                    report_state=row["report_state"],
                    fingerprint=row.get("fingerprint"),
                    proposal=row.get("proposal"),
                    metrics=row.get("metrics"),
                    platform_task_id=submission.get("task_id"),
                    artifacts=tuple(
                        {"name": a["name"], "uri": a["uri"], "sha256": a["sha256"]}
                        for a in row.get("artifacts", [])
                    ),
                )
            )
        return tuple(result)

    async def role_call(
        self,
        owner: PlanOwner,
        role: str,
        tool: str,
        arguments: dict,
        scope: dict,
        execute: Callable[..., Awaitable[dict]],
        read: Callable[..., Awaitable[bytes]],
        inspect: Callable[..., Awaitable[dict]] | None = None,
    ) -> dict:
        store = cast(DelegatedExecutionStore, self.store)
        row = await asyncio.to_thread(
            store.role_scope, owner, scope["run_id"], role, scope
        )
        if not row.get("task_spec"):
            raise PermissionError("this task uses the legacy preparation contract")
        policy = (row.get("task_spec") or {}).get("execution_policy") or {}
        submission_tool = policy.get("submission_tool", "df_submit_evaluation")
        if (
            tool in {"df_submit_evaluation", "df_submit_pipeline", "submit_workflow"}
            and tool != submission_tool
        ):
            raise PermissionError(
                "submission capability is not enabled for this execution"
            )
        if tool == "record_execution_checkpoint":
            checkpoint = await verify_checkpoint(
                owner, row, scope, arguments, read, inspect
            )
            await asyncio.to_thread(
                store.record_checkpoint, owner, row["id"], scope, checkpoint
            )
            return {
                "run_id": row["id"],
                "checkpoint": checkpoint,
                "state": "waiting_user",
            }
        if tool == "get_experiment_execution":
            if arguments.get("run_id", row["id"]) != row["id"]:
                raise PermissionError("execution belongs to another task")
            views = await asyncio.to_thread(self.list, owner)
            return next(
                item.model_dump(mode="json")
                for item in views
                if item.run_id == row["id"]
            )
        if tool == "analyze_task_failure":
            if arguments.get("task_id") not in {
                sub.get("task_id") for sub in row.get("submissions", [])
            }:
                raise PermissionError("analysis belongs to another execution")
        if tool == "df_check_progress":
            receipts = [
                sub.get("evidence", {}).get("receipt", {})
                for sub in row.get("submissions", [])
            ]
            if not any(
                arguments.get("output_dir") == receipt.get("output_dir")
                for receipt in receipts
            ):
                raise PermissionError("progress belongs to another execution")
        if tool == "df_submit_pipeline" and policy.get("require_review"):
            await require_approved_checkpoint(owner, row, arguments, read, inspect)
        if tool in {"df_submit_evaluation", "df_submit_pipeline"}:
            reserved = await asyncio.to_thread(
                store.managed_submission,
                owner,
                row["id"],
                scope,
                arguments
                if tool == "df_submit_evaluation"
                else {**arguments, "operation": "pipeline"},
            )
            if reserved.get("receipt"):
                receipt = reserved["receipt"]
                return {
                    "content": [{"type": "text", "text": json.dumps(receipt)}],
                    "details": receipt,
                    "is_error": bool(receipt.get("is_error")),
                }
            scope = {**scope, "operation_id": reserved["operation_id"]}
        result = await execute(owner, role, scope, tool, arguments)
        if tool in {"df_submit_evaluation", "df_submit_pipeline"}:
            details = result.get("details") or {}
            receipt = {
                key: details[key]
                for key in (
                    "run_id",
                    "task_id",
                    "output_dir",
                    "status",
                    "execution_revision",
                    "correlation",
                )
                if details.get(key) is not None
            }
            receipt["is_error"] = bool(result.get("is_error"))
            receipt["uncertain"] = "uncertain" in str(result.get("content", "")).lower()
            await asyncio.to_thread(
                store.managed_receipt,
                owner,
                row["id"],
                scope,
                str(arguments.get("operation", "inference"))
                if tool == "df_submit_evaluation"
                else "pipeline",
                receipt,
            )
        return result

    async def poll(
        self,
        query: Callable[..., Awaitable[dict]],
        read: Callable[..., Awaitable[bytes]],
    ) -> None:
        store = cast(DelegatedExecutionStore, self.store)
        failure_message = "Analyze the platform failure and continue the task."
        await asyncio.to_thread(store.apply_observations)
        conflicts = await asyncio.to_thread(store.conflicting_tasks)
        for pending in await asyncio.to_thread(store.pending_runs):
            run_id = pending["id"]
            rows_owner = PlanOwner(
                user_id=pending["user_id"],
                conversation_id=pending["conversation_id"],
                profile_id=pending["profile_id"],
            )
            rows = await asyncio.to_thread(store.list, rows_owner)
            row = next(row for row in rows if row["id"] == run_id)
            sub = row.get("submission") or {}
            conflict = sub.get("task_id") in conflicts
            if (
                not conflict
                and row["state"] == "failed"
                and row["report_state"] == "not_started"
                and sub.get("state") == "failed"
                and (
                    (row.get("attributes") or {}).get("current_submission_id")
                    or (row.get("attributes") or {}).get("workflow_request_id")
                )
            ):
                await asyncio.to_thread(
                    store.enqueue_report,
                    rows_owner,
                    run_id,
                    failure_message,
                    platform_task_id=sub["task_id"],
                )
                continue
            if not conflict and pending["state"] not in {
                "waiting_platform",
                "waiting_artifacts",
                "cancel_requested",
            }:
                continue
            due, failures = self._checks.get(run_id, (0, 0))
            if not conflict and due > time.monotonic():
                continue
            owner = PlanOwner(
                user_id=pending["user_id"],
                conversation_id=pending["conversation_id"],
                profile_id=pending["profile_id"],
            )
            if not sub.get("task_id"):
                continue
            try:
                status = str((await query(owner, sub["task_id"]))["status"]).lower()
                status = {
                    "success": "succeeded",
                    "error": "failed",
                    "terminated": "stopped",
                    "completed": "succeeded",
                }.get(status, status)
                await asyncio.to_thread(
                    store.ingest,
                    f"{sub['task_id']}:{time.time_ns()}",
                    sub["task_id"],
                    "query",
                    {"status": status},
                )
                await asyncio.to_thread(store.apply_observations)
                current = next(
                    row
                    for row in await asyncio.to_thread(store.list, owner)
                    if row["id"] == run_id
                )
                if (current.get("submission") or {}).get("id") != sub["id"]:
                    continue
                await asyncio.to_thread(store.note_wait, owner, run_id, "")
                if (
                    current["state"] == "waiting_artifacts"
                    and (
                        (current.get("task_spec") or {}).get("execution_policy") or {}
                    ).get("artifact_contract", "evaluation")
                    != "evaluation"
                ):
                    await asyncio.to_thread(
                        store.queue_artifact_report,
                        owner,
                        run_id,
                        platform_task_id=sub["task_id"],
                    )
                    self._checks[run_id] = (time.monotonic() + 30, 0)
                    continue
                if current["state"] == "waiting_artifacts":
                    primary = sub
                    operation = sub["evidence"].get("operation", sub["operation_key"])
                    if operation not in {"inference", "resume"}:
                        primary = next(
                            (
                                entry
                                for entry in current["submissions"]
                                if entry["evidence"].get(
                                    "operation", entry["operation_key"]
                                )
                                == "inference"
                                and entry["state"] == "succeeded"
                            ),
                            sub,
                        )
                    root = primary["evidence"]["receipt"]["output_dir"].rstrip("/")
                    files = {
                        name: await read(owner, root + "/" + name)
                        for name in ("metrics.json", "report.json", "predictions.jsonl")
                    }
                    metrics, report = (
                        json.loads(files["metrics.json"]),
                        json.loads(files["report.json"]),
                    )
                    count = sum(
                        bool(line.strip())
                        for line in files["predictions.jsonl"].splitlines()
                    )
                    if (
                        not isinstance(metrics, dict)
                        or not isinstance(report, dict)
                        or report.get("status") != "succeeded"
                        or report.get("evaluation_status") != "validated"
                        or report.get("metrics") != metrics
                        or count != metrics.get("total")
                        or not isinstance(metrics.get("evaluated_cases"), int)
                        or metrics["evaluated_cases"] <= 0
                        or metrics["evaluated_cases"]
                        + metrics.get("api_or_evaluation_errors", 0)
                        != count
                    ):
                        raise ValueError(
                            "evaluation artifacts do not prove valid coverage"
                        )
                    verified = [
                        {
                            "name": name,
                            "uri": root + "/" + name,
                            "sha256": hashlib.sha256(body).hexdigest(),
                        }
                        for name, body in files.items()
                    ]
                    await asyncio.to_thread(
                        store.register_artifacts,
                        owner,
                        run_id,
                        verified,
                        metrics,
                        platform_task_id=sub["task_id"],
                    )
                elif current["state"] == "failed":
                    await asyncio.to_thread(
                        store.enqueue_report,
                        owner,
                        run_id,
                        failure_message,
                        platform_task_id=sub["task_id"],
                    )
                interval = 120 if current["state"] == "waiting_platform" else 30
                self._checks[run_id] = (time.monotonic() + interval, 0)
            except PermissionError:
                await asyncio.to_thread(
                    store.note_wait,
                    owner,
                    run_id,
                    "等待重新授权：重新登录并在此会话继续。",
                )
                self._checks[run_id] = (time.monotonic() + 30, failures)
            except Exception:
                # Platform queries/late artifacts are safe to retry. This never
                # calls a submission tool or asks a model to repeat evaluation.
                self._checks[run_id] = (
                    time.monotonic() + min(300, 5 * 4 ** min(failures, 3)),
                    failures + 1,
                )
                unavailable = pending["state"] == "waiting_artifacts" and failures >= 3
                await asyncio.to_thread(
                    store.note_wait,
                    owner,
                    run_id,
                    "平台结果尚不能验证，正在重试读取产物或查询任务。"
                    if not unavailable
                    else "平台成功但产物未通过覆盖范围或格式校验；"
                    "不能宣告 baseline 有效。",
                    artifact_failure=unavailable,
                )
                if unavailable:
                    await asyncio.to_thread(
                        store.enqueue_report,
                        owner,
                        run_id,
                        "Investigate missing or invalid artifacts and continue.",
                        platform_task_id=sub["task_id"],
                    )
