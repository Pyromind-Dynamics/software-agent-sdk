"""Bounded delegated work; native execution remains behind a Harness port."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

from pyromind_runtime.domain.experiment_plans import PlanOwner
from pyromind_runtime.ports.experiment_execution import DelegatedExecutionStore
from pyromind_runtime.ports.experiment_plans import ExperimentPlanStore


Control = Callable[[PlanOwner, str, dict], Awaitable[dict]]


class DelegatedExecution:
    def __init__(
        self,
        plans: ExperimentPlanStore,
        store: DelegatedExecutionStore,
        control: Control,
    ):
        self.plans = plans
        self.store = store
        self.control = control

    async def steer(self, item: dict) -> dict:
        owner = PlanOwner.model_validate(item["payload"]["owner"])
        run_id = item["payload"]["run_id"]
        row = next(
            row
            for row in await asyncio.to_thread(self.store.list, owner)
            if row["id"] == run_id
        )
        role = row["role_id"]
        message = (
            item["payload"]["message"]
            or "Continue the previous task with its existing context."
        )
        while True:
            target = await asyncio.to_thread(
                self.store.guidance_target,
                owner,
                run_id,
                item["payload"]["target_work_id"],
            )
            reference = target["checkpoint"].get("execution_ref")
            if target["state"] != "running":
                break
            if not reference:
                await asyncio.sleep(0.1)
                continue
            scope = {
                "work_id": target["id"],
                "generation": target["generation"],
                "run_id": run_id,
            }
            try:
                result = await self.control(
                    owner,
                    role,
                    {
                        "action": "status",
                        "execution_ref": reference,
                        "scope": scope,
                    },
                )
            except (ValueError, PermissionError):
                latest_work = await asyncio.to_thread(
                    self.store.guidance_target, owner, run_id, target["id"]
                )
                if latest_work["state"] == "running":
                    raise
                break
            if result["status"] not in {"running", "queued"}:
                break
            try:
                result = await self.control(
                    owner,
                    role,
                    {
                        "action": "steer",
                        "execution_ref": reference,
                        "scope": scope,
                        "task": message,
                        "request_id": item["id"],
                    },
                )
            except Exception:
                # A child can finish between status and steer. Preserve the message
                # for a normal resume, and surface other delivery failures.
                latest = await self.control(
                    owner,
                    role,
                    {
                        "action": "status",
                        "execution_ref": reference,
                        "scope": scope,
                    },
                )
                if latest["status"] in {"running", "queued", "unknown"}:
                    raise
                break
            return {"execution_ref": reference, "delivery": "steered"}
        work_id = await asyncio.to_thread(
            self.store.enqueue_continuation, owner, run_id, item["id"], message
        )
        return {"delivery": "queued", "work_id": work_id}

    async def prepare(self, item: dict) -> dict:
        owner = PlanOwner.model_validate(item["payload"]["owner"])
        run_id = item["payload"]["run_id"]
        # The role is read from the server-owned run, never supplied by a tool call.
        rows = await asyncio.to_thread(self.store.list, owner)
        row = next(row for row in rows if row["id"] == run_id)
        role = row["role_id"]
        scope = {
            "work_id": item["id"],
            "generation": item["generation"],
            "run_id": run_id,
        }
        scoped = await asyncio.to_thread(
            self.store.role_scope, owner, run_id, role, scope
        )
        specification = row.get("task_spec")
        followup = item["payload"].get("platform_followup")
        if followup and followup != (scoped.get("submission") or {}).get("task_id"):
            return {"state": scoped["state"], "report_state": scoped["report_state"]}
        report = item["kind"] == "generate_report" or bool(followup)
        continuation = item["payload"].get("continuation", False)
        if specification:
            task_name = specification["task"]["name"]
            task_body = {
                "run_id": run_id,
                **specification,
                "phase": "report" if report else "execute",
                "metrics": row.get("metrics"),
                "checkpoint": (row.get("attributes") or {}).get("checkpoint"),
                "artifacts": row.get("artifacts", []),
                "submission": {
                    key: value
                    for key, value in (row.get("submission") or {}).items()
                    if key in {"task_id", "state", "evidence"}
                },
                "message": item["payload"].get("message", ""),
                "conversation_followup": continuation,
                "instruction": (
                    "Use your role Skills and tools for the supplied frozen task. "
                    "Execute when intent=execute; prepare only when intent=prepare. "
                    "Runtime delivers platform results to this task. Diagnose failures "
                    "and revise or rerun within the assigned scope. "
                    "Follow your role instructions for reporting outcomes or blockers. "
                    "No Graph confirmation is required. "
                    "For conversation_followup, respond to the Manager guidance in "
                    "this same context."
                ),
            }
        else:
            revision = await asyncio.to_thread(
                self.plans.revision, owner, row["revision"]
            )
            task = next(t for t in revision.plan.tasks if t.id == row["task_id"])
            task_name = task.name
            task_body = (
                {}
                if item["checkpoint"]
                else {
                    "run_id": run_id,
                    "goal": revision.plan.goal,
                    "task": task.model_dump(mode="json"),
                    "resources": [
                        r.model_dump(mode="json") for r in revision.plan.resources
                    ],
                    "message": item["payload"].get("message", ""),
                    "instruction": "Use submit_experiment_evaluation to prepare "
                    "a proposal. Do not start tasks.",
                }
            )
        display = {"task": task_name, "phase": "report" if report else "execute"}
        if (
            specification
            and specification.get("stage_id") is not None
            and specification.get("plan_sequence") is not None
        ):
            scope["execution"] = {
                "project_id": row["project_id"],
                "task_run_id": run_id,
                "stage_id": specification["stage_id"],
                "plan_sequence": specification["plan_sequence"],
            }
        await self.control(owner, role, {"action": "authorize", "scope": scope})
        checkpoint = item["checkpoint"]
        interruption_retried = checkpoint.get("interruption_retried", False)

        async def launch(prior: str | None, request_id: str) -> dict:
            action = "resume" if prior else "spawn"
            await asyncio.to_thread(
                self.store.save_work_checkpoint,
                item["id"],
                item["generation"],
                {
                    "launch_pending": True,
                    "launch_kind": action,
                    "launch_request_id": request_id,
                    "interruption_retried": interruption_retried,
                },
            )
            return await self.control(
                owner,
                role,
                {
                    "action": action,
                    "role": role,
                    **({"execution_ref": prior} if prior else {}),
                    "task": json.dumps(task_body, ensure_ascii=False),
                    "display": display,
                    "request_id": request_id,
                    "scope": scope,
                },
            )

        if checkpoint.get("launch_pending"):
            # A missing response is not permission to start a duplicate child.
            result = await self.control(
                owner,
                role,
                {
                    "action": "status",
                    "launch_request_id": checkpoint.get(
                        "launch_request_id", item["id"]
                    ),
                    "launch_kind": checkpoint.get("launch_kind", "spawn"),
                    "scope": scope,
                },
            )
        elif checkpoint.get("execution_ref"):
            result = await self.control(
                owner,
                role,
                {
                    "action": "status",
                    "execution_ref": checkpoint["execution_ref"],
                    "scope": scope,
                },
            )
            if (
                specification
                and checkpoint.get("generation", item["generation"])
                != item["generation"]
            ):
                while result.get("status") in {"running", "queued", "stopping"}:
                    await self.control(
                        owner,
                        role,
                        {
                            "action": "pause",
                            "execution_ref": checkpoint["execution_ref"],
                            "scope": scope,
                        },
                    )
                    await asyncio.sleep(0.5)
                    result = await self.control(
                        owner,
                        role,
                        {
                            "action": "status",
                            "execution_ref": checkpoint["execution_ref"],
                            "scope": scope,
                        },
                    )
                if result.get("status") in {"stopped", "paused"} and (
                    row["state"] in {"queued", "preparing"} or report or continuation
                ):
                    # A native hard stop seals its session. Once termination is
                    # proven, a fresh child can re-read the durable task facts;
                    # the existing submission ledger still forbids duplicates.
                    stopped = result["status"] == "stopped" and not continuation
                    result = await launch(
                        None if stopped else checkpoint["execution_ref"],
                        item["id"] + ":recovery:" + str(item["generation"]),
                    )
        else:
            prior = (
                await asyncio.to_thread(self.store.native_reference, owner, run_id)
                if specification or continuation
                else None
            )
            if report and prior and not continuation:
                previous = await self.control(
                    owner,
                    role,
                    {"action": "status", "execution_ref": prior, "scope": scope},
                )
                if previous["status"] == "stopped":
                    prior = None
                elif previous["status"] not in {"succeeded", "failed", "paused"}:
                    raise ValueError("report context requires reconciliation")
            if continuation and prior is None:
                raise ValueError(
                    "the original child context is unavailable; "
                    "delegate a fresh task explicitly"
                )
            result = await launch(prior, item["id"])
        reference = None
        try:
            while True:
                reference = result.get("execution_ref")
                if not reference or result.get("status") == "unknown":
                    raise ValueError("delegated launch needs reconciliation")
                await asyncio.to_thread(
                    self.store.save_work_checkpoint,
                    item["id"],
                    item["generation"],
                    {
                        "execution_ref": reference,
                        "generation": item["generation"],
                        "interruption_retried": interruption_retried,
                    },
                )
                while result["status"] in {"running", "queued", "stopping"}:
                    await asyncio.sleep(0.5)
                    result = await self.control(
                        owner,
                        role,
                        {
                            "action": "status",
                            "execution_ref": reference,
                            "scope": scope,
                        },
                    )
                if result["status"] not in {"succeeded", "failed", "stopped", "paused"}:
                    raise ValueError("delegated state requires reconciliation")
                current = await asyncio.to_thread(
                    self.store.role_scope, owner, run_id, role, scope
                )
                submission = current.get("submission") or {}
                report_only = (
                    report
                    and submission.get("evidence", {}).get("work_id") != item["id"]
                )
                if not (
                    report_only
                    and result["status"] == "failed"
                    and result.get("interrupted")
                    and not interruption_retried
                    and current["report_state"] != "ready"
                    and (current.get("submission") or {}).get("state") == "succeeded"
                ):
                    break
                interruption_retried = True
                previous_reference, reference = reference, None
                result = await launch(previous_reference, item["id"] + ":interruption")
            if specification:
                if not continuation or current["state"] in {"queued", "preparing"}:
                    result_text = (result.get("result_text") or "").strip()
                    current = await asyncio.to_thread(
                        self.store.complete_agent_turn,
                        owner,
                        run_id,
                        scope,
                        result_text or "The child ended without a result.",
                        report_only,
                        completed=bool(result_text) and result["status"] == "succeeded",
                    )
                return {
                    "execution_ref": reference,
                    "state": current["state"],
                    "report_state": current["report_state"],
                }
            if current["state"] != "awaiting_confirmation":
                await asyncio.to_thread(
                    self.store.preparation_needs_input,
                    owner,
                    run_id,
                    scope,
                    result.get("result_text")
                    or "Preparation needs input; inspect the execution details.",
                )
                return {
                    "execution_ref": reference,
                    "proposal_ready": False,
                    "needs_input": True,
                }
            return {"execution_ref": reference, "proposal_ready": True}
        except asyncio.CancelledError:
            # Stop intent is durable in the native owner; a subsequent recovery
            # still queries actual lifecycle proof before admitting more work.
            if reference:
                await self.control(
                    owner,
                    role,
                    {"action": "pause", "execution_ref": reference, "scope": scope},
                )
            raise

    async def reconcile(self, item: dict) -> bool:
        # Recovery reuses the same native reference (or launch correlation).
        # Do not replay a missing or ambiguous launch automatically.
        return bool(
            item["checkpoint"].get("execution_ref")
            or item["checkpoint"].get("launch_pending")
        )
