"""Bounded delegated work; native execution remains behind a Harness port."""

from __future__ import annotations

import asyncio
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
        await asyncio.to_thread(self.store.role_scope, owner, run_id, role, scope)
        specification = row.get("task_spec")
        report = item["kind"] == "generate_report"
        continuation = item["payload"].get("continuation", False)
        if specification:
            task_name = specification["task"]["name"]
            runtime_phase = "report" if report else "execute"
            task_inputs = specification["task"].get("inputs", {})
            child_phase = (
                task_inputs.get("phase", runtime_phase)
                if isinstance(task_inputs, dict)
                else runtime_phase
            )
            task_body = {
                "run_id": run_id,
                **specification,
                "phase": runtime_phase if report else child_phase,
                "runtime_phase": runtime_phase,
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
                "instruction": specification["task"]["instruction"],
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
            child_phase = "report" if report else "execute"
        display = {"task": task_name, "phase": child_phase}
        if (
            specification
            and specification.get("stage_id") is not None
            and specification.get("plan_sequence") is not None
        ):
            scope["execution"] = {
                "project_id": row["project_id"],
                "task_run_id": run_id,
                "stage_id": specification["stage_id"],
                "stage_run_id": specification.get("stage_run_id"),
                "plan_sequence": specification["plan_sequence"],
            }
        await self.control(owner, role, {"action": "authorize", "scope": scope})
        checkpoint = item["checkpoint"]
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
                    import json

                    # A native hard stop seals its session. Once termination is
                    # proven, a fresh child can re-read the durable task facts;
                    # the existing submission ledger still forbids duplicates.
                    stopped = result["status"] == "stopped" and not continuation
                    request_id = item["id"] + ":recovery:" + str(item["generation"])
                    await asyncio.to_thread(
                        self.store.save_work_checkpoint,
                        item["id"],
                        item["generation"],
                        {
                            "launch_pending": True,
                            "launch_kind": "spawn" if stopped else "resume",
                            "launch_request_id": request_id,
                        },
                    )
                    result = await self.control(
                        owner,
                        role,
                        {
                            "action": "spawn" if stopped else "resume",
                            "role": role,
                            **(
                                {}
                                if stopped
                                else {"execution_ref": checkpoint["execution_ref"]}
                            ),
                            "task": json.dumps(task_body, ensure_ascii=False),
                            "display": display,
                            "request_id": request_id,
                            "scope": scope,
                        },
                    )
        else:
            import json

            prompt = json.dumps(task_body, ensure_ascii=False)
            prior = (
                await asyncio.to_thread(self.store.native_reference, owner, run_id)
                if specification or continuation
                else None
            )
            if continuation and prior is None:
                raise ValueError(
                    "the original child context is unavailable; "
                    "delegate a fresh task explicitly"
                )
            await asyncio.to_thread(
                self.store.save_work_checkpoint,
                item["id"],
                item["generation"],
                {"launch_pending": True, "launch_kind": "resume" if prior else "spawn"},
            )
            result = await self.control(
                owner,
                role,
                {
                    "action": "resume" if prior else "spawn",
                    **({"execution_ref": prior} if prior else {}),
                    "role": role,
                    "task": prompt,
                    "display": display,
                    "request_id": item["id"],
                    "scope": scope,
                },
            )
        reference = result.get("execution_ref")
        if not reference or result.get("status") == "unknown":
            raise ValueError("delegated launch needs reconciliation")
        await asyncio.to_thread(
            self.store.save_work_checkpoint,
            item["id"],
            item["generation"],
            {"execution_ref": reference, "generation": item["generation"]},
        )
        try:
            while result["status"] not in {
                "succeeded",
                "failed",
                "stopped",
                "paused",
            }:
                await asyncio.sleep(0.5)
                result = await self.control(
                    owner,
                    role,
                    {"action": "status", "execution_ref": reference, "scope": scope},
                )
                if result.get("status") == "unknown":
                    raise ValueError("delegated state requires reconciliation")
            current = await asyncio.to_thread(
                self.store.role_scope, owner, run_id, role, scope
            )
            if specification:
                result_text = _delegated_result_text(
                    result,
                    succeeded_fallback=(
                        "The child ended without a submission or registered result."
                    ),
                )
                if not continuation or current["state"] in {"queued", "preparing"}:
                    current = await asyncio.to_thread(
                        self.store.complete_agent_turn,
                        owner,
                        run_id,
                        scope,
                        result_text,
                        report,
                    )
                if continuation:
                    notice_id = "guidance:" + item["id"]
                    await asyncio.to_thread(
                        self.store.enqueue_manager,
                        owner,
                        notice_id,
                        {
                            "notice_id": notice_id,
                            "result": {
                                "run_id": run_id,
                                "conversation_followup": True,
                                "state": current["state"],
                                "report_state": current["report_state"],
                                "summary": result_text[:16000],
                            },
                        },
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
                    _delegated_result_text(
                        result,
                        succeeded_fallback=(
                            "Preparation needs input; inspect the execution details."
                        ),
                    ),
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


def _delegated_result_text(result: dict, *, succeeded_fallback: str) -> str:
    status = str(result.get("status") or "unknown")
    result_text = str(result.get("result_text") or "").strip()
    if status == "succeeded":
        return result_text or succeeded_fallback
    outcome = {
        "paused": "paused before completing the task",
        "stopped": "stopped before completing the task",
        "failed": "failed before producing a valid result",
    }.get(status, f"ended with unrecognized status {status}")
    message = f"Delegated subagent {outcome}. This is not a successful business result."
    if result_text:
        message += "\n\nSubagent output:\n" + result_text
    return message
