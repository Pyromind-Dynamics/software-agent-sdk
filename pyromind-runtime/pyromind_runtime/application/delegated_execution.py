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
        if specification:
            task_body = {
                "run_id": run_id,
                **specification,
                "phase": "report" if report else "execute",
                "metrics": row.get("metrics"),
                "submission": {
                    key: value
                    for key, value in (row.get("submission") or {}).items()
                    if key in {"task_id", "state", "evidence"}
                },
                "message": item["payload"].get("message", ""),
                "instruction": (
                    "Use the evaluation Skill and its existing tools. "
                    "Execute when intent=execute; prepare only when intent=prepare. "
                    "After submission end this turn; Runtime waits for the platform "
                    "and resumes you for the report. Register a report or essential "
                    "blockers with record_experiment_result. "
                    "No Graph confirmation is required."
                ),
            }
        else:
            revision = await asyncio.to_thread(
                self.plans.revision, owner, row["revision"]
            )
            task = next(t for t in revision.plan.tasks if t.id == row["task_id"])
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
                    "instruction": "Use submit_experiment_evaluation to prepare "
                    "a proposal. Do not start tasks.",
                }
            )
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
                    row["state"] in {"queued", "preparing"} or report
                ):
                    import json

                    # A native hard stop seals its session. Once termination is
                    # proven, a fresh child can re-read the durable task facts;
                    # the existing submission ledger still forbids duplicates.
                    stopped = result["status"] == "stopped"
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
                            "request_id": request_id,
                            "scope": scope,
                        },
                    )
        else:
            import json

            prompt = json.dumps(task_body, ensure_ascii=False)
            prior = (
                await asyncio.to_thread(self.store.native_reference, owner, run_id)
                if specification
                else None
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
                current = await asyncio.to_thread(
                    self.store.complete_agent_turn,
                    owner,
                    run_id,
                    scope,
                    result.get("result_text")
                    or "The child ended without a submitted evaluation "
                    "or registered report.",
                    report,
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
