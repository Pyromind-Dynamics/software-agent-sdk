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
        revision = await asyncio.to_thread(self.plans.revision, owner, row["revision"])
        task = next(t for t in revision.plan.tasks if t.id == row["task_id"])
        await self.control(owner, role, {"action": "authorize", "scope": scope})
        checkpoint = item["checkpoint"]
        if checkpoint.get("launch_pending") and not checkpoint.get("execution_ref"):
            # A missing response is not permission to start a duplicate child.
            result = await self.control(
                owner,
                role,
                {"action": "status", "launch_request_id": item["id"], "scope": scope},
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
        else:
            await asyncio.to_thread(
                self.store.save_work_checkpoint,
                item["id"],
                item["generation"],
                {"launch_pending": True},
            )
            import json

            prompt = json.dumps(
                {
                    "run_id": run_id,
                    "goal": revision.plan.goal,
                    "task": task.model_dump(mode="json"),
                    "resources": [
                        r.model_dump(mode="json") for r in revision.plan.resources
                    ],
                    "instruction": (
                        "Use submit_experiment_evaluation to prepare a proposal. "
                        "Inspect declared inputs. Do not invent paths or start tasks. "
                        "Explain missing inputs in your final response."
                    ),
                },
                ensure_ascii=False,
            )
            result = await self.control(
                owner,
                role,
                {
                    "action": "spawn",
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
            {"execution_ref": reference},
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
                {"action": "stop", "execution_ref": reference, "scope": scope},
            )
            raise

    async def reconcile(self, item: dict) -> bool:
        # Recovery reuses the same native reference (or launch correlation).
        # Do not replay a missing or ambiguous launch automatically.
        return bool(
            item["checkpoint"].get("execution_ref")
            or item["checkpoint"].get("launch_pending")
        )
