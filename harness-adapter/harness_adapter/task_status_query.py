from __future__ import annotations

import os

import httpx
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.ports.task_monitor import TaskStatus

from pyromind_common.tools.workflow.inspect_workflow_task import (
    PRE_API_BASE,
    PROD_API_BASE,
    USER_AGENT,
    workflow_execution_state,
)


class PlatformTaskStatusQuery:
    async def query(self, task_id: str, context: RequestContext) -> TaskStatus:
        route = (context.x_cluster or "").strip()
        _, separator, routed_env = route.partition("#")
        env = (
            routed_env.strip().lower()
            if separator
            else "prod"
            if route
            else (os.getenv("PYROMIND_ENV") or os.getenv("APP_ENV", "dev")).lower()
        )
        base = (
            PROD_API_BASE if env in {"prod", "production", "online"} else PRE_API_BASE
        )
        if env == "pre2":
            base = PRE_API_BASE.replace("pre-api-portal", "pre2-api-portal")
        base = os.getenv("PYROMIND_API_BASE", "").strip() or base
        headers = {"accept": "application/json", "user-agent": USER_AGENT}
        headers.update(
            {
                name: value
                for name, value in (
                    ("cookie", context.cookie),
                    ("authorization", context.authorization),
                    ("x-cluster", route),
                )
                if value
            }
        )
        async with httpx.AsyncClient(timeout=25) as client:
            response = await client.get(
                f"{base.rstrip('/')}/api/task_workflow_result",
                params={"task_id": task_id},
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("expected a workflow status object")
        raw_status = payload.get("task_status")
        if not isinstance(raw_status, str):
            raise ValueError("workflow response has no task_status")
        status = workflow_execution_state(raw_status)
        if status in {"unknown", "skipped"}:
            raise ValueError("workflow response has no supported task status")
        return TaskStatus("stopped" if status == "cancelled" else status)
