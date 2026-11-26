"""Read the owner's submitted XYFlow definition from the Studio task archive."""

import httpx

from pyromind_common.tools.utils.platform_env import platform_portal_url
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext

from .training_metric_source import ReadBudget, bounded_json


async def read_accepted_workflow(task_id: str, context: RequestContext) -> JsonObject:
    if not task_id.isdecimal() or not (context.cookie or context.authorization):
        raise ValueError("Studio task and owner credentials required")
    headers = {"x-cluster": context.x_cluster or ""}
    if context.cookie:
        headers["cookie"] = context.cookie
    if context.authorization:
        headers["authorization"] = context.authorization
    base = platform_portal_url(
        "https://api-portal.pyromind.ai/std2/studio_api/", headers
    )
    async with httpx.AsyncClient(timeout=5, follow_redirects=False) as client:
        result = await bounded_json(
            client,
            "GET",
            base + "api/task_workflow_result",
            headers=headers,
            params={"task_id": task_id},
            budget=ReadBudget(4 * 1024 * 1024),
        )
    graph = result.get("workflow")
    if not isinstance(graph, dict):
        raise ValueError("Studio returned no submitted workflow")
    extra = graph.get("extra")
    if isinstance(extra, dict) and str(extra.get("taskId", task_id)) != task_id:
        raise ValueError("Studio returned a different task")
    return graph
