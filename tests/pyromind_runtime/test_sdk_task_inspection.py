"""Read task facts and logs through the deployed SDK terminal interpreter."""

from aiohttp import web


async def test_main_queries_raw_node_status_and_logs_without_inspection_tool(
    sdk_terminal_run,
):
    calls = []

    async def task(request):
        calls.append(("task", request.match_info["task_id"]))
        return web.json_response(
            {
                "success": True,
                "data": {
                    "task_id": "1304",
                    "name": "inspection-test",
                    "status": "Failed",
                    "nodes": [
                        {
                            "node_id": 43050,
                            "node_code": "sft",
                            "node_name": "Train",
                            "status": "Failed",
                        },
                        {
                            "node_id": 43051,
                            "node_name": "Merge",
                            "start_at": "2026-11-24T10:00:00Z",
                            "end_at": "2026-11-24T10:01:00Z",
                        },
                    ],
                },
            }
        )

    async def logs(request):
        calls.append(("logs", request.match_info["node_id"]))
        return web.json_response(
            {
                "success": True,
                "data": {"logs": ["setup\nstep one", "last diagnostic\nfailed"]},
            }
        )

    app = web.Application()
    app.router.add_get("/api/v1/studio/tasks/{task_id}", task)
    app.router.add_get("/api/v1/studio/tasks/1304/nodes/{node_id}/logs", logs)

    result = await sdk_terminal_run(
        app,
        """from pyromind_sdk import PyroMindAPIClient
from pyromind_common.platform.inspection import inspect_workflow_task

with PyroMindAPIClient(max_retries=0) as client:
    task = inspect_workflow_task(client, '1304')
    assert task.task_state == 'failed'
    assert task.failed_nodes[0].node_id == '43050'
    assert task.nodes[0].raw_status == 'Failed'
    assert task.nodes[1].state == 'unknown'
    assert task.node_status_available is False
    assert task.log is None
    selected = inspect_workflow_task(
        client, '1304', node_id='sft', include_log=True, tail_lines=2,
    )
    assert selected.selected_node.node_id == '43050'
    assert selected.log == 'last diagnostic\\nfailed'
    assert selected.node_log_available
    print('SDK-TASK-AND-NODE-INSPECTION-VERIFIED')
""",
    )

    assert calls == [("task", "1304"), ("task", "1304"), ("logs", "43050")]
    assert "SDK-TASK-AND-NODE-INSPECTION-VERIFIED" in result, result
