from functools import partial

import httpx
import pytest

from pyromind_agent_server.accepted_workflow import read_accepted_workflow
from pyromind_runtime.domain.context import RequestContext


@pytest.mark.parametrize("mismatch", [True, False])
async def test_studio_task_archive_reads_authenticated_scoped_graph(
    monkeypatch, mismatch
):
    graph = {
        "extra": {"taskId": "wrong" if mismatch else "21"},
        "nodes": [],
        "edges": [],
    }

    def handle(request):
        assert request.method == "GET"
        assert request.url.path.endswith("/studio_api/api/task_workflow_result")
        assert request.url.params["task_id"] == "21"
        assert request.headers["cookie"] == "session=owner"
        assert request.headers["x-cluster"] == "cn-east-1"
        return httpx.Response(200, json={"workflow": graph})

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        partial(httpx.AsyncClient, transport=httpx.MockTransport(handle)),
    )
    context = RequestContext(user_id="u", cookie="session=owner", x_cluster="cn-east-1")
    if mismatch:
        with pytest.raises(ValueError, match="different task"):
            await read_accepted_workflow("21", context)
    else:
        assert await read_accepted_workflow("21", context) == graph
