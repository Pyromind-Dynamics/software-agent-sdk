from harness_adapter.tool_context import current_user_from_context
from pyromind_agent_server.auth_context import request_context
from pyromind_common.platform.auth import CurrentLoginUser
from pyromind_runtime.domain.context import RequestContext
from starlette.requests import Request


def test_pi_tool_user_keeps_cookie_and_cluster_headers():
    context = RequestContext(
        user_id="42",
        cookie="auth_token=test-token; theme=dark",
        authorization="Bearer test-token",
        x_cluster="us-west-1#pre",
    )
    user = current_user_from_context(context)
    assert user is not None
    assert user.user_id == 42
    assert user.cookie == context.cookie
    assert user.x_cluster == context.x_cluster


def test_product_request_uses_header_instead_of_login_cluster():
    request = Request({"type": "http", "headers": [(b"x-cluster", b"us-west-2#pre")]})
    request.state.current_user = CurrentLoginUser(
        user_id=42,
        username="user-42",
        email="user-42@example.test",
        x_cluster="us-west-1",
    )
    context = request_context(request)
    assert context.user_id == "42"
    assert context.x_cluster == "us-west-2#pre"
