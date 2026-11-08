from harness_adapter.openhands_adapter.session_factory import request_from_context
from pyromind_agent_server.auth_context import request_context
from pyromind_runtime.domain.context import RequestContext

from openhands.agent_server.pyromind_auth import CurrentLoginUser


def test_legacy_request_keeps_cookie_and_cluster_headers() -> None:
    request = request_from_context(
        RequestContext(
            user_id="42",
            cookie="auth_token=test-token; theme=dark",
            authorization="Bearer test-token",
            x_cluster="us-west-1#pre",
            accept_language="zh-CN",
        )
    )

    assert request.headers["cookie"] == "auth_token=test-token; theme=dark"
    assert request.headers["authorization"] == "Bearer test-token"
    assert request.headers["x-cluster"] == "us-west-1#pre"
    assert request.state.current_user.cookie == "auth_token=test-token; theme=dark"
    assert request.state.current_user.x_cluster == "us-west-1#pre"


def test_product_request_uses_header_instead_of_login_cluster() -> None:
    request = request_from_context(
        RequestContext(user_id="42", x_cluster="us-west-2#pre")
    )
    request.state.current_user = CurrentLoginUser(
        user_id=42,
        username="user-42",
        email="user-42@example.test",
        x_cluster="us-west-1",
    )

    context = request_context(request)
    assert context.user_id == "42"
    assert context.x_cluster == "us-west-2#pre"
