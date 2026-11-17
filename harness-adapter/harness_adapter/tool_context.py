from __future__ import annotations

from pyromind_common.platform.auth import (
    CurrentLoginUser,
    get_current_login_user_from_token,
    parse_auth_token_from_cookie_header,
)
from pyromind_runtime.domain.context import RequestContext


def current_user_from_context(context: RequestContext) -> CurrentLoginUser | None:
    """为 Pi 业务工具构建请求级用户，保留 Cookie 与集群路由。"""
    token = parse_auth_token_from_cookie_header(context.cookie)
    current_user = get_current_login_user_from_token(token)
    if current_user is not None:
        return current_user.model_copy(
            update={"cookie": context.cookie, "x_cluster": context.x_cluster}
        )
    try:
        user_id = int(context.user_id)
    except ValueError:
        return None
    return CurrentLoginUser(
        username=f"debug-user-{user_id}",
        email=f"debug-user-{user_id}@example.test",
        user_id=user_id,
        group_id=0,
        cookie=context.cookie,
        x_cluster=context.x_cluster,
    )
