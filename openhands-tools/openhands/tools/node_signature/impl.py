"""Implementation of the node signature executor.

Fetches function signature from the Pyromind middleware agent-internal API.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast

import httpx

from openhands.sdk.logger import get_logger
from openhands.sdk.tool import ToolExecutor
from openhands.tools.node_signature.definition import (
    NodeSignatureAction,
    NodeSignatureObservation,
)
from openhands.tools.utils.pyromind_api_client import (
    _build_access_key_request_headers,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.impl.local_conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState

logger = get_logger(__name__)

PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET = "auth_token"

_PRE_URL = (
    "https://pre-api-portal.pyromind.ai/std2/studio_api/api/agent"
    "/nodes/function_signature"
)
_PROD_URL = (
    "https://api-portal.pyromind.ai/std2/studio_api/api/agent/nodes/function_signature"
)
_PROD_APP_ENVS = {"prod", "production", "online"}


def _resolve_endpoint(env: str | None) -> str:
    app_env = (env or os.getenv("APP_ENV", "dev")).strip().lower()
    if app_env in _PROD_APP_ENVS:
        return _PROD_URL
    return _PRE_URL


class NodeSignatureExecutor(
    ToolExecutor[NodeSignatureAction, NodeSignatureObservation]
):
    """Fetches a node's function signature from the middleware API."""

    def __init__(
        self,
        env: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._env = env
        self._headers = dict(headers or {})

    def __call__(
        self,
        action: NodeSignatureAction,
        conversation: LocalConversation | None = None,
    ) -> NodeSignatureObservation:
        """Fetch the function signature of a workflow node."""
        node_name = action.node_name
        node_type = action.node_type
        include_source = action.include_source

        if conversation is None:
            return NodeSignatureObservation(
                status="error",
                node_name=node_name,
                error_message=(
                    "get_node_function_signature requires a conversation context."
                ),
            )

        # Get auth token from conversation state
        state = cast("ConversationState", conversation.state)
        auth_token = state.secret_registry.get_secret_value(
            PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET
        )
        if not auth_token:
            return NodeSignatureObservation(
                status="error",
                node_name=node_name,
                error_message="No auth token available for middleware API access.",
            )

        # Build the endpoint URL
        endpoint = _resolve_endpoint(self._env)

        # Build auth headers
        headers = _build_access_key_request_headers(
            origin_headers=self._headers,
            auth_token=auth_token,
        )

        try:
            response = httpx.post(
                endpoint,
                json={
                    "node_name": node_name,
                    "node_type": node_type,
                    "include_source": include_source,
                    "max_source_lines": 300,
                },
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("success"):
                error_msg = data.get("message") or "Unknown error"
                return NodeSignatureObservation(
                    status="error",
                    node_name=node_name,
                    error_message=f"API error: {error_msg}",
                )

            result = data.get("data", {})
            return NodeSignatureObservation(
                status="success",
                node_name=node_name,
                function_signature=result.get("function_signature"),
                docstring=result.get("docstring"),
                parameters=result.get("parameters"),
                source_code=result.get("source_code"),
            )

        except httpx.HTTPError as e:
            return NodeSignatureObservation(
                status="error",
                node_name=node_name,
                error_message=f"HTTP error fetching function signature: {e}",
            )
        except Exception as e:
            logger.exception("Error fetching node function signature")
            return NodeSignatureObservation(
                status="error",
                node_name=node_name,
                error_message=f"Error fetching function signature: {e}",
            )
