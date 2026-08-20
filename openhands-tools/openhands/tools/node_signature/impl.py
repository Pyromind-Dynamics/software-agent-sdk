"""Implementation of the node signature executor.

Fetches function signature from the Pyromind middleware nodes API.
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
    decompress_gzip_base64_data,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.impl.local_conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState

logger = get_logger(__name__)

PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET = "auth_token"

_PRE_URL = "https://pre-api-portal.pyromind.ai/user_node/function_signature"
_PROD_URL = "https://api-portal.pyromind.ai/user_node/function_signature"
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
        """Fetch the function signature of one or more workflow nodes."""
        node_names = [n for n in (action.node_names or []) if n]
        if action.node_name and action.node_name not in node_names:
            node_names.insert(0, action.node_name)
        node_type = action.node_type
        include_source = action.include_source

        if conversation is None:
            return NodeSignatureObservation(
                status="error",
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
                error_message="No auth token available for middleware API access.",
            )

        if not node_names:
            return NodeSignatureObservation(
                status="error",
                error_message=(
                    "Either node_name or node_names with at least one entry is required."
                ),
            )

        return self._batch_query(
            auth_token=auth_token,
            endpoint=_resolve_endpoint(self._env),
            node_names=node_names,
            node_type=node_type,
            include_source=include_source,
        )

    def _batch_query(
        self,
        *,
        auth_token: str,
        endpoint: str,
        node_names: list[str],
        node_type: str | None,
        include_source: bool,
    ) -> NodeSignatureObservation:
        """Batch query: one HTTP request for all requested nodes."""
        headers = _build_access_key_request_headers(
            origin_headers=self._headers,
            auth_token=auth_token,
        )
        try:
            response = httpx.post(
                f"{endpoint}/batch",
                json={
                    "node_names": node_names,
                    "node_type": node_type,
                    "include_source": include_source,
                    "max_source_lines": 300,
                    "compressed": True,
                },
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            if not payload.get("success"):
                return NodeSignatureObservation(
                    status="error",
                    node_name=node_names[0],
                    error_message=f"API error: {payload.get('message') or 'Unknown error'}",
                )

            entries = payload.get("data", {})
            if isinstance(entries, str):
                entries = decompress_gzip_base64_data(entries)
            if not isinstance(entries, dict):
                return NodeSignatureObservation(
                    status="error",
                    node_name=node_names[0],
                    error_message="Unexpected signature response payload.",
                )
            results: list[dict] = []
            all_ok = True
            for name in node_names:
                entry = entries.get(name) or {}
                if entry.get("success"):
                    sig = entry.get("data", {})
                    results.append(
                        {
                            "node_name": name,
                            "success": True,
                            "function_signature": sig.get("function_signature"),
                            "docstring": sig.get("docstring"),
                            "parameters": sig.get("parameters"),
                            "source_code": sig.get("source_code"),
                        }
                    )
                else:
                    all_ok = False
                    results.append(
                        {
                            "node_name": name,
                            "success": False,
                            "error_message": entry.get("message") or "Unknown error",
                        }
                    )
            return NodeSignatureObservation(
                status="success" if all_ok else "error",
                results=results,
                error_message=(
                    None
                    if all_ok
                    else f"Failed for {len(node_names) - sum(1 for r in results if r['success'])} node(s)"
                ),
            )

        except httpx.HTTPError as e:
            return NodeSignatureObservation(
                status="error",
                node_name=node_names[0] if node_names else None,
                error_message=f"HTTP error fetching node signatures: {e}",
            )
        except Exception as e:
            logger.exception("Error fetching node function signatures")
            return NodeSignatureObservation(
                status="error",
                node_name=node_names[0] if node_names else None,
                error_message=f"Error fetching node signatures: {e}",
            )
