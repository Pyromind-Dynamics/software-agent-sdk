from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import UUID

from openhands.agent_server.config import Config
from openhands.agent_server.models import StartConversationRequest
from openhands.agent_server.persistence import PersistedSettings, get_settings_store
from openhands.agent_server.pyromind_constants import (
    PYROMIND_APP_TAG_KEY,
    PYROMIND_APP_TAG_VALUE,
)
from openhands.sdk.workspace import LocalWorkspace
from pyromind_runtime.contracts.harness import SessionSpec


class PersistedSettingsOpenHandsSessionFactory:
    """Build an OpenHands request from the server's existing persisted settings."""

    def __init__(self, config_provider: Callable[[], Config]) -> None:
        self._config_provider = config_provider

    async def __call__(self, spec: SessionSpec) -> StartConversationRequest:
        return await asyncio.to_thread(self._build, spec)

    def _build(self, spec: SessionSpec) -> StartConversationRequest:
        config = self._config_provider()
        settings = get_settings_store(config).load() or PersistedSettings()
        conversation_settings = settings.conversation_settings.model_copy(
            update={"agent_settings": settings.agent_settings}
        )
        return conversation_settings.create_request(
            StartConversationRequest,
            workspace=LocalWorkspace(working_dir=spec.workspace.root),
            conversation_id=UUID(spec.product_session_id),
            initial_message=None,
            tags={PYROMIND_APP_TAG_KEY: PYROMIND_APP_TAG_VALUE},
            user_id=spec.user_id,
        )
