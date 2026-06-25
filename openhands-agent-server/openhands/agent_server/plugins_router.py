"""Plugins router for OpenHands Agent Server.

HTTP API endpoints for plugin operations. Business logic is delegated to
``plugins_service.py``. This module exposes the plugins-only marketplace
catalog; installed-plugin management endpoints are added separately.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from openhands.agent_server.plugins_service import (
    MarketplacePluginInfo,
    service_get_plugins_marketplace_catalog,
)


plugins_router = APIRouter(prefix="/plugins", tags=["Plugins"])


class MarketplaceCatalogResponse(BaseModel):
    """Response containing the plugins marketplace catalog."""

    plugins: list[MarketplacePluginInfo]


@plugins_router.get("/marketplace", response_model=MarketplaceCatalogResponse)
def get_marketplace_catalog() -> MarketplaceCatalogResponse:
    """Get the plugins marketplace catalog with installation status.

    Returns the true plugins (entries whose source lives under ``./plugins/``)
    from the OpenHands extensions repository marketplace, each with attachable
    ``PluginSource`` coordinates (``source`` / ``ref`` / ``repo_path``) and an
    ``installed`` flag. This enables the front-end to render a plugins
    marketplace with install/installed state and to attach plugins to
    conversations.

    Returns:
        MarketplaceCatalogResponse containing the list of available plugins.
    """
    return MarketplaceCatalogResponse(plugins=service_get_plugins_marketplace_catalog())
