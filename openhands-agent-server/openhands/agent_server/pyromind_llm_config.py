"""Runtime LLM provider configuration for Pyromind conversations.

The server supports an ordered list of LLM providers configured in a JSON
file (``LLM_CONFIG_PATH``, default ``workspace/llm_config.json``). When more
than one provider is configured, the agent's LLM is wrapped in a
``FailoverRouter`` so a provider that fails with a transient error is skipped
automatically in favor of the next healthy provider.

Image-bearing requests can be routed to a separate ordered ``multimodal_llms``
queue so text-only providers never receive image content. When no config file
exists, the legacy single-provider environment variables (``LLM_MODEL`` /
``LLM_BASE_URL`` / ``OPENAI_API_KEY``) are used unchanged.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Final

from pydantic import SecretStr

from openhands.sdk.llm import LLM, FailoverRouter


LLM_CONFIG_PATH_ENV: Final[str] = "LLM_CONFIG_PATH"
LLM_FAILOVER_COOLDOWN_ENV: Final[str] = "LLM_FAILOVER_COOLDOWN_SECONDS"
DEFAULT_LLM_CONFIG_FILENAME: Final[str] = "llm_config.json"
DEFAULT_COOLDOWN_SECONDS: Final[float] = 300.0

_OPENAI_CHAT_COMPLETIONS_SUFFIX: Final[str] = "/chat/completions"


def normalize_openai_base_url(base_url: str | None) -> str | None:
    """Trim the URL and strip a Chat Completions suffix if present."""
    if base_url is None:
        return None
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        return None
    if normalized.endswith(_OPENAI_CHAT_COMPLETIONS_SUFFIX):
        return normalized[: -len(_OPENAI_CHAT_COMPLETIONS_SUFFIX)]
    return normalized


def default_config_path() -> Path:
    from openhands.agent_server.config import _default_workspace_root

    return _default_workspace_root() / DEFAULT_LLM_CONFIG_FILENAME


def _read_config_data() -> dict[str, Any] | list[Any] | None:
    configured = os.environ.get(LLM_CONFIG_PATH_ENV)
    if configured:
        path = Path(configured)
        if not path.exists() and path == default_config_path():
            # The startup script exports the default path unconditionally;
            # a missing default just means the legacy env mode is in use.
            return None
        if not path.exists():
            raise FileNotFoundError(f"LLM config file not found: {path}")
    else:
        path = default_config_path()
        if not path.exists():
            return None

    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return data
    raise ValueError(f"LLM config file must be a JSON object or list: {path}")


def _load_section(section: str, *, required: bool) -> list[dict[str, Any]]:
    """Return the ordered provider entries for a config section (primary first)."""
    data = _read_config_data()
    if data is None:
        return []
    entries = data.get(section) if isinstance(data, dict) else data
    if isinstance(data, list) and section != "llms":
        entries = None
    if entries is None and not required:
        return []
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"LLM config file must contain a non-empty '{section}' list")

    result: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"LLM config entry {index} must be a JSON object")
        if not entry.get("model"):
            raise ValueError(f"LLM config entry {index} is missing 'model'")
        prefix = "vision" if section == "multimodal_llms" else "provider"
        result.append(
            {**entry, "name": str(entry.get("name") or f"{prefix}-{index + 1}")}
        )
    return result


def load_config_entries() -> list[dict[str, Any]]:
    """Return the ordered text LLM provider entries (primary first).

    Returns an empty list when no config file is configured, keeping the
    legacy env-var driven single-provider behavior intact.
    """
    return _load_section("llms", required=True)


def load_multimodal_config_entries() -> list[dict[str, Any]]:
    """Return the optional multimodal (vision) provider entries.

    The ``multimodal_llms`` section is optional; an absent or empty list
    keeps image-bearing requests on the text cohort's vision-capable members.
    """
    return _load_section("multimodal_llms", required=False)


def _failover_cooldown_seconds() -> float:
    raw = os.environ.get(LLM_FAILOVER_COOLDOWN_ENV)
    if raw is None:
        return DEFAULT_COOLDOWN_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_COOLDOWN_SECONDS


def _coerce_secret(value: str | SecretStr | None) -> SecretStr | None:
    if isinstance(value, SecretStr):
        return value
    if isinstance(value, str) and value.strip():
        return SecretStr(value)
    return None


def _provider_api_key(entry: dict[str, Any]) -> str | None:
    """Resolve a provider API key from the config or the environment.

    Precedence: an explicit ``api_key`` field, the env var named by
    ``api_key_env``, the ``LLM_API_KEY_<NAME>`` convention, then the legacy
    ``OPENAI_API_KEY`` / ``LLM_API_KEY`` variables. Deployment manifests inject
    keys as environment variables (e.g. Kubernetes ``secretKeyRef``) so shared
    config files can stay free of secrets.
    """
    raw = entry.get("api_key")
    if isinstance(raw, str) and raw.strip():
        return raw
    env_name = entry.get("api_key_env")
    if isinstance(env_name, str) and env_name.strip():
        value = os.environ.get(env_name.strip())
        if value:
            return value
    name = entry.get("name")
    if isinstance(name, str) and name.strip():
        env_key = "LLM_API_KEY_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper()
        value = os.environ.get(env_key)
        if value:
            return value
    for env_key in ("OPENAI_API_KEY", "LLM_API_KEY"):
        value = os.environ.get(env_key)
        if value:
            return value
    return None


def _provider_llm(existing: LLM, entry: dict[str, Any], primary: bool) -> LLM:
    """Build one provider LLM from a config entry, layered on ``existing``."""
    api_key = _coerce_secret(_provider_api_key(entry)) or _coerce_secret(
        existing.api_key
    )

    usage_id = (
        existing.usage_id
        if primary
        else f"{existing.usage_id}-fallback-{entry['name']}"
    )
    return existing.model_copy(
        update={
            "model": entry["model"],
            "api_key": api_key,
            "base_url": normalize_openai_base_url(
                entry.get("base_url") or existing.base_url
            ),
            "persist_runtime_config": False,
            "usage_id": usage_id,
            "num_retries": int(entry.get("num_retries", 2)),
            "retry_min_wait": int(entry.get("retry_min_wait", 4)),
            "retry_max_wait": int(entry.get("retry_max_wait", 16)),
        }
    )


def build_runtime_llm(existing: LLM) -> LLM:
    """Build the runtime LLM for a Pyromind conversation.

    With multiple providers this returns a ``FailoverRouter``; with a single
    provider it returns that provider; without a config file it returns the
    legacy env-var-overridden copy of ``existing``. Vision-capable providers
    from the ``multimodal_llms`` section are attached to the router as a
    separate cohort for image-bearing requests.
    """
    entries = load_config_entries()
    if entries:
        providers = {
            entry["name"]: _provider_llm(existing, entry, primary=index == 0)
            for index, entry in enumerate(entries)
        }
        multimodal_entries = load_multimodal_config_entries()
        if multimodal_entries:
            duplicates = {entry["name"] for entry in multimodal_entries} & set(
                providers
            )
            if duplicates:
                raise ValueError(
                    f"multimodal_llms names must not overlap llms: {sorted(duplicates)}"
                )
        multimodal_llms = {
            entry["name"]: _provider_llm(existing, entry, primary=False)
            for entry in multimodal_entries
        }
        if len(providers) == 1 and not multimodal_llms:
            return next(iter(providers.values()))
        return FailoverRouter(
            llms_for_routing=providers,
            multimodal_llms=multimodal_llms,
            cooldown_seconds=_failover_cooldown_seconds(),
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    return existing.model_copy(
        update={
            "model": os.environ.get("LLM_MODEL") or existing.model,
            "api_key": SecretStr(api_key) if api_key is not None else existing.api_key,
            "base_url": normalize_openai_base_url(
                os.environ.get("LLM_BASE_URL") or existing.base_url
            ),
            "persist_runtime_config": False,
        }
    )
