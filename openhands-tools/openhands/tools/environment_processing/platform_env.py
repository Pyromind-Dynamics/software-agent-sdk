"""Shared platform control-plane env resolution for the edp submission tools.

The data-plane tools (``preview_dataset``, node signatures, ``df_stop_task``)
derive their endpoints from ``APP_ENV`` at construction time. The edp tools
submit platform workflow tasks and additionally need the control-plane env
(``prod`` / ``pre`` / ``pre2``); without a fallback they hard-fail whenever
the harness did not inject ``env`` explicitly, even though the same
deployment already expresses the environment through ``APP_ENV``.
"""

from __future__ import annotations

import os


_PROD_APP_ENVS = {"prod", "production", "online"}


def resolve_platform_env(env: str | None) -> str:
    """Resolve the control-plane env, falling back to ``APP_ENV``.

    An explicit non-empty ``env`` wins; otherwise ``APP_ENV`` maps the
    deployment's application environment onto the control-plane env the same
    way the data-plane URL defaults do (prod app envs -> ``prod``, anything
    else -> ``pre``).
    """
    if env and env.strip():
        return env.strip()
    app_env = os.getenv("APP_ENV", "dev").strip().lower()
    return "prod" if app_env in _PROD_APP_ENVS else "pre"
