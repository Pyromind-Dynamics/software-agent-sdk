from __future__ import annotations

import os


LOCAL_APP_ENVS = frozenset({"dev", "development", "local", "test"})
PI_TERMINAL_BACKENDS = frozenset({"local", "os-sandbox"})


def validate_pi_terminal_backend(value: str) -> str:
    backend = value.strip().lower()
    if backend not in PI_TERMINAL_BACKENDS:
        supported = ", ".join(sorted(PI_TERMINAL_BACKENDS))
        raise RuntimeError(
            "Unsupported PYROMIND_PI_TERMINAL_BACKEND: "
            f"{backend or '<empty>'}; expected one of: {supported}"
        )
    return backend


def resolve_pi_terminal_backend(
    *,
    app_env: str | None = None,
    terminal_backend: str | None = None,
) -> str:
    raw_app_env = os.getenv("APP_ENV") if app_env is None else app_env
    if not raw_app_env or not raw_app_env.strip():
        raise RuntimeError("APP_ENV is required when PYROMIND_HARNESS_BACKEND=pi")
    normalized_app_env = raw_app_env.strip().lower()

    raw_backend = (
        os.getenv("PYROMIND_PI_TERMINAL_BACKEND")
        if terminal_backend is None
        else terminal_backend
    )
    if not raw_backend or not raw_backend.strip():
        raise RuntimeError(
            "PYROMIND_PI_TERMINAL_BACKEND is required when "
            "PYROMIND_HARNESS_BACKEND=pi"
        )
    backend = validate_pi_terminal_backend(raw_backend)
    if normalized_app_env not in LOCAL_APP_ENVS and backend == "local":
        raise RuntimeError(
            "PYROMIND_PI_TERMINAL_BACKEND=local is allowed only for local "
            f"development environments; APP_ENV={normalized_app_env}"
        )
    return backend
