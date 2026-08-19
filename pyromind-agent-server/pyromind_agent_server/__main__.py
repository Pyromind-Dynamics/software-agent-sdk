from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from openhands.agent_server.logging_config import LOGGING_CONFIG
from openhands.sdk.logger import DEBUG


def _reload_excludes() -> list[str]:
    """Keep generated conversation Python files out of the dev reload watcher."""
    paths = {
        value
        for name in (
            "workspace_dir",
            "WORKSPACE_DIR",
            "OPENHANDS_CONFIG_DIR",
            "OH_CONVERSATIONS_PATH",
            "OH_WORKSPACE_PATH",
            "OH_BASH_EVENTS_DIR",
        )
        if (value := os.getenv(name))
    }
    return sorted(str(Path(path).resolve()) for path in paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pyromind Agent Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "pyromind_agent_server.app:api",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_excludes=_reload_excludes() if args.reload else None,
        log_level="debug" if DEBUG else "info",
        log_config=LOGGING_CONFIG,
        ws="wsproto",
    )


if __name__ == "__main__":
    main()
