from __future__ import annotations

import argparse

import uvicorn

from openhands.agent_server.logging_config import LOGGING_CONFIG
from openhands.sdk.logger import DEBUG


RELOAD_DIRS = [
    "openhands-agent-server",
    "openhands-sdk",
    "openhands-tools",
    "harness-adapter",
    "pyromind-agent-server",
    "pyromind-runtime",
]


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
        reload_dirs=RELOAD_DIRS if args.reload else None,
        reload_includes=["*.py"] if args.reload else None,
        log_level="debug" if DEBUG else "info",
        log_config=LOGGING_CONFIG,
        ws="wsproto",
    )


if __name__ == "__main__":
    main()
