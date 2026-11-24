from __future__ import annotations

import argparse
import asyncio
import logging
import os

import httpx


logger = logging.getLogger(__name__)
TOKEN_ENV = "PYROMIND_TASK_MONITOR_TOKEN"


async def run_worker(url: str, token: str, *, idle_interval: float = 5) -> None:
    """Drive the authenticated runtime checker independently of agent turns."""
    async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
        while True:
            try:
                response = await client.post(
                    f"{url.rstrip('/')}/internal/task-monitor/check",
                    headers={"x-task-monitor-token": token},
                )
                response.raise_for_status()
                if response.json().get("checked", 0) > 0:
                    continue
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("Task monitor worker retry: %s", type(exc).__name__)
            await asyncio.sleep(idle_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pyromind asynchronous task monitor")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    arguments = parser.parse_args()
    token = os.environ.get(TOKEN_ENV)
    if not token:
        parser.error(f"{TOKEN_ENV} is required")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker(arguments.url, token))


if __name__ == "__main__":
    main()
