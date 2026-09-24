"""Drive the Product API over HTTP to validate grace release and capacity.

Run against a server started with the Pi harness, for example::

    PYROMIND_HARNESS_BACKEND=pi PYROMIND_PI_TERMINAL_BACKEND=os-sandbox \
    PYROMIND_CONVERSATION_RELEASE_GRACE_SECONDS=5 \
    PYROMIND_MAX_ACTIVE_CONVERSATIONS=2 \
    uv run python -m pyromind_agent_server --host 127.0.0.1 --port 8012

then::

    uv run python .pr/pi_http_e2e.py --port 8012 --server-pid <pid>

The script prints a timeline of conversation status and live Pi runner
processes, then exercises a follow-up command after the runner was released.
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from typing import Any

import httpx
import psutil


def _runners(server_pid: int) -> list[psutil.Process]:
    try:
        root = psutil.Process(server_pid)
    except psutil.Error:
        return []
    alive: list[psutil.Process] = []
    for process in root.children(recursive=True):
        try:
            if process.is_running() and "index.js" in " ".join(process.cmdline()):
                alive.append(process)
        except psutil.Error:
            continue
    return alive


def _post(client: httpx.Client, path: str, payload: dict[str, Any]) -> httpx.Response:
    response = client.post(path, json=payload)
    if response.status_code >= 400:
        print(f"[http] POST {path} -> {response.status_code} {response.text[:400]}")
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--session-api-key", default=os.getenv("SESSION_API_KEY", ""))
    parser.add_argument("--message", default="Reply with exactly: ok")
    parser.add_argument("--grace-window", type=float, default=45.0)
    parser.add_argument("--capacity", type=int, default=0)
    parser.add_argument(
        "--lru-cap",
        type=int,
        default=0,
        help=(
            "Create cap+1 conversations, letting each go idle first, "
            "to show LRU eviction."
        ),
    )
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    headers = {"X-Session-API-Key": args.session_api_key}
    llm = {
        "model": os.getenv("LLM_MODEL", "openai/gpt-5"),
        "base_url": os.getenv("LLM_BASE_URL"),
        "api_key": os.getenv("OPENAI_API_KEY", "e2e-key"),
    }

    with httpx.Client(base_url=base, headers=headers, timeout=120.0) as client:
        started = time.monotonic()
        response = _post(
            client,
            "/api/v2/pyromind/conversations",
            {"llm": llm, "message": args.message},
        )
        print(f"[http] create -> {response.status_code}")
        if response.status_code != 201:
            print(response.text[:2000])
            return
        conversation_id = response.json()["conversation_id"]
        print(f"[setup] conversation={conversation_id}")

        released_at: float | None = None
        previous: str | None = None
        while time.monotonic() - started < args.grace_window:
            snapshot = client.get(
                f"/api/v2/pyromind/conversations/{conversation_id}/snapshot"
            )
            status = (
                snapshot.json().get("status")
                if snapshot.status_code == 200
                else f"http{snapshot.status_code}"
            )
            runners = len(_runners(args.server_pid))
            row = (
                f"t={time.monotonic() - started:5.1f}s status={status:<12} "
                f"runners={runners}"
            )
            if row != previous:
                print(f"[timeline] {row}")
                previous = row
            if status in {"idle", "finished", "error"} and runners == 0:
                released_at = time.monotonic()
                break
            time.sleep(0.5)
        else:
            print("[result] FAIL: runner was not released inside the grace window")

        if released_at is not None:
            grace_env = os.getenv("PYROMIND_CONVERSATION_RELEASE_GRACE_SECONDS")
            print(
                f"[result] released after {released_at - started:.1f}s "
                f"(grace env={grace_env})"
            )

        follow_up_started = time.monotonic()
        command = {
            "command_id": uuid.uuid4().hex,
            "type": "user_message",
            "content": [{"type": "text", "text": "Reply with exactly: ok again"}],
        }
        follow_up = _post(
            client,
            f"/api/v2/pyromind/conversations/{conversation_id}/commands",
            command,
        )
        accept_ms = (time.monotonic() - follow_up_started) * 1000
        print(
            f"[result] follow_up_accept_ms={accept_ms:.0f} "
            f"status={follow_up.status_code}"
        )
        print(f"[result] follow_up_body={follow_up.text[:300]}")

        reattach_started = time.monotonic()
        runners = 0
        while time.monotonic() - reattach_started < 60.0:
            runners = len(_runners(args.server_pid))
            if runners:
                break
            time.sleep(0.1)
        print(
            f"[result] runner_back_after={time.monotonic() - reattach_started:.1f}s "
            f"runners={runners}"
        )

        if args.capacity:
            print(f"[capacity] creating {args.capacity + 1} conversations")
            for index in range(args.capacity + 1):
                created = _post(
                    client,
                    "/api/v2/pyromind/conversations",
                    {"llm": llm, "message": args.message},
                )
                print(
                    f"[capacity] create#{index + 1} -> {created.status_code} "
                    f"runners={len(_runners(args.server_pid))} "
                    f"body={created.text[:200]}"
                )

        if args.lru_cap:
            created_ids: list[str] = []
            for index in range(args.lru_cap + 1):
                created = _post(
                    client,
                    "/api/v2/pyromind/conversations",
                    {"llm": llm, "message": args.message},
                )
                print(
                    f"[lru] create#{index + 1} -> {created.status_code} "
                    f"runners={len(_runners(args.server_pid))} "
                    f"body={created.text[:120]}"
                )
                if created.status_code == 201:
                    created_ids.append(created.json()["conversation_id"])
                if index < args.lru_cap and created_ids:
                    idle_started = time.monotonic()
                    while time.monotonic() - idle_started < 30.0:
                        snapshot = client.get(
                            f"/api/v2/pyromind/conversations/{created_ids[-1]}/snapshot"
                        )
                        if snapshot.json().get("status") in {
                            "idle",
                            "finished",
                            "error",
                        }:
                            print(
                                f"[lru] #{index + 1} idle after "
                                f"{time.monotonic() - idle_started:.1f}s"
                            )
                            break
                        time.sleep(0.3)
            print(f"[lru] conversation_ids={created_ids}")


if __name__ == "__main__":
    main()
