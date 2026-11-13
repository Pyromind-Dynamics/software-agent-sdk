"""Read experiment lineage through the same authorized API as the frontend."""

from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.parse import quote, urlsplit

import httpx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conversation_id")
    parser.add_argument("--experiment-id")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--metric")
    parser.add_argument("--comparison-group")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-prefix", default="/api/v2/pyromind/conversations")
    parser.add_argument("--user-id")
    parser.add_argument("--cluster")
    args = parser.parse_args(argv)
    if args.experiment_id and args.metrics:
        parser.error("--experiment-id and --metrics are mutually exclusive")
    base = urlsplit(args.base_url)
    if (
        base.scheme not in {"http", "https"}
        or not base.netloc
        or base.username
        or base.password
        or base.query
        or base.fragment
    ):
        parser.error("--base-url must be an HTTP(S) origin without credentials")
    if not args.api_prefix.startswith("/") or "?" in args.api_prefix:
        parser.error("--api-prefix must be an absolute API path")
    headers = {
        name: value
        for name, value in {
            "x-session-api-key": os.getenv("SESSION_API_KEY"),
            "authorization": os.getenv("PYROMIND_AUTHORIZATION"),
            "cookie": os.getenv("PYROMIND_COOKIE"),
            "x-cluster": args.cluster,
            "x-pyromind-debug-user-id": args.user_id,
        }.items()
        if value
    }
    resource = "experiment-metrics" if args.metrics else "experiments"
    if args.experiment_id:
        resource += "/" + quote(args.experiment_id, safe="")
    path = (
        args.api_prefix.rstrip("/")
        + "/"
        + quote(args.conversation_id, safe="")
        + "/"
        + resource
    )
    params = {
        key: value
        for key, value in {
            "metric": args.metric,
            "comparison_group": args.comparison_group,
        }.items()
        if value is not None and args.metrics
    }
    try:
        response = httpx.get(
            args.base_url.rstrip("/") + path,
            headers=headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except httpx.HTTPStatusError as exc:
        print(f"Lineage query failed: HTTP {exc.response.status_code}", file=sys.stderr)
        return 1
    except (httpx.RequestError, ValueError):
        print("Lineage query failed: response unavailable or invalid", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
