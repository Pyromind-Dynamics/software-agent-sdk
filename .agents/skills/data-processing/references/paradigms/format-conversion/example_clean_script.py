"""Map one source dataset to Pyromind training JSONL."""

from __future__ import annotations

import argparse
from typing import Any

from cleaning_utils import (
    DEFAULT_SYSTEM_PROMPT,
    ensure_system_message,
    run_cleaning,
    to_training_record,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser


def map_record(record: Any) -> dict[str, Any]:
    """Adapt this function to the source fields confirmed with the user."""
    if not isinstance(record, dict):
        raise TypeError("source record must be an object")
    cleaned = to_training_record(record)
    if "messages" in cleaned:
        cleaned["messages"] = ensure_system_message(
            cleaned["messages"], DEFAULT_SYSTEM_PROMPT
        )
    return cleaned


def main() -> int:
    args = build_parser().parse_args()
    run_cleaning(
        input_path=args.input,
        output_path=args.output,
        state_dir=args.state_dir,
        mapper=map_record,
        resume=args.resume,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
