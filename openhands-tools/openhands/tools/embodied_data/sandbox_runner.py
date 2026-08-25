"""Compatibility entrypoint for the standalone sandbox runtime."""

from collections.abc import Sequence

from openhands_embodied_runtime.sandbox_runner import (
    main as runtime_main,
    run_full,
    run_plan,
)


__all__ = ["main", "run_full", "run_plan"]


def main(argv: Sequence[str] | None = None) -> int:
    return runtime_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
