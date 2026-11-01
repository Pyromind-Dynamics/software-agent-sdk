from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from pyromind_agent_server.plan_database import resolve_database_url


def main() -> None:
    url = resolve_database_url()
    if url is None:
        raise ValueError("Experiment database is not configured")
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent))
    config.attributes["database_url"] = url
    command.upgrade(config, "head")


if __name__ == "__main__":
    main()
