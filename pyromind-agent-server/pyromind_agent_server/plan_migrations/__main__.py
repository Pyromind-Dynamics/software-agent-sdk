from __future__ import annotations

import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url


def main() -> None:
    url = make_url(os.environ["EXPERIMENT_PLAN_DATABASE_URL"])
    if url.drivername != "mysql+pymysql":
        raise ValueError("Plan migrations require mysql+pymysql")
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent))
    config.attributes["database_url"] = url
    command.upgrade(config, "head")


if __name__ == "__main__":
    main()
