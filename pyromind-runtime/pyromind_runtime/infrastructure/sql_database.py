"""SQLAlchemy configuration and short write transactions for both backends."""

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Connection, Engine, make_url


def _sqlite_connect(connection, _record):
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def configure_engine(engine: Engine) -> Engine:
    if engine.dialect.name == "sqlite" and not event.contains(
        engine, "connect", _sqlite_connect
    ):
        event.listen(engine, "connect", _sqlite_connect)
        with engine.connect() as connection:
            connection.exec_driver_sql("PRAGMA journal_mode=WAL")
            connection.exec_driver_sql("PRAGMA foreign_keys=ON")
            connection.exec_driver_sql("PRAGMA busy_timeout=5000")
    return engine


def create_database_engine(url) -> Engine:
    url = make_url(url)
    if url.drivername == "sqlite":
        if not url.database or url.database == ":memory:":
            raise ValueError("experiment database requires a persistent SQLite file")
        path = Path(url.database).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        url = url.set(database=str(path))
        options = {"connect_args": {"timeout": 5, "check_same_thread": False}}
    elif url.drivername in {"mysql+pymysql", "mariadb+pymysql"}:
        options = {
            "pool_pre_ping": True,
            "pool_recycle": 1800,
            "connect_args": {
                "connect_timeout": 10,
                "read_timeout": 30,
                "write_timeout": 30,
            },
        }
    else:
        raise ValueError("experiment database supports sqlite or MariaDB with PyMySQL")
    return configure_engine(create_engine(url, hide_parameters=True, **options))


def insert_if_absent(connection: Connection, table, values: dict, key: str) -> None:
    """Use the dialect's unique-key handling without replacing existing data."""
    if connection.dialect.name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert

        connection.execute(insert(table).values(**values).on_conflict_do_nothing())
    elif connection.dialect.name in {"mysql", "mariadb"}:
        from sqlalchemy.dialects.mysql import insert

        connection.execute(
            insert(table).values(**values).on_duplicate_key_update({key: values[key]})
        )
    else:
        raise ValueError("unsupported experiment database")


@contextmanager
def connection_transaction(connection: Connection):
    if connection.dialect.name == "sqlite":
        # Acquire the writer before reads; SELECT FOR UPDATE is not available.
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    else:
        with connection.begin():
            yield connection


@contextmanager
def write_transaction(engine: Engine):
    with engine.connect() as connection:
        with connection_transaction(connection):
            yield connection
