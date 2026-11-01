from alembic import context
from pyromind_runtime.infrastructure.sql_database import create_database_engine


config = context.config
engine = create_database_engine(config.attributes["database_url"])
try:
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            version_table="experiment_plan_schema_version",
            transaction_per_migration=True,
        )
        with context.begin_transaction():
            context.run_migrations()
finally:
    engine.dispose()
