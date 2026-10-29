from alembic import context
from sqlalchemy import create_engine


config = context.config
engine = create_engine(config.attributes["database_url"], hide_parameters=True)
with engine.connect() as connection:
    context.configure(
        connection=connection,
        version_table="experiment_plan_schema_version",
    )
    with context.begin_transaction():
        context.run_migrations()
engine.dispose()
