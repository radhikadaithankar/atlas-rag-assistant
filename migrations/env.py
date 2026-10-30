"""Alembic environment. The database URL comes from settings and is not printed."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context

from atlas.config import load_settings, require_database_url, to_psycopg_url
from atlas.database.models import Base
from atlas.database.session import make_engine

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    settings = load_settings()
    url = to_psycopg_url(require_database_url(settings))
    connectable = make_engine(url)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
