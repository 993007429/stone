import importlib
import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from sqlalchemy import create_engine

from alembic import context


def batch_import_model_from_modules():
    project_root_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(project_root_dir)
    for root, dirs, files in os.walk(os.path.join(project_root_dir, 'stone', 'modules')):
        for name in files:
            if name == 'models.py':
                module = os.path.join(root, name).replace('.py', '').replace('/', '.')
                if sys.platform == 'win32':
                    module = os.path.join(root, name).replace('.py', '').replace('\\', '.')
                importlib.import_module(module)


batch_import_model_from_modules()

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
import setting
from stone.seedwork.infrastructure.models import Base

config.set_main_option('sqlalchemy.url', setting.SQLALCHEMY_DATABASE_URI)
target_metadata = Base.metadata


def create_database():
    print(setting.SQLALCHEMY_DATABASE_URI)
    engine = create_engine(setting.SQLALCHEMY_DATABASE_URI.replace(f"/{setting.MYSQL_DATABASE}", ""))
    connection = engine.connect()
    connection.execute("CREATE DATABASE `stone` /*!40100 DEFAULT CHARACTER SET utf8mb4 */;")
    connection.close()
    print("stone created succeed")


try:
    create_database()
except Exception as e:
    print(e)

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
