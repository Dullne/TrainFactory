"""
Database connection management for TrainFactory.

Uses SQLModel for ORM and connection pooling.
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlmodel import Session, SQLModel, create_engine

from ..config import settings

logger = logging.getLogger(__name__)

# Global engine instance
_engine = None


def get_engine():
    """Get or create database engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.mysql_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            echo=settings.db_echo,
            pool_pre_ping=True,
        )
        logger.info(f"Database engine created: {settings.mysql_url.split('@')[-1]}")
    return _engine


def init_db():
    """Initialize database tables."""
    engine = get_engine()
    if settings.db_auto_migrate:
        # 失败时 run_migrations 会 raise（链式保留根因），此处不 fallback
        # 到 create_all（避免跳过约束/索引导致 schema 漂移）。
        run_migrations()
        logger.info("Database migrations applied, skipping SQLModel metadata create_all")
        return

    SQLModel.metadata.create_all(engine)
    logger.info("Database tables initialized via SQLModel metadata create_all")


def run_migrations() -> bool:
    """Run Alembic migrations if available."""
    try:
        from alembic import command
        from alembic.config import Config

        migrations_dir = Path(__file__).resolve().parent / "migrations"
        config_path = migrations_dir / "alembic.ini"
        alembic_cfg = Config(str(config_path))
        alembic_cfg.set_main_option("script_location", str(migrations_dir))
        logger.info("Running database migrations")
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations completed")
        return True
    except Exception as exc:
        logger.error("Database migration failed: %s", exc)
        raise RuntimeError(
            "Database migration failed; refusing to fallback to create_all. "
            "Fix the root cause (see exception chain) before restarting."
        ) from exc


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get database session context manager."""
    engine = get_engine()
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
