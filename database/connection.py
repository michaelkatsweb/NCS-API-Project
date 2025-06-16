"""
Database connection management for NeuroCluster Streamer API.

This module handles SQLAlchemy database connections, session management,
connection pooling, and database health monitoring.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, Generator, Optional
from urllib.parse import quote_plus

import asyncpg
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DisconnectionError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from app.exceptions import ConnectionException, DatabaseException

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections, sessions, and health monitoring.

    Provides both sync and async database operations with connection pooling,
    automatic reconnection, and comprehensive health checks.
    """

    def __init__(self):
        self.sync_engine: Optional[Engine] = None
        self.async_engine = None
        self.sync_session_factory = None
        self.async_session_factory = None
        self.connection_params = self._get_connection_params()
        self._health_check_cache = {}
        self._last_health_check = 0
        self._health_cache_duration = 30  # seconds

    def _get_connection_params(self) -> Dict[str, Any]:
        """Get database connection parameters from environment."""
        return {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "database": os.getenv("DB_NAME", "ncs_api"),
            "username": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "password"),
            "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
            "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
            "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", "30")),
            "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", "3600")),
            "echo": os.getenv("DB_ECHO", "false").lower() == "true",
        }

    def _build_database_url(self, async_driver: bool = False) -> str:
        """Build database URL with proper encoding."""
        params = self.connection_params
        password = quote_plus(params["password"])

        if async_driver:
            driver = "postgresql+asyncpg"
        else:
            driver = "postgresql+psycopg2"

        return (
            f"{driver}://{params['username']}:{password}@"
            f"{params['host']}:{params['port']}/{params['database']}"
        )

    def initialize_sync_engine(self) -> Engine:
        """Initialize synchronous SQLAlchemy engine with connection pooling."""
        if self.sync_engine is not None:
            return self.sync_engine

        try:
            params = self.connection_params
            database_url = self._build_database_url(async_driver=False)

            self.sync_engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=params["pool_size"],
                max_overflow=params["max_overflow"],
                pool_timeout=params["pool_timeout"],
                pool_recycle=params["pool_recycle"],
                pool_pre_ping=True,  # Validates connections before use
                echo=params["echo"],
                connect_args={
                    "connect_timeout": 10,
                    "command_timeout": 60,
                    "server_settings": {
                        "application_name": "ncs_api",
                        "jit": "off",  # Disable JIT for consistent performance
                    },
                },
            )

            # Add connection event listeners
            self._setup_engine_events(self.sync_engine)

            # Create session factory
            self.sync_session_factory = sessionmaker(
                bind=self.sync_engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
            )

            logger.info("Synchronous database engine initialized successfully")
            return self.sync_engine

        except Exception as e:
            logger.error(f"Failed to initialize sync database engine: {e}")
            raise ConnectionException(f"Database connection failed: {e}")

    async def initialize_async_engine(self):
        """Initialize asynchronous SQLAlchemy engine."""
        if self.async_engine is not None:
            return self.async_engine

        try:
            params = self.connection_params
            database_url = self._build_database_url(async_driver=True)

            self.async_engine = create_async_engine(
                database_url,
                pool_size=params["pool_size"],
                max_overflow=params["max_overflow"],
                pool_timeout=params["pool_timeout"],
                pool_recycle=params["pool_recycle"],
                pool_pre_ping=True,
                echo=params["echo"],
                connect_args={
                    "connect_timeout": 10,
                    "command_timeout": 60,
                    "server_settings": {
                        "application_name": "ncs_api_async",
                        "jit": "off",
                    },
                },
            )

            # Create async session factory
            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
            )

            logger.info("Asynchronous database engine initialized successfully")
            return self.async_engine

        except Exception as e:
            logger.error(f"Failed to initialize async database engine: {e}")
            raise ConnectionException(f"Async database connection failed: {e}")

    def _setup_engine_events(self, engine: Engine):
        """Setup SQLAlchemy engine events for monitoring and debugging."""

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Configure connection settings."""
            if engine.dialect.name == "postgresql":
                with dbapi_connection.cursor() as cursor:
                    # Set optimal PostgreSQL settings
                    cursor.execute("SET synchronous_commit = off")
                    cursor.execute("SET wal_buffers = '16MB'")
                    cursor.execute("SET checkpoint_completion_target = 0.9")

        @event.listens_for(engine, "before_cursor_execute")
        def receive_before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            """Log slow queries."""
            context._query_start_time = time.time()

        @event.listens_for(engine, "after_cursor_execute")
        def receive_after_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            """Log query execution time."""
            total = time.time() - context._query_start_time
            if total > 1.0:  # Log queries taking more than 1 second
                logger.warning(f"Slow query detected: {total:.2f}s - {statement[:100]}")

    @contextmanager
    def get_sync_session(self) -> Generator[Session, None, None]:
        """
        Get synchronous database session with automatic cleanup.

        Yields:
            Session: SQLAlchemy session

        Raises:
            DatabaseException: If session creation or operations fail
        """
        if self.sync_session_factory is None:
            self.initialize_sync_engine()

        session = self.sync_session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise DatabaseException(f"Database operation failed: {e}")
        finally:
            session.close()

    @asynccontextmanager
    async def get_async_session(self) -> AsyncSession:
        """
        Get asynchronous database session with automatic cleanup.

        Yields:
            AsyncSession: SQLAlchemy async session

        Raises:
            DatabaseException: If session creation or operations fail
        """
        if self.async_session_factory is None:
            await self.initialize_async_engine()

        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Async database session error: {e}")
            raise DatabaseException(f"Async database operation failed: {e}")
        finally:
            await session.close()

    async def check_health(self) -> Dict[str, Any]:
        """
        Comprehensive database health check.

        Returns:
            Dict containing health status, connection info, and performance metrics
        """
        current_time = time.time()

        # Return cached result if recent
        if (current_time - self._last_health_check) < self._health_cache_duration:
            return self._health_check_cache

        health_status = {"status": "healthy", "timestamp": current_time, "checks": {}}

        try:
            # Test synchronous connection
            sync_start = time.time()
            with self.get_sync_session() as session:
                result = session.execute(text("SELECT 1")).scalar()
                assert result == 1
            sync_duration = time.time() - sync_start

            health_status["checks"]["sync_connection"] = {
                "status": "ok",
                "response_time_ms": round(sync_duration * 1000, 2),
            }

            # Test asynchronous connection
            async_start = time.time()
            async with self.get_async_session() as session:
                result = await session.execute(text("SELECT 1"))
                assert result.scalar() == 1
            async_duration = time.time() - async_start

            health_status["checks"]["async_connection"] = {
                "status": "ok",
                "response_time_ms": round(async_duration * 1000, 2),
            }

            # Get connection pool status
            if self.sync_engine:
                pool = self.sync_engine.pool
                health_status["checks"]["connection_pool"] = {
                    "status": "ok",
                    "size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                    "total_connections": pool.checkedin() + pool.checkedout(),
                }

            # Database-specific metrics
            with self.get_sync_session() as session:
                # Get database size
                db_size_result = session.execute(
                    text("SELECT pg_size_pretty(pg_database_size(current_database()))")
                ).scalar()

                # Get active connections
                active_connections = session.execute(
                    text("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
                ).scalar()

                # Get database version
                db_version = session.execute(text("SELECT version()")).scalar()

                health_status["checks"]["database_metrics"] = {
                    "status": "ok",
                    "database_size": db_size_result,
                    "active_connections": active_connections,
                    "version": db_version.split()[0:2] if db_version else "Unknown",
                }

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)

        # Cache the result
        self._health_check_cache = health_status
        self._last_health_check = current_time

        return health_status

    async def close_connections(self):
        """Close all database connections and clean up resources."""
        try:
            if self.async_engine:
                await self.async_engine.dispose()
                logger.info("Async database engine disposed")

            if self.sync_engine:
                self.sync_engine.dispose()
                logger.info("Sync database engine disposed")

        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


# Global database manager instance
db_manager = DatabaseManager()


# FastAPI dependency functions
def get_database() -> DatabaseManager:
    """FastAPI dependency to get database manager."""
    return db_manager


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI dependency to get database session."""
    with db_manager.get_sync_session() as session:
        yield session


async def get_async_db_session() -> AsyncSession:
    """FastAPI dependency to get async database session."""
    async with db_manager.get_async_session() as session:
        yield session


# Convenience functions
def create_tables():
    """Create all database tables."""
    engine = db_manager.initialize_sync_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def drop_tables():
    """Drop all database tables."""
    engine = db_manager.initialize_sync_engine()
    Base.metadata.drop_all(bind=engine)
    logger.info("Database tables dropped successfully")


async def check_db_health() -> Dict[str, Any]:
    """Check database health status."""
    return await db_manager.check_health()
