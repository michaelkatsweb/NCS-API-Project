"""
Pytest configuration and shared fixtures for NCS API tests.

This module provides:
- Database test fixtures with isolation
- FastAPI test client configuration
- Authentication helpers and mock users
- Algorithm instance fixtures
- Common test data and utilities
"""

import os
import uuid
import asyncio
import tempfile
from typing import Dict, Any, Generator, AsyncGenerator
import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import redis

# Import application modules
from main_secure import app
from config import settings
from auth import create_access_token, get_current_user
from database.connection import get_database, get_db_session
from database.models import Base
from NCS_V8 import NCSClusteringAlgorithm
from app.dependencies import get_algorithm_instance

# Import test constants
from . import (
    TEST_DATABASE_URL,
    TEST_REDIS_URL,
    SAMPLE_DATA_POINTS,
    SAMPLE_CLUSTERING_CONFIG,
    TEST_USERS,
    TEST_TIMEOUTS,
)


# Configure pytest markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line("markers", "e2e: End-to-end tests for complete workflows")
    config.addinivalue_line("markers", "performance: Performance and load tests")
    config.addinivalue_line("markers", "security: Security and authentication tests")
    config.addinivalue_line("markers", "slow: Tests that take longer than 5 seconds")
    config.addinivalue_line(
        "markers", "database: Tests that require database connectivity"
    )
    config.addinivalue_line("markers", "external: Tests that require external services")


# Test database setup
@pytest.fixture(scope="session")
def test_database_url():
    """Provide test database URL."""
    # Use in-memory SQLite for tests
    return "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_engine(test_database_url):
    """Create test database engine."""
    engine = create_engine(
        test_database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,  # Set to True for SQL debugging
    )

    # Create all tables
    Base.metadata.create_all(bind=engine)

    yield engine

    # Cleanup
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def test_db_session(test_engine) -> Generator[Session, None, None]:
    """Provide isolated database session for each test."""
    connection = test_engine.connect()
    transaction = connection.begin()

    # Create session bound to the connection
    TestSessionLocal = sessionmaker(bind=connection)
    session = TestSessionLocal()

    yield session

    # Cleanup
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def override_get_db(test_db_session):
    """Override database dependency for tests."""

    def _override_get_db():
        try:
            yield test_db_session
        finally:
            pass

    app.dependency_overrides[get_db_session] = _override_get_db
    yield
    app.dependency_overrides.clear()


# Algorithm fixtures
@pytest.fixture(scope="function")
def test_algorithm():
    """Provide test NCS algorithm instance."""
    config = SAMPLE_CLUSTERING_CONFIG.copy()
    algorithm = NCSClusteringAlgorithm(**config)
    return algorithm


@pytest.fixture(scope="function")
def mock_algorithm():
    """Provide mock algorithm for testing API without algorithm logic."""
    mock = MagicMock(spec=NCSClusteringAlgorithm)

    # Configure mock responses
    mock.process_point.return_value = {
        "cluster_id": "cluster_1",
        "is_outlier": False,
        "confidence": 0.95,
        "processing_time_ms": 1.5,
    }

    mock.get_statistics.return_value = {
        "total_points": 100,
        "active_clusters": 5,
        "outliers_detected": 3,
        "avg_processing_time_ms": 2.1,
    }

    mock.get_clusters.return_value = [
        {
            "id": "cluster_1",
            "centroid": [1.0, 2.0, 3.0],
            "size": 25,
            "health": "healthy",
        }
    ]

    return mock


@pytest.fixture(scope="function")
def override_algorithm(mock_algorithm):
    """Override algorithm dependency for tests."""

    def _override_algorithm():
        return mock_algorithm

    app.dependency_overrides[get_algorithm_instance] = _override_algorithm
    yield mock_algorithm
    app.dependency_overrides.clear()


# FastAPI test client
@pytest.fixture(scope="function")
def test_client(override_get_db) -> TestClient:
    """Provide FastAPI test client."""
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="function")
def async_test_client(override_get_db):
    """Provide async test client."""
    from httpx import AsyncClient

    async def _async_client():
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

    return _async_client


# Authentication fixtures
@pytest.fixture(scope="function")
def test_tokens():
    """Generate test JWT tokens for different user types."""
    tokens = {}

    for user_type, user_data in TEST_USERS.items():
        token_data = {
            "sub": user_data["user_id"],
            "email": user_data["email"],
            "role": user_data["role"],
            "scopes": user_data["scopes"],
        }

        access_token = create_access_token(token_data)
        tokens[user_type] = {
            "access_token": access_token,
            "token_type": "bearer",
            "user_data": user_data,
        }

    return tokens


@pytest.fixture(scope="function")
def auth_headers(test_tokens):
    """Provide authorization headers for different user types."""
    headers = {}

    for user_type, token_info in test_tokens.items():
        headers[user_type] = {"Authorization": f"Bearer {token_info['access_token']}"}

    return headers


@pytest.fixture(scope="function")
def admin_headers(auth_headers):
    """Provide admin authorization headers."""
    return auth_headers["admin"]


@pytest.fixture(scope="function")
def user_headers(auth_headers):
    """Provide regular user authorization headers."""
    return auth_headers["user"]


@pytest.fixture(scope="function")
def readonly_headers(auth_headers):
    """Provide readonly user authorization headers."""
    return auth_headers["readonly"]


# Test data fixtures
@pytest.fixture(scope="function")
def sample_data_points():
    """Provide sample data points for testing."""
    return SAMPLE_DATA_POINTS.copy()


@pytest.fixture(scope="function")
def large_data_batch():
    """Generate large batch of test data points."""
    import random

    data_points = []
    for i in range(1000):
        # Generate clustered data
        if i < 300:  # Cluster 1
            base = [1.0, 2.0, 3.0]
        elif i < 600:  # Cluster 2
            base = [5.0, 6.0, 7.0]
        else:  # Cluster 3
            base = [10.0, 11.0, 12.0]

        # Add noise
        features = [base[j] + random.uniform(-0.5, 0.5) for j in range(len(base))]

        data_points.append({"id": f"test_point_{i}", "features": features})

    return data_points


@pytest.fixture(scope="function")
def clustering_config():
    """Provide test clustering configuration."""
    return SAMPLE_CLUSTERING_CONFIG.copy()


@pytest.fixture(scope="function")
def session_id():
    """Generate unique session ID for tests."""
    return str(uuid.uuid4())


# Mock external services
@pytest.fixture(scope="function")
def mock_redis():
    """Mock Redis connection for tests."""
    mock_redis = MagicMock()

    # In-memory storage for testing
    storage = {}

    def get(key):
        return storage.get(key)

    def set(key, value, ex=None):
        storage[key] = value
        return True

    def delete(key):
        return storage.pop(key, None) is not None

    def exists(key):
        return key in storage

    mock_redis.get = get
    mock_redis.set = set
    mock_redis.delete = delete
    mock_redis.exists = exists

    return mock_redis


@pytest.fixture(scope="function")
def mock_metrics_collector():
    """Mock metrics collector for tests."""
    mock = MagicMock()

    # Configure mock methods
    mock.record_request_metrics.return_value = None
    mock.record_algorithm_metrics.return_value = None
    mock.record_database_metrics.return_value = None
    mock.record_error.return_value = None

    return mock


# Performance test fixtures
@pytest.fixture(scope="session")
def performance_test_config():
    """Provide performance test configuration."""
    return {
        "target_response_time_ms": 200,
        "target_throughput_per_sec": 1000,
        "stress_test_duration": 60,  # seconds
        "concurrent_users": [1, 5, 10, 20],
        "batch_sizes": [100, 1000, 10000],
    }


# Async fixtures
@pytest_asyncio.fixture
async def async_db_session(test_engine):
    """Provide async database session for async tests."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    # This is a simplified async session for testing
    # In practice, you'd use an async-compatible engine
    session = test_engine.connect()
    yield session
    session.close()


# Utility fixtures
@pytest.fixture(scope="function")
def temp_file():
    """Provide temporary file for tests."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        yield f.name

    # Cleanup
    try:
        os.unlink(f.name)
    except FileNotFoundError:
        pass


@pytest.fixture(scope="function")
def temp_directory():
    """Provide temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="function")
def captured_logs():
    """Capture logs during tests."""
    import logging
    from io import StringIO

    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)

    # Add handler to root logger
    logging.getLogger().addHandler(handler)

    yield log_capture

    # Cleanup
    logging.getLogger().removeHandler(handler)


# Environment fixtures
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    original_env = os.environ.copy()

    # Set test environment variables
    os.environ.update(
        {
            "ENVIRONMENT": "testing",
            "DATABASE_URL": TEST_DATABASE_URL,
            "REDIS_URL": TEST_REDIS_URL,
            "JWT_SECRET_KEY": "test_secret_key_for_testing_only",
            "API_KEYS": "test_api_key_1,test_api_key_2",
            "LOG_LEVEL": "DEBUG",
        }
    )

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Database seeding fixtures
@pytest.fixture(scope="function")
def seed_test_data(test_db_session, session_id):
    """Seed database with test data."""
    from database.models import ProcessingSession, ClusterRecord, DataPointRecord
    from datetime import datetime

    # Create test session
    session = ProcessingSession(
        id=session_id,
        session_name="Test Session",
        user_id="test_user",
        algorithm_config=SAMPLE_CLUSTERING_CONFIG,
        algorithm_version="1.0.0",
        api_version="1.0.0",
        status="active",
    )
    test_db_session.add(session)

    # Create test cluster
    cluster = ClusterRecord(
        session_id=session_id,
        cluster_label="Test Cluster 1",
        centroid=[1.0, 2.0, 3.0],
        dimensionality=3,
        radius=1.0,
        min_points=2,
    )
    test_db_session.add(cluster)
    test_db_session.flush()

    # Create test data points
    for i, point_data in enumerate(SAMPLE_DATA_POINTS):
        data_point = DataPointRecord(
            session_id=session_id,
            point_id=point_data["id"],
            features=point_data["features"],
            dimensionality=len(point_data["features"]),
            processing_order=i,
            algorithm_version="1.0.0",
            cluster_id=cluster.id if i < 2 else None,
            is_outlier=i >= 4,
        )
        test_db_session.add(data_point)

    test_db_session.commit()

    yield {
        "session": session,
        "cluster": cluster,
        "data_points": test_db_session.query(DataPointRecord)
        .filter(DataPointRecord.session_id == session_id)
        .all(),
    }


# Error simulation fixtures
@pytest.fixture(scope="function")
def simulate_database_error():
    """Simulate database errors for testing error handling."""

    def _simulate_error():
        from sqlalchemy.exc import SQLAlchemyError

        raise SQLAlchemyError("Simulated database error")

    return _simulate_error


@pytest.fixture(scope="function")
def simulate_algorithm_error():
    """Simulate algorithm errors for testing error handling."""

    def _simulate_error():
        raise RuntimeError("Simulated algorithm error")

    return _simulate_error


# Test event loop for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Cleanup fixture
@pytest.fixture(scope="function", autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield

    # Clear any global state
    # Reset mocks
    # Clear caches
    pass
