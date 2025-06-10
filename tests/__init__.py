"""
Test suite for NeuroCluster Streamer API.

This package contains comprehensive tests for all API components:
- API endpoint tests
- Authentication and security tests
- Algorithm functionality tests
- Database integration tests
- Performance benchmarks
- Integration tests

Test Categories:
- Unit tests: Individual component testing
- Integration tests: Component interaction testing
- End-to-end tests: Full workflow testing
- Performance tests: Load and benchmark testing
- Security tests: Authentication and authorization testing
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_DATABASE_URL = "sqlite:///./test_ncs_api.db"
TEST_REDIS_URL = "redis://localhost:6379/1"

# Test data constants
SAMPLE_DATA_POINTS = [
    {"id": "test_1", "features": [1.0, 2.0, 3.0]},
    {"id": "test_2", "features": [1.1, 2.1, 3.1]},
    {"id": "test_3", "features": [5.0, 6.0, 7.0]},
    {"id": "test_4", "features": [5.1, 6.1, 7.1]},
    {"id": "test_5", "features": [10.0, 11.0, 12.0]},
]

SAMPLE_CLUSTERING_CONFIG = {
    "similarity_threshold": 0.85,
    "min_cluster_size": 2,
    "max_clusters": 100,
    "outlier_threshold": 0.75,
    "adaptive_threshold": True,
}

# Test user credentials
TEST_USERS = {
    "admin": {
        "user_id": "test_admin",
        "email": "admin@test.com",
        "password": "admin_password_123",
        "role": "admin",
        "scopes": ["read", "write", "admin"],
    },
    "user": {
        "user_id": "test_user",
        "email": "user@test.com",
        "password": "user_password_123",
        "role": "user",
        "scopes": ["read", "write"],
    },
    "readonly": {
        "user_id": "test_readonly",
        "email": "readonly@test.com",
        "password": "readonly_password_123",
        "role": "readonly",
        "scopes": ["read"],
    },
}

# API endpoint constants
API_ENDPOINTS = {
    "health": "/health",
    "auth_login": "/auth/login",
    "auth_refresh": "/auth/refresh",
    "process_point": "/process-point",
    "process_batch": "/process-batch",
    "get_clusters": "/clusters",
    "get_session": "/session/{session_id}",
    "get_statistics": "/statistics",
    "get_metrics": "/metrics",
}

# Test timeouts and limits
TEST_TIMEOUTS = {
    "api_request": 30,  # seconds
    "algorithm_processing": 10,  # seconds
    "database_operation": 5,  # seconds
    "auth_operation": 3,  # seconds
}

# Performance test parameters
PERFORMANCE_TEST_CONFIG = {
    "small_batch": 100,
    "medium_batch": 1000,
    "large_batch": 10000,
    "stress_batch": 50000,
    "concurrent_users": [1, 5, 10, 20],
    "target_response_time_ms": 200,
    "target_throughput_per_sec": 1000,
}

# Test markers for pytest
TEST_MARKERS = {
    "unit": "Unit tests for individual components",
    "integration": "Integration tests for component interactions",
    "e2e": "End-to-end tests for complete workflows",
    "performance": "Performance and load tests",
    "security": "Security and authentication tests",
    "slow": "Tests that take longer than 5 seconds",
    "database": "Tests that require database connectivity",
    "external": "Tests that require external services",
}

__version__ = "1.0.0"
__author__ = "NCS Development Team"
__description__ = "Comprehensive test suite for NeuroCluster Streamer API"
