"""
Database package initialization for NeuroCluster Streamer API.

This module provides database connection management, ORM models, and CRUD operations
for the NCS clustering algorithm data persistence layer.
"""

from .connection import (
    get_database,
    get_db_session,
    create_tables,
    drop_tables,
    check_db_health,
    DatabaseManager,
)

from .models import (
    Base,
    DataPointRecord,
    ClusterRecord,
    ProcessingSession,
    PerformanceMetric,
    AuditLog,
    UserActivity,
    SystemConfiguration,
)

from .crud import (
    DataPointCRUD,
    ClusterCRUD,
    SessionCRUD,
    MetricsCRUD,
    AuditCRUD,
    UserCRUD,
    ConfigCRUD,
)

__all__ = [
    # Connection management
    "get_database",
    "get_db_session",
    "create_tables",
    "drop_tables",
    "check_db_health",
    "DatabaseManager",
    # Models
    "Base",
    "DataPointRecord",
    "ClusterRecord",
    "ProcessingSession",
    "PerformanceMetric",
    "AuditLog",
    "UserActivity",
    "SystemConfiguration",
    # CRUD operations
    "DataPointCRUD",
    "ClusterCRUD",
    "SessionCRUD",
    "MetricsCRUD",
    "AuditCRUD",
    "UserCRUD",
    "ConfigCRUD",
]

# Version info
__version__ = "1.0.0"
__author__ = "NCS Development Team"
__description__ = "Database layer for NeuroCluster Streamer API"
