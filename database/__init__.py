"""
Database package initialization for NeuroCluster Streamer API.

This module provides database connection management, ORM models, and CRUD operations
for the NCS clustering algorithm data persistence layer.
"""

from .connection import (
    DatabaseManager,
    check_db_health,
    create_tables,
    drop_tables,
    get_database,
    get_db_session,
)
from .crud import (
    AuditCRUD,
    ClusterCRUD,
    ConfigCRUD,
    DataPointCRUD,
    MetricsCRUD,
    SessionCRUD,
    UserCRUD,
)
from .models import (
    AuditLog,
    Base,
    ClusterRecord,
    DataPointRecord,
    PerformanceMetric,
    ProcessingSession,
    SystemConfiguration,
    UserActivity,
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
