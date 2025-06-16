"""
SQLAlchemy database models for NeuroCluster Streamer API.

This module defines all database tables and relationships for storing
clustering data, performance metrics, audit logs, and system configuration.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    DECIMAL,
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

Base = declarative_base()


class TimestampMixin:
    """Mixin for adding timestamp columns to models."""

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class DataPointRecord(Base, TimestampMixin):
    """
    Records individual data points processed by the clustering algorithm.

    Stores the raw data points, their features, cluster assignments,
    and processing metadata for analysis and debugging.
    """

    __tablename__ = "data_points"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True), ForeignKey("processing_sessions.id"), nullable=False
    )
    point_id = Column(String(255), nullable=False)  # External point identifier

    # Data point features and metadata
    features = Column(JSONB, nullable=False)  # Raw feature vector
    normalized_features = Column(JSONB)  # Normalized feature vector
    dimensionality = Column(SmallInteger, nullable=False)

    # Clustering results
    cluster_id = Column(UUID(as_uuid=True), ForeignKey("clusters.id"), nullable=True)
    is_outlier = Column(Boolean, default=False, nullable=False)
    outlier_score = Column(Float)  # Outlier detection score
    confidence_score = Column(Float)  # Cluster assignment confidence

    # Processing metadata
    processing_order = Column(
        BigInteger, nullable=False
    )  # Order in which point was processed
    processing_time_ms = Column(Float)  # Time taken to process this point
    algorithm_version = Column(String(50), nullable=False)

    # Similarity metrics
    nearest_cluster_distance = Column(Float)
    second_nearest_distance = Column(Float)
    silhouette_score = Column(Float)

    # Quality metrics
    stability_score = Column(Float)  # How stable is the cluster assignment
    novelty_score = Column(Float)  # How novel/different is this point

    # Relationships
    session = relationship("ProcessingSession", back_populates="data_points")
    cluster = relationship("ClusterRecord", back_populates="data_points")

    __table_args__ = (
        Index("idx_data_points_session_order", "session_id", "processing_order"),
        Index("idx_data_points_cluster", "cluster_id"),
        Index("idx_data_points_outlier", "is_outlier"),
        Index("idx_data_points_created", "created_at"),
        UniqueConstraint("session_id", "point_id", name="uq_session_point"),
        CheckConstraint("dimensionality > 0", name="ck_positive_dimensionality"),
        CheckConstraint(
            "outlier_score >= 0 AND outlier_score <= 1", name="ck_outlier_score_range"
        ),
        CheckConstraint(
            "confidence_score >= 0 AND confidence_score <= 1",
            name="ck_confidence_range",
        ),
    )

    @validates("features")
    def validate_features(self, key, features):
        """Validate that features is a non-empty list of numbers."""
        if not isinstance(features, list) or len(features) == 0:
            raise ValueError("Features must be a non-empty list")
        if not all(isinstance(x, (int, float)) for x in features):
            raise ValueError("All features must be numeric")
        return features

    @hybrid_property
    def cluster_stability(self):
        """Calculate cluster assignment stability based on distances."""
        if (
            self.nearest_cluster_distance is None
            or self.second_nearest_distance is None
        ):
            return None
        if self.second_nearest_distance == 0:
            return 1.0
        return 1.0 - (self.nearest_cluster_distance / self.second_nearest_distance)


class ClusterRecord(Base, TimestampMixin):
    """
    Records cluster information and evolution over time.

    Tracks cluster properties, health metrics, and statistical summaries
    for monitoring algorithm performance and cluster quality.
    """

    __tablename__ = "clusters"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True), ForeignKey("processing_sessions.id"), nullable=False
    )
    cluster_label = Column(String(100), nullable=False)  # Human-readable cluster name

    # Cluster properties
    centroid = Column(JSONB, nullable=False)  # Cluster center coordinates
    dimensionality = Column(SmallInteger, nullable=False)
    radius = Column(Float, nullable=False)  # Cluster radius/spread
    density = Column(Float)  # Point density within cluster

    # Statistical properties
    point_count = Column(Integer, default=0, nullable=False)
    min_points = Column(Integer, nullable=False)  # Minimum points required
    max_points = Column(Integer)  # Maximum points allowed

    # Quality metrics
    cohesion = Column(Float)  # Intra-cluster similarity
    separation = Column(Float)  # Inter-cluster distance
    silhouette_avg = Column(Float)  # Average silhouette score
    stability_score = Column(Float)  # Temporal stability

    # Health and lifecycle
    health_status = Column(String(20), default="healthy", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_updated = Column(DateTime(timezone=True), server_default=func.now())
    merge_candidate = Column(Boolean, default=False)
    split_candidate = Column(Boolean, default=False)

    # Evolution tracking
    parent_cluster_id = Column(UUID(as_uuid=True), ForeignKey("clusters.id"))
    generation = Column(SmallInteger, default=0, nullable=False)
    split_count = Column(SmallInteger, default=0)
    merge_count = Column(SmallInteger, default=0)

    # Performance metrics
    last_access_time = Column(DateTime(timezone=True))
    access_frequency = Column(Integer, default=0)
    update_frequency = Column(Integer, default=0)

    # Relationships
    session = relationship("ProcessingSession", back_populates="clusters")
    data_points = relationship(
        "DataPointRecord", back_populates="cluster", cascade="all, delete-orphan"
    )
    parent_cluster = relationship("ClusterRecord", remote_side=[id])

    __table_args__ = (
        Index("idx_clusters_session", "session_id"),
        Index("idx_clusters_active", "is_active"),
        Index("idx_clusters_health", "health_status"),
        Index("idx_clusters_label", "cluster_label"),
        Index("idx_clusters_updated", "last_updated"),
        CheckConstraint("point_count >= 0", name="ck_non_negative_points"),
        CheckConstraint("radius > 0", name="ck_positive_radius"),
        CheckConstraint(
            "health_status IN ('healthy', 'warning', 'critical', 'inactive')",
            name="ck_valid_health_status",
        ),
        CheckConstraint("generation >= 0", name="ck_non_negative_generation"),
    )

    @validates("centroid")
    def validate_centroid(self, key, centroid):
        """Validate centroid is a list of numbers matching dimensionality."""
        if not isinstance(centroid, list):
            raise ValueError("Centroid must be a list")
        if len(centroid) != self.dimensionality:
            raise ValueError(f"Centroid must have {self.dimensionality} dimensions")
        return centroid

    @hybrid_property
    def is_stable(self):
        """Check if cluster is considered stable."""
        return (
            self.stability_score is not None
            and self.stability_score > 0.7
            and self.point_count >= self.min_points
        )

    @hybrid_property
    def quality_score(self):
        """Calculate overall cluster quality score."""
        if None in [self.cohesion, self.separation, self.silhouette_avg]:
            return None
        return self.cohesion * 0.4 + self.separation * 0.3 + self.silhouette_avg * 0.3


class ProcessingSession(Base, TimestampMixin):
    """
    Records clustering processing sessions and their configuration.

    Tracks session parameters, performance metrics, and overall results
    for each clustering run or continuous processing period.
    """

    __tablename__ = "processing_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_name = Column(String(255), nullable=False)
    user_id = Column(String(255), nullable=False)  # User who initiated session

    # Session configuration
    algorithm_config = Column(JSONB, nullable=False)  # NCS algorithm parameters
    input_source = Column(String(500))  # Data source description
    session_type = Column(
        String(50), default="batch", nullable=False
    )  # batch, streaming, etc.

    # Status and lifecycle
    status = Column(String(20), default="active", nullable=False)
    start_time = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    end_time = Column(DateTime(timezone=True))
    total_duration_seconds = Column(Float)

    # Data statistics
    total_points_processed = Column(BigInteger, default=0, nullable=False)
    total_points_clustered = Column(BigInteger, default=0, nullable=False)
    total_outliers_detected = Column(BigInteger, default=0, nullable=False)
    unique_clusters_created = Column(Integer, default=0, nullable=False)

    # Performance metrics
    avg_processing_time_ms = Column(Float)
    max_processing_time_ms = Column(Float)
    min_processing_time_ms = Column(Float)
    throughput_points_per_sec = Column(Float)

    # Quality metrics
    overall_silhouette_score = Column(Float)
    cluster_purity = Column(Float)
    noise_ratio = Column(Float)  # Percentage of outliers

    # Resource utilization
    peak_memory_usage_mb = Column(Float)
    avg_cpu_usage_percent = Column(Float)
    disk_io_mb = Column(Float)

    # Error handling
    error_count = Column(Integer, default=0, nullable=False)
    warning_count = Column(Integer, default=0, nullable=False)
    last_error_message = Column(Text)
    last_error_time = Column(DateTime(timezone=True))

    # Configuration tracking
    algorithm_version = Column(String(50), nullable=False)
    api_version = Column(String(20), nullable=False)

    # Relationships
    data_points = relationship(
        "DataPointRecord", back_populates="session", cascade="all, delete-orphan"
    )
    clusters = relationship(
        "ClusterRecord", back_populates="session", cascade="all, delete-orphan"
    )
    performance_metrics = relationship(
        "PerformanceMetric", back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_sessions_user", "user_id"),
        Index("idx_sessions_status", "status"),
        Index("idx_sessions_start_time", "start_time"),
        Index("idx_sessions_type", "session_type"),
        CheckConstraint(
            "status IN ('active', 'completed', 'failed', 'cancelled')",
            name="ck_valid_status",
        ),
        CheckConstraint(
            "total_points_processed >= 0", name="ck_non_negative_processed"
        ),
        CheckConstraint(
            "throughput_points_per_sec >= 0", name="ck_non_negative_throughput"
        ),
    )

    @hybrid_property
    def is_completed(self):
        """Check if session is completed."""
        return self.status in ["completed", "failed", "cancelled"]

    @hybrid_property
    def clustering_efficiency(self):
        """Calculate clustering efficiency (clustered/processed ratio)."""
        if self.total_points_processed == 0:
            return 0.0
        return self.total_points_clustered / self.total_points_processed


class PerformanceMetric(Base, TimestampMixin):
    """
    Detailed performance metrics collected during processing.

    Stores fine-grained performance data for monitoring, optimization,
    and troubleshooting of the clustering algorithm.
    """

    __tablename__ = "performance_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True), ForeignKey("processing_sessions.id"), nullable=False
    )
    metric_name = Column(String(100), nullable=False)
    metric_category = Column(
        String(50), nullable=False
    )  # timing, memory, throughput, etc.

    # Metric values
    numeric_value = Column(DECIMAL(20, 6))
    string_value = Column(String(500))
    json_value = Column(JSONB)

    # Measurement context
    measurement_timestamp = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    measurement_window_seconds = Column(Float)  # Time window for the measurement
    sample_size = Column(Integer)  # Number of samples for aggregated metrics

    # Statistical properties
    min_value = Column(DECIMAL(20, 6))
    max_value = Column(DECIMAL(20, 6))
    avg_value = Column(DECIMAL(20, 6))
    std_dev = Column(DECIMAL(20, 6))
    percentile_95 = Column(DECIMAL(20, 6))
    percentile_99 = Column(DECIMAL(20, 6))

    # Metadata
    tags = Column(JSONB)  # Additional categorization tags
    notes = Column(Text)

    # Relationships
    session = relationship("ProcessingSession", back_populates="performance_metrics")

    __table_args__ = (
        Index("idx_metrics_session_name", "session_id", "metric_name"),
        Index("idx_metrics_category", "metric_category"),
        Index("idx_metrics_timestamp", "measurement_timestamp"),
        Index("idx_metrics_session_time", "session_id", "measurement_timestamp"),
        CheckConstraint("sample_size > 0", name="ck_positive_sample_size"),
        CheckConstraint("measurement_window_seconds > 0", name="ck_positive_window"),
    )


class AuditLog(Base, TimestampMixin):
    """
    Comprehensive audit trail for security and compliance.

    Records all significant actions, API calls, and system events
    for security monitoring, compliance, and troubleshooting.
    """

    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Event identification
    event_type = Column(String(100), nullable=False)  # api_call, config_change, etc.
    event_category = Column(String(50), nullable=False)  # security, data, system, etc.
    action = Column(String(100), nullable=False)  # create, update, delete, access, etc.
    resource_type = Column(String(100))  # session, cluster, configuration, etc.
    resource_id = Column(String(255))  # ID of affected resource

    # Actor information
    user_id = Column(String(255))
    user_email = Column(String(320))
    user_role = Column(String(100))
    api_key_id = Column(String(255))
    client_ip = Column(String(45))  # IPv6-compatible
    user_agent = Column(String(1000))

    # Request context
    request_id = Column(String(255))
    session_id = Column(String(255))
    endpoint = Column(String(500))
    http_method = Column(String(10))
    request_size_bytes = Column(BigInteger)
    response_size_bytes = Column(BigInteger)
    response_status = Column(SmallInteger)
    processing_time_ms = Column(Float)

    # Event details
    description = Column(Text, nullable=False)
    old_values = Column(JSONB)  # Previous values for changes
    new_values = Column(JSONB)  # New values for changes
    metadata = Column(JSONB)  # Additional context data

    # Security context
    security_level = Column(
        String(20), default="info", nullable=False
    )  # info, warning, critical
    risk_score = Column(SmallInteger)  # 0-100 risk assessment
    threat_indicators = Column(ARRAY(String))

    # Compliance and retention
    retention_policy = Column(String(50), default="standard")
    compliance_tags = Column(ARRAY(String))

    __table_args__ = (
        Index("idx_audit_user", "user_id"),
        Index("idx_audit_event_type", "event_type"),
        Index("idx_audit_timestamp", "created_at"),
        Index("idx_audit_resource", "resource_type", "resource_id"),
        Index("idx_audit_security", "security_level"),
        Index("idx_audit_ip", "client_ip"),
        CheckConstraint(
            "security_level IN ('info', 'warning', 'critical')",
            name="ck_valid_security_level",
        ),
        CheckConstraint(
            "risk_score >= 0 AND risk_score <= 100", name="ck_valid_risk_score"
        ),
    )


class UserActivity(Base, TimestampMixin):
    """
    User activity tracking for analytics and usage monitoring.

    Records user interactions, preferences, and usage patterns
    for improving user experience and system optimization.
    """

    __tablename__ = "user_activities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False)

    # Activity details
    activity_type = Column(
        String(100), nullable=False
    )  # login, api_call, config_change
    activity_name = Column(String(200))  # Specific activity description
    category = Column(
        String(50), nullable=False
    )  # authentication, processing, configuration

    # Context and metadata
    session_token = Column(String(255))
    client_info = Column(JSONB)  # Browser, OS, device info
    location_info = Column(JSONB)  # Geographic location if available

    # Performance and usage metrics
    duration_seconds = Column(Float)
    data_processed_mb = Column(Float)
    api_calls_count = Column(Integer, default=0)
    success_rate = Column(Float)  # Success percentage for batch operations

    # Resource utilization
    cpu_time_ms = Column(Float)
    memory_peak_mb = Column(Float)
    network_io_mb = Column(Float)

    # User preferences and behavior
    preferred_settings = Column(JSONB)
    feature_usage = Column(JSONB)  # Which features were used
    interaction_path = Column(ARRAY(String))  # Sequence of actions

    # Outcome and feedback
    completion_status = Column(String(20), default="completed")
    error_message = Column(Text)
    user_satisfaction_score = Column(SmallInteger)  # 1-5 rating if provided
    feedback_text = Column(Text)

    __table_args__ = (
        Index("idx_user_activity_user", "user_id"),
        Index("idx_user_activity_type", "activity_type"),
        Index("idx_user_activity_timestamp", "created_at"),
        Index("idx_user_activity_category", "category"),
        CheckConstraint(
            "completion_status IN ('completed', 'failed', 'cancelled', 'timeout')",
            name="ck_valid_completion_status",
        ),
        CheckConstraint(
            "user_satisfaction_score >= 1 AND user_satisfaction_score <= 5",
            name="ck_valid_satisfaction_score",
        ),
    )


class SystemConfiguration(Base, TimestampMixin):
    """
    System configuration management and versioning.

    Stores system settings, algorithm parameters, and configuration
    history for change tracking and rollback capabilities.
    """

    __tablename__ = "system_configurations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    config_key = Column(String(200), nullable=False)
    config_category = Column(
        String(100), nullable=False
    )  # algorithm, security, performance

    # Configuration values
    config_value = Column(JSONB, nullable=False)
    default_value = Column(JSONB)
    data_type = Column(
        String(50), nullable=False
    )  # string, integer, float, boolean, json

    # Validation and constraints
    validation_schema = Column(JSONB)  # JSON schema for validation
    min_value = Column(Float)
    max_value = Column(Float)
    allowed_values = Column(ARRAY(String))

    # Metadata and documentation
    description = Column(Text)
    documentation_url = Column(String(1000))
    example_value = Column(JSONB)

    # Change management
    version = Column(Integer, default=1, nullable=False)
    changed_by = Column(String(255))
    change_reason = Column(Text)
    previous_value = Column(JSONB)

    # Status and lifecycle
    is_active = Column(Boolean, default=True, nullable=False)
    is_sensitive = Column(
        Boolean, default=False, nullable=False
    )  # Contains sensitive data
    requires_restart = Column(Boolean, default=False, nullable=False)
    environment = Column(String(50), default="production")  # dev, staging, production

    # Impact and dependencies
    affects_components = Column(ARRAY(String))  # Which components use this config
    dependencies = Column(ARRAY(String))  # Other configs this depends on

    __table_args__ = (
        Index("idx_config_key", "config_key"),
        Index("idx_config_category", "config_category"),
        Index("idx_config_active", "is_active"),
        Index("idx_config_environment", "environment"),
        UniqueConstraint("config_key", "environment", name="uq_config_key_env"),
        CheckConstraint("version > 0", name="ck_positive_version"),
        CheckConstraint(
            "environment IN ('development', 'staging', 'production')",
            name="ck_valid_environment",
        ),
    )

    @validates("config_value")
    def validate_config_value(self, key, value):
        """Validate configuration value against constraints."""
        if self.data_type == "integer" and not isinstance(value, int):
            raise ValueError("Value must be an integer")
        elif self.data_type == "float" and not isinstance(value, (int, float)):
            raise ValueError("Value must be a number")
        elif self.data_type == "boolean" and not isinstance(value, bool):
            raise ValueError("Value must be a boolean")
        elif self.data_type == "string" and not isinstance(value, str):
            raise ValueError("Value must be a string")
        return value
