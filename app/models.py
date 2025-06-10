"""
NeuroCluster Streamer API - Pydantic Models
==========================================
Comprehensive data models for request/response validation and serialization

This module contains all Pydantic models used throughout the NCS API for:
- Request/response validation
- Data serialization/deserialization
- Type safety and documentation
- Automatic OpenAPI schema generation

Author: NCS API Development Team
Year: 2025
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Literal
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator, ConfigDict
import numpy as np


# =============================================================================
# Base Models and Configurations
# =============================================================================


class BaseAPIModel(BaseModel):
    """Base model with common configuration for all API models."""

    model_config = ConfigDict(
        # Allow extra fields for forward compatibility
        extra="forbid",
        # Use enum values instead of enum objects
        use_enum_values=True,
        # Validate assignment to help catch errors
        validate_assignment=True,
        # Generate JSON schema
        json_schema_extra={"examples": []},
    )


class TimestampMixin(BaseModel):
    """Mixin for models that need timestamp tracking."""

    created_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the record was created",
    )
    updated_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the record was last updated",
    )


# =============================================================================
# Enums and Constants
# =============================================================================


class ProcessingStatus(str, Enum):
    """Status of data point processing."""

    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    TIMEOUT = "timeout"


class ClusterHealthStatus(str, Enum):
    """Health status of clusters."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    INACTIVE = "inactive"


class ErrorCode(str, Enum):
    """Standardized error codes."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    RESOURCE_ERROR = "RESOURCE_ERROR"
    ALGORITHM_ERROR = "ALGORITHM_ERROR"
    SECURITY_ERROR = "SECURITY_ERROR"
    RATE_LIMIT_ERROR = "RATE_LIMIT_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# =============================================================================
# Core Data Models
# =============================================================================


class DataPoint(BaseAPIModel):
    """Model for a single data point with validation and metadata."""

    coordinates: List[float] = Field(
        ...,
        description="Numeric coordinates representing the data point's features",
        min_length=1,
        max_length=1000,
        examples=[[1.0, 2.0, 3.0], [0.5, -1.2, 2.8, 4.1]],
    )

    point_id: Optional[str] = Field(
        None,
        description="Optional unique identifier for the data point",
        max_length=255,
        examples=["point_001", "sensor_reading_12345"],
    )

    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata associated with the data point",
        examples=[{"sensor_id": "temp_01", "location": "factory_floor"}],
    )

    timestamp: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the data point was created",
    )

    @validator("coordinates")
    def validate_coordinates(cls, v):
        """Validate that coordinates are finite numbers."""
        if not v:
            raise ValueError("Coordinates cannot be empty")

        for i, coord in enumerate(v):
            if not isinstance(coord, (int, float)):
                raise ValueError(f"Coordinate at index {i} must be a number")
            if not np.isfinite(coord):
                raise ValueError(
                    f"Coordinate at index {i} must be finite (not NaN or infinity)"
                )

        return v

    @validator("metadata")
    def validate_metadata(cls, v):
        """Validate metadata size and content."""
        if v is not None:
            # Convert to JSON string and check size (approximate)
            import json

            json_str = json.dumps(v)
            if len(json_str) > 10000:  # 10KB limit
                raise ValueError("Metadata too large (max 10KB)")

        return v


class ProcessPointsRequest(BaseAPIModel):
    """Model for batch point processing request."""

    points: List[List[float]] = Field(
        ...,
        description="List of data points for batch processing",
        min_length=1,
        max_length=10000,
    )

    options: Optional["BatchProcessingOptions"] = Field(
        None, description="Optional processing options for the batch"
    )

    request_id: Optional[str] = Field(
        None, description="Optional unique identifier for the request", max_length=255
    )

    @validator("points")
    def validate_points(cls, v):
        """Validate batch of points."""
        if not v:
            raise ValueError("Points list cannot be empty")

        # Check consistency of dimensions
        first_point_dim = len(v[0]) if v else 0
        for i, point in enumerate(v):
            if len(point) != first_point_dim:
                raise ValueError(
                    f"Point at index {i} has {len(point)} dimensions, "
                    f"expected {first_point_dim}"
                )

            # Validate individual point
            for j, coord in enumerate(point):
                if not isinstance(coord, (int, float)):
                    raise ValueError(f"Point {i}, coordinate {j} must be a number")
                if not np.isfinite(coord):
                    raise ValueError(f"Point {i}, coordinate {j} must be finite")

        return v


class ProcessPointResult(BaseAPIModel):
    """Model for individual point processing result."""

    input_point: List[float] = Field(..., description="The original input data point")

    cluster_id: int = Field(
        ..., description="Assigned cluster ID (-1 for outliers)", ge=-1
    )

    is_outlier: bool = Field(..., description="True if classified as outlier")

    outlier_score: float = Field(
        ..., description="Outlier confidence score [0-1]", ge=0.0, le=1.0
    )

    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds", ge=0.0
    )

    confidence: Optional[float] = Field(
        None, description="Clustering confidence score [0-1]", ge=0.0, le=1.0
    )

    status: ProcessingStatus = Field(
        ProcessingStatus.SUCCESS, description="Processing status"
    )

    point_id: Optional[str] = Field(None, description="Original point ID if provided")


class ClusterInfo(BaseAPIModel):
    """Detailed cluster information."""

    cluster_id: int = Field(..., description="Unique cluster identifier", ge=0)

    centroid: List[float] = Field(..., description="Cluster centroid coordinates")

    confidence: float = Field(
        ..., description="Cluster confidence score [0-1]", ge=0.0, le=1.0
    )

    age: float = Field(..., description="Cluster age (number of updates)", ge=0.0)

    update_count: int = Field(
        ..., description="Number of points assigned to this cluster", ge=0
    )

    health_score: float = Field(
        ..., description="Overall cluster health score [0-1]", ge=0.0, le=1.0
    )

    health_status: ClusterHealthStatus = Field(
        ..., description="Categorical health status"
    )

    last_updated: datetime = Field(..., description="Timestamp of last cluster update")

    bounding_radius: Optional[float] = Field(
        None, description="Estimated bounding radius of the cluster", ge=0.0
    )

    @validator("health_status", pre=False, always=True)
    def determine_health_status(cls, v, values):
        """Automatically determine health status from health score."""
        if "health_score" in values:
            health_score = values["health_score"]
            if health_score >= 0.8:
                return ClusterHealthStatus.HEALTHY
            elif health_score >= 0.5:
                return ClusterHealthStatus.WARNING
            elif health_score >= 0.2:
                return ClusterHealthStatus.CRITICAL
            else:
                return ClusterHealthStatus.INACTIVE
        return v


class ClustersSummary(BaseAPIModel):
    """Comprehensive cluster summary response."""

    num_active_clusters: int = Field(
        ..., description="Number of currently active clusters", ge=0
    )

    cluster_ids: List[int] = Field(..., description="List of active cluster IDs")

    clusters_info: List[ClusterInfo] = Field(
        ..., description="Detailed information for each cluster"
    )

    total_points_processed: int = Field(
        ..., description="Total number of points processed", ge=0
    )

    average_cluster_confidence: float = Field(
        ..., description="Average confidence across all clusters", ge=0.0, le=1.0
    )

    cluster_distribution: Dict[ClusterHealthStatus, int] = Field(
        ..., description="Distribution of clusters by health status"
    )

    summary_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when summary was generated",
    )


class AlgorithmStatus(BaseAPIModel):
    """Real-time algorithm status and performance metrics."""

    current_dynamic_threshold: float = Field(
        ..., description="Current adaptive threshold value", ge=0.0, le=1.0
    )

    clustering_quality: float = Field(
        ..., description="Overall clustering quality score [0-1]", ge=0.0, le=1.0
    )

    global_stability: float = Field(
        ..., description="Global algorithm stability score [0-1]", ge=0.0, le=1.0
    )

    adaptation_rate: float = Field(
        ..., description="Current adaptation rate multiplier", ge=0.0
    )

    total_points_processed: int = Field(
        ..., description="Total points processed since startup", ge=0
    )

    average_processing_time_ms: float = Field(
        ..., description="Average processing time per point in milliseconds", ge=0.0
    )

    max_processing_time_ms: float = Field(
        ..., description="Maximum processing time recorded in milliseconds", ge=0.0
    )

    memory_usage_mb: float = Field(
        ..., description="Current memory usage in megabytes", ge=0.0
    )

    uptime_seconds: float = Field(
        ..., description="Algorithm uptime in seconds", ge=0.0
    )

    throughput_points_per_second: Optional[float] = Field(
        None, description="Current throughput in points per second", ge=0.0
    )

    error_rate: Optional[float] = Field(
        None, description="Current error rate [0-1]", ge=0.0, le=1.0
    )

    drift_detected: bool = Field(
        False, description="Whether concept drift has been detected"
    )


# =============================================================================
# API Response Models
# =============================================================================


class APIResponse(BaseAPIModel):
    """Generic API response wrapper."""

    success: bool = Field(True, description="Whether the request was successful")

    message: Optional[str] = Field(None, description="Human-readable message")

    request_id: Optional[str] = Field(
        None, description="Unique request identifier for tracking"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )

    execution_time_ms: Optional[float] = Field(
        None, description="Request execution time in milliseconds", ge=0.0
    )


class HealthResponse(BaseAPIModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., description="Overall system health status"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Health check timestamp"
    )

    version: str = Field(..., description="API version")

    algorithm_ready: bool = Field(
        ..., description="Whether the clustering algorithm is ready"
    )

    security_features: List[str] = Field(
        ..., description="List of enabled security features"
    )

    uptime_seconds: float = Field(..., description="System uptime in seconds", ge=0.0)

    components: Dict[str, str] = Field(
        default_factory=dict, description="Status of individual system components"
    )


class ErrorResponse(BaseAPIModel):
    """Standardized error response."""

    success: bool = Field(False, description="Always false for error responses")

    error_code: ErrorCode = Field(..., description="Standardized error code")

    message: str = Field(..., description="Human-readable error message")

    details: Optional[str] = Field(None, description="Additional error details")

    request_id: Optional[str] = Field(
        None, description="Request identifier for debugging"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )

    path: Optional[str] = Field(None, description="API path where error occurred")

    suggestion: Optional[str] = Field(
        None, description="Suggested action to resolve the error"
    )


class ValidationErrorDetail(BaseAPIModel):
    """Detailed validation error information."""

    field: str = Field(..., description="Field name that failed validation")

    message: str = Field(..., description="Validation error message")

    invalid_value: Any = Field(None, description="The invalid value that was provided")

    constraint: Optional[str] = Field(
        None, description="The validation constraint that was violated"
    )


# =============================================================================
# Configuration Models
# =============================================================================


class PaginationParams(BaseAPIModel):
    """Pagination parameters for list endpoints."""

    page: int = Field(1, description="Page number (1-based)", ge=1)

    size: int = Field(50, description="Number of items per page", ge=1, le=1000)

    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.size


class BatchProcessingOptions(BaseAPIModel):
    """Options for batch processing operations."""

    enable_parallel_processing: bool = Field(
        True, description="Whether to enable parallel processing for the batch"
    )

    batch_timeout_seconds: int = Field(
        300, description="Maximum time to wait for batch processing", ge=1, le=3600
    )

    outlier_detection_enabled: bool = Field(
        True, description="Whether to enable outlier detection"
    )

    adaptive_threshold: bool = Field(
        True, description="Whether to use adaptive thresholding"
    )

    return_detailed_metrics: bool = Field(
        False, description="Whether to include detailed processing metrics"
    )


class ClusteringConfiguration(BaseAPIModel):
    """Configuration parameters for the clustering algorithm."""

    base_threshold: float = Field(
        0.71,
        description="Base similarity threshold for cluster assignment",
        ge=0.0,
        le=1.0,
    )

    learning_rate: float = Field(
        0.06, description="Learning rate for centroid updates", ge=0.001, le=1.0
    )

    max_clusters: int = Field(
        30, description="Maximum number of clusters allowed", ge=1, le=1000
    )

    outlier_threshold: float = Field(
        0.2, description="Threshold for outlier detection", ge=0.0, le=1.0
    )

    performance_mode: bool = Field(
        True, description="Whether to enable performance optimizations"
    )

    @validator("learning_rate")
    def validate_learning_rate(cls, v):
        """Validate that learning rate is reasonable."""
        if v > 0.5:
            import warnings

            warnings.warn("High learning rate may cause instability")
        return v


# =============================================================================
# Batch Operation Models
# =============================================================================


class BatchProcessingResult(BaseAPIModel):
    """Result of batch processing operation."""

    total_points: int = Field(
        ..., description="Total number of points in the batch", ge=0
    )

    successful_points: int = Field(
        ..., description="Number of successfully processed points", ge=0
    )

    failed_points: int = Field(
        ..., description="Number of points that failed processing", ge=0
    )

    outliers_detected: int = Field(
        ..., description="Number of outliers detected in the batch", ge=0
    )

    clusters_created: int = Field(
        ..., description="Number of new clusters created", ge=0
    )

    processing_time_ms: float = Field(
        ..., description="Total processing time for the batch", ge=0.0
    )

    throughput_points_per_second: float = Field(
        ..., description="Processing throughput for this batch", ge=0.0
    )

    results: List[ProcessPointResult] = Field(
        ..., description="Individual results for each point"
    )


# =============================================================================
# Monitoring and Metrics Models
# =============================================================================


class PerformanceMetrics(BaseAPIModel):
    """Performance metrics for monitoring."""

    requests_per_second: float = Field(
        ..., description="Current requests per second", ge=0.0
    )

    average_response_time_ms: float = Field(
        ..., description="Average response time in milliseconds", ge=0.0
    )

    p95_response_time_ms: float = Field(
        ..., description="95th percentile response time in milliseconds", ge=0.0
    )

    error_rate: float = Field(
        ..., description="Current error rate [0-1]", ge=0.0, le=1.0
    )

    memory_usage_mb: float = Field(
        ..., description="Current memory usage in megabytes", ge=0.0
    )

    cpu_usage_percent: float = Field(
        ..., description="Current CPU usage percentage", ge=0.0, le=100.0
    )


# =============================================================================
# Model Relationships and Forward References
# =============================================================================

# Update forward references for models that reference other models
ProcessPointsRequest.model_rebuild()
BatchProcessingOptions.model_rebuild()


# =============================================================================
# Model Validation Utilities
# =============================================================================


def validate_data_point_model(point: List[float]) -> DataPoint:
    """Validate and convert a raw data point to DataPoint model."""
    return DataPoint(coordinates=point)


def create_error_response_model(
    error_code: ErrorCode,
    message: str,
    details: Optional[str] = None,
    suggestion: Optional[str] = None,
) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        error_code=error_code, message=message, details=details, suggestion=suggestion
    )


# Export all models for easy importing
__all__ = [
    # Base models
    "BaseAPIModel",
    "TimestampMixin",
    # Enums
    "ProcessingStatus",
    "ClusterHealthStatus",
    "ErrorCode",
    "LogLevel",
    # Core data models
    "DataPoint",
    "ProcessPointsRequest",
    "ProcessPointResult",
    "ClusterInfo",
    "ClustersSummary",
    "AlgorithmStatus",
    # Response models
    "APIResponse",
    "HealthResponse",
    "ErrorResponse",
    "ValidationErrorDetail",
    # Configuration models
    "PaginationParams",
    "BatchProcessingOptions",
    "ClusteringConfiguration",
    # Batch operation models
    "BatchProcessingResult",
    # Monitoring models
    "PerformanceMetrics",
    # Utility functions
    "validate_data_point_model",
    "create_error_response_model",
]
