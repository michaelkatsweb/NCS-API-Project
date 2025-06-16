"""
Pydantic models for NCS API
Data validation and serialization models
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, root_validator, validator


class ClusteringAlgorithm(str, Enum):
    """Supported clustering algorithms"""

    NCS = "ncs"
    KMEANS = "kmeans"
    DBSCAN = "dbscan"


class JobStatus(str, Enum):
    """Job status enumeration"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ClusteringParams(BaseModel):
    """Parameters for clustering algorithms"""

    # Common parameters
    n_clusters: Optional[int] = Field(
        None, ge=1, le=100, description="Number of clusters"
    )
    quality_threshold: float = Field(
        0.85, ge=0.0, le=1.0, description="Minimum quality threshold"
    )
    max_iterations: int = Field(100, ge=1, le=1000, description="Maximum iterations")
    random_seed: Optional[int] = Field(
        None, description="Random seed for reproducibility"
    )

    # NCS-specific parameters
    auto_cluster_detection: bool = Field(
        True, description="Enable automatic cluster detection"
    )
    convergence_tolerance: float = Field(
        1e-6, gt=0, description="Convergence tolerance"
    )

    # DBSCAN-specific parameters
    eps: float = Field(
        0.5, gt=0, description="Maximum distance between points in neighborhood"
    )
    min_samples: int = Field(5, ge=1, description="Minimum samples in neighborhood")

    # K-Means specific parameters
    init_method: str = Field(
        "k-means++", regex="^(k-means\+\+|random)$", description="Initialization method"
    )

    @validator("n_clusters")
    def validate_n_clusters(cls, v, values):
        """Validate n_clusters based on algorithm"""
        if v is not None and v < 1:
            raise ValueError("Number of clusters must be at least 1")
        return v


class ClusterRequest(BaseModel):
    """Request model for clustering operations"""

    data: List[List[float]] = Field(
        ...,
        min_items=1,
        max_items=100000,
        description="Input data points as list of coordinates",
    )
    algorithm: ClusteringAlgorithm = Field(
        ClusteringAlgorithm.NCS, description="Clustering algorithm to use"
    )
    params: ClusteringParams = Field(
        default_factory=ClusteringParams, description="Algorithm-specific parameters"
    )
    job_id: Optional[str] = Field(None, description="Custom job ID")
    priority: int = Field(
        0, ge=0, le=10, description="Job priority (0=lowest, 10=highest)"
    )
    callback_url: Optional[str] = Field(
        None, description="Webhook URL for job completion"
    )

    @validator("data")
    def validate_data(cls, v):
        """Validate input data"""
        if not v:
            raise ValueError("Data cannot be empty")

        if len(v) < 2:
            raise ValueError("Need at least 2 data points for clustering")

        # Check dimensions are consistent
        if len(v) > 1:
            first_dim = len(v[0])
            if first_dim < 1:
                raise ValueError("Data points must have at least 1 dimension")

            for i, row in enumerate(v[1:], 1):
                if len(row) != first_dim:
                    raise ValueError(
                        f"Row {i} has {len(row)} dimensions, expected {first_dim}"
                    )

        # Check for valid numbers
        for i, row in enumerate(v):
            for j, val in enumerate(row):
                if not isinstance(val, (int, float)) or val != val:  # NaN check
                    raise ValueError(f"Invalid value at row {i}, column {j}: {val}")

        return v

    @root_validator
    def validate_algorithm_params(cls, values):
        """Validate parameters based on selected algorithm"""
        algorithm = values.get("algorithm")
        params = values.get("params")

        if not params:
            return values

        if algorithm == ClusteringAlgorithm.KMEANS:
            if params.n_clusters is None:
                raise ValueError("n_clusters is required for K-Means algorithm")

        elif algorithm == ClusteringAlgorithm.DBSCAN:
            if params.n_clusters is not None:
                # DBSCAN doesn't use n_clusters, set to None
                params.n_clusters = None

        return values


class ClusterResponse(BaseModel):
    """Response model for clustering operations"""

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Job status")
    clusters: Optional[List[int]] = Field(
        None, description="Cluster assignments for each point"
    )
    centroids: Optional[List[List[float]]] = Field(
        None, description="Cluster centroids"
    )
    quality_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Clustering quality score"
    )
    n_clusters: Optional[int] = Field(None, description="Number of clusters found")
    processing_time_ms: Optional[float] = Field(
        None, description="Processing time in milliseconds"
    )
    algorithm: Optional[ClusteringAlgorithm] = Field(None, description="Algorithm used")
    error: Optional[str] = Field(None, description="Error message if failed")
    warnings: Optional[List[str]] = Field(None, description="Warning messages")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)

    @validator("clusters")
    def validate_clusters(cls, v):
        """Validate cluster assignments"""
        if v is not None:
            if not all(isinstance(x, int) for x in v):
                raise ValueError("All cluster assignments must be integers")
            if any(x < -1 for x in v):  # -1 allowed for noise in DBSCAN
                raise ValueError("Cluster assignments must be >= -1")
        return v


class JobStatusResponse(BaseModel):
    """Response model for job status queries"""

    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: float = Field(
        0.0, ge=0.0, le=1.0, description="Job progress (0.0 to 1.0)"
    )
    result: Optional[ClusterResponse] = Field(
        None, description="Job result if completed"
    )
    error: Optional[str] = Field(None, description="Error message if failed")

    # Timestamps
    created_at: datetime = Field(..., description="Job creation time")
    updated_at: datetime = Field(..., description="Last update time")
    started_at: Optional[datetime] = Field(None, description="Job start time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")

    # Metadata
    algorithm: Optional[ClusteringAlgorithm] = Field(None, description="Algorithm used")
    data_points: Optional[int] = Field(None, description="Number of input data points")
    estimated_time_remaining_ms: Optional[float] = Field(
        None, description="Estimated time remaining"
    )


class BatchClusterRequest(BaseModel):
    """Request model for batch clustering operations"""

    jobs: List[ClusterRequest] = Field(..., min_items=1, max_items=100)
    batch_id: Optional[str] = Field(None, description="Custom batch ID")
    priority: int = Field(0, ge=0, le=10, description="Batch priority")

    @validator("jobs")
    def validate_unique_job_ids(cls, v):
        """Ensure job IDs are unique within batch"""
        job_ids = [job.job_id for job in v if job.job_id is not None]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("Job IDs must be unique within batch")
        return v


class BatchClusterResponse(BaseModel):
    """Response model for batch clustering operations"""

    batch_id: str = Field(..., description="Batch identifier")
    jobs: List[JobStatusResponse] = Field(
        ..., description="Status of each job in batch"
    )
    overall_status: JobStatus = Field(..., description="Overall batch status")
    completed_jobs: int = Field(0, description="Number of completed jobs")
    failed_jobs: int = Field(0, description="Number of failed jobs")
    total_jobs: int = Field(..., description="Total number of jobs in batch")

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AlgorithmInfo(BaseModel):
    """Information about a clustering algorithm"""

    name: str = Field(..., description="Algorithm name")
    description: str = Field(..., description="Algorithm description")
    supports_auto_clusters: bool = Field(
        ..., description="Supports automatic cluster detection"
    )
    quality_optimization: bool = Field(..., description="Supports quality optimization")
    parameters: Dict[str, str] = Field(..., description="Parameter descriptions")
    performance_notes: Optional[str] = Field(
        None, description="Performance characteristics"
    )


class HealthResponse(BaseModel):
    """Health check response model"""

    status: str = Field("healthy", description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field("1.0.0", description="API version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")

    # Component health
    components: Dict[str, str] = Field(
        default_factory=lambda: {
            "clustering_engine": "healthy",
            "websocket_manager": "healthy",
            "cache": "healthy",
        }
    )


class MetricsResponse(BaseModel):
    """Metrics response model"""

    active_jobs: int = Field(0, description="Currently processing jobs")
    total_jobs: int = Field(0, description="Total jobs processed")
    completed_jobs: int = Field(0, description="Successfully completed jobs")
    failed_jobs: int = Field(0, description="Failed jobs")

    # Performance metrics
    average_processing_time_ms: Optional[float] = Field(
        None, description="Average processing time"
    )
    points_processed_per_second: Optional[float] = Field(
        None, description="Processing throughput"
    )

    # System metrics
    active_websocket_connections: int = Field(
        0, description="Active WebSocket connections"
    )
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    cpu_usage_percent: Optional[float] = Field(None, description="CPU usage percentage")

    uptime_seconds: float = Field(..., description="Service uptime")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WebSocketMessage(BaseModel):
    """WebSocket message model"""

    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message data")
    timestamp: float = Field(default_factory=lambda: datetime.utcnow().timestamp())
    message_id: Optional[str] = Field(None, description="Unique message ID")

    class Config:
        extra = "allow"  # Allow additional fields


class SubscriptionRequest(BaseModel):
    """WebSocket subscription request"""

    action: str = Field(..., regex="^(subscribe|unsubscribe)$")
    topic: str = Field(..., min_length=1, max_length=100)

    # Optional filters
    filters: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request identifier")

    # Validation errors
    validation_errors: Optional[List[Dict[str, str]]] = Field(
        None, description="Field validation errors"
    )


# Export commonly used models
__all__ = [
    "ClusteringAlgorithm",
    "JobStatus",
    "ClusteringParams",
    "ClusterRequest",
    "ClusterResponse",
    "JobStatusResponse",
    "BatchClusterRequest",
    "BatchClusterResponse",
    "AlgorithmInfo",
    "HealthResponse",
    "MetricsResponse",
    "WebSocketMessage",
    "SubscriptionRequest",
    "ErrorResponse",
]
