"""
NeuroCluster Streamer API - Core Application Package
==================================================
Core application modules supporting the NCS API functionality

This package contains:
- Pydantic models for request/response validation
- FastAPI dependencies for dependency injection
- Custom exceptions for error handling
- Utility functions for common operations

Author: NCS API Development Team
Year: 2025
"""

from .models import *
from .exceptions import *
from .utils import *

__version__ = "1.0.0"
__author__ = "NCS API Development Team"
__email__ = "dev@ncs-api.com"

# Package metadata
__all__ = [
    # Models
    "DataPoint",
    "ProcessPointsRequest", 
    "ProcessPointResult",
    "ClusterInfo",
    "ClustersSummary",
    "AlgorithmStatus",
    "HealthResponse",
    "APIResponse",
    "ErrorResponse",
    "ValidationErrorDetail",
    "PaginationParams",
    "BatchProcessingOptions",
    "ClusteringConfiguration",
    
    # Exceptions
    "NCSAPIException",
    "AlgorithmException",
    "ValidationException",
    "ProcessingException",
    "ResourceException",
    "ConfigurationException",
    "SecurityException",
    "RateLimitException",
    
    # Utilities
    "generate_request_id",
    "validate_data_point",
    "validate_batch_size",
    "format_processing_time",
    "calculate_throughput",
    "estimate_memory_usage",
    "create_error_response",
    "sanitize_input",
    "measure_execution_time",
    "cache_key_generator",
    "validate_clustering_params",
]