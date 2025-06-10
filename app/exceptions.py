"""
NeuroCluster Streamer API - Custom Exceptions
============================================
Custom exception classes for structured error handling

This module provides:
- Hierarchical exception classes for different error types
- Rich error information with context and suggestions
- Integration with FastAPI error handling
- Standardized error codes and messages

Author: NCS API Development Team
Year: 2025
"""

import time
import traceback
from typing import Dict, Any, Optional, List, Union
from enum import Enum

from fastapi import HTTPException, status
from .models import ErrorCode


# =============================================================================
# Base Exception Classes
# =============================================================================


class NCSAPIException(Exception):
    """
    Base exception class for all NCS API exceptions.

    Provides structured error information and context for debugging.
    """

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.PROCESSING_ERROR,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize NCS API exception.

        Args:
            message: Human-readable error message
            error_code: Standardized error code
            status_code: HTTP status code
            details: Additional error details
            suggestion: Suggested action to resolve the error
            context: Additional context information
        """
        super().__init__(message)

        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.suggestion = suggestion
        self.context = context or {}
        self.timestamp = time.time()
        self.traceback_info = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary format.

        Returns:
            Dictionary representation of the exception
        """
        return {
            "success": False,
            "error_code": self.error_code.value,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion,
            "context": self.context,
            "timestamp": self.timestamp,
        }

    def to_http_exception(self) -> HTTPException:
        """
        Convert to FastAPI HTTPException.

        Returns:
            HTTPException for FastAPI error handling
        """
        return HTTPException(status_code=self.status_code, detail=self.to_dict())

    def __str__(self) -> str:
        """String representation of the exception."""
        return f"{self.error_code.value}: {self.message}"

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return (
            f"{self.__class__.__name__}(message='{self.message}', "
            f"error_code={self.error_code.value}, "
            f"status_code={self.status_code})"
        )


# =============================================================================
# Algorithm-Related Exceptions
# =============================================================================


class AlgorithmException(NCSAPIException):
    """Exception raised when the clustering algorithm encounters errors."""

    def __init__(
        self, message: str, algorithm_state: Optional[Dict[str, Any]] = None, **kwargs
    ):
        """
        Initialize algorithm exception.

        Args:
            message: Error message
            algorithm_state: Current algorithm state for debugging
            **kwargs: Additional arguments passed to base class
        """
        kwargs.setdefault("error_code", ErrorCode.ALGORITHM_ERROR)
        kwargs.setdefault("status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
        kwargs.setdefault(
            "suggestion", "Check algorithm parameters and input data quality"
        )

        if algorithm_state:
            kwargs.setdefault("context", {}).update(
                {"algorithm_state": algorithm_state}
            )

        super().__init__(message, **kwargs)


class AlgorithmInitializationException(AlgorithmException):
    """Exception raised during algorithm initialization."""

    def __init__(
        self, message: str, config_params: Optional[Dict[str, Any]] = None, **kwargs
    ):
        kwargs.setdefault("suggestion", "Check algorithm configuration parameters")

        if config_params:
            kwargs.setdefault("context", {}).update({"config_params": config_params})

        super().__init__(message, **kwargs)


class AlgorithmConfigurationException(AlgorithmException):
    """Exception raised for invalid algorithm configuration."""

    def __init__(
        self, message: str, invalid_params: Optional[List[str]] = None, **kwargs
    ):
        kwargs.setdefault(
            "suggestion", "Review and correct algorithm configuration parameters"
        )

        if invalid_params:
            kwargs.setdefault("details", {}).update(
                {"invalid_parameters": invalid_params}
            )

        super().__init__(message, **kwargs)


class AlgorithmPerformanceException(AlgorithmException):
    """Exception raised when algorithm performance degrades significantly."""

    def __init__(
        self,
        message: str,
        performance_metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        kwargs.setdefault("suggestion", "Check system resources and algorithm load")

        if performance_metrics:
            kwargs.setdefault("context", {}).update(
                {"performance_metrics": performance_metrics}
            )

        super().__init__(message, **kwargs)


# =============================================================================
# Validation Exceptions
# =============================================================================


class ValidationException(NCSAPIException):
    """Exception raised for input validation errors."""

    def __init__(
        self, message: str, field_errors: Optional[Dict[str, str]] = None, **kwargs
    ):
        """
        Initialize validation exception.

        Args:
            message: Validation error message
            field_errors: Dictionary of field-specific errors
            **kwargs: Additional arguments passed to base class
        """
        kwargs.setdefault("error_code", ErrorCode.VALIDATION_ERROR)
        kwargs.setdefault("status_code", status.HTTP_422_UNPROCESSABLE_ENTITY)
        kwargs.setdefault("suggestion", "Check input data format and constraints")

        if field_errors:
            kwargs.setdefault("details", {}).update({"field_errors": field_errors})

        super().__init__(message, **kwargs)


class DataPointValidationException(ValidationException):
    """Exception for data point validation errors."""

    def __init__(
        self,
        message: str,
        point_index: Optional[int] = None,
        invalid_coordinates: Optional[List[int]] = None,
        **kwargs,
    ):
        if point_index is not None:
            kwargs.setdefault("context", {}).update({"point_index": point_index})

        if invalid_coordinates:
            kwargs.setdefault("details", {}).update(
                {"invalid_coordinates": invalid_coordinates}
            )

        kwargs.setdefault("suggestion", "Ensure all coordinates are finite numbers")

        super().__init__(message, **kwargs)


class BatchSizeValidationException(ValidationException):
    """Exception for batch size validation errors."""

    def __init__(
        self,
        message: str,
        provided_size: Optional[int] = None,
        max_allowed: Optional[int] = None,
        **kwargs,
    ):
        if provided_size is not None and max_allowed is not None:
            kwargs.setdefault("details", {}).update(
                {"provided_size": provided_size, "max_allowed": max_allowed}
            )

        kwargs.setdefault(
            "suggestion", f"Reduce batch size to {max_allowed or 'allowed limit'}"
        )

        super().__init__(message, **kwargs)


class DimensionMismatchException(ValidationException):
    """Exception for dimension mismatch errors."""

    def __init__(
        self,
        message: str,
        expected_dimensions: Optional[int] = None,
        provided_dimensions: Optional[int] = None,
        **kwargs,
    ):
        if expected_dimensions is not None and provided_dimensions is not None:
            kwargs.setdefault("details", {}).update(
                {
                    "expected_dimensions": expected_dimensions,
                    "provided_dimensions": provided_dimensions,
                }
            )

        kwargs.setdefault(
            "suggestion",
            f"Ensure all points have {expected_dimensions or 'consistent'} dimensions",
        )

        super().__init__(message, **kwargs)


# =============================================================================
# Processing Exceptions
# =============================================================================


class ProcessingException(NCSAPIException):
    """Exception raised during data processing operations."""

    def __init__(self, message: str, processing_stage: Optional[str] = None, **kwargs):
        """
        Initialize processing exception.

        Args:
            message: Processing error message
            processing_stage: Stage where processing failed
            **kwargs: Additional arguments passed to base class
        """
        kwargs.setdefault("error_code", ErrorCode.PROCESSING_ERROR)
        kwargs.setdefault("status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
        kwargs.setdefault("suggestion", "Retry the operation or check input data")

        if processing_stage:
            kwargs.setdefault("context", {}).update(
                {"processing_stage": processing_stage}
            )

        super().__init__(message, **kwargs)


class BatchProcessingException(ProcessingException):
    """Exception for batch processing errors."""

    def __init__(
        self,
        message: str,
        failed_points: Optional[List[int]] = None,
        successful_points: Optional[int] = None,
        **kwargs,
    ):
        if failed_points:
            kwargs.setdefault("details", {}).update(
                {"failed_point_indices": failed_points}
            )

        if successful_points is not None:
            kwargs.setdefault("details", {}).update(
                {"successful_points": successful_points}
            )

        kwargs.setdefault("suggestion", "Check failed points and retry if appropriate")

        super().__init__(message, **kwargs)


class TimeoutException(ProcessingException):
    """Exception for operation timeouts."""

    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        kwargs.setdefault("status_code", status.HTTP_408_REQUEST_TIMEOUT)

        if timeout_seconds:
            kwargs.setdefault("details", {}).update(
                {"timeout_seconds": timeout_seconds}
            )

        kwargs.setdefault("suggestion", "Increase timeout or reduce request size")

        super().__init__(message, **kwargs)


# =============================================================================
# Resource Exceptions
# =============================================================================


class ResourceException(NCSAPIException):
    """Exception raised for resource-related errors."""

    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        """
        Initialize resource exception.

        Args:
            message: Resource error message
            resource_type: Type of resource (memory, cpu, disk, etc.)
            **kwargs: Additional arguments passed to base class
        """
        kwargs.setdefault("error_code", ErrorCode.RESOURCE_ERROR)
        kwargs.setdefault("status_code", status.HTTP_507_INSUFFICIENT_STORAGE)
        kwargs.setdefault("suggestion", "Check system resources and try again later")

        if resource_type:
            kwargs.setdefault("context", {}).update({"resource_type": resource_type})

        super().__init__(message, **kwargs)


class MemoryException(ResourceException):
    """Exception for memory-related errors."""

    def __init__(
        self,
        message: str,
        memory_usage_mb: Optional[float] = None,
        memory_limit_mb: Optional[float] = None,
        **kwargs,
    ):
        kwargs.setdefault("resource_type", "memory")

        if memory_usage_mb and memory_limit_mb:
            kwargs.setdefault("details", {}).update(
                {"memory_usage_mb": memory_usage_mb, "memory_limit_mb": memory_limit_mb}
            )

        kwargs.setdefault("suggestion", "Reduce batch size or restart the service")

        super().__init__(message, **kwargs)


class ConcurrencyException(ResourceException):
    """Exception for concurrency limit errors."""

    def __init__(
        self,
        message: str,
        active_requests: Optional[int] = None,
        max_concurrent: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault("resource_type", "concurrency")
        kwargs.setdefault("status_code", status.HTTP_503_SERVICE_UNAVAILABLE)

        if active_requests and max_concurrent:
            kwargs.setdefault("details", {}).update(
                {"active_requests": active_requests, "max_concurrent": max_concurrent}
            )

        kwargs.setdefault("suggestion", "Wait and retry the request")

        super().__init__(message, **kwargs)


# =============================================================================
# Configuration Exceptions
# =============================================================================


class ConfigurationException(NCSAPIException):
    """Exception raised for configuration errors."""

    def __init__(self, message: str, config_section: Optional[str] = None, **kwargs):
        """
        Initialize configuration exception.

        Args:
            message: Configuration error message
            config_section: Section of configuration with error
            **kwargs: Additional arguments passed to base class
        """
        kwargs.setdefault("error_code", ErrorCode.CONFIGURATION_ERROR)
        kwargs.setdefault("status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
        kwargs.setdefault("suggestion", "Check application configuration")

        if config_section:
            kwargs.setdefault("context", {}).update({"config_section": config_section})

        super().__init__(message, **kwargs)


class MissingConfigurationException(ConfigurationException):
    """Exception for missing required configuration."""

    def __init__(
        self, message: str, missing_keys: Optional[List[str]] = None, **kwargs
    ):
        if missing_keys:
            kwargs.setdefault("details", {}).update({"missing_keys": missing_keys})

        kwargs.setdefault("suggestion", "Set required configuration values")

        super().__init__(message, **kwargs)


class InvalidConfigurationException(ConfigurationException):
    """Exception for invalid configuration values."""

    def __init__(
        self, message: str, invalid_values: Optional[Dict[str, Any]] = None, **kwargs
    ):
        if invalid_values:
            kwargs.setdefault("details", {}).update({"invalid_values": invalid_values})

        kwargs.setdefault("suggestion", "Correct invalid configuration values")

        super().__init__(message, **kwargs)


# =============================================================================
# Security Exceptions
# =============================================================================


class SecurityException(NCSAPIException):
    """Exception raised for security-related errors."""

    def __init__(
        self, message: str, security_context: Optional[Dict[str, Any]] = None, **kwargs
    ):
        """
        Initialize security exception.

        Args:
            message: Security error message
            security_context: Security-related context (sanitized)
            **kwargs: Additional arguments passed to base class
        """
        kwargs.setdefault("error_code", ErrorCode.SECURITY_ERROR)
        kwargs.setdefault("status_code", status.HTTP_403_FORBIDDEN)
        kwargs.setdefault("suggestion", "Check authentication and permissions")

        if security_context:
            # Sanitize sensitive information
            sanitized_context = {
                k: v
                for k, v in security_context.items()
                if k not in ["password", "token", "secret"]
            }
            kwargs.setdefault("context", {}).update(
                {"security_context": sanitized_context}
            )

        super().__init__(message, **kwargs)


class AuthenticationException(SecurityException):
    """Exception for authentication errors."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("status_code", status.HTTP_401_UNAUTHORIZED)
        kwargs.setdefault("suggestion", "Provide valid authentication credentials")

        super().__init__(message, **kwargs)


class AuthorizationException(SecurityException):
    """Exception for authorization errors."""

    def __init__(
        self,
        message: str,
        required_scopes: Optional[List[str]] = None,
        user_scopes: Optional[List[str]] = None,
        **kwargs,
    ):
        if required_scopes and user_scopes:
            kwargs.setdefault("details", {}).update(
                {"required_scopes": required_scopes, "user_scopes": user_scopes}
            )

        kwargs.setdefault("suggestion", "Obtain required permissions")

        super().__init__(message, **kwargs)


class RateLimitException(SecurityException):
    """Exception for rate limiting errors."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        current_rate: Optional[float] = None,
        limit: Optional[float] = None,
        **kwargs,
    ):
        kwargs.setdefault("status_code", status.HTTP_429_TOO_MANY_REQUESTS)
        kwargs.setdefault("error_code", ErrorCode.RATE_LIMIT_ERROR)

        if retry_after:
            kwargs.setdefault("details", {}).update(
                {"retry_after_seconds": retry_after}
            )

        if current_rate and limit:
            kwargs.setdefault("details", {}).update(
                {"current_rate": current_rate, "rate_limit": limit}
            )

        kwargs.setdefault(
            "suggestion", f"Wait {retry_after or 60} seconds before retrying"
        )

        super().__init__(message, **kwargs)


# =============================================================================
# External Service Exceptions
# =============================================================================


class ExternalServiceException(NCSAPIException):
    """Exception for external service errors."""

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        service_status: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize external service exception.

        Args:
            message: Service error message
            service_name: Name of the external service
            service_status: Current status of the service
            **kwargs: Additional arguments passed to base class
        """
        kwargs.setdefault("status_code", status.HTTP_502_BAD_GATEWAY)
        kwargs.setdefault("suggestion", "Check external service availability")

        if service_name:
            kwargs.setdefault("context", {}).update({"service_name": service_name})

        if service_status:
            kwargs.setdefault("context", {}).update({"service_status": service_status})

        super().__init__(message, **kwargs)


class DatabaseException(ExternalServiceException):
    """Exception for database errors."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("service_name", "database")
        kwargs.setdefault("suggestion", "Check database connectivity and try again")

        super().__init__(message, **kwargs)


class CacheException(ExternalServiceException):
    """Exception for cache service errors."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("service_name", "cache")
        kwargs.setdefault(
            "suggestion", "Cache service unavailable, continuing without caching"
        )
        kwargs.setdefault(
            "status_code", status.HTTP_200_OK
        )  # Cache errors are often non-critical

        super().__init__(message, **kwargs)


# =============================================================================
# Exception Utilities
# =============================================================================


def create_validation_exception(
    field_name: str, error_message: str, invalid_value: Any = None
) -> ValidationException:
    """
    Create a validation exception for a specific field.

    Args:
        field_name: Name of the field with validation error
        error_message: Validation error message
        invalid_value: The invalid value provided

    Returns:
        ValidationException instance
    """
    field_errors = {field_name: error_message}
    details = {}

    if invalid_value is not None:
        details["invalid_value"] = str(invalid_value)

    return ValidationException(
        message=f"Validation error in field '{field_name}': {error_message}",
        field_errors=field_errors,
        details=details,
    )


def create_http_exception_from_ncs_exception(exc: NCSAPIException) -> HTTPException:
    """
    Convert NCS exception to FastAPI HTTPException.

    Args:
        exc: NCS API exception

    Returns:
        HTTPException for FastAPI
    """
    return HTTPException(
        status_code=exc.status_code,
        detail=exc.to_dict(),
        headers={"X-Error-Code": exc.error_code.value},
    )


def log_exception(exc: NCSAPIException, logger=None):
    """
    Log exception with appropriate level and context.

    Args:
        exc: Exception to log
        logger: Logger instance (optional)
    """
    import logging

    if logger is None:
        logger = logging.getLogger(__name__)

    # Determine log level based on exception type
    if isinstance(
        exc, (ValidationException, AuthenticationException, AuthorizationException)
    ):
        log_level = logging.WARNING
    elif isinstance(exc, (ResourceException, ExternalServiceException)):
        log_level = logging.ERROR
    else:
        log_level = logging.ERROR

    # Log with context
    logger.log(
        log_level,
        f"{exc.error_code.value}: {exc.message}",
        extra={
            "error_code": exc.error_code.value,
            "status_code": exc.status_code,
            "details": exc.details,
            "context": exc.context,
            "timestamp": exc.timestamp,
        },
    )


# =============================================================================
# Exception Handler Registry
# =============================================================================

EXCEPTION_HANDLERS = {
    ValidationException: lambda exc: exc.to_http_exception(),
    AlgorithmException: lambda exc: exc.to_http_exception(),
    ProcessingException: lambda exc: exc.to_http_exception(),
    ResourceException: lambda exc: exc.to_http_exception(),
    ConfigurationException: lambda exc: exc.to_http_exception(),
    SecurityException: lambda exc: exc.to_http_exception(),
    ExternalServiceException: lambda exc: exc.to_http_exception(),
}


def get_exception_handler(exc_type: type) -> Optional[callable]:
    """
    Get exception handler for given exception type.

    Args:
        exc_type: Exception type

    Returns:
        Exception handler function or None
    """
    return EXCEPTION_HANDLERS.get(exc_type)


# =============================================================================
# Export All Exception Classes
# =============================================================================

__all__ = [
    # Base exceptions
    "NCSAPIException",
    # Algorithm exceptions
    "AlgorithmException",
    "AlgorithmInitializationException",
    "AlgorithmConfigurationException",
    "AlgorithmPerformanceException",
    # Validation exceptions
    "ValidationException",
    "DataPointValidationException",
    "BatchSizeValidationException",
    "DimensionMismatchException",
    # Processing exceptions
    "ProcessingException",
    "BatchProcessingException",
    "TimeoutException",
    # Resource exceptions
    "ResourceException",
    "MemoryException",
    "ConcurrencyException",
    # Configuration exceptions
    "ConfigurationException",
    "MissingConfigurationException",
    "InvalidConfigurationException",
    # Security exceptions
    "SecurityException",
    "AuthenticationException",
    "AuthorizationException",
    "RateLimitException",
    # External service exceptions
    "ExternalServiceException",
    "DatabaseException",
    "CacheException",
    # Utility functions
    "create_validation_exception",
    "create_http_exception_from_ncs_exception",
    "log_exception",
    "get_exception_handler",
]
