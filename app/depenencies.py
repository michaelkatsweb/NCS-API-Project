"""
NeuroCluster Streamer API - FastAPI Dependencies
===============================================
Dependency injection functions for cross-cutting concerns

This module provides dependency functions for:
- Authentication and authorization
- Request validation and rate limiting
- Algorithm instance management
- Performance monitoring and metrics
- Error handling and logging

Author: NCS API Development Team
Year: 2025
"""

import time
import uuid
import asyncio
from typing import Optional, Dict, Any, Callable, Generator, AsyncGenerator
from functools import wraps
from contextlib import asynccontextmanager

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as aioredis

from config import get_settings
from auth import (
    get_current_active_user, verify_api_key_dependency,
    User, APIKeyInfo, rate_limiter
)
from .models import (
    PaginationParams, BatchProcessingOptions, ClusteringConfiguration,
    ErrorCode, ProcessingStatus
)
from .exceptions import (
    NCSAPIException, ValidationException, ResourceException,
    RateLimitException, ProcessingException
)
from .utils import generate_request_id, measure_execution_time

# Get application settings
settings = get_settings()

# Security scheme for dependency injection
security = HTTPBearer(auto_error=False)


# =============================================================================
# Core Algorithm Dependencies
# =============================================================================

def get_ncs_algorithm():
    """
    Dependency to get the NeuroCluster Streamer algorithm instance.
    
    Returns:
        NCS algorithm instance
        
    Raises:
        HTTPException: If algorithm is not ready
    """
    from main_secure import api_state
    
    if not api_state.is_ready or api_state.ncs_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NCS algorithm not ready. Please try again later."
        )
    
    return api_state.ncs_instance


def get_algorithm_with_health_check():
    """
    Dependency that returns algorithm instance with health validation.
    
    Returns:
        Tuple of (algorithm_instance, health_status)
    """
    algorithm = get_ncs_algorithm()
    
    # Basic health check
    try:
        stats = algorithm.get_statistics()
        health_status = {
            "healthy": True,
            "clusters": stats.get('num_clusters', 0),
            "quality": stats.get('clustering_quality', 0.0),
            "memory_mb": stats.get('memory_usage_estimate_mb', 0.0)
        }
        
        # Check for warning conditions
        if stats.get('error_count', 0) > 10:
            health_status["healthy"] = False
            health_status["warning"] = "High error count"
        
        if stats.get('memory_usage_estimate_mb', 0) > settings.ncs.memory_warning_threshold_mb:
            health_status["healthy"] = False
            health_status["warning"] = "High memory usage"
        
        return algorithm, health_status
        
    except Exception as e:
        raise ProcessingException(f"Algorithm health check failed: {str(e)}")


# =============================================================================
# Authentication and Authorization Dependencies
# =============================================================================

async def get_current_user_with_scopes(
    required_scopes: list = None,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Dependency for user authentication with scope validation.
    
    Args:
        required_scopes: List of required scopes
        credentials: HTTP Bearer credentials
        
    Returns:
        Authenticated user
        
    Raises:
        HTTPException: If authentication or authorization fails
    """
    if required_scopes is None:
        required_scopes = ["read"]
    
    # Get current user
    user = await get_current_active_user(credentials)
    
    # Check scopes
    user_scopes = set(user.scopes)
    required_scopes_set = set(required_scopes)
    
    if not required_scopes_set.issubset(user_scopes):
        missing_scopes = required_scopes_set - user_scopes
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions. Missing scopes: {list(missing_scopes)}"
        )
    
    return user


async def get_api_key_with_scopes(
    required_scopes: list = None,
    api_key_info: APIKeyInfo = Depends(verify_api_key_dependency)
) -> APIKeyInfo:
    """
    Dependency for API key authentication with scope validation.
    
    Args:
        required_scopes: List of required scopes
        api_key_info: API key information
        
    Returns:
        API key info
        
    Raises:
        HTTPException: If authorization fails
    """
    if required_scopes is None:
        required_scopes = ["read"]
    
    # Check scopes
    api_key_scopes = set(api_key_info.scopes)
    required_scopes_set = set(required_scopes)
    
    if not required_scopes_set.issubset(api_key_scopes):
        missing_scopes = required_scopes_set - api_key_scopes
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient API key permissions. Missing scopes: {list(missing_scopes)}"
        )
    
    return api_key_info


def require_read_scope():
    """Dependency that requires 'read' scope."""
    return Depends(lambda: get_current_user_with_scopes(["read"]))


def require_write_scope():
    """Dependency that requires 'read' and 'write' scopes."""
    return Depends(lambda: get_current_user_with_scopes(["read", "write"]))


def require_admin_scope():
    """Dependency that requires 'admin' scope.""" 
    return Depends(lambda: get_current_user_with_scopes(["admin"]))


# =============================================================================
# Request Validation Dependencies
# =============================================================================

def validate_pagination(
    page: int = 1,
    size: int = 50
) -> PaginationParams:
    """
    Dependency for pagination parameter validation.
    
    Args:
        page: Page number (1-based)
        size: Items per page
        
    Returns:
        Validated pagination parameters
        
    Raises:
        ValidationException: If parameters are invalid
    """
    try:
        return PaginationParams(page=page, size=size)
    except Exception as e:
        raise ValidationException(f"Invalid pagination parameters: {str(e)}")


def validate_batch_options(
    enable_parallel: bool = True,
    timeout: int = 300,
    outlier_detection: bool = True,
    adaptive_threshold: bool = True,
    detailed_metrics: bool = False
) -> BatchProcessingOptions:
    """
    Dependency for batch processing options validation.
    
    Args:
        enable_parallel: Enable parallel processing
        timeout: Processing timeout in seconds
        outlier_detection: Enable outlier detection
        adaptive_threshold: Use adaptive thresholding
        detailed_metrics: Return detailed metrics
        
    Returns:
        Validated batch processing options
        
    Raises:
        ValidationException: If options are invalid
    """
    try:
        return BatchProcessingOptions(
            enable_parallel_processing=enable_parallel,
            batch_timeout_seconds=timeout,
            outlier_detection_enabled=outlier_detection,
            adaptive_threshold=adaptive_threshold,
            return_detailed_metrics=detailed_metrics
        )
    except Exception as e:
        raise ValidationException(f"Invalid batch options: {str(e)}")


def validate_clustering_config(
    base_threshold: float = 0.71,
    learning_rate: float = 0.06,
    max_clusters: int = 30,
    outlier_threshold: float = 0.2,
    performance_mode: bool = True
) -> ClusteringConfiguration:
    """
    Dependency for clustering configuration validation.
    
    Args:
        base_threshold: Base similarity threshold
        learning_rate: Learning rate for updates
        max_clusters: Maximum number of clusters
        outlier_threshold: Outlier detection threshold
        performance_mode: Enable performance optimizations
        
    Returns:
        Validated clustering configuration
        
    Raises:
        ValidationException: If configuration is invalid
    """
    try:
        return ClusteringConfiguration(
            base_threshold=base_threshold,
            learning_rate=learning_rate,
            max_clusters=max_clusters,
            outlier_threshold=outlier_threshold,
            performance_mode=performance_mode
        )
    except Exception as e:
        raise ValidationException(f"Invalid clustering configuration: {str(e)}")


# =============================================================================
# Rate Limiting Dependencies
# =============================================================================

async def check_rate_limit(request: Request) -> Dict[str, Any]:
    """
    Dependency for rate limiting with detailed information.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Rate limit information
        
    Raises:
        RateLimitException: If rate limit is exceeded
    """
    from auth import get_client_ip
    
    client_ip = get_client_ip(request)
    endpoint = request.url.path
    
    # Get endpoint-specific limits
    limits = _get_endpoint_rate_limits(endpoint)
    
    # Check rate limit
    is_allowed, rate_info = rate_limiter.is_allowed(client_ip, endpoint, limits)
    
    if not is_allowed:
        # Determine retry after time
        retry_after = 60  # Default
        if "reset_time" in rate_info:
            retry_after = max(1, int(rate_info["reset_time"] - time.time()))
        
        raise RateLimitException(
            message=f"Rate limit exceeded for {endpoint}",
            retry_after=retry_after,
            details={
                "client_ip": client_ip,
                "endpoint": endpoint,
                "rate_info": rate_info
            }
        )
    
    return rate_info


def _get_endpoint_rate_limits(endpoint: str) -> Dict[str, int]:
    """Get rate limits for specific endpoint."""
    endpoint_limits = {
        "/api/v1/process_points": {
            "60": 100,     # 100 requests per minute
            "300": 400,    # 400 requests per 5 minutes  
            "3600": 2000   # 2000 requests per hour
        },
        "/api/v1/clusters_summary": {
            "60": 300,
            "300": 1200,
            "3600": 5000
        },
        "/api/v1/algorithm_status": {
            "60": 600,
            "300": 2400,
            "3600": 10000
        }
    }
    
    # Default limits
    default_limits = {
        "60": settings.security.rate_limit_per_minute,
        "300": settings.security.rate_limit_per_minute * 3,
        "3600": settings.security.rate_limit_per_minute * 20
    }
    
    return endpoint_limits.get(endpoint, default_limits)


# =============================================================================
# Performance Monitoring Dependencies
# =============================================================================

async def track_request_performance(request: Request) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Dependency for tracking request performance metrics.
    
    Args:
        request: FastAPI request object
        
    Yields:
        Performance tracking context
    """
    start_time = time.perf_counter()
    request_id = generate_request_id()
    
    # Store request metadata
    context = {
        "request_id": request_id,
        "endpoint": request.url.path,
        "method": request.method,
        "start_time": start_time,
        "client_ip": request.client.host
    }
    
    try:
        yield context
    finally:
        # Calculate final metrics
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        context.update({
            "end_time": end_time,
            "duration_ms": duration_ms,
            "success": True
        })
        
        # Log performance metrics (could be sent to monitoring system)
        if duration_ms > 1000:  # Log slow requests
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Slow request detected: {context['endpoint']} took {duration_ms:.2f}ms"
            )


def measure_processing_time():
    """
    Dependency decorator for measuring processing time.
    
    Returns:
        Function decorator that measures execution time
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with measure_execution_time() as timer:
                result = await func(*args, **kwargs)
            
            # Add timing information to result if it's a dict
            if isinstance(result, dict):
                result["execution_time_ms"] = timer.elapsed_ms
            
            return result
        
        return wrapper
    return decorator


# =============================================================================
# Resource Management Dependencies
# =============================================================================

async def check_system_resources() -> Dict[str, Any]:
    """
    Dependency for checking system resource availability.
    
    Returns:
        System resource information
        
    Raises:
        ResourceException: If resources are critically low
    """
    import psutil
    import os
    
    try:
        # Get memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        disk_free_gb = disk_usage.free / (1024 ** 3)
        
        resource_info = {
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent,
            "disk_free_gb": disk_free_gb,
            "healthy": True
        }
        
        # Check for resource constraints
        warnings = []
        
        if memory_mb > settings.ncs.memory_warning_threshold_mb:
            warnings.append(f"High memory usage: {memory_mb:.1f}MB")
            if memory_mb > settings.ncs.memory_warning_threshold_mb * 2:
                resource_info["healthy"] = False
        
        if cpu_percent > 90:
            warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
            if cpu_percent > 95:
                resource_info["healthy"] = False
        
        if disk_free_gb < 1.0:
            warnings.append(f"Low disk space: {disk_free_gb:.1f}GB")
            if disk_free_gb < 0.5:
                resource_info["healthy"] = False
        
        resource_info["warnings"] = warnings
        
        # Raise exception if resources are critically low
        if not resource_info["healthy"]:
            raise ResourceException(
                message="System resources critically low",
                details={
                    "resource_info": resource_info,
                    "warnings": warnings
                }
            )
        
        return resource_info
        
    except psutil.Error as e:
        raise ResourceException(f"Failed to check system resources: {str(e)}")


async def get_connection_pool():
    """
    Dependency for getting database/cache connection pool.
    
    Returns:
        Connection pool or None if not available
    """
    # This would return actual connection pool in production
    # For now, return a mock object
    return None


# =============================================================================
# Caching Dependencies
# =============================================================================

async def get_cache_client() -> Optional[aioredis.Redis]:
    """
    Dependency for getting Redis cache client.
    
    Returns:
        Redis client or None if not available
    """
    if not settings.redis.redis_host:
        return None
    
    try:
        redis_client = aioredis.from_url(
            settings.redis.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=settings.redis.redis_timeout,
            socket_connect_timeout=settings.redis.redis_timeout
        )
        
        # Test connection
        await redis_client.ping()
        return redis_client
        
    except Exception:
        # Cache is optional, don't fail if unavailable
        return None


@asynccontextmanager
async def cache_context(key_prefix: str = "ncs"):
    """
    Async context manager for cache operations.
    
    Args:
        key_prefix: Prefix for cache keys
        
    Yields:
        Cache operations object
    """
    cache_client = await get_cache_client()
    
    class CacheOps:
        def __init__(self, client, prefix):
            self.client = client
            self.prefix = prefix
        
        async def get(self, key: str) -> Optional[str]:
            if not self.client:
                return None
            return await self.client.get(f"{self.prefix}:{key}")
        
        async def set(self, key: str, value: str, ttl: int = 3600):
            if not self.client:
                return
            await self.client.setex(f"{self.prefix}:{key}", ttl, value)
        
        async def delete(self, key: str):
            if not self.client:
                return
            await self.client.delete(f"{self.prefix}:{key}")
    
    cache_ops = CacheOps(cache_client, key_prefix)
    
    try:
        yield cache_ops
    finally:
        if cache_client:
            await cache_client.close()


# =============================================================================
# Request Context Dependencies
# =============================================================================

def get_request_context(request: Request) -> Dict[str, Any]:
    """
    Dependency for extracting request context information.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Request context dictionary
    """
    from auth import get_client_ip
    
    return {
        "request_id": generate_request_id(),
        "client_ip": get_client_ip(request),
        "user_agent": request.headers.get("user-agent", ""),
        "endpoint": request.url.path,
        "method": request.method,
        "query_params": dict(request.query_params),
        "timestamp": time.time()
    }


def add_response_headers(response: Response, context: Dict[str, Any] = None):
    """
    Dependency for adding standard response headers.
    
    Args:
        response: FastAPI response object
        context: Request context (optional)
    """
    # Add standard headers
    response.headers["X-API-Version"] = "1.0.0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    
    if context:
        response.headers["X-Request-ID"] = context.get("request_id", "")
        
        # Add processing time if available
        if "duration_ms" in context:
            response.headers["X-Processing-Time"] = f"{context['duration_ms']:.2f}ms"


# =============================================================================
# Health Check Dependencies
# =============================================================================

async def get_health_status() -> Dict[str, Any]:
    """
    Dependency for comprehensive health status check.
    
    Returns:
        Health status information
    """
    health = {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {}
    }
    
    # Check algorithm
    try:
        algorithm = get_ncs_algorithm()
        stats = algorithm.get_statistics()
        health["checks"]["algorithm"] = {
            "status": "healthy",
            "clusters": stats.get('num_clusters', 0),
            "quality": stats.get('clustering_quality', 0.0)
        }
    except Exception as e:
        health["checks"]["algorithm"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health["status"] = "degraded"
    
    # Check cache
    try:
        cache_client = await get_cache_client()
        if cache_client:
            await cache_client.ping()
            health["checks"]["cache"] = {"status": "healthy"}
            await cache_client.close()
        else:
            health["checks"]["cache"] = {"status": "unavailable"}
    except Exception as e:
        health["checks"]["cache"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health["status"] = "degraded"
    
    # Check resources
    try:
        resource_info = await check_system_resources()
        health["checks"]["resources"] = {
            "status": "healthy" if resource_info["healthy"] else "warning",
            "memory_mb": resource_info["memory_mb"],
            "cpu_percent": resource_info["cpu_percent"]
        }
        
        if not resource_info["healthy"]:
            health["status"] = "degraded"
            
    except Exception as e:
        health["checks"]["resources"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health["status"] = "unhealthy"
    
    return health


# =============================================================================
# Utility Dependencies
# =============================================================================

def get_settings_dependency():
    """Dependency for accessing application settings."""
    return settings


def create_dependency_chain(*dependencies):
    """
    Create a chain of dependencies that execute in sequence.
    
    Args:
        *dependencies: Functions to chain as dependencies
        
    Returns:
        Combined dependency function
    """
    async def combined_dependency():
        results = []
        for dep in dependencies:
            if asyncio.iscoroutinefunction(dep):
                result = await dep()
            else:
                result = dep()
            results.append(result)
        return results
    
    return Depends(combined_dependency)


# =============================================================================
# Export Dependencies
# =============================================================================

__all__ = [
    # Algorithm dependencies
    "get_ncs_algorithm",
    "get_algorithm_with_health_check",
    
    # Authentication dependencies
    "get_current_user_with_scopes",
    "get_api_key_with_scopes",
    "require_read_scope",
    "require_write_scope", 
    "require_admin_scope",
    
    # Validation dependencies
    "validate_pagination",
    "validate_batch_options",
    "validate_clustering_config",
    
    # Rate limiting dependencies
    "check_rate_limit",
    
    # Performance monitoring dependencies
    "track_request_performance",
    "measure_processing_time",
    
    # Resource management dependencies
    "check_system_resources",
    "get_connection_pool",
    
    # Caching dependencies
    "get_cache_client",
    "cache_context",
    
    # Request context dependencies
    "get_request_context",
    "add_response_headers",
    
    # Health check dependencies
    "get_health_status",
    
    # Utility dependencies
    "get_settings_dependency",
    "create_dependency_chain",
]