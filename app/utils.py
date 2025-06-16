"""
NeuroCluster Streamer API - Utility Functions
============================================
Common utility functions for data processing, validation, and formatting

This module provides:
- Data validation and sanitization utilities
- Performance measurement and profiling tools
- Caching and memoization helpers
- String formatting and conversion functions
- Mathematical and statistical utilities

Author: NCS API Development Team
Year: 2025
"""

import asyncio
import functools
import hashlib
import json
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from pydantic import ValidationError

from .exceptions import ProcessingException, ValidationException
from .models import DataPoint, ErrorCode

# Type variables for generic functions
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# Configure logger
logger = logging.getLogger(__name__)


# =============================================================================
# ID and Token Generation
# =============================================================================


def generate_request_id() -> str:
    """
    Generate a unique request ID for tracking.

    Returns:
        Unique request identifier
    """
    return f"req_{uuid.uuid4().hex[:16]}"


def generate_session_id() -> str:
    """
    Generate a unique session ID.

    Returns:
        Unique session identifier
    """
    return f"sess_{uuid.uuid4().hex[:24]}"


def generate_api_key_id() -> str:
    """
    Generate a unique API key identifier.

    Returns:
        Unique API key identifier
    """
    return f"key_{uuid.uuid4().hex[:12]}"


def generate_cluster_id() -> str:
    """
    Generate a unique cluster identifier.

    Returns:
        Unique cluster identifier
    """
    return f"cluster_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


def create_hash(data: Union[str, bytes, Dict[str, Any]]) -> str:
    """
    Create SHA-256 hash of data.

    Args:
        data: Data to hash

    Returns:
        Hexadecimal hash string
    """
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    elif isinstance(data, str):
        data = data.encode("utf-8")

    return hashlib.sha256(data).hexdigest()


# =============================================================================
# Data Validation Utilities
# =============================================================================


def validate_data_point(
    point: Union[List[float], np.ndarray, Dict[str, Any]],
) -> DataPoint:
    """
    Validate and convert a data point to DataPoint model.

    Args:
        point: Input data point in various formats

    Returns:
        Validated DataPoint instance

    Raises:
        ValidationException: If validation fails
    """
    try:
        if isinstance(point, dict):
            return DataPoint(**point)
        elif isinstance(point, (list, np.ndarray)):
            coordinates = point.tolist() if isinstance(point, np.ndarray) else point
            return DataPoint(coordinates=coordinates)
        else:
            raise ValueError(f"Unsupported point type: {type(point)}")

    except (ValidationError, ValueError) as e:
        raise ValidationException(f"Invalid data point: {str(e)}")


def validate_batch_size(batch_size: int, max_size: int = 10000) -> bool:
    """
    Validate batch size against limits.

    Args:
        batch_size: Size of the batch
        max_size: Maximum allowed batch size

    Returns:
        True if valid

    Raises:
        ValidationException: If batch size is invalid
    """
    if batch_size <= 0:
        raise ValidationException("Batch size must be positive")

    if batch_size > max_size:
        raise ValidationException(
            f"Batch size {batch_size} exceeds maximum {max_size}",
            details={"provided_size": batch_size, "max_size": max_size},
        )

    return True


def validate_coordinates(coordinates: List[float], max_dimensions: int = 1000) -> bool:
    """
    Validate coordinate list.

    Args:
        coordinates: List of coordinate values
        max_dimensions: Maximum allowed dimensions

    Returns:
        True if valid

    Raises:
        ValidationException: If coordinates are invalid
    """
    if not coordinates:
        raise ValidationException("Coordinates cannot be empty")

    if len(coordinates) > max_dimensions:
        raise ValidationException(
            f"Too many dimensions: {len(coordinates)} > {max_dimensions}"
        )

    for i, coord in enumerate(coordinates):
        if not isinstance(coord, (int, float)):
            raise ValidationException(f"Coordinate {i} must be numeric")

        if not np.isfinite(coord):
            raise ValidationException(f"Coordinate {i} must be finite")

    return True


def validate_clustering_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate clustering algorithm parameters.

    Args:
        params: Parameter dictionary

    Returns:
        Validated parameters

    Raises:
        ValidationException: If parameters are invalid
    """
    validated = {}

    # Validate threshold parameters
    threshold_params = ["base_threshold", "merge_threshold", "outlier_threshold"]
    for param in threshold_params:
        if param in params:
            value = params[param]
            if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                raise ValidationException(f"{param} must be between 0.0 and 1.0")
            validated[param] = float(value)

    # Validate rate parameters
    rate_params = ["learning_rate", "decay_rate"]
    for param in rate_params:
        if param in params:
            value = params[param]
            if not isinstance(value, (int, float)) or not (0.0 < value <= 1.0):
                raise ValidationException(f"{param} must be between 0.0 and 1.0")
            validated[param] = float(value)

    # Validate integer parameters
    int_params = ["max_clusters", "stability_window", "validation_window"]
    for param in int_params:
        if param in params:
            value = params[param]
            if not isinstance(value, int) or value <= 0:
                raise ValidationException(f"{param} must be a positive integer")
            validated[param] = int(value)

    # Validate boolean parameters
    bool_params = ["performance_mode"]
    for param in bool_params:
        if param in params:
            validated[param] = bool(params[param])

    return validated


# =============================================================================
# Input Sanitization
# =============================================================================


def sanitize_string(text: str, max_length: int = 1000) -> str:
    """
    Sanitize string input for security.

    Args:
        text: Input text
        max_length: Maximum allowed length

    Returns:
        Sanitized string
    """
    if not isinstance(text, str):
        text = str(text)

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]

    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

    # Remove potential script tags
    text = re.sub(
        r"<script[^>]*>.*?</script>", "", text, flags=re.IGNORECASE | re.DOTALL
    )

    return text.strip()


def sanitize_input(data: Any) -> Any:
    """
    Recursively sanitize input data.

    Args:
        data: Input data structure

    Returns:
        Sanitized data
    """
    if isinstance(data, str):
        return sanitize_string(data)
    elif isinstance(data, dict):
        return {sanitize_string(k): sanitize_input(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_input(item) for item in data]
    elif isinstance(data, (int, float, bool, type(None))):
        return data
    else:
        return sanitize_string(str(data))


def extract_numeric_values(data: Any) -> List[float]:
    """
    Extract numeric values from nested data structure.

    Args:
        data: Input data

    Returns:
        List of numeric values
    """
    values = []

    def _extract(item):
        if isinstance(item, (int, float)) and np.isfinite(item):
            values.append(float(item))
        elif isinstance(item, (list, tuple)):
            for sub_item in item:
                _extract(sub_item)
        elif isinstance(item, dict):
            for value in item.values():
                _extract(value)

    _extract(data)
    return values


# =============================================================================
# Performance Measurement
# =============================================================================


@dataclass
class ExecutionTimer:
    """Context manager for measuring execution time."""

    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def __enter__(self) -> "ExecutionTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.perf_counter()
        return end - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed_seconds * 1000


@contextmanager
def measure_execution_time() -> Iterator[ExecutionTimer]:
    """
    Context manager for measuring execution time.

    Yields:
        ExecutionTimer instance

    Example:
        with measure_execution_time() as timer:
            # do something
            pass
        print(f"Elapsed: {timer.elapsed_ms:.2f}ms")
    """
    timer = ExecutionTimer()
    with timer:
        yield timer


def timing_decorator(func: F) -> F:
    """
    Decorator to measure function execution time.

    Args:
        func: Function to decorate

    Returns:
        Decorated function that logs execution time
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with measure_execution_time() as timer:
            result = func(*args, **kwargs)

        logger.debug(f"{func.__name__} took {timer.elapsed_ms:.2f}ms")
        return result

    return wrapper


def async_timing_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to measure async function execution time.

    Args:
        func: Async function to decorate

    Returns:
        Decorated async function that logs execution time
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        with measure_execution_time() as timer:
            result = await func(*args, **kwargs)

        logger.debug(f"{func.__name__} took {timer.elapsed_ms:.2f}ms")
        return result

    return wrapper


# =============================================================================
# Formatting Utilities
# =============================================================================


def format_processing_time(time_ms: float) -> str:
    """
    Format processing time for display.

    Args:
        time_ms: Time in milliseconds

    Returns:
        Formatted time string
    """
    if time_ms < 1:
        return f"{time_ms:.3f}ms"
    elif time_ms < 1000:
        return f"{time_ms:.2f}ms"
    else:
        return f"{time_ms / 1000:.2f}s"


def format_memory_size(size_bytes: float) -> str:
    """
    Format memory size for display.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)

    for unit in units:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0

    return f"{size:.1f}PB"


def format_number(number: Union[int, float], precision: int = 2) -> str:
    """
    Format number with appropriate precision and thousand separators.

    Args:
        number: Number to format
        precision: Decimal precision

    Returns:
        Formatted number string
    """
    if isinstance(number, int):
        return f"{number:,}"
    else:
        return f"{number:,.{precision}f}"


def format_percentage(value: float, precision: int = 1) -> str:
    """
    Format percentage value.

    Args:
        value: Value between 0 and 1
        precision: Decimal precision

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{precision}f}%"


def format_timestamp(
    timestamp: Union[float, datetime], format_str: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """
    Format timestamp for display.

    Args:
        timestamp: Unix timestamp or datetime object
        format_str: Format string

    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp)
    else:
        dt = timestamp

    return dt.strftime(format_str)


# =============================================================================
# Mathematical and Statistical Utilities
# =============================================================================


def calculate_throughput(count: int, duration_seconds: float) -> float:
    """
    Calculate throughput rate.

    Args:
        count: Number of items processed
        duration_seconds: Duration in seconds

    Returns:
        Throughput in items per second
    """
    if duration_seconds <= 0:
        return 0.0
    return count / duration_seconds


def calculate_percentiles(
    values: List[float], percentiles: List[float] = None
) -> Dict[str, float]:
    """
    Calculate percentiles for a list of values.

    Args:
        values: List of numeric values
        percentiles: List of percentile values (0-100)

    Returns:
        Dictionary of percentile values
    """
    if not values:
        return {}

    if percentiles is None:
        percentiles = [50, 75, 90, 95, 99]

    np_values = np.array(values)
    result = {}

    for p in percentiles:
        result[f"p{p}"] = float(np.percentile(np_values, p))

    return result


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of values.

    Args:
        values: List of numeric values

    Returns:
        Dictionary of statistical measures
    """
    if not values:
        return {}

    np_values = np.array(values)

    return {
        "count": len(values),
        "mean": float(np.mean(np_values)),
        "median": float(np.median(np_values)),
        "std": float(np.std(np_values)),
        "min": float(np.min(np_values)),
        "max": float(np.max(np_values)),
        "sum": float(np.sum(np_values)),
    }


def estimate_memory_usage(data: Any) -> int:
    """
    Estimate memory usage of data structure in bytes.

    Args:
        data: Data structure to analyze

    Returns:
        Estimated memory usage in bytes
    """
    import sys

    def _estimate(obj, seen=None):
        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return 0

        seen.add(obj_id)
        size = sys.getsizeof(obj)

        if isinstance(obj, dict):
            size += sum(_estimate(k, seen) + _estimate(v, seen) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set)):
            size += sum(_estimate(item, seen) for item in obj)
        elif hasattr(obj, "__dict__"):
            size += _estimate(obj.__dict__, seen)

        return size

    return _estimate(data)


def normalize_vector(vector: List[float]) -> List[float]:
    """
    Normalize vector to unit length.

    Args:
        vector: Input vector

    Returns:
        Normalized vector
    """
    np_vector = np.array(vector)
    norm = np.linalg.norm(np_vector)

    if norm == 0:
        return vector

    return (np_vector / norm).tolist()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (-1 to 1)
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same length")

    np1 = np.array(vec1)
    np2 = np.array(vec2)

    dot_product = np.dot(np1, np2)
    norm1 = np.linalg.norm(np1)
    norm2 = np.linalg.norm(np2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


# =============================================================================
# Caching Utilities
# =============================================================================


class LRUCache(Generic[T]):
    """Simple LRU cache implementation."""

    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.cache: Dict[str, T] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None

    def put(self, key: str, value: T) -> None:
        """Put value in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]

            self.cache[key] = value
            self.access_order.append(key)

    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)


def cache_key_generator(*args, **kwargs) -> str:
    """
    Generate cache key from function arguments.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Cache key string
    """
    # Convert arguments to hashable form
    key_parts = []

    for arg in args:
        if isinstance(arg, (str, int, float, bool, type(None))):
            key_parts.append(str(arg))
        elif isinstance(arg, (list, tuple)):
            key_parts.append(str(tuple(arg)))
        elif isinstance(arg, dict):
            key_parts.append(str(sorted(arg.items())))
        else:
            key_parts.append(str(hash(str(arg))))

    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")

    return hashlib.md5("_".join(key_parts).encode()).hexdigest()


def memoize(maxsize: int = 128):
    """
    Memoization decorator with LRU cache.

    Args:
        maxsize: Maximum cache size

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        cache = LRUCache[Any](maxsize)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = cache_key_generator(*args, **kwargs)
            result = cache.get(key)

            if result is None:
                result = func(*args, **kwargs)
                cache.put(key, result)

            return result

        # Add cache management methods
        wrapper.cache_clear = cache.clear
        wrapper.cache_size = cache.size

        return wrapper

    return decorator


# =============================================================================
# Error Response Utilities
# =============================================================================


def create_error_response(
    error_code: ErrorCode,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    suggestion: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create standardized error response.

    Args:
        error_code: Error code
        message: Error message
        details: Additional details
        suggestion: Suggested action
        request_id: Request identifier

    Returns:
        Error response dictionary
    """
    return {
        "success": False,
        "error_code": error_code.value,
        "message": message,
        "details": details or {},
        "suggestion": suggestion,
        "request_id": request_id,
        "timestamp": time.time(),
    }


def create_success_response(
    data: Any,
    message: Optional[str] = None,
    request_id: Optional[str] = None,
    execution_time_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Create standardized success response.

    Args:
        data: Response data
        message: Success message
        request_id: Request identifier
        execution_time_ms: Execution time

    Returns:
        Success response dictionary
    """
    response = {"success": True, "data": data, "timestamp": time.time()}

    if message:
        response["message"] = message

    if request_id:
        response["request_id"] = request_id

    if execution_time_ms is not None:
        response["execution_time_ms"] = execution_time_ms

    return response


# =============================================================================
# Async Utilities
# =============================================================================


async def run_in_thread_pool(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run blocking function in thread pool.

    Args:
        func: Function to run
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Function result
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(func, **kwargs), *args)


@asynccontextmanager
async def async_timeout(seconds: float):
    """
    Async context manager for timeouts.

    Args:
        seconds: Timeout in seconds

    Raises:
        asyncio.TimeoutError: If timeout is exceeded
    """
    try:
        async with asyncio.timeout(seconds):
            yield
    except asyncio.TimeoutError:
        raise ProcessingException(f"Operation timed out after {seconds} seconds")


def create_task_with_timeout(coro, timeout_seconds: float) -> asyncio.Task:
    """
    Create asyncio task with timeout.

    Args:
        coro: Coroutine
        timeout_seconds: Timeout in seconds

    Returns:
        Asyncio task
    """

    async def wrapper():
        async with async_timeout(timeout_seconds):
            return await coro

    return asyncio.create_task(wrapper())


# =============================================================================
# Configuration Utilities
# =============================================================================


def parse_env_bool(value: str) -> bool:
    """
    Parse boolean value from environment variable.

    Args:
        value: String value

    Returns:
        Boolean value
    """
    return value.lower() in ("true", "1", "yes", "on", "enabled")


def parse_env_list(value: str, separator: str = ",") -> List[str]:
    """
    Parse list from environment variable.

    Args:
        value: String value
        separator: List separator

    Returns:
        List of strings
    """
    if not value:
        return []

    return [item.strip() for item in value.split(separator) if item.strip()]


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge configuration dictionaries.

    Args:
        *configs: Configuration dictionaries

    Returns:
        Merged configuration
    """
    merged = {}

    for config in configs:
        for key, value in config.items():
            if (
                isinstance(value, dict)
                and key in merged
                and isinstance(merged[key], dict)
            ):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value

    return merged


# =============================================================================
# Export All Utilities
# =============================================================================

__all__ = [
    # ID generation
    "generate_request_id",
    "generate_session_id",
    "generate_api_key_id",
    "generate_cluster_id",
    "create_hash",
    # Data validation
    "validate_data_point",
    "validate_batch_size",
    "validate_coordinates",
    "validate_clustering_params",
    # Input sanitization
    "sanitize_string",
    "sanitize_input",
    "extract_numeric_values",
    # Performance measurement
    "ExecutionTimer",
    "measure_execution_time",
    "timing_decorator",
    "async_timing_decorator",
    # Formatting
    "format_processing_time",
    "format_memory_size",
    "format_number",
    "format_percentage",
    "format_timestamp",
    # Mathematical utilities
    "calculate_throughput",
    "calculate_percentiles",
    "calculate_statistics",
    "estimate_memory_usage",
    "normalize_vector",
    "cosine_similarity",
    # Caching
    "LRUCache",
    "cache_key_generator",
    "memoize",
    # Response utilities
    "create_error_response",
    "create_success_response",
    # Async utilities
    "run_in_thread_pool",
    "async_timeout",
    "create_task_with_timeout",
    # Configuration utilities
    "parse_env_bool",
    "parse_env_list",
    "merge_configs",
]
