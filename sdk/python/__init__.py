#!/usr/bin/env python3
"""
NeuroCluster Streamer Python SDK
===============================
Official Python client library for the NeuroCluster Streamer API

This package provides both synchronous and asynchronous clients for
interacting with the NCS API, along with comprehensive data models
and error handling.

Author: NCS API Development Team
Year: 2025
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "NCS API Development Team"
__email__ = "sdk@yourdomain.com"
__description__ = "Official Python client library for the NeuroCluster Streamer API"
__license__ = "MIT"
__url__ = "https://github.com/your-org/ncs-api"

from .async_client import (  # Async client; Async utilities
    AsyncNCSClient,
    AsyncRateLimiter,
    StreamingConnection,
    async_client_context,
    create_async_client,
)

# Import all exceptions with consistent naming
# Import main client classes
from .ncs_client import (  # Main client; Data models; Type definitions; Utilities
    AlgorithmStatus,
    AuthenticationError,
    Cluster,
    ConnectionError,
    HealthStatus,
    NCSClient,
    NCSError,
    Point,
    Points,
    ProcessingError,
    ProcessingResult,
    RateLimitError,
    ValidationError,
    create_client,
)


# Package-level convenience functions
def get_version() -> str:
    """Get the current package version."""
    return __version__


def get_client_info() -> dict:
    """Get comprehensive client information."""
    return {
        "name": "ncs-python-sdk",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "license": __license__,
        "url": __url__,
        "python_requires": ">=3.8",
    }


# Configuration helpers
def configure_logging(level: str = "INFO"):
    """
    Configure logging for the NCS client.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    import logging

    # Configure root logger for NCS
    logger = logging.getLogger("ncs_client")
    logger.setLevel(getattr(logging, level.upper()))

    # Add handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info(f"NCS SDK logging configured at {level} level")


def load_config_from_env() -> dict:
    """
    Load client configuration from environment variables.

    Returns:
        Configuration dictionary suitable for client initialization
    """
    import os

    config = {}

    # Core settings
    if os.getenv("NCS_API_URL"):
        config["base_url"] = os.getenv("NCS_API_URL")

    if os.getenv("NCS_API_KEY"):
        config["api_key"] = os.getenv("NCS_API_KEY")

    if os.getenv("NCS_JWT_TOKEN"):
        config["jwt_token"] = os.getenv("NCS_JWT_TOKEN")

    # Optional settings with defaults
    config["timeout"] = float(os.getenv("NCS_TIMEOUT", "30.0"))
    config["max_retries"] = int(os.getenv("NCS_MAX_RETRIES", "3"))
    config["retry_delay"] = float(os.getenv("NCS_RETRY_DELAY", "1.0"))
    config["verify_ssl"] = os.getenv("NCS_VERIFY_SSL", "true").lower() == "true"
    config["log_level"] = os.getenv("NCS_LOG_LEVEL", "INFO")

    return {k: v for k, v in config.items() if v is not None}


def create_client_from_env(**kwargs) -> NCSClient:
    """
    Create a client using environment variable configuration.

    Args:
        **kwargs: Additional configuration to override environment settings

    Returns:
        Configured NCSClient instance

    Raises:
        ValueError: If required environment variables are missing
    """
    import os

    config = load_config_from_env()
    config.update(kwargs)

    if "base_url" not in config:
        raise ValueError("NCS_API_URL environment variable is required")

    if "api_key" not in config and "jwt_token" not in config:
        raise ValueError(
            "Either NCS_API_KEY or NCS_JWT_TOKEN environment variable is required"
        )

    return NCSClient(**config)


async def create_async_client_from_env(**kwargs) -> AsyncNCSClient:
    """
    Create an async client using environment variable configuration.

    Args:
        **kwargs: Additional configuration to override environment settings

    Returns:
        Configured AsyncNCSClient instance

    Raises:
        ValueError: If required environment variables are missing
    """
    import os

    config = load_config_from_env()
    config.update(kwargs)

    if "base_url" not in config:
        raise ValueError("NCS_API_URL environment variable is required")

    if "api_key" not in config and "jwt_token" not in config:
        raise ValueError(
            "Either NCS_API_KEY or NCS_JWT_TOKEN environment variable is required"
        )

    return AsyncNCSClient(**config)


# Version checking
def check_compatibility():
    """Check if the current environment is compatible with the SDK."""
    import sys
    import warnings

    # Check Python version
    if sys.version_info < (3, 8):
        raise RuntimeError("NCS Python SDK requires Python 3.8 or higher")

    # Check for optional dependencies
    try:
        import numpy
    except ImportError:
        warnings.warn(
            "NumPy is not installed. Some performance features may be limited.",
            UserWarning,
        )

    try:
        import pandas
    except ImportError:
        warnings.warn(
            "Pandas is not installed. DataFrame processing features are not available.",
            UserWarning,
        )


# Run compatibility check on import
try:
    check_compatibility()
except Exception as e:
    import warnings

    warnings.warn(f"Compatibility check failed: {e}", UserWarning)

# Package-level constants
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
MAX_BATCH_SIZE = 10000
DEFAULT_RATE_LIMIT = 100  # calls per minute

# Export everything
__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "__license__",
    "__url__",
    # Main classes
    "NCSClient",
    "AsyncNCSClient",
    # Data models
    "Cluster",
    "ProcessingResult",
    "AlgorithmStatus",
    "HealthStatus",
    # Streaming
    "StreamingConnection",
    "AsyncRateLimiter",
    # Exceptions
    "NCSError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "ProcessingError",
    "ConnectionError",
    # Type definitions
    "Point",
    "Points",
    # Factory functions
    "create_client",
    "create_async_client",
    "async_client_context",
    # Configuration utilities
    "get_version",
    "get_client_info",
    "configure_logging",
    "load_config_from_env",
    "create_client_from_env",
    "create_async_client_from_env",
    "check_compatibility",
    # Constants
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_DELAY",
    "MAX_BATCH_SIZE",
    "DEFAULT_RATE_LIMIT",
]

# Package initialization message
import logging

logger = logging.getLogger(__name__)
logger.debug(f"NCS Python SDK v{__version__} initialized")

# Backwards compatibility aliases
NCSSyncClient = NCSClient  # For backwards compatibility
