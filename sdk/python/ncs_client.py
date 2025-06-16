#!/usr/bin/env python3
"""
NeuroCluster Streamer Python SDK - Main Client
============================================
Official Python client library for the NCS API

This module provides both synchronous and asynchronous clients for
interacting with the NeuroCluster Streamer API.

Author: NCS API Development Team
Year: 2025
"""

import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

import httpx
import jwt
import requests
from ratelimit import limits, sleep_and_retry
from requests.adapters import HTTPAdapter
from tenacity import retry, stop_after_attempt, wait_exponential
from urllib3.util.retry import Retry

# Type definitions
Point = List[Union[int, float]]
Points = List[Point]

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# Data Models
# =============================================================================


@dataclass
class Cluster:
    """Represents a data cluster from the NCS algorithm."""

    id: int
    center: Point
    points: Points
    size: int
    quality: float
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at and isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(
                self.created_at.replace("Z", "+00:00")
            )
        if self.last_updated and isinstance(self.last_updated, str):
            self.last_updated = datetime.fromisoformat(
                self.last_updated.replace("Z", "+00:00")
            )


@dataclass
class ProcessingResult:
    """Result from processing data points."""

    clusters: List[Cluster]
    outliers: Points
    processing_time_ms: float
    algorithm_quality: float
    request_id: str
    total_points: int = 0

    def __post_init__(self):
        if not self.total_points:
            self.total_points = sum(
                len(cluster.points) for cluster in self.clusters
            ) + len(self.outliers)


@dataclass
class AlgorithmStatus:
    """Current status of the NCS algorithm."""

    is_ready: bool
    active_clusters: int
    total_points_processed: int
    clustering_quality: float
    memory_usage_mb: float
    last_processing_time_ms: float
    error_count: int = 0
    uptime_seconds: float = 0


@dataclass
class HealthStatus:
    """API health status information."""

    status: str
    timestamp: datetime
    version: str
    algorithm_ready: bool
    uptime_seconds: float
    components: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# Exception Classes
# =============================================================================


class NCSError(Exception):
    """Base exception for NCS client errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.request_id = request_id


class AuthenticationError(NCSError):
    """Authentication failed."""

    pass


class RateLimitError(NCSError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ValidationError(NCSError):
    """Request validation failed."""

    pass


class ProcessingError(NCSError):
    """Algorithm processing error."""

    pass


class ConnectionError(NCSError):
    """Network connection error."""

    pass


# =============================================================================
# Main Client Classes
# =============================================================================


class NCSClient:
    """Synchronous client for the NeuroCluster Streamer API."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        verify_ssl: bool = True,
        headers: Optional[Dict[str, str]] = None,
        log_level: str = "INFO",
    ):
        """
        Initialize the NCS client.

        Args:
            base_url: Base URL of the NCS API
            api_key: API key for authentication
            jwt_token: JWT token for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            verify_ssl: Whether to verify SSL certificates
            headers: Additional headers to send with requests
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.jwt_token = jwt_token
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.verify_ssl = verify_ssl
        self.logger = self._setup_logging(log_level)

        # Setup session with retry strategy
        self.session = self._create_session()
        self._setup_authentication()

        # Add custom headers
        if headers:
            self.session.headers.update(headers)

    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup logging for the client."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        logger.setLevel(getattr(logging, log_level.upper()))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy."""
        session = requests.Session()

        # Setup retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=[
                "HEAD",
                "GET",
                "POST",
                "PUT",
                "DELETE",
                "OPTIONS",
                "TRACE",
            ],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default headers
        session.headers.update(
            {
                "User-Agent": f"ncs-python-sdk/1.0.0",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

        return session

    def _setup_authentication(self):
        """Setup authentication headers."""
        if self.api_key:
            self.session.headers["X-API-Key"] = self.api_key
        elif self.jwt_token:
            self.session.headers["Authorization"] = f"Bearer {self.jwt_token}"

    def authenticate(self, username: str, password: str) -> str:
        """
        Authenticate with username/password and get JWT token.

        Args:
            username: Username for authentication
            password: Password for authentication

        Returns:
            JWT token

        Raises:
            AuthenticationError: If authentication fails
        """
        self.logger.info("Authenticating with username/password")

        auth_data = {"username": username, "password": password}

        try:
            response = self.session.post(
                f"{self.base_url}/auth/login",
                data=auth_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self.timeout,
                verify=self.verify_ssl,
            )

            if response.status_code == 200:
                token_data = response.json()
                self.jwt_token = token_data["access_token"]
                self.session.headers["Authorization"] = f"Bearer {self.jwt_token}"
                self.logger.info("Authentication successful")
                return self.jwt_token
            else:
                raise AuthenticationError(
                    "Authentication failed",
                    status_code=response.status_code,
                    request_id=response.headers.get("X-Request-ID"),
                )

        except requests.RequestException as e:
            raise ConnectionError(f"Authentication request failed: {str(e)}")

    @sleep_and_retry
    @limits(calls=100, period=60)  # Rate limiting: 100 calls per minute
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> requests.Response:
        """
        Make HTTP request with error handling and rate limiting.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request body data
            params: URL parameters

        Returns:
            Response object

        Raises:
            Various NCSError subclasses based on response
        """
        url = f"{self.base_url}{endpoint}"

        try:
            self.logger.debug(f"Making {method} request to {url}")

            response = self.session.request(
                method=method,
                url=url,
                json=data if data else None,
                params=params,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )

            # Handle different response codes
            if response.status_code == 200:
                return response
            elif response.status_code == 401:
                raise AuthenticationError(
                    "Authentication required or token expired",
                    status_code=response.status_code,
                    request_id=response.headers.get("X-Request-ID"),
                )
            elif response.status_code == 403:
                raise AuthenticationError(
                    "Insufficient permissions",
                    status_code=response.status_code,
                    request_id=response.headers.get("X-Request-ID"),
                )
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                raise RateLimitError(
                    "Rate limit exceeded",
                    status_code=response.status_code,
                    retry_after=retry_after,
                    request_id=response.headers.get("X-Request-ID"),
                )
            elif response.status_code == 422:
                error_detail = response.json().get("detail", "Validation error")
                raise ValidationError(
                    error_detail,
                    status_code=response.status_code,
                    request_id=response.headers.get("X-Request-ID"),
                )
            elif response.status_code >= 500:
                raise ProcessingError(
                    "Server error occurred",
                    status_code=response.status_code,
                    request_id=response.headers.get("X-Request-ID"),
                )
            else:
                raise NCSError(
                    f"Unexpected response: {response.status_code}",
                    status_code=response.status_code,
                    request_id=response.headers.get("X-Request-ID"),
                )

        except requests.RequestException as e:
            raise ConnectionError(f"Request failed: {str(e)}")

    def health_check(self) -> HealthStatus:
        """
        Check API health status.

        Returns:
            HealthStatus object
        """
        self.logger.debug("Checking API health")

        response = self._make_request("GET", "/health")
        data = response.json()

        return HealthStatus(
            status=data["status"],
            timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")),
            version=data["version"],
            algorithm_ready=data["algorithm_ready"],
            uptime_seconds=data["uptime_seconds"],
            components=data.get("components", {}),
        )

    def process_points(
        self, points: Points, options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Process data points through the clustering algorithm.

        Args:
            points: List of data points to process
            options: Additional processing options

        Returns:
            ProcessingResult with clusters and outliers
        """
        self.logger.info(f"Processing {len(points)} data points")

        request_data = {"points": points}

        if options:
            request_data.update(options)

        response = self._make_request(
            "POST", "/api/v1/process_points", data=request_data
        )
        data = response.json()

        # Parse clusters
        clusters = [
            Cluster(
                id=cluster["id"],
                center=cluster["center"],
                points=cluster["points"],
                size=cluster["size"],
                quality=cluster["quality"],
                created_at=cluster.get("created_at"),
                last_updated=cluster.get("last_updated"),
            )
            for cluster in data["clusters"]
        ]

        result = ProcessingResult(
            clusters=clusters,
            outliers=data["outliers"],
            processing_time_ms=data["processing_time_ms"],
            algorithm_quality=data["algorithm_quality"],
            request_id=data["request_id"],
        )

        self.logger.info(
            f"Processing complete: {len(clusters)} clusters, {len(data['outliers'])} outliers"
        )
        return result

    def get_clusters_summary(
        self, filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get summary of current clusters.

        Args:
            filters: Optional filters for cluster selection

        Returns:
            Cluster summary data
        """
        self.logger.debug("Getting clusters summary")

        params = filters if filters else {}
        response = self._make_request("GET", "/api/v1/clusters_summary", params=params)

        return response.json()

    def get_algorithm_status(self) -> AlgorithmStatus:
        """
        Get current algorithm status.

        Returns:
            AlgorithmStatus object
        """
        self.logger.debug("Getting algorithm status")

        response = self._make_request("GET", "/api/v1/algorithm_status")
        data = response.json()

        return AlgorithmStatus(
            is_ready=data["is_ready"],
            active_clusters=data["active_clusters"],
            total_points_processed=data["total_points_processed"],
            clustering_quality=data["clustering_quality"],
            memory_usage_mb=data["memory_usage_mb"],
            last_processing_time_ms=data["last_processing_time_ms"],
            error_count=data.get("error_count", 0),
            uptime_seconds=data.get("uptime_seconds", 0),
        )

    def process_points_batch(
        self, points_batch: Points, batch_options: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Process a large batch of points with optimizations.

        Args:
            points_batch: Large batch of points to process
            batch_options: Batch-specific options (timeout, chunk_size, etc.)

        Returns:
            ProcessingResult for the entire batch
        """
        self.logger.info(f"Processing batch of {len(points_batch)} points")

        # Default batch options
        options = {
            "batch_mode": True,
            "timeout": batch_options.get("timeout", 60) if batch_options else 60,
        }

        if batch_options:
            options.update(batch_options)

        return self.process_points(points_batch, options)

    def close(self):
        """Close the client session."""
        if hasattr(self, "session"):
            self.session.close()
            self.logger.debug("Client session closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NCSClient":
        """
        Create client from configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Configured NCSClient instance
        """
        return cls(**config)


# =============================================================================
# Client Factory and Utilities
# =============================================================================


def create_client(base_url: str, api_key: Optional[str] = None, **kwargs) -> NCSClient:
    """
    Factory function to create an NCS client.

    Args:
        base_url: Base URL of the NCS API
        api_key: API key for authentication
        **kwargs: Additional client options

    Returns:
        Configured NCSClient instance
    """
    return NCSClient(base_url=base_url, api_key=api_key, **kwargs)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main classes
    "NCSClient",
    # Data models
    "Cluster",
    "ProcessingResult",
    "AlgorithmStatus",
    "HealthStatus",
    # Exceptions
    "NCSError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "ProcessingError",
    "ConnectionError",
    # Utilities
    "create_client",
    # Type definitions
    "Point",
    "Points",
]
