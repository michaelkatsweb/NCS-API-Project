#!/usr/bin/env python3
"""
NeuroCluster Streamer Python SDK - Async Client
==============================================
Asynchronous client for high-performance NCS API interactions

This module provides async/await support for the NCS API, enabling
high-throughput concurrent operations and streaming capabilities.

Author: NCS API Development Team
Year: 2025
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

import aiofiles
import httpx
import websockets
from tenacity import AsyncRetrying, retry, stop_after_attempt, wait_exponential

# Import shared types and models from main client
from .ncs_client import (
    AlgorithmStatus,
    AuthenticationError,
    Cluster,
    ConnectionError,
    HealthStatus,
    NCSError,
    Point,
    Points,
    ProcessingError,
    ProcessingResult,
    RateLimitError,
    ValidationError,
)

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# Async Rate Limiter
# =============================================================================


class AsyncRateLimiter:
    """Async rate limiter for API calls."""

    def __init__(self, calls: int, period: float):
        self.calls = calls
        self.period = period
        self.calls_made = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire rate limit permission."""
        async with self._lock:
            now = asyncio.get_event_loop().time()

            # Remove old calls outside the period
            self.calls_made = [
                call_time
                for call_time in self.calls_made
                if now - call_time < self.period
            ]

            # Check if we can make a call
            if len(self.calls_made) >= self.calls:
                sleep_time = self.period - (now - self.calls_made[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    return await self.acquire()

            self.calls_made.append(now)


# =============================================================================
# Streaming Support
# =============================================================================


class StreamingConnection:
    """WebSocket connection for real-time streaming."""

    def __init__(self, websocket, on_message: Optional[Callable] = None):
        self.websocket = websocket
        self.on_message = on_message
        self._running = False
        self._task = None

    async def start_listening(self):
        """Start listening for incoming messages."""
        self._running = True
        self._task = asyncio.create_task(self._listen_loop())

    async def _listen_loop(self):
        """Main listening loop for WebSocket messages."""
        try:
            async for message in self.websocket:
                if not self._running:
                    break

                try:
                    data = json.loads(message)
                    if self.on_message:
                        await self.on_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON: {message}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    async def send_point(self, point: Point):
        """Send a data point through the stream."""
        message = json.dumps({"type": "point", "data": point})
        await self.websocket.send(message)

    async def send_batch(self, points: Points):
        """Send a batch of points through the stream."""
        message = json.dumps({"type": "batch", "data": points})
        await self.websocket.send(message)

    async def stop(self):
        """Stop the streaming connection."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.websocket.close()


# =============================================================================
# Async Client Class
# =============================================================================


class AsyncNCSClient:
    """Asynchronous client for the NeuroCluster Streamer API."""

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
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
    ):
        """
        Initialize the async NCS client.

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
            max_connections: Maximum number of concurrent connections
            max_keepalive_connections: Maximum number of keepalive connections
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.jwt_token = jwt_token
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.verify_ssl = verify_ssl
        self.logger = self._setup_logging(log_level)

        # Setup HTTP client
        self.client = self._create_client(max_connections, max_keepalive_connections)
        self._setup_authentication()

        # Add custom headers
        if headers:
            self.client.headers.update(headers)

        # Rate limiter (100 calls per minute)
        self.rate_limiter = AsyncRateLimiter(calls=100, period=60)

        # Streaming connections
        self._streaming_connections = {}

    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup logging for the async client."""
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

    def _create_client(
        self, max_connections: int, max_keepalive_connections: int
    ) -> httpx.AsyncClient:
        """Create async HTTP client with connection pooling."""
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        )

        timeout = httpx.Timeout(self.timeout, connect=10.0)

        return httpx.AsyncClient(
            limits=limits,
            timeout=timeout,
            verify=self.verify_ssl,
            headers={
                "User-Agent": "ncs-python-sdk-async/1.0.0",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )

    def _setup_authentication(self):
        """Setup authentication headers."""
        if self.api_key:
            self.client.headers["X-API-Key"] = self.api_key
        elif self.jwt_token:
            self.client.headers["Authorization"] = f"Bearer {self.jwt_token}"

    async def authenticate(self, username: str, password: str) -> str:
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
            response = await self.client.post(
                f"{self.base_url}/auth/login",
                data=auth_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            if response.status_code == 200:
                token_data = response.json()
                self.jwt_token = token_data["access_token"]
                self.client.headers["Authorization"] = f"Bearer {self.jwt_token}"
                self.logger.info("Authentication successful")
                return self.jwt_token
            else:
                raise AuthenticationError(
                    "Authentication failed",
                    status_code=response.status_code,
                    request_id=response.headers.get("X-Request-ID"),
                )

        except httpx.RequestError as e:
            raise ConnectionError(f"Authentication request failed: {str(e)}")

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        use_rate_limit: bool = True,
    ) -> httpx.Response:
        """
        Make async HTTP request with error handling and rate limiting.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request body data
            params: URL parameters
            use_rate_limit: Whether to apply rate limiting

        Returns:
            Response object

        Raises:
            Various NCSError subclasses based on response
        """
        if use_rate_limit:
            await self.rate_limiter.acquire()

        url = f"{self.base_url}{endpoint}"

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.retry_delay, min=1, max=10),
        ):
            with attempt:
                try:
                    self.logger.debug(f"Making async {method} request to {url}")

                    response = await self.client.request(
                        method=method,
                        url=url,
                        json=data if data else None,
                        params=params,
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

                except httpx.RequestError as e:
                    raise ConnectionError(f"Request failed: {str(e)}")

    async def health_check(self) -> HealthStatus:
        """
        Check API health status.

        Returns:
            HealthStatus object
        """
        self.logger.debug("Checking API health")

        response = await self._make_request("GET", "/health")
        data = response.json()

        return HealthStatus(
            status=data["status"],
            timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")),
            version=data["version"],
            algorithm_ready=data["algorithm_ready"],
            uptime_seconds=data["uptime_seconds"],
            components=data.get("components", {}),
        )

    async def process_points(
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

        response = await self._make_request(
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

    async def get_clusters_summary(
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
        response = await self._make_request(
            "GET", "/api/v1/clusters_summary", params=params
        )

        return response.json()

    async def get_algorithm_status(self) -> AlgorithmStatus:
        """
        Get current algorithm status.

        Returns:
            AlgorithmStatus object
        """
        self.logger.debug("Getting algorithm status")

        response = await self._make_request("GET", "/api/v1/algorithm_status")
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

    async def process_points_concurrent(
        self, points_batches: List[Points], max_concurrent: int = 5
    ) -> List[ProcessingResult]:
        """
        Process multiple batches of points concurrently.

        Args:
            points_batches: List of point batches to process
            max_concurrent: Maximum number of concurrent requests

        Returns:
            List of ProcessingResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch(batch: Points) -> ProcessingResult:
            async with semaphore:
                return await self.process_points(batch)

        tasks = [process_batch(batch) for batch in points_batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch {i} failed: {result}")
                raise result
            processed_results.append(result)

        return processed_results

    async def start_streaming(
        self, on_message: Optional[Callable] = None, connection_id: str = "default"
    ) -> StreamingConnection:
        """
        Start a WebSocket streaming connection.

        Args:
            on_message: Callback function for incoming messages
            connection_id: Unique identifier for this connection

        Returns:
            StreamingConnection object
        """
        ws_url = self.base_url.replace("http", "ws") + "/ws/stream"

        # Add authentication to WebSocket URL
        if self.api_key:
            ws_url += f"?api_key={self.api_key}"
        elif self.jwt_token:
            ws_url += f"?token={self.jwt_token}"

        websocket = await websockets.connect(ws_url)
        connection = StreamingConnection(websocket, on_message)

        self._streaming_connections[connection_id] = connection
        await connection.start_listening()

        self.logger.info(f"Started streaming connection: {connection_id}")
        return connection

    async def stop_streaming(self, connection_id: str = "default"):
        """
        Stop a streaming connection.

        Args:
            connection_id: ID of the connection to stop
        """
        if connection_id in self._streaming_connections:
            await self._streaming_connections[connection_id].stop()
            del self._streaming_connections[connection_id]
            self.logger.info(f"Stopped streaming connection: {connection_id}")

    async def process_stream(
        self,
        points_stream: AsyncGenerator[Points, None],
        batch_size: int = 100,
        on_result: Optional[Callable[[ProcessingResult], None]] = None,
    ):
        """
        Process a stream of points in batches.

        Args:
            points_stream: Async generator yielding points
            batch_size: Size of processing batches
            on_result: Callback for processing results
        """
        batch = []

        async for points in points_stream:
            batch.extend(points)

            if len(batch) >= batch_size:
                result = await self.process_points(batch[:batch_size])
                if on_result:
                    await on_result(result)
                batch = batch[batch_size:]

        # Process remaining points
        if batch:
            result = await self.process_points(batch)
            if on_result:
                await on_result(result)

    async def close(self):
        """Close all connections and cleanup resources."""
        # Stop all streaming connections
        for connection_id in list(self._streaming_connections.keys()):
            await self.stop_streaming(connection_id)

        # Close HTTP client
        await self.client.aclose()
        self.logger.debug("Async client closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AsyncNCSClient":
        """
        Create async client from configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Configured AsyncNCSClient instance
        """
        return cls(**config)


# =============================================================================
# Async Utilities
# =============================================================================


async def create_async_client(
    base_url: str, api_key: Optional[str] = None, **kwargs
) -> AsyncNCSClient:
    """
    Factory function to create an async NCS client.

    Args:
        base_url: Base URL of the NCS API
        api_key: API key for authentication
        **kwargs: Additional client options

    Returns:
        Configured AsyncNCSClient instance
    """
    return AsyncNCSClient(base_url=base_url, api_key=api_key, **kwargs)


@asynccontextmanager
async def async_client_context(base_url: str, api_key: Optional[str] = None, **kwargs):
    """
    Async context manager for NCS client.

    Args:
        base_url: Base URL of the NCS API
        api_key: API key for authentication
        **kwargs: Additional client options

    Yields:
        Configured AsyncNCSClient instance
    """
    client = AsyncNCSClient(base_url=base_url, api_key=api_key, **kwargs)
    try:
        yield client
    finally:
        await client.close()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main classes
    "AsyncNCSClient",
    "StreamingConnection",
    "AsyncRateLimiter",
    # Utilities
    "create_async_client",
    "async_client_context",
]
