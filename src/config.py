"""
Configuration management for NCS API
Handles environment variables and application settings
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Server Configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    debug: bool = Field(False, env="DEBUG")
    workers: int = Field(1, env="WORKERS")

    # Security
    require_api_key: bool = Field(False, env="REQUIRE_API_KEY")
    api_key: Optional[str] = Field(None, env="API_KEY")
    allowed_origins: List[str] = Field(["*"], env="ALLOWED_ORIGINS")
    allowed_hosts: List[str] = Field(["*"], env="ALLOWED_HOSTS")

    # Algorithm Settings
    max_clusters: int = Field(100, env="MAX_CLUSTERS")
    quality_threshold: float = Field(0.85, env="QUALITY_THRESHOLD")
    batch_size: int = Field(1000, env="BATCH_SIZE")
    max_data_points: int = Field(100000, env="MAX_DATA_POINTS")

    # Database/Cache
    redis_url: Optional[str] = Field(None, env="REDIS_URL")
    cache_ttl: int = Field(3600, env="CACHE_TTL")  # seconds

    # Monitoring
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Rate Limiting
    rate_limit_requests: int = Field(100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: str = Field("1 minute", env="RATE_LIMIT_WINDOW")

    # Performance
    max_request_size: int = Field(10 * 1024 * 1024, env="MAX_REQUEST_SIZE")  # 10MB
    request_timeout: int = Field(300, env="REQUEST_TIMEOUT")  # seconds

    # WebSocket
    max_websocket_connections: int = Field(100, env="MAX_WEBSOCKET_CONNECTIONS")
    websocket_ping_interval: int = Field(30, env="WEBSOCKET_PING_INTERVAL")

    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    @validator("allowed_origins")
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("allowed_hosts")
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v

    @validator("quality_threshold")
    def validate_quality_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Quality threshold must be between 0.0 and 1.0")
        return v

    @validator("max_clusters")
    def validate_max_clusters(cls, v):
        if v < 1:
            raise ValueError("Max clusters must be at least 1")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class DevelopmentSettings(Settings):
    """Development environment settings"""

    debug: bool = True
    log_level: str = "DEBUG"
    require_api_key: bool = False


class ProductionSettings(Settings):
    """Production environment settings"""

    debug: bool = False
    log_level: str = "INFO"
    require_api_key: bool = True
    allowed_origins: List[str] = []  # Must be explicitly set
    allowed_hosts: List[str] = []  # Must be explicitly set


class TestingSettings(Settings):
    """Testing environment settings"""

    debug: bool = True
    log_level: str = "DEBUG"
    require_api_key: bool = False
    redis_url: Optional[str] = None  # Use in-memory for tests


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings based on environment
    Cached for performance
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()

    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Configuration validation
def validate_settings(settings: Settings) -> None:
    """Validate settings and raise errors if invalid"""

    if settings.require_api_key and not settings.api_key:
        raise ValueError("API_KEY must be set when REQUIRE_API_KEY is True")

    if settings.workers < 1:
        raise ValueError("WORKERS must be at least 1")

    if settings.port < 1 or settings.port > 65535:
        raise ValueError("PORT must be between 1 and 65535")

    # Validate Redis URL format if provided
    if settings.redis_url:
        if not settings.redis_url.startswith(("redis://", "rediss://")):
            raise ValueError("REDIS_URL must start with redis:// or rediss://")


# Environment-specific configurations
ALGORITHM_CONFIG = {
    "ncs": {
        "default_quality_threshold": 0.85,
        "max_iterations": 1000,
        "convergence_tolerance": 1e-6,
        "parallel_processing": True,
    },
    "kmeans": {"max_iterations": 300, "tolerance": 1e-4, "init_method": "k-means++"},
    "dbscan": {"default_eps": 0.5, "default_min_samples": 5, "metric": "euclidean"},
}

PERFORMANCE_CONFIG = {
    "batch_processing": {
        "chunk_size": 1000,
        "max_parallel_jobs": 4,
        "memory_limit_mb": 512,
    },
    "caching": {
        "enable_result_cache": True,
        "cache_large_results": False,
        "max_cache_size_mb": 100,
    },
}

# Export settings instance
settings = get_settings()

# Validate on import
try:
    validate_settings(settings)
except ValueError as e:
    raise RuntimeError(f"Invalid configuration: {e}")
