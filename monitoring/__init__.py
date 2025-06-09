"""
Monitoring package for NeuroCluster Streamer API.

This package provides comprehensive observability tools including:
- Performance metrics collection and reporting
- Structured logging with correlation IDs
- Health checks and status monitoring
- Alerting and notification systems
- Dashboard configurations and data aggregation
"""

from .metrics import (
    MetricsCollector,
    PrometheusMetrics,
    CustomMetrics,
    PerformanceTracker,
    get_metrics_collector
)

from .logging import (
    setup_logging,
    get_logger,
    LoggingMiddleware,
    CorrelationIdFilter,
    StructuredLogger,
    audit_logger,
    performance_logger,
    security_logger
)

from .health import (
    HealthChecker,
    ComponentHealthCheck,
    SystemHealthMonitor,
    get_health_checker,
    health_check_registry
)

from .alerts import (
    AlertManager,
    AlertRule,
    NotificationChannel,
    EmailNotifier,
    SlackNotifier,
    WebhookNotifier,
    get_alert_manager
)

from .dashboard import (
    DashboardDataAggregator,
    MetricsDashboard,
    RealtimeMonitor,
    SessionAnalytics,
    get_dashboard_data
)

__all__ = [
    # Metrics
    "MetricsCollector",
    "PrometheusMetrics", 
    "CustomMetrics",
    "PerformanceTracker",
    "get_metrics_collector",
    
    # Logging
    "setup_logging",
    "get_logger",
    "LoggingMiddleware",
    "CorrelationIdFilter", 
    "StructuredLogger",
    "audit_logger",
    "performance_logger",
    "security_logger",
    
    # Health Checks
    "HealthChecker",
    "ComponentHealthCheck",
    "SystemHealthMonitor", 
    "get_health_checker",
    "health_check_registry",
    
    # Alerts
    "AlertManager",
    "AlertRule",
    "NotificationChannel",
    "EmailNotifier",
    "SlackNotifier", 
    "WebhookNotifier",
    "get_alert_manager",
    
    # Dashboard
    "DashboardDataAggregator",
    "MetricsDashboard",
    "RealtimeMonitor",
    "SessionAnalytics", 
    "get_dashboard_data"
]

# Package metadata
__version__ = "1.0.0"
__author__ = "NCS Development Team"
__description__ = "Comprehensive monitoring and observability for NeuroCluster Streamer API"

# Initialize global monitoring components
_metrics_collector = None
_health_checker = None
_alert_manager = None

def initialize_monitoring(config: dict = None):
    """
    Initialize all monitoring components with configuration.
    
    Args:
        config: Optional configuration dictionary
    """
    global _metrics_collector, _health_checker, _alert_manager
    
    # Initialize metrics collection
    _metrics_collector = MetricsCollector(config)
    
    # Initialize health checking
    _health_checker = HealthChecker(config)
    
    # Initialize alerting
    _alert_manager = AlertManager(config)
    
    # Setup logging
    setup_logging(config)

def shutdown_monitoring():
    """Gracefully shutdown all monitoring components."""
    global _metrics_collector, _health_checker, _alert_manager
    
    if _metrics_collector:
        _metrics_collector.shutdown()
    if _health_checker:
        _health_checker.shutdown()
    if _alert_manager:
        _alert_manager.shutdown()