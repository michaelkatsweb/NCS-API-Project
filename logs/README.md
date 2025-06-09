# 📄 NCS API - Application Logs Directory

This directory contains application logs for the NeuroCluster Streamer API. The logging system is designed for comprehensive observability, debugging, and performance monitoring in both development and production environments.

## 📁 Directory Structure

```
logs/
├── app/                     # Application logs
│   ├── ncs-api.log         # Main application log (current)
│   ├── ncs-api.1.log       # Rotated log (previous day)
│   ├── ncs-api.2.log       # Rotated log (2 days ago)
│   └── ...
├── access/                  # HTTP access logs
│   ├── access.log          # HTTP request/response logs
│   ├── access.1.log        # Rotated access logs
│   └── ...
├── error/                   # Error-specific logs
│   ├── error.log           # Error and exception logs
│   ├── error.1.log         # Rotated error logs
│   └── ...
├── security/                # Security-related logs
│   ├── auth.log            # Authentication events
│   ├── audit.log           # Security audit trail
│   └── ...
├── performance/             # Performance monitoring logs
│   ├── metrics.log         # Performance metrics
│   ├── slow-queries.log    # Database slow queries
│   └── ...
└── debug/                   # Development and debugging logs
    ├── debug.log           # Detailed debug information
    ├── database.log        # Database query logs
    └── algorithm.log       # Algorithm-specific logs
```

## 🎯 Log Categories

### Application Logs (`app/`)
**Primary application logs containing:**
- Service startup/shutdown events
- Business logic execution
- Algorithm processing results
- Configuration changes
- Health check results

**Format:** Structured JSON logging for easy parsing
```json
{
  "timestamp": "2025-06-07T14:30:45.123Z",
  "level": "INFO",
  "logger": "ncs.api.main",
  "message": "Processing clustering request",
  "request_id": "req_123e4567-e89b-12d3-a456-426614174000",
  "user_id": "user_987fcdeb-51a2-43d7-9c8b-123456789abc",
  "processing_time": 1.234,
  "points_count": 1000,
  "clusters_found": 5
}
```

### Access Logs (`access/`)
**HTTP request/response logs following the Extended Log Format:**
- Request method, URL, and parameters
- Response status codes and sizes
- Client IP addresses and user agents
- Request processing times
- Authentication status

**Format:** Apache Common Log Format + Custom Fields
```
192.168.1.100 - user123 [07/Jun/2025:14:30:45 +0000] "POST /api/v1/process_points HTTP/1.1" 200 2048 "https://client-app.com" "Python-SDK/1.0.0" req_123e4567 1.234
```

### Error Logs (`error/`)
**Error and exception logs including:**
- Application exceptions and stack traces
- HTTP error responses (4xx, 5xx)
- Database connection errors
- External service failures
- Critical system alerts

**Format:** Enhanced error logging with context
```json
{
  "timestamp": "2025-06-07T14:30:45.123Z",
  "level": "ERROR",
  "logger": "ncs.api.clustering",
  "message": "Clustering algorithm failed",
  "request_id": "req_123e4567-e89b-12d3-a456-426614174000",
  "exception": {
    "type": "ValueError",
    "message": "Invalid input dimensions",
    "stack_trace": "Traceback (most recent call last):\n  File..."
  },
  "context": {
    "input_size": 1000,
    "algorithm": "ncs_v8",
    "parameters": {"threshold": 0.5}
  }
}
```

### Security Logs (`security/`)
**Security-related events including:**
- Authentication attempts (success/failure)
- API key usage and violations
- Rate limiting events
- Suspicious activity detection
- Access control violations

**Format:** Security-focused structured logging
```json
{
  "timestamp": "2025-06-07T14:30:45.123Z",
  "level": "WARN",
  "logger": "ncs.security.auth",
  "event_type": "failed_authentication",
  "message": "Failed login attempt",
  "client_ip": "192.168.1.100",
  "user_agent": "curl/7.68.0",
  "attempted_username": "admin",
  "failure_reason": "invalid_credentials",
  "request_id": "req_123e4567-e89b-12d3-a456-426614174000"
}
```

### Performance Logs (`performance/`)
**Performance monitoring and metrics:**
- Response time distributions
- Database query performance
- Algorithm execution times
- Memory and CPU usage
- Cache hit/miss rates

**Format:** Metrics-focused logging
```json
{
  "timestamp": "2025-06-07T14:30:45.123Z",
  "level": "INFO",
  "logger": "ncs.performance.metrics",
  "metric_type": "response_time",
  "endpoint": "/api/v1/process_points",
  "duration_ms": 1234,
  "status_code": 200,
  "request_size": 2048,
  "response_size": 512,
  "database_queries": 3,
  "cache_hits": 2,
  "cache_misses": 1
}
```

### Debug Logs (`debug/`)
**Detailed debugging information:**
- Detailed execution flow
- Variable values and state changes
- Algorithm step-by-step processing
- Database query details
- External API calls

**Format:** Verbose debugging with context
```json
{
  "timestamp": "2025-06-07T14:30:45.123Z",
  "level": "DEBUG",
  "logger": "ncs.algorithm.clustering",
  "message": "Starting distance calculation phase",
  "request_id": "req_123e4567-e89b-12d3-a456-426614174000",
  "algorithm_state": {
    "phase": "distance_calculation",
    "processed_points": 750,
    "total_points": 1000,
    "current_clusters": 3,
    "iteration": 15
  },
  "performance": {
    "memory_usage_mb": 156.7,
    "cpu_usage_percent": 45.2
  }
}
```

## 📊 Log Levels

The NCS API uses standard log levels with specific use cases:

| Level | Purpose | Examples |
|-------|---------|----------|
| **DEBUG** | Detailed diagnostic information | Variable values, algorithm steps, flow control |
| **INFO** | General application flow | Request processing, successful operations, status updates |
| **WARNING** | Potentially harmful situations | Deprecated API usage, recoverable errors, rate limiting |
| **ERROR** | Error events but application continues | Failed requests, invalid input, external service errors |
| **CRITICAL** | Serious error events | Database connectivity loss, service unavailability |

## ⚙️ Configuration

### Environment Variables

Control logging behavior through environment variables:

```bash
# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# Log format (json, text)
LOG_FORMAT=json

# Enable/disable specific log categories
LOG_ACCESS_ENABLED=true
LOG_SECURITY_ENABLED=true
LOG_PERFORMANCE_ENABLED=true
LOG_DEBUG_ENABLED=false

# Log file settings
LOG_FILE_MAX_SIZE=100MB
LOG_FILE_BACKUP_COUNT=7
LOG_TO_STDOUT=true
LOG_TO_FILE=true

# Structured logging settings
LOG_INCLUDE_TRACE_ID=true
LOG_INCLUDE_USER_ID=true
LOG_INCLUDE_REQUEST_ID=true
```

### Configuration Example

```python
# config.py logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            'class': 'ncs.logging.JSONFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        },
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        }
    },
    'handlers': {
        'app_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/app/ncs-api.log',
            'maxBytes': 100 * 1024 * 1024,  # 100MB
            'backupCount': 7,
            'formatter': 'json'
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/error/error.log',
            'maxBytes': 100 * 1024 * 1024,
            'backupCount': 7,
            'formatter': 'json'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        }
    },
    'loggers': {
        'ncs': {
            'level': 'INFO',
            'handlers': ['app_file', 'console'],
            'propagate': False
        },
        'ncs.security': {
            'level': 'INFO',
            'handlers': ['app_file', 'error_file'],
            'propagate': False
        }
    }
}
```

## 🔄 Log Rotation and Retention

### Automatic Rotation
- **Size-based rotation:** 100MB per log file
- **Time-based rotation:** Daily rotation for high-volume logs
- **Backup retention:** 7 days for development, 30 days for production
- **Compression:** Older logs are automatically compressed

### Retention Policy

| Log Type | Development | Staging | Production |
|----------|-------------|---------|------------|
| Application | 7 days | 14 days | 30 days |
| Access | 7 days | 14 days | 60 days |
| Error | 14 days | 30 days | 90 days |
| Security | 14 days | 30 days | 365 days |
| Performance | 7 days | 14 days | 30 days |
| Debug | 3 days | 7 days | Not enabled |

### Manual Log Management

```bash
# View recent application logs
tail -f logs/app/ncs-api.log

# Search for specific errors
grep -r "ERROR" logs/error/

# Monitor real-time access logs
tail -f logs/access/access.log

# Analyze performance metrics
grep "response_time" logs/performance/metrics.log | jq '.duration_ms'

# Clean old logs manually (if needed)
find logs/ -name "*.log.*" -mtime +30 -delete

# Compress old logs
gzip logs/app/ncs-api.*.log
```

## 🔍 Log Analysis and Monitoring

### Integration with Monitoring Systems

**Prometheus Integration:**
- Log-based metrics extraction
- Error rate monitoring
- Performance metrics collection
- Custom alerting rules

**ELK Stack Integration:**
```yaml
# Filebeat configuration for ELK stack
filebeat.inputs:
- type: log
  paths:
    - /app/logs/app/*.log
    - /app/logs/error/*.log
  fields:
    service: ncs-api
    environment: production
  fields_under_root: true
  json.keys_under_root: true
  json.add_error_key: true
```

**Grafana Dashboards:**
- Real-time log volume monitoring
- Error rate trends
- Performance metrics visualization
- Security event tracking

### Common Analysis Patterns

**Error Analysis:**
```bash
# Find most common errors
grep "ERROR" logs/error/error.log | jq -r '.exception.type' | sort | uniq -c | sort -nr

# Analyze error trends
grep "ERROR" logs/error/error.log | jq -r '.timestamp' | cut -d'T' -f1 | sort | uniq -c

# Find errors by endpoint
grep "ERROR" logs/error/error.log | jq -r '.context.endpoint' | sort | uniq -c | sort -nr
```

**Performance Analysis:**
```bash
# Average response times by endpoint
grep "response_time" logs/performance/metrics.log | jq -r '"\(.endpoint) \(.duration_ms)"' | awk '{sum[$1]+=$2; count[$1]++} END {for(i in sum) print i, sum[i]/count[i]}'

# Find slow requests
grep "response_time" logs/performance/metrics.log | jq 'select(.duration_ms > 5000)'

# Database performance analysis
grep "database" logs/debug/database.log | jq 'select(.execution_time > 1000)'
```

**Security Analysis:**
```bash
# Failed authentication attempts
grep "failed_authentication" logs/security/auth.log | jq -r '.client_ip' | sort | uniq -c | sort -nr

# Rate limiting events
grep "rate_limit" logs/security/audit.log | jq -r '"\(.timestamp) \(.client_ip) \(.endpoint)"'

# Suspicious activity detection
grep "suspicious" logs/security/audit.log | jq -r '"\(.timestamp) \(.event_type) \(.client_ip)"'
```

## 🛡️ Security Considerations

### Log Sanitization
- **PII Removal:** Personal identifiable information is automatically scrubbed
- **Secret Masking:** API keys, passwords, and tokens are masked in logs
- **Data Anonymization:** User data is anonymized or hashed when logged

### Access Control
- Log files are readable only by the application user and administrators
- Sensitive logs (security, audit) have restricted access permissions
- Log rotation maintains appropriate file permissions

### Compliance
- Logs support audit requirements for SOC 2, ISO 27001
- Retention policies comply with GDPR and data protection regulations
- Security logs provide forensic capabilities for incident response

## 🚨 Alerting and Notifications

### Critical Alerts
Configure alerts for these log patterns:

```yaml
# Example alerting rules
groups:
- name: ncs-api-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(ncs_api_errors_total[5m]) > 0.1
    for: 2m
    annotations:
      summary: High error rate detected in NCS API

  - alert: DatabaseConnectionLoss
    expr: increase(ncs_database_connection_errors[1m]) > 0
    for: 0m
    annotations:
      summary: Database connection error detected

  - alert: SecurityIncident
    expr: increase(ncs_security_violations[1m]) > 0
    for: 0m
    annotations:
      summary: Security violation detected
```

### Log-based Metrics

```python
# Example log-based metrics extraction
from prometheus_client import Counter, Histogram, Gauge

# Metrics derived from logs
REQUEST_COUNT = Counter('ncs_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('ncs_api_request_duration_seconds', 'Request duration in seconds', ['endpoint'])
ACTIVE_USERS = Gauge('ncs_api_active_users', 'Number of active users')
ERROR_COUNT = Counter('ncs_api_errors_total', 'Total errors', ['error_type', 'endpoint'])
```

## 🔧 Troubleshooting

### Common Issues

**Log Files Not Created:**
1. Check directory permissions: `ls -la logs/`
2. Verify disk space: `df -h`
3. Check log configuration: `cat config.py | grep LOG`

**Log Rotation Not Working:**
1. Verify logrotate configuration
2. Check file permissions on log directory
3. Ensure sufficient disk space for rotation

**Missing Log Entries:**
1. Check log level configuration
2. Verify logger initialization
3. Check for exceptions in logging setup

**High Disk Usage:**
1. Review retention policies
2. Check for log rotation issues
3. Consider log compression
4. Monitor log volume growth

### Debug Commands

```bash
# Check log file sizes
du -sh logs/*/*.log

# Monitor log file creation in real-time
watch -n 1 'ls -la logs/app/'

# Test log configuration
python -c "import logging; logging.getLogger('ncs').info('Test message')"

# Validate JSON log format
tail -1 logs/app/ncs-api.log | jq .

# Check log permissions
find logs/ -type f -exec ls -la {} \;
```

## 📈 Best Practices

### Development
- Use DEBUG level for detailed troubleshooting
- Include context information in log messages
- Test log output in development environment
- Regularly review and clean up debug statements

### Production
- Use INFO level or higher in production
- Implement log aggregation and centralized monitoring
- Set up automated alerts for critical errors
- Regular log analysis for performance optimization

### Security
- Never log sensitive information (passwords, tokens, PII)
- Implement log access controls and audit trails
- Use structured logging for better security analysis
- Regular security log reviews and incident response

---

## 📞 Support

For logging-related issues:
1. Check this documentation first
2. Review the [Troubleshooting Guide](../docs/TROUBLESHOOTING.md)
3. Check existing GitHub issues
4. Create a new issue with log samples and configuration details

**Note:** When sharing logs for support, always remove sensitive information and use log samples from development/test environments when possible.