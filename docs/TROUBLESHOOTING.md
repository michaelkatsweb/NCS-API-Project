# NeuroCluster Streamer API Troubleshooting Guide

Comprehensive troubleshooting documentation for diagnosing and resolving common issues with the NCS API, from development to production environments.

## 📋 Table of Contents

- [Quick Diagnostic Checklist](#-quick-diagnostic-checklist)
- [Common Issues & Solutions](#-common-issues--solutions)
- [Application Issues](#-application-issues)
- [Performance Problems](#-performance-problems)
- [Database Issues](#-database-issues)
- [Authentication & Security](#-authentication--security)
- [Deployment Issues](#-deployment-issues)
- [Monitoring & Debugging](#-monitoring--debugging)
- [Log Analysis](#-log-analysis)
- [Performance Debugging](#-performance-debugging)

## 🔍 Quick Diagnostic Checklist

Before diving into specific issues, run through this quick checklist:

### Health Check Commands
```bash
# 1. Basic connectivity test
curl -f http://localhost:8000/health
echo "Exit code: $?"

# 2. Check API with authentication
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/algorithm_status

# 3. Test database connectivity
curl http://localhost:8000/health | jq '.components.database'

# 4. Check service status (Docker)
docker ps | grep ncs-api
docker logs ncs-api --tail 50

# 5. Check service status (Kubernetes)
kubectl get pods -n ncs-api
kubectl logs deployment/ncs-api -n ncs-api --tail 50

# 6. Resource usage check
docker stats ncs-api --no-stream
# or
kubectl top pods -n ncs-api
```

### Quick Status Overview
```bash
#!/bin/bash
# Quick system status check script

echo "=== NCS API System Status ==="
echo "Date: $(date)"
echo ""

# API Health
echo "🏥 API Health:"
if curl -sf http://localhost:8000/health > /dev/null; then
    echo "  ✅ API is responding"
    curl -s http://localhost:8000/health | jq '.status, .components'
else
    echo "  ❌ API is not responding"
fi
echo ""

# Database Status
echo "🗄️ Database Status:"
if pg_isready -h ${POSTGRES_HOST:-localhost} -p ${POSTGRES_PORT:-5432} > /dev/null 2>&1; then
    echo "  ✅ PostgreSQL is reachable"
else
    echo "  ❌ PostgreSQL connection failed"
fi

# Redis Status
echo "🔄 Cache Status:"
if redis-cli -h ${REDIS_HOST:-localhost} -p ${REDIS_PORT:-6379} ping > /dev/null 2>&1; then
    echo "  ✅ Redis is reachable"
else
    echo "  ❌ Redis connection failed"
fi

# Memory and CPU
echo "💻 Resource Usage:"
if command -v free > /dev/null; then
    echo "  Memory: $(free -m | awk 'NR==2{printf "%.1f%% (%s/%s MB)\n", $3*100/$2, $3, $2}')"
fi
if command -v top > /dev/null; then
    echo "  CPU Load: $(uptime | awk -F'load average:' '{print $2}')"
fi
```

## ❗ Common Issues & Solutions

### Issue: API Returns 503 Service Unavailable

**Symptoms:**
```
HTTP/1.1 503 Service Unavailable
{
  "detail": "Service temporarily unavailable"
}
```

**Common Causes & Solutions:**

1. **Database Connection Failed**
   ```bash
   # Check database connectivity
   pg_isready -h $POSTGRES_HOST -p $POSTGRES_PORT
   
   # Test connection with credentials
   psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB -c "\l"
   
   # Solution: Fix database configuration
   export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
   ```

2. **Redis Connection Failed**
   ```bash
   # Check Redis connectivity
   redis-cli -h $REDIS_HOST -p $REDIS_PORT ping
   
   # Solution: Fix Redis configuration
   export REDIS_URL="redis://host:6379/0"
   ```

3. **Resource Exhaustion**
   ```bash
   # Check memory usage
   free -m
   docker stats --no-stream
   
   # Check disk space
   df -h
   
   # Solution: Increase resources or optimize configuration
   ```

### Issue: Authentication Errors (401 Unauthorized)

**Symptoms:**
```json
{
  "detail": "Authentication failed",
  "error_code": "AUTHENTICATION_ERROR"
}
```

**Solutions:**

1. **JWT Token Issues**
   ```bash
   # Verify token is not expired
   python -c "
   import jwt
   token = '$YOUR_TOKEN'
   decoded = jwt.decode(token, options={'verify_signature': False})
   import datetime
   exp = datetime.datetime.fromtimestamp(decoded['exp'])
   print(f'Token expires: {exp}')
   print(f'Current time: {datetime.datetime.now()}')
   "
   
   # Get new token
   curl -X POST "http://localhost:8000/auth/login" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "username=admin&password=admin123"
   ```

2. **API Key Issues**
   ```bash
   # Verify API key format
   echo $API_KEY | wc -c  # Should be 32+ characters
   
   # Check API key in logs
   grep "API key" logs/app.log
   
   # Test with curl
   curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/algorithm_status
   ```

3. **Invalid Credentials**
   ```bash
   # Reset admin password (development)
   python scripts/reset_password.py --username admin --password new_password
   
   # Check user exists in database
   psql $DATABASE_URL -c "SELECT username, is_active FROM users WHERE username='admin';"
   ```

### Issue: Rate Limiting (429 Too Many Requests)

**Symptoms:**
```json
{
  "detail": "Rate limit exceeded",
  "retry_after": 60
}
```

**Solutions:**

1. **Adjust Rate Limits**
   ```bash
   # Check current rate limit configuration
   grep RATE_LIMIT .env
   
   # Increase limits temporarily
   export RATE_LIMIT_PER_MINUTE=2000
   
   # Or use different authentication method
   # API keys typically have higher limits than JWT tokens
   ```

2. **Implement Request Batching**
   ```python
   # Instead of individual requests
   for point in points:
       response = client.process_points([point])
   
   # Batch requests
   batch_size = 100
   for i in range(0, len(points), batch_size):
       batch = points[i:i + batch_size]
       response = client.process_points(batch)
   ```

### Issue: Validation Errors (422 Unprocessable Entity)

**Symptoms:**
```json
{
  "detail": [
    {
      "loc": ["body", "points", 0],
      "msg": "Point must contain at least 1 dimension",
      "type": "value_error"
    }
  ]
}
```

**Common Validation Issues:**

1. **Invalid Data Point Format**
   ```python
   # ❌ Incorrect formats
   {"points": [1, 2, 3]}           # Should be array of arrays
   {"points": [["a", "b", "c"]]}   # Should be numbers
   {"points": [[]]}                # Empty points not allowed
   
   # ✅ Correct format
   {"points": [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]}
   ```

2. **Missing Required Fields**
   ```python
   # ❌ Missing points field
   {}
   
   # ✅ Include required fields
   {"points": [[1.0, 2.0, 3.0]]}
   ```

3. **Out of Range Values**
   ```python
   # ❌ Values too large/small
   {"points": [[1e20, 2e20, 3e20]]}
   
   # ✅ Reasonable value ranges
   {"points": [[1.0, 2.0, 3.0]]}
   ```

## 🚀 Application Issues

### Application Won't Start

**Diagnostic Steps:**

1. **Check Environment Variables**
   ```bash
   # Verify required environment variables
   echo "SECRET_KEY: ${SECRET_KEY:0:8}..."
   echo "DATABASE_URL: $DATABASE_URL"
   echo "ENVIRONMENT: $ENVIRONMENT"
   
   # Load environment file
   source .env
   env | grep -E "^(SECRET_KEY|DATABASE|REDIS|NCS_)" | head -10
   ```

2. **Check Python Dependencies**
   ```bash
   # Verify Python version
   python --version  # Should be 3.11+
   
   # Check package installation
   pip list | grep -E "(fastapi|pydantic|sqlalchemy)"
   
   # Reinstall if needed
   pip install -r requirements.txt --force-reinstall
   ```

3. **Check Port Availability**
   ```bash
   # Check if port is in use
   netstat -tlnp | grep :8000
   lsof -i :8000
   
   # Use different port
   export PORT=8001
   uvicorn main_secure:app --port $PORT
   ```

### Algorithm Performance Issues

**Symptoms:**
- High processing latency
- Memory usage growing over time
- Poor clustering quality

**Diagnostic & Solutions:**

1. **Check Algorithm Parameters**
   ```bash
   # View current algorithm configuration
   curl -H "Authorization: Bearer $TOKEN" \
        http://localhost:8000/api/v1/algorithm_status | jq
   
   # Adjust parameters
   export NCS_BASE_THRESHOLD=0.65  # Lower = more clusters
   export NCS_LEARNING_RATE=0.08   # Higher = faster adaptation
   export NCS_MAX_CLUSTERS=50      # Allow more clusters
   ```

2. **Memory Leak Detection**
   ```python
   # Monitor memory usage over time
   import psutil
   import time
   
   process = psutil.Process()
   
   for i in range(100):
       # Make API call
       response = client.process_points(test_points)
       
       # Check memory
       memory_mb = process.memory_info().rss / 1024 / 1024
       print(f"Iteration {i}: {memory_mb:.1f}MB")
       
       if memory_mb > 100:  # Alert if over 100MB
           print("⚠️ High memory usage detected")
       
       time.sleep(1)
   ```

3. **Performance Profiling**
   ```python
   # Enable profiling
   import cProfile
   import io
   import pstats
   
   pr = cProfile.Profile()
   pr.enable()
   
   # Process points
   result = client.process_points(large_dataset)
   
   pr.disable()
   s = io.StringIO()
   ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
   ps.print_stats()
   print(s.getvalue())
   ```

### Import Errors

**Common Import Issues:**

1. **Missing Dependencies**
   ```bash
   # Install missing packages
   pip install numpy scipy scikit-learn
   
   # For development dependencies
   pip install -r requirements-dev.txt
   ```

2. **Python Path Issues**
   ```bash
   # Add current directory to Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   
   # Or run from project root
   cd /path/to/ncs-api
   python -m main_secure
   ```

3. **Virtual Environment Issues**
   ```bash
   # Recreate virtual environment
   rm -rf venv
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## ⚡ Performance Problems

### High Latency Issues

**Diagnostic Tools:**

1. **Request Timing Analysis**
   ```bash
   # Use curl with timing
   curl -w "@curl-format.txt" -H "Authorization: Bearer $TOKEN" \
        -X POST http://localhost:8000/api/v1/process_points \
        -d '{"points": [[1,2,3]]}' -H "Content-Type: application/json"
   
   # curl-format.txt content:
   cat > curl-format.txt << 'EOF'
        time_namelookup:  %{time_namelookup}\n
           time_connect:  %{time_connect}\n
        time_appconnect:  %{time_appconnect}\n
       time_pretransfer:  %{time_pretransfer}\n
          time_redirect:  %{time_redirect}\n
     time_starttransfer:  %{time_starttransfer}\n
                        ----------\n
             time_total:  %{time_total}\n
   EOF
   ```

2. **Database Query Performance**
   ```sql
   -- Enable query logging
   ALTER SYSTEM SET log_statement = 'all';
   ALTER SYSTEM SET log_min_duration_statement = 100;  -- Log queries > 100ms
   SELECT pg_reload_conf();
   
   -- Check slow queries
   SELECT query, mean_time, calls, total_time 
   FROM pg_stat_statements 
   ORDER BY mean_time DESC 
   LIMIT 10;
   
   -- Analyze specific query
   EXPLAIN ANALYZE SELECT * FROM clusters WHERE created_at > NOW() - INTERVAL '1 hour';
   ```

3. **Application Profiling**
   ```python
   # Simple timing decorator
   import time
   import functools
   
   def timing(func):
       @functools.wraps(func)
       def wrapper(*args, **kwargs):
           start = time.time()
           result = func(*args, **kwargs)
           end = time.time()
           print(f"{func.__name__} took {end - start:.3f} seconds")
           return result
       return wrapper
   
   # Use on critical functions
   @timing
   def process_data_points(points):
       # Your processing logic
       pass
   ```

### Memory Issues

**Memory Leak Detection:**

1. **Python Memory Profiling**
   ```bash
   # Install memory profiler
   pip install memory-profiler psutil
   
   # Run with memory monitoring
   python -m memory_profiler main_secure.py
   ```

2. **Container Memory Monitoring**
   ```bash
   # Docker memory usage
   docker stats ncs-api --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
   
   # Kubernetes memory usage
   kubectl top pods -n ncs-api --sort-by=memory
   
   # Set memory alerts
   kubectl get hpa -n ncs-api
   ```

3. **Memory Optimization**
   ```python
   # Clear caches periodically
   import gc
   import sys
   
   def cleanup_memory():
       """Force garbage collection and clear caches."""
       gc.collect()
       
       # Clear algorithm caches if implemented
       if hasattr(algorithm, 'clear_cache'):
           algorithm.clear_cache()
       
       print(f"Memory usage: {sys.getsizeof(gc.get_objects())} bytes")
   
   # Call periodically
   cleanup_memory()
   ```

### Throughput Bottlenecks

**Optimization Strategies:**

1. **Worker Process Scaling**
   ```bash
   # Increase worker processes
   export WORKERS=8  # 2x CPU cores
   
   # Or use auto-scaling
   gunicorn main_secure:app \
     --workers 4 \
     --worker-class uvicorn.workers.UvicornWorker \
     --max-requests 10000 \
     --max-requests-jitter 1000 \
     --preload-app
   ```

2. **Database Connection Pooling**
   ```python
   # Optimize connection pool
   DATABASE_CONFIG = {
       "pool_size": 20,
       "max_overflow": 40,
       "pool_timeout": 30,
       "pool_recycle": 3600,
       "pool_pre_ping": True
   }
   ```

3. **Redis Optimization**
   ```bash
   # Redis configuration tuning
   redis-cli CONFIG SET maxmemory-policy allkeys-lru
   redis-cli CONFIG SET timeout 300
   redis-cli CONFIG SET tcp-keepalive 60
   ```

## 🗄️ Database Issues

### Connection Problems

**Diagnostic Steps:**

1. **Basic Connectivity**
   ```bash
   # Test database connection
   pg_isready -h $POSTGRES_HOST -p $POSTGRES_PORT -d $POSTGRES_DB
   
   # Test with credentials
   PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT 1;"
   
   # Check connection limits
   PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB -c "SHOW max_connections;"
   ```

2. **Connection Pool Issues**
   ```python
   # Check active connections
   import sqlalchemy as sa
   
   engine = sa.create_engine(DATABASE_URL)
   with engine.connect() as conn:
       result = conn.execute(sa.text("""
           SELECT count(*) as active_connections 
           FROM pg_stat_activity 
           WHERE datname = current_database()
       """))
       print(f"Active connections: {result.fetchone()[0]}")
   ```

3. **Firewall/Network Issues**
   ```bash
   # Test network connectivity
   telnet $POSTGRES_HOST $POSTGRES_PORT
   
   # Check DNS resolution
   nslookup $POSTGRES_HOST
   
   # Test from container
   docker exec ncs-api pg_isready -h $POSTGRES_HOST -p $POSTGRES_PORT
   ```

### Migration Issues

**Common Migration Problems:**

1. **Failed Migrations**
   ```bash
   # Check migration status
   python database/migrate.py --status
   
   # Force migration
   python database/migrate.py --force
   
   # Rollback if needed
   python database/migrate.py --rollback
   ```

2. **Schema Conflicts**
   ```sql
   -- Check existing schema
   \d+ clusters
   \d+ users
   
   -- Fix schema conflicts
   DROP TABLE IF EXISTS problematic_table CASCADE;
   
   -- Recreate from migration
   python database/migrate.py --recreate
   ```

### Performance Issues

1. **Slow Queries**
   ```sql
   -- Find slow queries
   SELECT query, mean_time, calls, total_time, rows, 100.0 * shared_blks_hit /
          nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
   FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 5;
   
   -- Create missing indexes
   CREATE INDEX CONCURRENTLY idx_clusters_created_at ON clusters(created_at);
   CREATE INDEX CONCURRENTLY idx_points_cluster_id ON points(cluster_id);
   ```

2. **Lock Issues**
   ```sql
   -- Check for locks
   SELECT blocked_locks.pid AS blocked_pid,
          blocking_locks.pid AS blocking_pid,
          blocked_activity.usename AS blocked_user,
          blocking_activity.usename AS blocking_user,
          blocked_activity.query AS blocked_statement,
          blocking_activity.query AS current_statement_in_blocking_process
   FROM pg_catalog.pg_locks blocked_locks
   JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
   JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
   AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
   WHERE NOT blocked_locks.granted;
   ```

## 🔐 Authentication & Security

### JWT Token Issues

**Token Validation Problems:**

1. **Expired Tokens**
   ```python
   # Check token expiration
   import jwt
   from datetime import datetime
   
   try:
       decoded = jwt.decode(token, options={"verify_signature": False})
       exp_time = datetime.fromtimestamp(decoded['exp'])
       print(f"Token expires: {exp_time}")
       print(f"Time remaining: {exp_time - datetime.now()}")
   except jwt.InvalidTokenError as e:
       print(f"Token error: {e}")
   ```

2. **Invalid Secret Key**
   ```bash
   # Verify secret key length
   echo $SECRET_KEY | wc -c  # Should be 32+ characters
   
   # Generate new secret key
   openssl rand -base64 32
   
   # Update configuration
   export SECRET_KEY="$(openssl rand -base64 32)"
   ```

3. **Token Format Issues**
   ```bash
   # Check token format
   echo $TOKEN | cut -d. -f1 | base64 -d  # Header
   echo $TOKEN | cut -d. -f2 | base64 -d  # Payload
   
   # Validate token structure
   python -c "
   import jwt
   token = '$TOKEN'
   header = jwt.get_unverified_header(token)
   print(f'Algorithm: {header.get(\"alg\")}')
   print(f'Type: {header.get(\"typ\")}')
   "
   ```

### API Key Issues

1. **Key Format Validation**
   ```bash
   # Check API key format
   echo "API Key length: $(echo $API_KEY | wc -c)"
   echo "API Key prefix: $(echo $API_KEY | cut -c1-4)"
   
   # Generate new API key
   python scripts/generate_api_key.py
   ```

2. **Permission Issues**
   ```sql
   -- Check API key permissions
   SELECT api_key_id, permissions, is_active, expires_at 
   FROM api_keys 
   WHERE key_hash = encode(sha256($API_KEY::bytea), 'hex');
   ```

### CORS Issues

**Cross-Origin Problems:**

1. **Origin Not Allowed**
   ```bash
   # Check CORS configuration
   echo $ALLOWED_ORIGINS
   
   # Test CORS
   curl -H "Origin: https://example.com" \
        -H "Access-Control-Request-Method: POST" \
        -H "Access-Control-Request-Headers: Authorization" \
        -X OPTIONS http://localhost:8000/api/v1/process_points
   ```

2. **Preflight Issues**
   ```javascript
   // Debug CORS in browser
   fetch('http://localhost:8000/api/v1/process_points', {
     method: 'OPTIONS',
     headers: {
       'Origin': 'https://example.com',
       'Access-Control-Request-Method': 'POST',
       'Access-Control-Request-Headers': 'Authorization,Content-Type'
     }
   }).then(response => {
     console.log('CORS preflight response:', response.headers);
   });
   ```

## 🚢 Deployment Issues

### Docker Issues

**Container Won't Start:**

1. **Build Issues**
   ```bash
   # Check build logs
   docker build -t ncs-api . --no-cache
   
   # Debug build step by step
   docker build -t ncs-api . --progress=plain
   
   # Check final image
   docker run -it ncs-api /bin/bash
   ```

2. **Runtime Issues**
   ```bash
   # Check container logs
   docker logs ncs-api --timestamps
   
   # Debug container environment
   docker exec ncs-api env | grep -E "(SECRET|DATABASE|REDIS)"
   
   # Check file permissions
   docker exec ncs-api ls -la /app
   ```

3. **Network Issues**
   ```bash
   # Check container network
   docker network ls
   docker inspect bridge
   
   # Test connectivity between containers
   docker exec ncs-api ping postgres
   docker exec ncs-api nc -zv redis 6379
   ```

### Kubernetes Issues

**Pod Problems:**

1. **Pod Won't Start**
   ```bash
   # Check pod status
   kubectl get pods -n ncs-api -o wide
   kubectl describe pod ncs-api-xxx -n ncs-api
   
   # Check events
   kubectl get events -n ncs-api --sort-by='.lastTimestamp'
   
   # Check resource limits
   kubectl top pods -n ncs-api
   ```

2. **Configuration Issues**
   ```bash
   # Check config maps
   kubectl get configmap -n ncs-api
   kubectl describe configmap ncs-api-config -n ncs-api
   
   # Check secrets
   kubectl get secrets -n ncs-api
   kubectl describe secret ncs-api-secrets -n ncs-api
   ```

3. **Service Issues**
   ```bash
   # Check service connectivity
   kubectl get services -n ncs-api
   kubectl describe service ncs-api-service -n ncs-api
   
   # Test internal connectivity
   kubectl run debug --image=busybox -it --rm -- /bin/sh
   # Inside the pod:
   nslookup ncs-api-service.ncs-api.svc.cluster.local
   ```

## 📊 Monitoring & Debugging

### Log Analysis

**Structured Log Analysis:**

1. **Find Error Patterns**
   ```bash
   # Search for errors in logs
   grep -E "(ERROR|CRITICAL)" logs/app.log | tail -20
   
   # Count error types
   grep "ERROR" logs/app.log | awk '{print $4}' | sort | uniq -c | sort -nr
   
   # Find authentication failures
   grep "authentication_failed" logs/app.log | jq '.timestamp,.user_id,.source_ip'
   ```

2. **Performance Log Analysis**
   ```bash
   # Find slow requests
   grep "processing_time_ms" logs/app.log | jq '. | select(.processing_time_ms > 1000)'
   
   # Average response times
   grep "request_completed" logs/app.log | jq '.response_time_ms' | \
   awk '{sum+=$1; count++} END {print "Average:", sum/count "ms"}'
   
   # Request rate by endpoint
   grep "request_started" logs/app.log | jq -r '.endpoint' | sort | uniq -c | sort -nr
   ```

3. **Security Log Analysis**
   ```bash
   # Failed login attempts by IP
   grep "login_failed" logs/security.log | jq -r '.source_ip' | sort | uniq -c | sort -nr
   
   # Rate limit violations
   grep "rate_limit_exceeded" logs/app.log | jq '.source_ip,.endpoint,.violation_count'
   
   # Suspicious activity
   grep "suspicious_activity" logs/security.log | jq '.event_type,.details'
   ```

### Real-time Monitoring

**Live Monitoring Commands:**

1. **Real-time Logs**
   ```bash
   # Follow application logs
   tail -f logs/app.log | jq 'select(.level == "ERROR")'
   
   # Docker logs
   docker logs -f ncs-api | grep ERROR
   
   # Kubernetes logs
   kubectl logs -f deployment/ncs-api -n ncs-api | grep ERROR
   ```

2. **Resource Monitoring**
   ```bash
   # Watch resource usage
   watch -n 2 'docker stats ncs-api --no-stream'
   
   # Kubernetes resource watch
   watch -n 2 'kubectl top pods -n ncs-api'
   
   # System resources
   watch -n 2 'free -m && echo "---" && df -h | head -5'
   ```

3. **API Monitoring**
   ```bash
   # Continuous health checks
   watch -n 5 'curl -s http://localhost:8000/health | jq ".status,.components"'
   
   # Monitor response times
   while true; do
     time curl -s http://localhost:8000/health > /dev/null
     sleep 5
   done
   ```

## 🔧 Performance Debugging

### Profiling Tools

**Application Profiling:**

1. **Python Profiling**
   ```python
   # CPU profiling
   import cProfile
   import pstats
   import io
   
   pr = cProfile.Profile()
   pr.enable()
   
   # Your code here
   process_large_dataset()
   
   pr.disable()
   s = io.StringIO()
   ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
   ps.print_stats()
   
   with open('profile_results.txt', 'w') as f:
       f.write(s.getvalue())
   ```

2. **Memory Profiling**
   ```python
   # Memory line-by-line profiling
   from memory_profiler import profile
   
   @profile
   def memory_intensive_function():
       # Your code here
       pass
   
   # Run with: python -m memory_profiler your_script.py
   ```

3. **Request Profiling**
   ```python
   # Custom middleware for request profiling
   import time
   import psutil
   from fastapi import Request
   
   @app.middleware("http")
   async def profile_requests(request: Request, call_next):
       start_time = time.time()
       start_memory = psutil.Process().memory_info().rss
       
       response = await call_next(request)
       
       end_time = time.time()
       end_memory = psutil.Process().memory_info().rss
       
       logger.info(
           "request_profile",
           path=request.url.path,
           method=request.method,
           duration_ms=(end_time - start_time) * 1000,
           memory_delta_mb=(end_memory - start_memory) / 1024 / 1024
       )
       
       return response
   ```

### Load Testing

**Stress Testing:**

1. **Basic Load Test**
   ```bash
   # Apache Bench
   ab -n 1000 -c 10 -H "Authorization: Bearer $TOKEN" http://localhost:8000/health
   
   # wrk load testing
   wrk -t12 -c400 -d30s --header "Authorization: Bearer $TOKEN" http://localhost:8000/health
   ```

2. **Advanced Load Testing**
   ```python
   # Using locust for complex scenarios
   from locust import HttpUser, task, between
   
   class NCSAPIUser(HttpUser):
       wait_time = between(1, 3)
       
       def on_start(self):
           # Login to get token
           response = self.client.post("/auth/login", data={
               "username": "test_user",
               "password": "test_password"
           })
           self.token = response.json()["access_token"]
           self.headers = {"Authorization": f"Bearer {self.token}"}
       
       @task(3)
       def process_points(self):
           data = {"points": [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]}
           self.client.post("/api/v1/process_points", json=data, headers=self.headers)
       
       @task(1)
       def get_status(self):
           self.client.get("/api/v1/algorithm_status", headers=self.headers)
   
   # Run with: locust -f load_test.py --host=http://localhost:8000
   ```

### Database Performance

**Query Optimization:**

1. **Query Analysis**
   ```sql
   -- Enable query statistics
   CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
   
   -- Reset statistics
   SELECT pg_stat_statements_reset();
   
   -- After running workload, check slow queries
   SELECT 
       query,
       calls,
       total_time,
       mean_time,
       rows,
       100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
   FROM pg_stat_statements 
   ORDER BY total_time DESC 
   LIMIT 10;
   ```

2. **Index Optimization**
   ```sql
   -- Find missing indexes
   SELECT 
       schemaname,
       tablename,
       attname,
       n_distinct,
       correlation
   FROM pg_stats
   WHERE tablename IN ('clusters', 'points', 'users')
   ORDER BY n_distinct DESC;
   
   -- Check index usage
   SELECT 
       indexrelname AS index_name,
       idx_scan AS index_scans,
       idx_tup_read AS tuples_read,
       idx_tup_fetch AS tuples_fetched
   FROM pg_stat_user_indexes
   ORDER BY idx_scan DESC;
   ```

---

## 📞 Getting Help

### Support Channels

- **Documentation**: Complete guides at [docs/](../docs/)
- **GitHub Issues**: [Bug reports and feature requests](https://github.com/your-org/ncs-api/issues)
- **GitHub Discussions**: [Community support](https://github.com/your-org/ncs-api/discussions)
- **Security Issues**: security@yourdomain.com

### Before Seeking Help

1. **Check the logs** for error messages
2. **Verify configuration** matches requirements
3. **Test basic connectivity** (health checks)
4. **Review recent changes** that might have caused issues
5. **Search existing issues** on GitHub

### Creating Bug Reports

Include the following information:

```markdown
## Environment
- NCS API Version: 
- Deployment Method: (Docker/Kubernetes/Local)
- OS: 
- Python Version: 

## Issue Description
Brief description of the problem

## Steps to Reproduce
1. 
2. 
3. 

## Expected Behavior
What should happen

## Actual Behavior
What actually happened

## Logs
```
# Relevant log entries
```

## Configuration
```bash
# Relevant environment variables (redact secrets)
```
```

### Emergency Procedures

For critical production issues:

1. **Immediate Response**: Check health endpoints and restart if necessary
2. **Escalation**: Contact on-call engineer (+1-555-ONCALL)
3. **Rollback**: Use deployment rollback procedures if recent change
4. **Communication**: Update status page and notify stakeholders
5. **Investigation**: Preserve logs and system state for post-mortem

Remember: **When in doubt, restart the service and investigate after ensuring stability.**