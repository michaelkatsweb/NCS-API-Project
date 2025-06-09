# NeuroCluster Streamer API Deployment Guide

Comprehensive guide for deploying the NeuroCluster Streamer API across different environments and platforms, from development to production-scale deployments.

## 📋 Table of Contents

- [Prerequisites](#-prerequisites)
- [Quick Start Options](#-quick-start-options)
- [Docker Deployment](#-docker-deployment)
- [Kubernetes Deployment](#-kubernetes-deployment)
- [Cloud Platform Deployment](#-cloud-platform-deployment)
- [Traditional Server Deployment](#-traditional-server-deployment)
- [Configuration Management](#-configuration-management)
- [Monitoring & Observability](#-monitoring--observability)
- [Scaling & Performance](#-scaling--performance)
- [Troubleshooting](#-troubleshooting)

## 🔧 Prerequisites

### System Requirements

#### Minimum Requirements (Development)
- **CPU**: 2 cores, 2.0 GHz
- **Memory**: 4 GB RAM
- **Storage**: 10 GB available space
- **Network**: 100 Mbps

#### Recommended Requirements (Production)
- **CPU**: 8+ cores, 3.0+ GHz
- **Memory**: 16+ GB RAM
- **Storage**: 100+ GB SSD with IOPS > 3000
- **Network**: 1+ Gbps with low latency

### Software Dependencies

#### Core Dependencies
- **Python 3.11+** (for development and traditional deployment)
- **Docker 24.0+** and **Docker Compose 2.0+**
- **Kubernetes 1.24+** (for K8s deployment)
- **PostgreSQL 13+** (database)
- **Redis 6+** (caching and session management)

#### Development Tools
- **Git 2.30+**
- **Make** (GNU Make 4.0+)
- **curl** (for testing)
- **jq** (for JSON processing)
- **OpenSSL** (for certificate generation)

### Platform Compatibility

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux (Ubuntu 20.04+)** | ✅ Fully Supported | Recommended for production |
| **Linux (RHEL 8+/CentOS 8+)** | ✅ Fully Supported | Enterprise deployments |
| **macOS 11+** | ✅ Development Only | Local development |
| **Windows 11 + WSL2** | ⚠️ Development Only | Limited support |
| **Docker Desktop** | ✅ Cross-platform | All platforms |

## 🚀 Quick Start Options

### Option 1: Docker Compose (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/your-org/ncs-api.git
cd ncs-api

# Copy environment template
cp .env.example .env

# Generate secrets
./scripts/generate_secrets.py

# Start all services
docker-compose up -d

# Verify deployment
curl http://localhost:8000/health
```

**Access Points:**
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090

### Option 2: Kubernetes (Production Ready)

```bash
# Apply all manifests
kubectl apply -f k8s/

# Wait for deployment
kubectl wait --for=condition=available --timeout=300s deployment/ncs-api -n ncs-api

# Check status
kubectl get pods -n ncs-api

# Get external IP (if using LoadBalancer)
kubectl get service ncs-api-service -n ncs-api
```

### Option 3: One-Click Cloud Deployment

#### AWS (ECS + Fargate)
```bash
# Deploy with automated script
./scripts/deploy.sh aws production

# Or use CloudFormation
aws cloudformation create-stack \
  --stack-name ncs-api-production \
  --template-body file://aws/cloudformation.yml \
  --capabilities CAPABILITY_IAM
```

#### Google Cloud Platform
```bash
# Deploy to Cloud Run
./scripts/deploy.sh gcp production

# Or use gcloud directly
gcloud run deploy ncs-api \
  --image gcr.io/your-project/ncs-api:latest \
  --platform managed \
  --region us-central1
```

## 🐳 Docker Deployment

### Development Environment

#### Quick Setup
```bash
# Start development environment
docker-compose up -d

# View logs
docker-compose logs -f ncs-api

# Stop environment
docker-compose down
```

#### Custom Configuration
```bash
# Use custom environment file
docker-compose --env-file .env.custom up -d

# Override specific services
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

### Production Environment

#### Production Compose File
```bash
# Deploy with production overrides
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale API service
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale ncs-api=3
```

#### Production Configuration Example
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  ncs-api:
    image: your-registry.com/ncs-api:v1.0.0
    environment:
      ENVIRONMENT: production
      WORKERS: 8
      LOG_LEVEL: WARNING
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Building Custom Images

#### Multi-Stage Production Build
```dockerfile
# docker/Dockerfile
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash ncs
WORKDIR /app

# Copy dependencies
COPY --from=builder /root/.local /home/ncs/.local
COPY --chown=ncs:ncs . .

USER ncs
ENV PATH=/home/ncs/.local/bin:$PATH

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "main_secure.py"]
```

#### Build and Push
```bash
# Build production image
docker build -f docker/Dockerfile -t ncs-api:latest .

# Tag for registry
docker tag ncs-api:latest your-registry.com/ncs-api:v1.0.0

# Push to registry
docker push your-registry.com/ncs-api:v1.0.0
```

## ☸️ Kubernetes Deployment

### Prerequisites

#### Cluster Requirements
- **Kubernetes Version**: 1.24+
- **Node Resources**: 8 vCPU, 16 GB RAM minimum per node
- **Storage Classes**: Support for persistent volumes
- **Ingress Controller**: NGINX, Traefik, or cloud provider
- **Cert Manager**: For TLS certificate management

#### Required RBAC Permissions
```yaml
# Required for deployment
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: ncs-api-deployer
rules:
- apiGroups: ["apps", ""]
  resources: ["deployments", "services", "configmaps", "secrets"]
  verbs: ["create", "get", "list", "update", "patch", "delete"]
- apiGroups: ["networking.k8s.io"]
  resources: ["ingresses"]
  verbs: ["create", "get", "list", "update", "patch", "delete"]
```

### Step-by-Step Deployment

#### 1. Namespace and Configuration
```bash
# Create namespace
kubectl create namespace ncs-api

# Apply configuration
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
```

#### 2. Secrets Management
```bash
# Create secrets (DO NOT use default values in production)
kubectl create secret generic ncs-api-secrets \
  --from-literal=secret-key="$(openssl rand -base64 32)" \
  --from-literal=postgres-password="$(openssl rand -base64 16)" \
  --from-literal=redis-password="$(openssl rand -base64 16)" \
  -n ncs-api

# Or apply from template
envsubst < k8s/secrets.yaml | kubectl apply -f -
```

#### 3. Database and Cache
```bash
# Deploy PostgreSQL
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/postgres-service.yaml

# Deploy Redis
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/redis-service.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n ncs-api --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n ncs-api --timeout=300s
```

#### 4. Application Deployment
```bash
# Deploy API service
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Wait for deployment
kubectl wait --for=condition=available deployment/ncs-api -n ncs-api --timeout=300s

# Check pods
kubectl get pods -n ncs-api
```

#### 5. Ingress and TLS
```bash
# Deploy ingress
kubectl apply -f k8s/ingress.yaml

# Check ingress status
kubectl get ingress -n ncs-api

# Verify TLS certificate (if using cert-manager)
kubectl describe certificate ncs-api-tls -n ncs-api
```

### Production Kubernetes Configuration

#### Deployment Manifest
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ncs-api
  namespace: ncs-api
  labels:
    app: ncs-api
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: ncs-api
  template:
    metadata:
      labels:
        app: ncs-api
        version: v1.0.0
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: ncs-api
        image: your-registry.com/ncs-api:v1.0.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: ncs-api-secrets
              key: secret-key
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ncs-api-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
```

#### Horizontal Pod Autoscaler
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ncs-api-hpa
  namespace: ncs-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ncs-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
```

### Kubernetes Monitoring

#### Resource Monitoring
```bash
# Check resource usage
kubectl top pods -n ncs-api
kubectl top nodes

# Check HPA status
kubectl get hpa -n ncs-api

# View pod logs
kubectl logs -l app=ncs-api -n ncs-api --tail=100 -f

# Check events
kubectl get events -n ncs-api --sort-by='.lastTimestamp'
```

## ☁️ Cloud Platform Deployment

### Amazon Web Services (AWS)

#### ECS Fargate Deployment
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name ncs-api-production

# Register task definition
aws ecs register-task-definition --cli-input-json file://aws/task-definition.json

# Create service
aws ecs create-service \
  --cluster ncs-api-production \
  --service-name ncs-api \
  --task-definition ncs-api:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration file://aws/network-config.json
```

#### EKS Deployment
```bash
# Create EKS cluster
eksctl create cluster --name ncs-api-cluster --version 1.24

# Apply manifests
kubectl apply -f k8s/

# Configure ALB Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.4.4/docs/install/iam_policy.json
```

### Google Cloud Platform (GCP)

#### Cloud Run Deployment
```bash
# Build and push image
gcloud builds submit --tag gcr.io/your-project/ncs-api:latest

# Deploy to Cloud Run
gcloud run deploy ncs-api \
  --image gcr.io/your-project/ncs-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 1Gi \
  --cpu 2 \
  --max-instances 10 \
  --set-env-vars ENVIRONMENT=production
```

#### GKE Deployment
```bash
# Create GKE cluster
gcloud container clusters create ncs-api-cluster \
  --zone us-central1-a \
  --machine-type e2-standard-4 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10

# Get credentials
gcloud container clusters get-credentials ncs-api-cluster --zone us-central1-a

# Deploy application
kubectl apply -f k8s/
```

### Microsoft Azure

#### Container Instances
```bash
# Create resource group
az group create --name ncs-api-rg --location eastus

# Deploy container instance
az container create \
  --resource-group ncs-api-rg \
  --name ncs-api \
  --image your-registry.azurecr.io/ncs-api:latest \
  --cpu 2 \
  --memory 4 \
  --port 8000 \
  --environment-variables ENVIRONMENT=production
```

#### AKS Deployment
```bash
# Create AKS cluster
az aks create \
  --resource-group ncs-api-rg \
  --name ncs-api-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10

# Get credentials
az aks get-credentials --resource-group ncs-api-rg --name ncs-api-cluster

# Deploy application
kubectl apply -f k8s/
```

## 🖥️ Traditional Server Deployment

### Ubuntu/Debian Installation

#### System Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.11 python3.11-dev python3.11-venv \
  postgresql-client redis-tools nginx supervisor curl

# Create application user
sudo useradd --system --home /opt/ncs-api --shell /bin/bash ncs
sudo mkdir -p /opt/ncs-api
sudo chown ncs:ncs /opt/ncs-api
```

#### Application Installation
```bash
# Switch to application user
sudo -u ncs -i

# Clone repository
git clone https://github.com/your-org/ncs-api.git /opt/ncs-api/app
cd /opt/ncs-api/app

# Create virtual environment
python3.11 -m venv /opt/ncs-api/venv
source /opt/ncs-api/venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy configuration
cp .env.example .env
# Edit .env with production values
```

#### Service Configuration
```bash
# Create systemd service
sudo tee /etc/systemd/system/ncs-api.service << EOF
[Unit]
Description=NeuroCluster Streamer API
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=ncs
Group=ncs
WorkingDirectory=/opt/ncs-api/app
Environment=PATH=/opt/ncs-api/venv/bin
ExecStart=/opt/ncs-api/venv/bin/python main_secure.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ncs-api
sudo systemctl start ncs-api
```

#### Nginx Configuration
```bash
# Create Nginx configuration
sudo tee /etc/nginx/sites-available/ncs-api << EOF
upstream ncs_api {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://ncs_api;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/ncs-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### RHEL/CentOS Installation

#### System Preparation
```bash
# Enable EPEL
sudo dnf install -y epel-release

# Install dependencies
sudo dnf install -y python3.11 python3.11-devel postgresql-devel \
  redis nginx supervisor curl

# Create application user
sudo useradd --system --home /opt/ncs-api --shell /bin/bash ncs
```

## ⚙️ Configuration Management

### Environment Variables

#### Core Configuration
```bash
# Application Settings
ENVIRONMENT=production                    # development, staging, production
DEBUG=false                              # Enable debug mode
SECRET_KEY=your-32-character-secret-key  # JWT signing key
HOST=0.0.0.0                            # Bind address
PORT=8000                               # Bind port
WORKERS=8                               # Number of worker processes

# Database Configuration
DATABASE_URL=postgresql://user:pass@host:5432/dbname
DB_POOL_SIZE=20                         # Connection pool size
DB_MAX_OVERFLOW=40                      # Max overflow connections
DB_POOL_TIMEOUT=30                      # Pool timeout seconds

# Redis Configuration  
REDIS_URL=redis://user:pass@host:6379/0
REDIS_POOL_SIZE=50                      # Connection pool size
CACHE_TTL_SECONDS=3600                  # Default cache TTL

# Security Configuration
JWT_ALGORITHM=HS256                     # JWT algorithm
ACCESS_TOKEN_EXPIRE_MINUTES=30          # Token expiration
RATE_LIMIT_PER_MINUTE=1000             # Rate limiting
ALLOWED_ORIGINS=https://yourdomain.com  # CORS origins

# Algorithm Configuration
NCS_BASE_THRESHOLD=0.71                 # Clustering threshold
NCS_LEARNING_RATE=0.06                  # Adaptation rate
NCS_MAX_CLUSTERS=30                     # Maximum clusters
```

#### Environment-Specific Files
```bash
# Development (.env.development)
DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://localhost/ncs_dev
REDIS_URL=redis://localhost:6379/0
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080

# Staging (.env.staging)
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=postgresql://staging-db:5432/ncs_staging
REDIS_URL=redis://staging-redis:6379/0

# Production (.env.production)
DEBUG=false
LOG_LEVEL=WARNING
# Use external secret management for production
```

### Secret Management

#### Using External Secret Managers

**AWS Secrets Manager:**
```python
# config.py
import boto3

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']

SECRET_KEY = get_secret('ncs-api/secret-key')
DATABASE_URL = get_secret('ncs-api/database-url')
```

**HashiCorp Vault:**
```python
import hvac

def get_vault_secret(path):
    client = hvac.Client(url='https://vault.company.com')
    client.token = os.getenv('VAULT_TOKEN')
    response = client.secrets.kv.v2.read_secret_version(path=path)
    return response['data']['data']

secrets = get_vault_secret('ncs-api/production')
SECRET_KEY = secrets['secret_key']
```

**Kubernetes Secrets:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ncs-api-secrets
type: Opaque
data:
  secret-key: <base64-encoded-secret>
  database-url: <base64-encoded-url>
```

### Database Migrations

#### Automatic Migration Script
```bash
#!/bin/bash
# scripts/db_migrate.py

# Wait for database to be available
python -c "
import psycopg2
import time
import os

def wait_for_db():
    max_retries = 30
    for i in range(max_retries):
        try:
            conn = psycopg2.connect(os.getenv('DATABASE_URL'))
            conn.close()
            print('Database is ready')
            return
        except psycopg2.OperationalError:
            print(f'Waiting for database... ({i+1}/{max_retries})')
            time.sleep(2)
    raise Exception('Database not available')

wait_for_db()
"

# Run migrations
python database/migrate.py
```

## 📊 Monitoring & Observability

### Prometheus Configuration

#### Metrics Collection
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert-rules.yml"

scrape_configs:
  - job_name: 'ncs-api'
    static_configs:
      - targets: ['ncs-api:8000']
    metrics_path: /metrics
    scrape_interval: 10s
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

#### Alert Rules
```yaml
# monitoring/alert-rules.yml
groups:
  - name: ncs-api-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
```

### Grafana Dashboards

#### API Performance Dashboard
```json
{
  "dashboard": {
    "title": "NCS API Performance",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{handler}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### Health Checks

#### Kubernetes Health Checks
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
```

#### Load Balancer Health Checks
```bash
# Nginx upstream health check
upstream ncs_api {
    server 10.0.1.100:8000 max_fails=3 fail_timeout=30s;
    server 10.0.1.101:8000 max_fails=3 fail_timeout=30s;
    server 10.0.1.102:8000 max_fails=3 fail_timeout=30s;
}
```

## 📈 Scaling & Performance

### Horizontal Scaling

#### Kubernetes Autoscaling
```yaml
# HPA configuration for CPU and memory
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ncs-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ncs-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Docker Swarm Scaling
```bash
# Scale service
docker service scale ncs-api=5

# Update service with rolling update
docker service update --image ncs-api:v1.1.0 ncs-api
```

### Performance Optimization

#### Application Tuning
```python
# main_secure.py - Production configuration
if __name__ == "__main__":
    import uvicorn
    
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 8,                    # CPU cores * 2
        "worker_class": "uvicorn.workers.UvicornWorker",
        "worker_connections": 1000,
        "max_requests": 10000,
        "max_requests_jitter": 1000,
        "preload_app": True,
        "keepalive": 5,
        "access_log": False,             # Disable for performance
        "log_level": "warning"
    }
    
    uvicorn.run("main_secure:app", **config)
```

#### Database Optimization
```sql
-- PostgreSQL performance tuning
-- postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200

-- Create indexes for performance
CREATE INDEX CONCURRENTLY idx_clusters_created_at ON clusters(created_at);
CREATE INDEX CONCURRENTLY idx_points_cluster_id ON points(cluster_id);
```

#### Redis Optimization
```bash
# redis.conf
maxmemory 1gb
maxmemory-policy allkeys-lru
save ""  # Disable persistence for cache-only usage
tcp-keepalive 60
timeout 300
```

### Load Testing

#### Basic Load Test
```bash
# Using Apache Bench
ab -n 10000 -c 100 -H "Authorization: Bearer $TOKEN" \
   -T "application/json" \
   -p post_data.json \
   http://localhost:8000/api/v1/process_points

# Using wrk
wrk -t12 -c400 -d30s --script=load_test.lua http://localhost:8000/
```

#### Advanced Load Testing
```python
# load_test.py using locust
from locust import HttpUser, task, between

class NCSAPIUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Authenticate
        response = self.client.post("/auth/login", data={
            "username": "test_user",
            "password": "test_password"
        })
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(3)
    def process_points(self):
        data = {
            "points": [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]
        }
        self.client.post("/api/v1/process_points", 
                        json=data, headers=self.headers)
    
    @task(1)
    def get_status(self):
        self.client.get("/api/v1/algorithm_status", 
                       headers=self.headers)
```

## 🔧 Troubleshooting

### Common Issues

#### Application Won't Start
```bash
# Check logs
docker logs ncs-api
kubectl logs deployment/ncs-api -n ncs-api

# Common causes:
# 1. Missing environment variables
# 2. Database connection issues
# 3. Port already in use
# 4. Insufficient permissions

# Debug steps:
# Check configuration
python -c "from config import settings; print(settings.dict())"

# Test database connection
python -c "import psycopg2; psycopg2.connect('$DATABASE_URL')"

# Check port availability
netstat -tlnp | grep :8000
```

#### High Memory Usage
```bash
# Check memory usage
docker stats ncs-api
kubectl top pods -n ncs-api

# Common causes:
# 1. Memory leaks in algorithm
# 2. Too many cached objects
# 3. Large request payloads

# Solutions:
# 1. Restart pods/containers
# 2. Adjust cache TTL
# 3. Implement request size limits
```

#### Database Connection Issues
```bash
# Test connectivity
pg_isready -h $POSTGRES_HOST -p $POSTGRES_PORT

# Check connection pool
SELECT count(*) FROM pg_stat_activity WHERE datname = 'ncs_api';

# Common solutions:
# 1. Increase connection pool size
# 2. Check firewall rules
# 3. Verify credentials
```

#### Performance Degradation
```bash
# Check metrics
curl -s http://localhost:8000/metrics | grep http_request_duration

# Common causes:
# 1. Database query performance
# 2. Memory pressure
# 3. Network latency
# 4. Algorithm complexity

# Debug steps:
# 1. Check database query plans
# 2. Monitor resource usage
# 3. Review algorithm parameters
# 4. Analyze request patterns
```

### Monitoring and Debugging

#### Application Logs
```bash
# View logs in real-time
docker logs -f ncs-api
kubectl logs -f deployment/ncs-api -n ncs-api

# Filter logs by level
docker logs ncs-api 2>&1 | grep ERROR
kubectl logs deployment/ncs-api -n ncs-api | grep WARNING
```

#### Performance Profiling
```python
# Enable profiling in development
import cProfile
import pstats

def profile_request():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your code here
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats()
```

#### Database Debugging
```sql
-- Check slow queries
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Check active connections
SELECT pid, usename, application_name, client_addr, state
FROM pg_stat_activity
WHERE datname = 'ncs_api';
```

### Recovery Procedures

#### Rollback Deployment
```bash
# Kubernetes rollback
kubectl rollout undo deployment/ncs-api -n ncs-api

# Docker Compose rollback
docker-compose down
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Check rollback status
kubectl rollout status deployment/ncs-api -n ncs-api
```

#### Database Recovery
```bash
# Restore from backup
pg_restore -h $POSTGRES_HOST -U $POSTGRES_USER -d ncs_api backup.sql

# Point-in-time recovery
pg_basebackup -h $POSTGRES_HOST -D backup -U $POSTGRES_USER -v -W
```

---

## 📞 Support and Resources

### Getting Help
- **Documentation**: [Complete guides and references](README.md)
- **GitHub Issues**: [Bug reports and feature requests](https://github.com/your-org/ncs-api/issues)
- **Community Support**: [Discussions and questions](https://github.com/your-org/ncs-api/discussions)
- **Enterprise Support**: contact@yourdomain.com

### Additional Resources
- [Security Guide](SECURITY_GUIDE.md) - Security best practices
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Troubleshooting Guide](TROUBLESHOOTING.md) - Common issues and solutions
- [Performance Tuning](examples/production_setup.md) - Optimization guide

### Contributing
We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to get involved.