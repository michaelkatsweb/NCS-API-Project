# NeuroCluster Streamer API Security Guide

Comprehensive security documentation covering authentication, authorization, security configurations, and best practices for securing the NCS API in production environments.

## 📋 Table of Contents

- [Security Overview](#-security-overview)
- [Authentication & Authorization](#-authentication--authorization)
- [API Security](#-api-security)
- [Network Security](#-network-security)
- [Data Protection](#-data-protection)
- [Infrastructure Security](#-infrastructure-security)
- [Monitoring & Incident Response](#-monitoring--incident-response)
- [Compliance & Auditing](#-compliance--auditing)
- [Security Best Practices](#-security-best-practices)

## 🔒 Security Overview

The NeuroCluster Streamer API implements defense-in-depth security with multiple layers of protection:

### Security Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    External Layer                           │
├─────────────────────────────────────────────────────────────┤
│ CDN/WAF │ Load Balancer │ TLS Termination │ DDoS Protection │
├─────────────────────────────────────────────────────────────┤
│                    Network Layer                            │
├─────────────────────────────────────────────────────────────┤
│ Firewall │ VPC/Network Isolation │ Security Groups          │
├─────────────────────────────────────────────────────────────┤
│                   Application Layer                         │
├─────────────────────────────────────────────────────────────┤
│ Authentication │ Authorization │ Rate Limiting │ Validation  │
├─────────────────────────────────────────────────────────────┤
│                     Data Layer                              │
├─────────────────────────────────────────────────────────────┤
│ Encryption at Rest │ Encryption in Transit │ Key Management │
└─────────────────────────────────────────────────────────────┘
```

### Security Principles

- **Zero Trust**: Never trust, always verify
- **Least Privilege**: Minimal required permissions
- **Defense in Depth**: Multiple security layers
- **Fail Secure**: Secure defaults when systems fail
- **Security by Design**: Built-in security, not bolted-on

### Threat Model

| Threat Category | Risk Level | Mitigation |
|----------------|------------|------------|
| **Unauthorized Access** | High | Multi-factor authentication, RBAC |
| **Data Breach** | High | Encryption, access controls, auditing |
| **DDoS Attacks** | Medium | Rate limiting, CDN, scaling |
| **Injection Attacks** | Medium | Input validation, parameterized queries |
| **Man-in-the-Middle** | Medium | TLS 1.3, certificate pinning |
| **Insider Threats** | Low | Audit logging, least privilege |

## 🔐 Authentication & Authorization

### Authentication Methods

#### 1. JWT Token Authentication (Recommended for Users)

**Implementation:**
```python
# JWT Configuration
JWT_SECRET_KEY = "your-256-bit-secret-key"
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Token structure
{
  "sub": "user_id_12345",           # Subject (user ID)
  "email": "user@example.com",      # User email
  "role": "user",                   # User role
  "scopes": ["read", "write"],      # Permissions
  "iat": 1642248000,                # Issued at
  "exp": 1642249800,                # Expires at
  "jti": "token_unique_id"          # JWT ID for revocation
}
```

**Login Process:**
```bash
# 1. Authenticate with credentials
curl -X POST "https://api.yourdomain.com/auth/login" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=user@example.com&password=secure_password"

# Response
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "scope": "read write"
}

# 2. Use token in subsequent requests
curl -X GET "https://api.yourdomain.com/api/v1/clusters_summary" \
     -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
```

**Token Refresh:**
```bash
# Refresh expired access token
curl -X POST "https://api.yourdomain.com/auth/refresh" \
     -H "Authorization: Bearer REFRESH_TOKEN"
```

#### 2. API Key Authentication (Recommended for Services)

**Configuration:**
```python
# API Key Settings
API_KEY_HEADER = "X-API-Key"
API_KEY_LENGTH = 32
API_KEY_PREFIX = "ncs_"

# Example API key: ncs_1234567890abcdef1234567890abcdef
```

**Usage:**
```bash
# Include API key in header
curl -X POST "https://api.yourdomain.com/api/v1/process_points" \
     -H "X-API-Key: ncs_1234567890abcdef1234567890abcdef" \
     -H "Content-Type: application/json" \
     -d '{"points": [[1.0, 2.0, 3.0]]}'
```

### Role-Based Access Control (RBAC)

#### User Roles

| Role | Permissions | Description |
|------|-------------|-------------|
| **admin** | Full access | System administration, user management |
| **user** | Read/Write data | Normal API operations |
| **readonly** | Read-only | Monitoring, reporting access |
| **service** | API operations | Service-to-service communication |

#### Permission Scopes

```python
# Scope definitions
SCOPES = {
    "read": "Read access to data and status",
    "write": "Write access to process data",
    "admin": "Administrative operations",
    "monitor": "Monitoring and metrics access"
}

# Role scope mapping
ROLE_SCOPES = {
    "admin": ["read", "write", "admin", "monitor"],
    "user": ["read", "write"],
    "readonly": ["read", "monitor"],
    "service": ["read", "write"]
}
```

#### Endpoint Protection

```python
# FastAPI dependency for role-based access
from fastapi import Depends, HTTPException, status
from functools import wraps

def require_role(required_role: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(current_user = Depends(get_current_user), *args, **kwargs):
            if not has_required_role(current_user, required_role):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            return await func(current_user, *args, **kwargs)
        return wrapper
    return decorator

# Usage
@app.get("/admin/users")
@require_role("admin")
async def list_users():
    return {"users": []}
```

### Authentication Security

#### Password Security

```python
# Password requirements
PASSWORD_MIN_LENGTH = 12
PASSWORD_REQUIRE_UPPERCASE = True
PASSWORD_REQUIRE_LOWERCASE = True
PASSWORD_REQUIRE_DIGITS = True
PASSWORD_REQUIRE_SPECIAL = True
PASSWORD_HISTORY_COUNT = 5  # Prevent reuse of last 5 passwords

# Secure password hashing
import bcrypt

def hash_password(password: str) -> str:
    """Hash password using bcrypt with random salt."""
    salt = bcrypt.gensalt(rounds=12)  # CPU cost factor
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )
```

#### Multi-Factor Authentication (MFA)

```python
# TOTP-based MFA implementation
import pyotp
import qrcode

def generate_mfa_secret(user_id: str) -> str:
    """Generate MFA secret for user."""
    secret = pyotp.random_base32()
    
    # Store secret securely (encrypted)
    store_encrypted_mfa_secret(user_id, secret)
    
    return secret

def generate_qr_code(user_email: str, secret: str) -> str:
    """Generate QR code for MFA setup."""
    totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
        name=user_email,
        issuer_name="NCS API"
    )
    
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(totp_uri)
    qr.make(fit=True)
    
    return qr.get_matrix()

def verify_mfa_token(user_id: str, token: str) -> bool:
    """Verify MFA token."""
    secret = get_decrypted_mfa_secret(user_id)
    totp = pyotp.TOTP(secret)
    return totp.verify(token, valid_window=1)
```

## 🛡️ API Security

### Security Headers

#### Essential Security Headers

```python
# Middleware for security headers
from fastapi import FastAPI
from fastapi.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Enable XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Enforce HTTPS
        response.headers["Strict-Transport-Security"] = \
            "max-age=31536000; includeSubDomains; preload"
        
        # Content Security Policy
        response.headers["Content-Security-Policy"] = \
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions Policy
        response.headers["Permissions-Policy"] = \
            "geolocation=(), microphone=(), camera=()"
        
        return response

app.add_middleware(SecurityHeadersMiddleware)
```

### Cross-Origin Resource Sharing (CORS)

```python
# Secure CORS configuration
from fastapi.middleware.cors import CORSMiddleware

# Production CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "https://app.yourdomain.com"
    ],  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-API-Key",
        "X-Request-ID"
    ],
    expose_headers=["X-Request-ID"],
    max_age=3600
)
```

### Rate Limiting

#### Multi-Tier Rate Limiting

```python
# Rate limiting configuration
RATE_LIMITS = {
    "global": "10000/hour",           # Global limit
    "per_user": "1000/hour",          # Per authenticated user
    "per_ip": "100/hour",             # Per IP address
    "per_endpoint": {
        "/api/v1/process_points": "100/minute",
        "/api/v1/clusters_summary": "300/minute",
        "/auth/login": "10/minute"
    }
}

# Implementation with Redis backend
import redis
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def is_allowed(self, key: str, limit: int, window: int) -> tuple[bool, dict]:
        """Check if request is within rate limit."""
        now = datetime.utcnow()
        pipeline = self.redis.pipeline()
        
        # Sliding window implementation
        window_start = now - timedelta(seconds=window)
        
        # Remove old entries
        pipeline.zremrangebyscore(key, 0, window_start.timestamp())
        
        # Count current requests
        pipeline.zcard(key)
        
        # Add current request
        pipeline.zadd(key, {str(now.timestamp()): now.timestamp()})
        
        # Set expiry
        pipeline.expire(key, window)
        
        results = pipeline.execute()
        current_requests = results[1]
        
        return current_requests < limit, {
            "limit": limit,
            "remaining": max(0, limit - current_requests),
            "reset": (now + timedelta(seconds=window)).timestamp()
        }
```

#### Rate Limiting Headers

```python
# Add rate limit headers to responses
@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    # Get rate limit info
    is_allowed, rate_info = rate_limiter.is_allowed(
        get_rate_limit_key(request),
        get_rate_limit(request),
        60  # 1 minute window
    )
    
    if not is_allowed:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"},
            headers={
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(rate_info["reset"])),
                "Retry-After": str(60)
            }
        )
    
    response = await call_next(request)
    
    # Add rate limit headers
    response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
    response.headers["X-RateLimit-Reset"] = str(int(rate_info["reset"]))
    
    return response
```

### Input Validation & Sanitization

#### Request Validation

```python
# Pydantic models with validation
from pydantic import BaseModel, validator, Field
from typing import List, Optional
import re

class ProcessPointsRequest(BaseModel):
    points: List[List[float]] = Field(
        ...,
        min_items=1,
        max_items=10000,
        description="Array of data points"
    )
    batch_mode: Optional[bool] = Field(
        default=False,
        description="Enable batch processing mode"
    )
    timeout: Optional[int] = Field(
        default=30000,
        ge=1000,
        le=300000,
        description="Request timeout in milliseconds"
    )
    
    @validator('points')
    def validate_points(cls, v):
        for i, point in enumerate(v):
            if len(point) == 0:
                raise ValueError(f"Point {i} must have at least 1 dimension")
            if len(point) > 1000:
                raise ValueError(f"Point {i} has too many dimensions (max 1000)")
            
            for j, coord in enumerate(point):
                if not isinstance(coord, (int, float)):
                    raise ValueError(f"Point {i}, coordinate {j} must be a number")
                if not (-1e10 <= coord <= 1e10):
                    raise ValueError(f"Point {i}, coordinate {j} is out of range")
        
        return v

# SQL injection prevention
def sanitize_sql_input(input_str: str) -> str:
    """Sanitize input for SQL queries."""
    # Remove dangerous characters
    dangerous_chars = ["'", '"', ";", "--", "/*", "*/", "xp_", "sp_"]
    for char in dangerous_chars:
        input_str = input_str.replace(char, "")
    
    return input_str.strip()
```

#### Output Sanitization

```python
# Sanitize output data
import html
import json

def sanitize_output(data: any) -> any:
    """Sanitize output data to prevent XSS."""
    if isinstance(data, str):
        return html.escape(data)
    elif isinstance(data, dict):
        return {key: sanitize_output(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_output(item) for item in data]
    else:
        return data
```

## 🌐 Network Security

### TLS/SSL Configuration

#### TLS Best Practices

```nginx
# Nginx TLS configuration
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;
    
    # TLS certificates
    ssl_certificate /etc/ssl/certs/api.yourdomain.com.crt;
    ssl_certificate_key /etc/ssl/private/api.yourdomain.com.key;
    
    # TLS protocol versions
    ssl_protocols TLSv1.2 TLSv1.3;
    
    # Cipher suites (Mozilla Intermediate)
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    
    # OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/ssl/certs/ca-certificates.crt;
    
    # Session settings
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_session_tickets off;
    
    location / {
        proxy_pass http://ncs-api-backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

### Firewall Configuration

#### iptables Rules

```bash
#!/bin/bash
# Basic firewall configuration

# Flush existing rules
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (limit connections)
iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --set
iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --update --seconds 60 --hitcount 4 -j DROP
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow API port (internal only)
iptables -A INPUT -p tcp --dport 8000 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 8000 -s 172.16.0.0/12 -j ACCEPT
iptables -A INPUT -p tcp --dport 8000 -s 192.168.0.0/16 -j ACCEPT

# Drop everything else
iptables -A INPUT -j DROP

# Save rules
iptables-save > /etc/iptables/rules.v4
```

### Network Segmentation

#### Kubernetes Network Policies

```yaml
# Network policy for API pods
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ncs-api-network-policy
  namespace: ncs-api
spec:
  podSelector:
    matchLabels:
      app: ncs-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - podSelector:
        matchLabels:
          app: load-balancer
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to: []  # Allow DNS
    ports:
    - protocol: UDP
      port: 53
```

## 🔐 Data Protection

### Encryption at Rest

#### Database Encryption

```sql
-- PostgreSQL encryption configuration
-- postgresql.conf

# Enable SSL
ssl = on
ssl_cert_file = '/etc/ssl/certs/postgres.crt'
ssl_key_file = '/etc/ssl/private/postgres.key'
ssl_ca_file = '/etc/ssl/certs/ca.crt'

# Require SSL
ssl_min_protocol_version = 'TLSv1.2'
ssl_prefer_server_ciphers = on
ssl_ciphers = 'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS'

# Transparent Data Encryption (if available)
# wal_level = replica
# archive_mode = on
# archive_command = 'pgbackrest --stanza=main archive-push %p'
```

#### Application-Level Encryption

```python
# Sensitive data encryption
from cryptography.fernet import Fernet
import base64
import os

class DataEncryption:
    def __init__(self, key: bytes = None):
        if key is None:
            key = base64.urlsafe_b64decode(os.getenv('ENCRYPTION_KEY'))
        self.cipher_suite = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data."""
        encrypted_data = self.cipher_suite.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
        return decrypted_data.decode()

# Usage for storing sensitive user data
encryptor = DataEncryption()

# Encrypt PII before storing
encrypted_email = encryptor.encrypt("user@example.com")
encrypted_api_key = encryptor.encrypt("api_key_value")
```

### Encryption in Transit

#### Application-Level Encryption

```python
# TLS configuration for HTTP clients
import ssl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

class TLSAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM')
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

# Use secure HTTP client
session = requests.Session()
session.mount('https://', TLSAdapter())
```

### Key Management

#### HashiCorp Vault Integration

```python
# Vault integration for key management
import hvac

class VaultKeyManager:
    def __init__(self, vault_url: str, vault_token: str):
        self.client = hvac.Client(url=vault_url, token=vault_token)
        
    def get_secret(self, path: str, key: str) -> str:
        """Retrieve secret from Vault."""
        response = self.client.secrets.kv.v2.read_secret_version(path=path)
        return response['data']['data'][key]
    
    def store_secret(self, path: str, secret_dict: dict):
        """Store secret in Vault."""
        self.client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret=secret_dict
        )
    
    def rotate_key(self, path: str):
        """Rotate encryption key."""
        new_key = Fernet.generate_key()
        self.store_secret(path, {'key': new_key.decode()})
        return new_key

# Usage
vault = VaultKeyManager('https://vault.company.com', vault_token)
encryption_key = vault.get_secret('ncs-api/keys', 'encryption_key')
```

### Data Anonymization

```python
# Data anonymization for logs and exports
import hashlib
import re

class DataAnonymizer:
    @staticmethod
    def hash_pii(data: str, salt: str = None) -> str:
        """Hash PII data for anonymization."""
        if salt is None:
            salt = os.getenv('PII_SALT', 'default_salt')
        
        return hashlib.sha256(f"{data}{salt}".encode()).hexdigest()[:16]
    
    @staticmethod
    def mask_email(email: str) -> str:
        """Mask email address."""
        if '@' not in email:
            return email
        
        username, domain = email.split('@', 1)
        masked_username = username[0] + '*' * (len(username) - 2) + username[-1]
        return f"{masked_username}@{domain}"
    
    @staticmethod
    def anonymize_ip(ip: str) -> str:
        """Anonymize IP address."""
        # IPv4
        if '.' in ip:
            parts = ip.split('.')
            return f"{parts[0]}.{parts[1]}.xxx.xxx"
        # IPv6
        elif ':' in ip:
            parts = ip.split(':')
            return ':'.join(parts[:3] + ['xxxx'] * (len(parts) - 3))
        
        return 'xxx.xxx.xxx.xxx'
```

## 🏢 Infrastructure Security

### Container Security

#### Secure Dockerfile

```dockerfile
# Multi-stage build for security
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r ncs && useradd -r -g ncs ncs

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy application
WORKDIR /app
COPY --from=builder /root/.local /home/ncs/.local
COPY --chown=ncs:ncs . .

# Security hardening
RUN chmod -R 755 /app \
    && find /app -name "*.py" -exec chmod 644 {} \;

# Use non-root user
USER ncs

# Security labels
LABEL security.scan="enabled" \
      security.policy="restricted"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["python", "main_secure.py"]
```

#### Container Scanning

```yaml
# GitHub Actions security scanning
name: Container Security Scan

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t ncs-api:latest .
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'ncs-api:latest'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

### Kubernetes Security

#### Pod Security Standards

```yaml
# Pod Security Policy
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: ncs-api-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

#### Security Context

```yaml
# Secure pod security context
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ncs-api
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      containers:
      - name: ncs-api
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1000
          capabilities:
            drop:
            - ALL
            add:
            - NET_BIND_SERVICE
```

## 📊 Monitoring & Incident Response

### Security Monitoring

#### Audit Logging

```python
# Comprehensive audit logging
import structlog
from datetime import datetime
from typing import Dict, Any

logger = structlog.get_logger()

class SecurityAuditLogger:
    @staticmethod
    def log_authentication_event(
        user_id: str,
        event_type: str,
        source_ip: str,
        user_agent: str,
        success: bool,
        details: Dict[str, Any] = None
    ):
        """Log authentication events."""
        logger.info(
            "authentication_event",
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            event_type=event_type,  # login, logout, token_refresh, mfa_verify
            source_ip=source_ip,
            user_agent=user_agent,
            success=success,
            details=details or {}
        )
    
    @staticmethod
    def log_authorization_event(
        user_id: str,
        resource: str,
        action: str,
        granted: bool,
        source_ip: str
    ):
        """Log authorization events."""
        logger.info(
            "authorization_event",
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            resource=resource,
            action=action,
            granted=granted,
            source_ip=source_ip
        )
    
    @staticmethod
    def log_security_event(
        event_type: str,
        severity: str,
        source_ip: str,
        details: Dict[str, Any]
    ):
        """Log security events."""
        logger.warning(
            "security_event",
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,  # rate_limit_exceeded, suspicious_activity
            severity=severity,      # low, medium, high, critical
            source_ip=source_ip,
            details=details
        )
```

#### Intrusion Detection

```python
# Simple intrusion detection system
from collections import defaultdict, deque
from datetime import datetime, timedelta

class IntrusionDetector:
    def __init__(self):
        self.failed_attempts = defaultdict(deque)
        self.suspicious_ips = set()
        
    def record_failed_login(self, ip_address: str):
        """Record failed login attempt."""
        now = datetime.utcnow()
        
        # Clean old attempts
        cutoff = now - timedelta(minutes=15)
        while (self.failed_attempts[ip_address] and 
               self.failed_attempts[ip_address][0] < cutoff):
            self.failed_attempts[ip_address].popleft()
        
        # Add current attempt
        self.failed_attempts[ip_address].append(now)
        
        # Check for suspicious activity
        if len(self.failed_attempts[ip_address]) >= 5:
            self.mark_suspicious(ip_address)
    
    def mark_suspicious(self, ip_address: str):
        """Mark IP as suspicious."""
        self.suspicious_ips.add(ip_address)
        
        SecurityAuditLogger.log_security_event(
            event_type="suspicious_activity",
            severity="medium",
            source_ip=ip_address,
            details={
                "reason": "multiple_failed_logins",
                "attempt_count": len(self.failed_attempts[ip_address])
            }
        )
    
    def is_suspicious(self, ip_address: str) -> bool:
        """Check if IP is marked as suspicious."""
        return ip_address in self.suspicious_ips
```

### Incident Response

#### Automated Response

```python
# Automated incident response
class IncidentResponder:
    def __init__(self, rate_limiter, notification_service):
        self.rate_limiter = rate_limiter
        self.notification_service = notification_service
    
    def handle_security_incident(self, incident_type: str, details: Dict[str, Any]):
        """Handle security incidents automatically."""
        
        if incident_type == "brute_force_attack":
            self._handle_brute_force(details)
        elif incident_type == "rate_limit_exceeded":
            self._handle_rate_limit(details)
        elif incident_type == "suspicious_payload":
            self._handle_suspicious_payload(details)
    
    def _handle_brute_force(self, details: Dict[str, Any]):
        """Handle brute force attack."""
        ip_address = details.get("source_ip")
        
        # Temporarily block IP
        self.rate_limiter.block_ip(ip_address, duration=3600)  # 1 hour
        
        # Alert security team
        self.notification_service.send_alert(
            severity="high",
            message=f"Brute force attack detected from {ip_address}",
            details=details
        )
    
    def _handle_rate_limit(self, details: Dict[str, Any]):
        """Handle rate limit violations."""
        if details.get("violation_count", 0) > 10:
            # Extended block for persistent violators
            self.rate_limiter.block_ip(
                details.get("source_ip"),
                duration=7200  # 2 hours
            )
```

## 📋 Compliance & Auditing

### Compliance Standards

#### GDPR Compliance

```python
# GDPR compliance features
class GDPRCompliance:
    def __init__(self, db_session, encryptor):
        self.db = db_session
        self.encryptor = encryptor
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all user data (Right of Access)."""
        user_data = self.db.query(User).filter(User.id == user_id).first()
        user_sessions = self.db.query(UserSession).filter(
            UserSession.user_id == user_id
        ).all()
        
        return {
            "personal_data": {
                "user_id": user_data.id,
                "email": self.encryptor.decrypt(user_data.encrypted_email),
                "created_at": user_data.created_at.isoformat(),
                "last_login": user_data.last_login.isoformat()
            },
            "sessions": [
                {
                    "session_id": session.id,
                    "created_at": session.created_at.isoformat(),
                    "ip_address": session.ip_address
                }
                for session in user_sessions
            ]
        }
    
    def delete_user_data(self, user_id: str) -> bool:
        """Delete user data (Right to be Forgotten)."""
        try:
            # Anonymize instead of delete for audit trail
            user = self.db.query(User).filter(User.id == user_id).first()
            if user:
                user.encrypted_email = self.encryptor.encrypt("deleted@example.com")
                user.status = "deleted"
                user.deleted_at = datetime.utcnow()
                
                # Remove sessions
                self.db.query(UserSession).filter(
                    UserSession.user_id == user_id
                ).delete()
                
                self.db.commit()
                return True
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to delete user data", user_id=user_id, error=str(e))
            return False
```

#### SOC 2 Compliance

```python
# SOC 2 compliance monitoring
class SOC2Monitor:
    def __init__(self):
        self.access_logs = []
        self.data_processing_logs = []
    
    def log_data_access(
        self,
        user_id: str,
        data_type: str,
        operation: str,
        timestamp: datetime
    ):
        """Log data access for SOC 2 compliance."""
        self.access_logs.append({
            "user_id": user_id,
            "data_type": data_type,
            "operation": operation,
            "timestamp": timestamp.isoformat(),
            "compliance_requirement": "SOC2_CC6.1"
        })
    
    def generate_compliance_report(self, start_date: datetime, end_date: datetime):
        """Generate compliance report."""
        relevant_logs = [
            log for log in self.access_logs
            if start_date <= datetime.fromisoformat(log["timestamp"]) <= end_date
        ]
        
        return {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_access_events": len(relevant_logs),
            "unique_users": len(set(log["user_id"] for log in relevant_logs)),
            "data_types_accessed": list(set(log["data_type"] for log in relevant_logs)),
            "compliance_status": "compliant"
        }
```

## 🔐 Security Best Practices

### Development Security

#### Secure Development Lifecycle (SDLC)

```yaml
# Security gates in CI/CD pipeline
name: Security Pipeline

on: [push, pull_request]

jobs:
  security-checks:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    # Static Application Security Testing (SAST)
    - name: Run Bandit Security Scan
      run: |
        pip install bandit
        bandit -r . -f json -o bandit-report.json
    
    # Dependency vulnerability scanning
    - name: Run Safety Check
      run: |
        pip install safety
        safety check --json --output safety-report.json
    
    # Secret scanning
    - name: Run Secrets Scan
      uses: trufflesecurity/trufflehog@main
      with:
        path: ./
        base: main
        head: HEAD
    
    # Container security scanning
    - name: Build and Scan Docker Image
      run: |
        docker build -t ncs-api:${{ github.sha }} .
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
          aquasec/trivy image ncs-api:${{ github.sha }}
```

### Operational Security

#### Security Configuration Checklist

```yaml
# Production security checklist
production_security_checklist:
  authentication:
    - [ ] JWT tokens use strong secrets (32+ characters)
    - [ ] Token expiration is configured appropriately
    - [ ] MFA is enabled for admin accounts
    - [ ] API keys are rotated regularly
    - [ ] Password policy is enforced
  
  authorization:
    - [ ] RBAC is properly configured
    - [ ] Principle of least privilege is applied
    - [ ] Endpoint permissions are validated
    - [ ] Admin functions require elevated privileges
  
  network_security:
    - [ ] TLS 1.2+ is enforced
    - [ ] Strong cipher suites are configured
    - [ ] HSTS headers are enabled
    - [ ] Network segmentation is implemented
    - [ ] Firewall rules are restrictive
  
  data_protection:
    - [ ] Encryption at rest is enabled
    - [ ] Encryption in transit is enforced
    - [ ] Key management is secure
    - [ ] PII is properly anonymized
    - [ ] Backups are encrypted
  
  monitoring:
    - [ ] Security logging is comprehensive
    - [ ] Alerts are configured for threats
    - [ ] Intrusion detection is active
    - [ ] Audit trails are maintained
    - [ ] Incident response plan exists
  
  compliance:
    - [ ] GDPR requirements are met
    - [ ] SOC 2 controls are implemented
    - [ ] Security assessments are regular
    - [ ] Vulnerability management process exists
    - [ ] Security training is provided
```

### Security Maintenance

#### Regular Security Tasks

```bash
#!/bin/bash
# Security maintenance script

# 1. Update system packages
apt update && apt upgrade -y

# 2. Rotate API keys (monthly)
python scripts/rotate_api_keys.py

# 3. Review access logs
python scripts/analyze_security_logs.py --days 7

# 4. Check for vulnerable dependencies
pip-audit --desc --format=json

# 5. Verify TLS certificates
openssl x509 -in /etc/ssl/certs/api.crt -dates -noout

# 6. Test backup restoration
python scripts/test_backup_restore.py

# 7. Security configuration review
python scripts/security_config_check.py

# 8. Update security documentation
git add docs/SECURITY_INCIDENTS.md
git commit -m "Update security incident log"
```

### Security Training

#### Developer Security Guidelines

1. **Secure Coding Practices**
   - Input validation and sanitization
   - Output encoding
   - Parameterized queries
   - Error handling without information disclosure

2. **Authentication Security**
   - Secure password storage
   - Token management
   - Session security
   - Multi-factor authentication

3. **Authorization Best Practices**
   - Principle of least privilege
   - Role-based access control
   - Resource-level permissions
   - Privilege escalation prevention

4. **Data Protection**
   - Encryption requirements
   - Key management
   - PII handling
   - Data retention policies

---

## 📞 Security Support

### Reporting Security Issues

- **Security Email**: security@yourdomain.com
- **PGP Key**: Available on company website
- **Response Time**: 24 hours for critical issues
- **Bug Bounty Program**: Details at https://security.yourdomain.com/bounty

### Security Resources

- [OWASP Top 10](https://owasp.org/Top10/) - Web application security risks
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework) - Security guidelines
- [CIS Controls](https://www.cisecurity.org/controls/) - Security best practices
- [SANS Security Guidelines](https://www.sans.org/) - Security resources

### Emergency Contacts

- **Security Team**: +1-555-SEC-TEAM
- **Incident Commander**: +1-555-INC-CMDR
- **Legal Team**: +1-555-LEGAL-01