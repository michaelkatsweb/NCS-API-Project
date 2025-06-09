# 🌐 NCS API Documentation

**Complete API reference for the NeuroCluster Streamer clustering service**

Welcome to the official API documentation for the NeuroCluster Streamer (NCS) API. This high-performance clustering service provides real-time data point processing with adaptive intelligence capabilities.

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Authentication](#-authentication)
3. [Base URL & Endpoints](#-base-url--endpoints)
4. [Core Endpoints](#-core-endpoints)
5. [Request/Response Formats](#-requestresponse-formats)
6. [Error Handling](#-error-handling)
7. [Rate Limiting](#-rate-limiting)
8. [SDKs & Code Examples](#-sdks--code-examples)
9. [Webhooks](#-webhooks)
10. [API Versioning](#-api-versioning)
11. [Status Codes](#-status-codes)
12. [Best Practices](#-best-practices)

---

## ⚡ Quick Start

### 30-Second Integration

```bash
# Get your API key from the dashboard
curl -X POST "https://api.ncs-cluster.com/api/v1/process_points" \
     -H "X-API-Key: YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "points": [
         [1.0, 2.0, 3.0],
         [1.1, 2.1, 3.1],
         [5.0, 6.0, 7.0]
       ]
     }'
```

**Response:**
```json
{
  "request_id": "req_123e4567-e89b-12d3-a456-426614174000",
  "clusters": [
    {
      "point": [1.0, 2.0, 3.0],
      "cluster_id": 0,
      "outlier_score": 0.12
    },
    {
      "point": [1.1, 2.1, 3.1],
      "cluster_id": 0,
      "outlier_score": 0.08
    }
  ],
  "outliers": [
    {
      "point": [5.0, 6.0, 7.0],
      "outlier_score": 0.89
    }
  ],
  "algorithm_quality": 0.923,
  "processing_time_ms": 1.23,
  "points_processed": 3,
  "clusters_found": 1
}
```

---

## 🔐 Authentication

The NCS API supports two authentication methods:

### API Key Authentication (Recommended)

**Header:** `X-API-Key`

```bash
curl -H "X-API-Key: your_api_key_here" \
     "https://api.ncs-cluster.com/api/v1/clusters_summary"
```

### JWT Token Authentication

**Header:** `Authorization: Bearer <token>`

```bash
# 1. Get JWT token
curl -X POST "https://api.ncs-cluster.com/auth/login" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=your_username&password=your_password"

# 2. Use token in requests
curl -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." \
     "https://api.ncs-cluster.com/api/v1/clusters_summary"
```

### Getting Your API Key

1. **Sign up** at [https://api.ncs-cluster.com/signup](https://api.ncs-cluster.com/signup)
2. **Verify your email** and complete account setup
3. **Generate API key** in your dashboard under "API Keys"
4. **Copy your key** and keep it secure

---

## 🌐 Base URL & Endpoints

### Base URL
```
https://api.ncs-cluster.com
```

### Available Versions
- **v1** (Current): `https://api.ncs-cluster.com/api/v1/`

### Core Endpoints Overview

| Endpoint | Method | Purpose | Auth Required |
|----------|--------|---------|---------------|
| `/health` | GET | Service health check | ❌ |
| `/api/v1/process_points` | POST | Process data points | ✅ |
| `/api/v1/clusters_summary` | GET | Get cluster information | ✅ |
| `/api/v1/algorithm_status` | GET | Algorithm status & metrics | ✅ |
| `/auth/login` | POST | User authentication | ❌ |
| `/auth/refresh` | POST | Refresh JWT token | ✅ |
| `/auth/api-keys` | POST | Create API key | ✅ |

---

## 🔧 Core Endpoints

### 1. Process Data Points

**Endpoint:** `POST /api/v1/process_points`

Process a batch of data points through the NCS clustering algorithm.

#### Request Body

```json
{
  "points": [
    [1.0, 2.0, 3.0],
    [1.1, 2.1, 3.1],
    [5.0, 6.0, 7.0]
  ],
  "batch_mode": false,
  "algorithm_params": {
    "base_threshold": 0.71,
    "learning_rate": 0.06,
    "max_clusters": 30
  }
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `points` | array | ✅ | Array of data points (max 10,000) |
| `batch_mode` | boolean | ❌ | Enable batch processing mode |
| `algorithm_params` | object | ❌ | Custom algorithm parameters |

#### Algorithm Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `base_threshold` | float | 0.1-0.95 | 0.71 | Similarity threshold for clustering |
| `learning_rate` | float | 0.01-0.3 | 0.06 | Adaptation speed |
| `max_clusters` | integer | 5-1000 | 30 | Maximum clusters to maintain |
| `outlier_threshold` | float | 0.1-0.9 | 0.2 | Outlier detection threshold |

#### Response

```json
{
  "request_id": "req_123e4567-e89b-12d3-a456-426614174000",
  "clusters": [
    {
      "point": [1.0, 2.0, 3.0],
      "cluster_id": 0,
      "outlier_score": 0.12
    }
  ],
  "outliers": [
    {
      "point": [5.0, 6.0, 7.0],
      "outlier_score": 0.89
    }
  ],
  "algorithm_quality": 0.923,
  "processing_time_ms": 1.23,
  "points_processed": 3,
  "clusters_found": 1,
  "timestamp": "2025-06-07T14:30:45.123Z"
}
```

#### cURL Example

```bash
curl -X POST "https://api.ncs-cluster.com/api/v1/process_points" \
     -H "X-API-Key: YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "points": [
         [1.0, 2.0, 3.0],
         [1.1, 2.1, 3.1],
         [5.0, 6.0, 7.0]
       ],
       "algorithm_params": {
         "base_threshold": 0.75,
         "max_clusters": 50
       }
     }'
```

#### Python Example

```python
import requests

response = requests.post(
    'https://api.ncs-cluster.com/api/v1/process_points',
    headers={
        'X-API-Key': 'YOUR_API_KEY',
        'Content-Type': 'application/json'
    },
    json={
        'points': [
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [5.0, 6.0, 7.0]
        ]
    }
)

result = response.json()
print(f"Found {result['clusters_found']} clusters")
```

#### JavaScript Example

```javascript
const response = await fetch('https://api.ncs-cluster.com/api/v1/process_points', {
  method: 'POST',
  headers: {
    'X-API-Key': 'YOUR_API_KEY',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    points: [
      [1.0, 2.0, 3.0],
      [1.1, 2.1, 3.1],
      [5.0, 6.0, 7.0]
    ]
  })
});

const result = await response.json();
console.log(`Found ${result.clusters_found} clusters`);
```

---

### 2. Get Cluster Summary

**Endpoint:** `GET /api/v1/clusters_summary`

Retrieve information about current clusters and algorithm statistics.

#### Response

```json
{
  "total_clusters": 5,
  "active_clusters": 4,
  "clusters": [
    {
      "cluster_id": 0,
      "centroid": [1.05, 2.05, 3.05],
      "size": 150,
      "confidence": 0.89,
      "age": 3600,
      "last_updated": "2025-06-07T14:29:12.456Z"
    },
    {
      "cluster_id": 1,
      "centroid": [5.12, 6.08, 7.15],
      "size": 89,
      "confidence": 0.76,
      "age": 2400,
      "last_updated": "2025-06-07T14:25:33.789Z"
    }
  ],
  "algorithm_stats": {
    "total_points_processed": 10000,
    "clustering_quality": 0.918,
    "memory_usage_mb": 12.4,
    "avg_processing_time_ms": 0.22
  },
  "timestamp": "2025-06-07T14:30:45.123Z"
}
```

#### cURL Example

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
     "https://api.ncs-cluster.com/api/v1/clusters_summary"
```

---

### 3. Algorithm Status

**Endpoint:** `GET /api/v1/algorithm_status`

Get current algorithm status, performance metrics, and configuration.

#### Response

```json
{
  "is_ready": true,
  "algorithm_version": "8.0",
  "uptime_seconds": 86400,
  "performance_metrics": {
    "points_per_second": 6309,
    "p95_latency_ms": 0.18,
    "p99_latency_ms": 0.31,
    "memory_usage_mb": 12.4,
    "cpu_usage_percent": 15.2
  },
  "current_parameters": {
    "base_threshold": 0.71,
    "learning_rate": 0.06,
    "max_clusters": 30,
    "outlier_threshold": 0.2
  },
  "health_score": 0.95,
  "last_restart": "2025-06-06T14:30:45.123Z",
  "timestamp": "2025-06-07T14:30:45.123Z"
}
```

#### cURL Example

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
     "https://api.ncs-cluster.com/api/v1/algorithm_status"
```

---

### 4. Health Check

**Endpoint:** `GET /health`

Check service health and availability. No authentication required.

#### Response

```json
{
  "status": "healthy",
  "timestamp": 1717681845.123,
  "uptime": 86400,
  "version": "1.0.0",
  "algorithm_ready": true,
  "database_connected": true,
  "cache_connected": true
}
```

#### cURL Example

```bash
curl "https://api.ncs-cluster.com/health"
```

---

### 5. Authentication Endpoints

#### Login

**Endpoint:** `POST /auth/login`

Authenticate with username/password to get JWT token.

**Request Body:**
```json
{
  "username": "your_username",
  "password": "your_password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

#### Create API Key

**Endpoint:** `POST /auth/api-keys`

Create a new API key for your account.

**Request Body:**
```json
{
  "name": "My Application Key",
  "scopes": ["read", "write"],
  "expires_in_days": 90
}
```

**Response:**
```json
{
  "api_key": "ncs_1234567890abcdef1234567890abcdef",
  "name": "My Application Key",
  "scopes": ["read", "write"],
  "expires_at": "2025-09-05T14:30:45.123Z",
  "created_at": "2025-06-07T14:30:45.123Z"
}
```

---

## 📝 Request/Response Formats

### Request Headers

#### Required Headers

```http
Content-Type: application/json
X-API-Key: your_api_key_here
```

#### Optional Headers

```http
X-Request-ID: custom_request_id_for_tracking
User-Agent: YourApp/1.0
Accept: application/json
```

### Response Headers

All responses include these headers:

```http
Content-Type: application/json
X-Request-ID: req_123e4567-e89b-12d3-a456-426614174000
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1717681845
```

### Data Types

#### Point Format

Points must be arrays of numeric values:

```json
// Valid points
[1.0, 2.0, 3.0]          // 3D point
[1, 2, 3, 4, 5]          // 5D point
[0.5, -2.1, 3.14159]     // Decimal values

// Invalid points
["1", "2", "3"]          // Strings not allowed
[1, 2, null]             // Null values not allowed
[1, 2, Infinity]         // Infinite values not allowed
```

#### Constraints

| Constraint | Limit | Description |
|------------|-------|-------------|
| **Max Points per Request** | 10,000 | Maximum points in single request |
| **Max Dimensions** | 1,000 | Maximum dimensions per point |
| **Value Range** | ±1,000,000 | Coordinate value range |
| **Request Size** | 10 MB | Maximum request body size |

---

## 🚨 Error Handling

### Error Response Format

```json
{
  "error": "ValidationError",
  "detail": "Points list cannot be empty",
  "request_id": "req_123e4567-e89b-12d3-a456-426614174000",
  "timestamp": "2025-06-07T14:30:45.123Z",
  "code": "INVALID_INPUT",
  "documentation_url": "https://docs.ncs-cluster.com/errors#invalid-input"
}
```

### Common Error Codes

#### 400 Bad Request

| Code | Description | Solution |
|------|-------------|----------|
| `INVALID_INPUT` | Invalid request format | Check request body format |
| `VALIDATION_ERROR` | Data validation failed | Verify data types and ranges |
| `EMPTY_POINTS` | Points array is empty | Provide at least one point |
| `TOO_MANY_POINTS` | Too many points in request | Reduce batch size (max 10,000) |
| `DIMENSION_MISMATCH` | Inconsistent point dimensions | Ensure all points have same dimensions |
| `INVALID_COORDINATES` | Invalid coordinate values | Check for NaN, Infinity, or out-of-range values |

#### 401 Unauthorized

| Code | Description | Solution |
|------|-------------|----------|
| `MISSING_API_KEY` | No API key provided | Include X-API-Key header |
| `INVALID_API_KEY` | API key is invalid | Check your API key is correct |
| `EXPIRED_TOKEN` | JWT token has expired | Refresh your token |
| `MALFORMED_TOKEN` | Token format is invalid | Check token format |

#### 403 Forbidden

| Code | Description | Solution |
|------|-------------|----------|
| `INSUFFICIENT_PERMISSIONS` | Missing required permissions | Check your account permissions |
| `ACCOUNT_SUSPENDED` | Account is suspended | Contact support |
| `FEATURE_NOT_AVAILABLE` | Feature not available in your plan | Upgrade your plan |

#### 422 Unprocessable Entity

| Code | Description | Solution |
|------|-------------|----------|
| `ALGORITHM_PARAMS_INVALID` | Invalid algorithm parameters | Check parameter ranges |
| `DATA_QUALITY_POOR` | Data quality too poor for processing | Improve data preprocessing |
| `CLUSTERING_IMPOSSIBLE` | Cannot perform clustering on data | Check data characteristics |

#### 429 Too Many Requests

| Code | Description | Solution |
|------|-------------|----------|
| `RATE_LIMIT_EXCEEDED` | Rate limit exceeded | Wait before retrying (see Retry-After header) |
| `QUOTA_EXCEEDED` | Monthly quota exceeded | Upgrade plan or wait for reset |

#### 500 Internal Server Error

| Code | Description | Solution |
|------|-------------|----------|
| `INTERNAL_ERROR` | Unexpected server error | Retry request or contact support |
| `ALGORITHM_ERROR` | Algorithm processing failed | Check input data or contact support |
| `DATABASE_ERROR` | Database connectivity issue | Retry request |

#### 503 Service Unavailable

| Code | Description | Solution |
|------|-------------|----------|
| `SERVICE_MAINTENANCE` | Service under maintenance | Wait for maintenance to complete |
| `ALGORITHM_UNAVAILABLE` | Algorithm temporarily unavailable | Retry with exponential backoff |
| `OVERLOADED` | Service temporarily overloaded | Reduce request rate |

### Error Handling Examples

#### Python

```python
import requests
from requests.exceptions import RequestException

def handle_ncs_request(points):
    try:
        response = requests.post(
            'https://api.ncs-cluster.com/api/v1/process_points',
            headers={'X-API-Key': 'your_key'},
            json={'points': points},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        
        # Handle specific error codes
        error_data = response.json()
        
        if response.status_code == 400:
            print(f"Bad request: {error_data['detail']}")
        elif response.status_code == 401:
            print("Authentication failed - check your API key")
        elif response.status_code == 429:
            retry_after = response.headers.get('Retry-After', 60)
            print(f"Rate limited - retry after {retry_after} seconds")
        elif response.status_code == 503:
            print("Service unavailable - try again later")
        else:
            print(f"API error: {error_data['detail']}")
            
    except RequestException as e:
        print(f"Network error: {e}")
    except KeyError:
        print("Unexpected response format")
```

#### JavaScript

```javascript
async function handleNCSRequest(points) {
  try {
    const response = await fetch('https://api.ncs-cluster.com/api/v1/process_points', {
      method: 'POST',
      headers: {
        'X-API-Key': 'your_key',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ points })
    });

    if (response.ok) {
      return await response.json();
    }

    const errorData = await response.json();

    switch (response.status) {
      case 400:
        console.error(`Bad request: ${errorData.detail}`);
        break;
      case 401:
        console.error('Authentication failed - check your API key');
        break;
      case 429:
        const retryAfter = response.headers.get('Retry-After') || 60;
        console.error(`Rate limited - retry after ${retryAfter} seconds`);
        break;
      case 503:
        console.error('Service unavailable - try again later');
        break;
      default:
        console.error(`API error: ${errorData.detail}`);
    }

  } catch (error) {
    console.error(`Network error: ${error.message}`);
  }
}
```

---

## ⏱️ Rate Limiting

The NCS API implements multiple layers of rate limiting to ensure fair usage and optimal performance.

### Rate Limit Tiers

| Tier | Limit | Window | Scope |
|------|-------|--------|-------|
| **Global** | 10,000 requests | 1 hour | All users |
| **Per User** | 1,000 requests | 1 hour | Individual account |
| **Per IP** | 100 requests | 1 hour | IP address |
| **Per Endpoint** | Varies | 1 minute | Specific endpoints |

### Endpoint-Specific Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/api/v1/process_points` | 100 requests | 1 minute |
| `/api/v1/clusters_summary` | 300 requests | 1 minute |
| `/api/v1/algorithm_status` | 60 requests | 1 minute |
| `/auth/login` | 10 requests | 1 minute |

### Rate Limit Headers

Every response includes rate limit information:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1717681845
X-RateLimit-Retry-After: 3600
```

### Handling Rate Limits

#### Exponential Backoff Strategy

```python
import time
import random

def exponential_backoff_request(url, data, max_retries=5):
    for attempt in range(max_retries):
        response = requests.post(url, json=data)
        
        if response.status_code != 429:
            return response
        
        # Calculate backoff delay
        base_delay = 2 ** attempt
        jitter = random.uniform(0.5, 1.5)
        delay = base_delay * jitter
        
        print(f"Rate limited, retrying in {delay:.1f} seconds...")
        time.sleep(delay)
    
    raise Exception("Max retries exceeded")
```

#### Respect Retry-After Header

```python
def handle_rate_limit(response):
    if response.status_code == 429:
        retry_after = int(response.headers.get('Retry-After', 60))
        print(f"Rate limited. Waiting {retry_after} seconds...")
        time.sleep(retry_after)
        return True
    return False
```

---

## 📚 SDKs & Code Examples

### Official SDKs

#### Python SDK

```bash
pip install ncs-client
```

```python
from ncs_client import NCSClient

client = NCSClient(api_key="your_api_key")
result = client.process_points([[1, 2, 3], [4, 5, 6]])
print(f"Found {result.clusters_found} clusters")
```

#### JavaScript SDK

```bash
npm install @ncs/client
```

```javascript
import { NCSClient } from '@ncs/client';

const client = new NCSClient({ apiKey: 'your_api_key' });
const result = await client.processPoints([[1, 2, 3], [4, 5, 6]]);
console.log(`Found ${result.clustersFound} clusters`);
```

### Language Examples

#### Python with requests

```python
import requests
import json

def cluster_points(points, api_key):
    """Process points using NCS API."""
    
    response = requests.post(
        'https://api.ncs-cluster.com/api/v1/process_points',
        headers={
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        },
        json={'points': points}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API error: {response.status_code}")

# Usage
points = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = cluster_points(points, "your_api_key")
print(json.dumps(result, indent=2))
```

#### Node.js with fetch

```javascript
async function clusterPoints(points, apiKey) {
  const response = await fetch('https://api.ncs-cluster.com/api/v1/process_points', {
    method: 'POST',
    headers: {
      'X-API-Key': apiKey,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ points })
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

// Usage
const points = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
const result = await clusterPoints(points, "your_api_key");
console.log(JSON.stringify(result, null, 2));
```

#### PHP with cURL

```php
<?php
function clusterPoints($points, $apiKey) {
    $url = 'https://api.ncs-cluster.com/api/v1/process_points';
    $data = json_encode(['points' => $points]);
    
    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, $url);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $data);
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'X-API-Key: ' . $apiKey,
        'Content-Type: application/json'
    ]);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    
    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);
    
    if ($httpCode === 200) {
        return json_decode($response, true);
    } else {
        throw new Exception("API error: " . $httpCode);
    }
}

// Usage
$points = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
$result = clusterPoints($points, "your_api_key");
echo json_encode($result, JSON_PRETTY_PRINT);
?>
```

#### Java with HttpClient

```java
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.URI;
import java.time.Duration;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.Map;
import java.util.List;

public class NCSClient {
    private final HttpClient client;
    private final String apiKey;
    private final ObjectMapper mapper;
    
    public NCSClient(String apiKey) {
        this.apiKey = apiKey;
        this.client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(30))
            .build();
        this.mapper = new ObjectMapper();
    }
    
    public Map<String, Object> processPoints(List<List<Double>> points) throws Exception {
        String requestBody = mapper.writeValueAsString(Map.of("points", points));
        
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create("https://api.ncs-cluster.com/api/v1/process_points"))
            .header("X-API-Key", apiKey)
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(requestBody))
            .build();
        
        HttpResponse<String> response = client.send(request, 
            HttpResponse.BodyHandlers.ofString());
        
        if (response.statusCode() == 200) {
            return mapper.readValue(response.body(), Map.class);
        } else {
            throw new RuntimeException("API error: " + response.statusCode());
        }
    }
}

// Usage
NCSClient client = new NCSClient("your_api_key");
List<List<Double>> points = List.of(
    List.of(1.0, 2.0, 3.0),
    List.of(4.0, 5.0, 6.0),
    List.of(7.0, 8.0, 9.0)
);
Map<String, Object> result = client.processPoints(points);
System.out.println(result);
```

#### Go with net/http

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
    "time"
)

type NCSClient struct {
    APIKey     string
    BaseURL    string
    HTTPClient *http.Client
}

type ProcessRequest struct {
    Points [][]float64 `json:"points"`
}

type ProcessResponse struct {
    RequestID       string                   `json:"request_id"`
    Clusters        []map[string]interface{} `json:"clusters"`
    Outliers        []map[string]interface{} `json:"outliers"`
    AlgorithmQuality float64                 `json:"algorithm_quality"`
    ProcessingTimeMS float64                 `json:"processing_time_ms"`
    ClustersFound   int                      `json:"clusters_found"`
}

func NewNCSClient(apiKey string) *NCSClient {
    return &NCSClient{
        APIKey:  apiKey,
        BaseURL: "https://api.ncs-cluster.com",
        HTTPClient: &http.Client{
            Timeout: 30 * time.Second,
        },
    }
}

func (c *NCSClient) ProcessPoints(points [][]float64) (*ProcessResponse, error) {
    req := ProcessRequest{Points: points}
    jsonData, err := json.Marshal(req)
    if err != nil {
        return nil, err
    }

    httpReq, err := http.NewRequest("POST", c.BaseURL+"/api/v1/process_points", 
        bytes.NewBuffer(jsonData))
    if err != nil {
        return nil, err
    }

    httpReq.Header.Set("X-API-Key", c.APIKey)
    httpReq.Header.Set("Content-Type", "application/json")

    resp, err := c.HTTPClient.Do(httpReq)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("API error: %d", resp.StatusCode)
    }

    var result ProcessResponse
    err = json.NewDecoder(resp.Body).Decode(&result)
    return &result, err
}

// Usage
func main() {
    client := NewNCSClient("your_api_key")
    
    points := [][]float64{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
    }
    
    result, err := client.ProcessPoints(points)
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    
    fmt.Printf("Found %d clusters\n", result.ClustersFound)
    fmt.Printf("Algorithm quality: %.3f\n", result.AlgorithmQuality)
}
```

---

## 🔗 Webhooks

Configure webhooks to receive real-time notifications about clustering events.

### Webhook Events

| Event | Description | Payload |
|-------|-------------|---------|
| `cluster.created` | New cluster formed | Cluster details |
| `outlier.detected` | Outlier point identified | Point and score |
| `algorithm.quality_changed` | Quality score updated | New quality metrics |
| `processing.completed` | Batch processing finished | Processing summary |

### Webhook Configuration

#### Create Webhook

```bash
curl -X POST "https://api.ncs-cluster.com/api/v1/webhooks" \
     -H "X-API-Key: YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://your-app.com/webhooks/ncs",
       "events": ["cluster.created", "outlier.detected"],
       "secret": "your_webhook_secret"
     }'
```

#### Webhook Payload Example

```json
{
  "event": "cluster.created",
  "timestamp": "2025-06-07T14:30:45.123Z",
  "data": {
    "cluster_id": 5,
    "centroid": [2.1, 3.5, 4.8],
    "initial_points": 3,
    "confidence": 0.85
  },
  "webhook_id": "wh_123e4567-e89b-12d3-a456-426614174000"
}
```

#### Webhook Verification

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    """Verify webhook signature."""
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(f"sha256={expected_signature}", signature)

# In your webhook handler
def handle_webhook(request):
    signature = request.headers.get('X-NCS-Signature')
    payload = request.body
    
    if verify_webhook(payload, signature, "your_webhook_secret"):
        # Process webhook
        event_data = json.loads(payload)
        print(f"Received event: {event_data['event']}")
    else:
        print("Invalid webhook signature")
```

---

## 📈 API Versioning

### Current Version: v1

The NCS API uses URL-based versioning with semantic versioning principles.

#### Version Format

```
https://api.ncs-cluster.com/api/v{major}/
```

#### Version History

| Version | Status | Release Date | End of Life |
|---------|--------|--------------|-------------|
| **v1** | Current | 2025-01-15 | TBD |

#### Backward Compatibility

- **Minor Updates**: Backward-compatible changes (new optional fields)
- **Major Updates**: Breaking changes require new version
- **Deprecation**: 12-month notice before removing features

#### Version Headers

You can specify the API version in headers:

```http
X-API-Version: v1
Accept: application/vnd.ncs.v1+json
```

---

## 📊 Status Codes

### Success Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Request succeeded |
| 201 | Created | Resource created successfully |
| 202 | Accepted | Request accepted for processing |

### Client Error Codes

| Code | Status | Description |
|------|--------|-------------|
| 400 | Bad Request | Invalid request format |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Request validation failed |
| 429 | Too Many Requests | Rate limit exceeded |

### Server Error Codes

| Code | Status | Description |
|------|--------|-------------|
| 500 | Internal Server Error | Unexpected server error |
| 502 | Bad Gateway | Upstream service error |
| 503 | Service Unavailable | Service temporarily unavailable |
| 504 | Gateway Timeout | Request timeout |

---

## ✅ Best Practices

### 1. Authentication

- **Use API Keys** for server-to-server communication
- **Store securely** - never expose API keys in client-side code
- **Rotate regularly** - rotate API keys every 90 days
- **Use environment variables** for storing credentials

```bash
# Good - Environment variable
export NCS_API_KEY="your_secret_key"

# Bad - Hardcoded in code
api_key = "ncs_1234567890abcdef"  # Never do this!
```

### 2. Request Optimization

- **Batch processing** - send multiple points in single request
- **Optimal batch size** - use 500-2000 points per batch
- **Compression** - enable gzip compression for large requests
- **Connection reuse** - reuse HTTP connections when possible

```python
# Good - Batch processing
points = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = client.process_points(points)

# Bad - Individual requests
for point in points:
    result = client.process_points([point])  # Don't do this!
```

### 3. Error Handling

- **Implement retries** with exponential backoff
- **Handle rate limits** gracefully
- **Log errors** with request IDs for debugging
- **Validate input** before sending to API

```python
import time
import random

def safe_api_call(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            time.sleep(e.retry_after)
        except ServiceError:
            if attempt == max_retries - 1:
                raise
            delay = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
```

### 4. Performance

- **Cache results** when appropriate
- **Use async/await** for high-throughput scenarios
- **Monitor response times** and adjust batch sizes
- **Implement connection pooling** for production

```python
# Good - Async processing
async def process_multiple_batches(batches):
    tasks = [client.process_points(batch) for batch in batches]
    results = await asyncio.gather(*tasks)
    return results
```

### 5. Data Preparation

- **Normalize data** for better clustering results
- **Remove outliers** before processing if appropriate
- **Check data quality** before sending to API
- **Use consistent dimensions** across all points

```python
from sklearn.preprocessing import StandardScaler

# Good - Normalize data
scaler = StandardScaler()
normalized_points = scaler.fit_transform(raw_points)
result = client.process_points(normalized_points.tolist())
```

### 6. Monitoring

- **Track API usage** and quotas
- **Monitor response times** and success rates
- **Set up alerts** for high error rates
- **Log request IDs** for debugging

```python
import logging

logger = logging.getLogger(__name__)

def monitored_api_call(points):
    start_time = time.time()
    try:
        result = client.process_points(points)
        duration = time.time() - start_time
        
        logger.info(f"API call succeeded", extra={
            'request_id': result.get('request_id'),
            'duration_ms': duration * 1000,
            'points_processed': len(points),
            'clusters_found': result.get('clusters_found')
        })
        
        return result
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"API call failed", extra={
            'error': str(e),
            'duration_ms': duration * 1000,
            'points_attempted': len(points)
        })
        raise
```

---

## 🆘 Support

### Documentation & Resources

- **API Documentation**: [https://docs.ncs-cluster.com](https://docs.ncs-cluster.com)
- **Developer Guide**: [https://docs.ncs-cluster.com/dev-guide](https://docs.ncs-cluster.com/dev-guide)
- **Status Page**: [https://status.ncs-cluster.com](https://status.ncs-cluster.com)
- **Changelog**: [https://docs.ncs-cluster.com/changelog](https://docs.ncs-cluster.com/changelog)

### Getting Help

- **Community Forum**: [https://community.ncs-cluster.com](https://community.ncs-cluster.com)
- **GitHub Issues**: [https://github.com/ncs-api/issues](https://github.com/ncs-api/issues)
- **Email Support**: [support@ncs-cluster.com](mailto:support@ncs-cluster.com)
- **Emergency Support**: Available for Enterprise plans

### Response Times

| Plan | Response Time | Channels |
|------|---------------|----------|
| **Free** | 48-72 hours | Community, Email |
| **Pro** | 24 hours | Email, Priority Support |
| **Enterprise** | 4 hours | Phone, Email, Dedicated Support |

---

## 🔄 Changelog

### v1.0.0 (Current)
- Initial API release
- Core clustering endpoints
- Authentication system
- Rate limiting
- Comprehensive error handling

### Upcoming Features
- Streaming WebSocket API
- Advanced algorithm parameters
- Bulk data processing
- Enhanced monitoring

---

**Ready to get started?** [Sign up for your free API key](https://api.ncs-cluster.com/signup) and begin clustering your data in minutes!

For questions or support, reach out to our team at [support@ncs-cluster.com](mailto:support@ncs-cluster.com).