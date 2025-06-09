# NeuroCluster Streamer API Reference

Complete reference documentation for the NeuroCluster Streamer API endpoints, data models, authentication, and integration patterns.

## 📋 Table of Contents

- [Base Information](#-base-information)
- [Authentication](#-authentication)
- [Core Endpoints](#-core-endpoints)
- [WebSocket Streaming](#-websocket-streaming)
- [Data Models](#-data-models)
- [Error Handling](#-error-handling)
- [Rate Limiting](#-rate-limiting)
- [Examples](#-examples)

## 🌐 Base Information

### API Base URL
```
Production: https://api.yourdomain.com
Development: http://localhost:8000
```

### API Version
- **Current Version**: `v1`
- **API Prefix**: `/api/v1`
- **Protocol**: `HTTPS` (production), `HTTP` (development)

### Content Types
- **Request**: `application/json`
- **Response**: `application/json`
- **Authentication**: `application/x-www-form-urlencoded` (login only)

### Standards Compliance
- **OpenAPI 3.0.3**: Full specification available at `/docs`
- **RFC 7519**: JWT token format
- **RFC 6750**: OAuth 2.0 Bearer Token Usage

## 🔐 Authentication

The NCS API supports two authentication methods: JWT tokens for user authentication and API keys for service-to-service communication.

### JWT Authentication

#### Login Endpoint
```http
POST /auth/login
Content-Type: application/x-www-form-urlencoded

username=your_username&password=your_password
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "scope": "read write"
}
```

#### Using JWT Tokens
Include the JWT token in the Authorization header:
```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

### API Key Authentication

Include the API key in the request header:
```http
X-API-Key: your-api-key-here
```

### Authentication Examples

#### cURL - JWT
```bash
# Login to get token
TOKEN=$(curl -s -X POST "http://localhost:8000/auth/login" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=admin&password=admin123" | jq -r '.access_token')

# Use token in requests
curl -X GET "http://localhost:8000/api/v1/algorithm_status" \
     -H "Authorization: Bearer $TOKEN"
```

#### cURL - API Key
```bash
curl -X GET "http://localhost:8000/api/v1/algorithm_status" \
     -H "X-API-Key: your-api-key"
```

#### Python
```python
import requests

# JWT Authentication
response = requests.post('http://localhost:8000/auth/login', data={
    'username': 'admin',
    'password': 'admin123'
})
token = response.json()['access_token']

headers = {'Authorization': f'Bearer {token}'}

# API Key Authentication
headers = {'X-API-Key': 'your-api-key'}
```

#### JavaScript
```javascript
// JWT Authentication
const response = await fetch('http://localhost:8000/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  body: 'username=admin&password=admin123'
});
const { access_token } = await response.json();

const headers = { 'Authorization': `Bearer ${access_token}` };

// API Key Authentication
const headers = { 'X-API-Key': 'your-api-key' };
```

## 🔧 Core Endpoints

### Health Check

#### `GET /health`

Check the overall health and status of the API service.

**Authentication:** None required

**Response Schema:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00Z",
  "version": "1.0.0",
  "algorithm_ready": true,
  "uptime_seconds": 86400.5,
  "components": {
    "database": "healthy",
    "redis": "healthy",
    "algorithm": "healthy"
  }
}
```

**Example:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Status Codes:**
- `200`: Service is healthy
- `503`: Service is unhealthy or degraded

---

### Process Data Points

#### `POST /api/v1/process_points`

Process a collection of data points through the clustering algorithm.

**Authentication:** Required (JWT or API Key)

**Request Schema:**
```json
{
  "points": [
    [1.0, 2.0, 3.0],
    [1.1, 2.1, 3.1],
    [5.0, 6.0, 7.0]
  ],
  "batch_mode": false,
  "timeout": 30000,
  "clustering_config": {
    "threshold": 0.71,
    "max_clusters": 30
  }
}
```

**Request Parameters:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `points` | `Array<Array<number>>` | ✅ | Array of data points (each point is an array of numbers) |
| `batch_mode` | `boolean` | ❌ | Enable batch processing mode (default: false) |
| `timeout` | `integer` | ❌ | Request timeout in milliseconds (default: 30000) |
| `clustering_config` | `object` | ❌ | Optional clustering configuration overrides |

**Response Schema:**
```json
{
  "clusters": [
    {
      "id": 1,
      "center": [1.05, 2.05, 3.05],
      "points": [
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1]
      ],
      "size": 2,
      "quality": 0.95,
      "created_at": "2025-01-15T10:30:00Z",
      "last_updated": "2025-01-15T10:30:00Z"
    }
  ],
  "outliers": [
    [5.0, 6.0, 7.0]
  ],
  "processing_time_ms": 15.2,
  "algorithm_quality": 0.92,
  "request_id": "req_12345678",
  "total_points": 3
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/process_points" \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "points": [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [5.0, 6.0, 7.0]],
       "batch_mode": false
     }'
```

**Status Codes:**
- `200`: Points processed successfully
- `400`: Invalid request data
- `401`: Authentication required
- `413`: Request payload too large
- `422`: Validation error
- `429`: Rate limit exceeded
- `500`: Processing error

---

### Get Clusters Summary

#### `GET /api/v1/clusters_summary`

Retrieve a summary of current clusters with optional filtering.

**Authentication:** Required (JWT or API Key)

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `min_size` | `integer` | ❌ | Minimum cluster size to include |
| `quality_threshold` | `float` | ❌ | Minimum quality score (0.0-1.0) |
| `cluster_id` | `integer` | ❌ | Specific cluster ID to retrieve |
| `health_status` | `string` | ❌ | Filter by health status (healthy, warning, critical) |
| `limit` | `integer` | ❌ | Maximum number of clusters to return (default: 50) |
| `offset` | `integer` | ❌ | Pagination offset (default: 0) |

**Response Schema:**
```json
{
  "num_active_clusters": 5,
  "cluster_ids": [1, 2, 3, 4, 5],
  "clusters_info": [
    {
      "cluster_id": 1,
      "size": 150,
      "center": [1.5, 2.5, 3.5],
      "confidence": 0.95,
      "health_status": "healthy",
      "quality_score": 0.92,
      "drift_coefficient": 0.02,
      "last_updated": "2025-01-15T10:30:00Z",
      "created_at": "2025-01-15T09:00:00Z"
    }
  ],
  "total_points_processed": 1000,
  "average_cluster_confidence": 0.89,
  "cluster_distribution": {
    "healthy": 4,
    "warning": 1,
    "critical": 0
  },
  "summary_timestamp": "2025-01-15T10:30:00Z"
}
```

**Example:**
```bash
curl -X GET "http://localhost:8000/api/v1/clusters_summary?min_size=10&quality_threshold=0.8" \
     -H "Authorization: Bearer $TOKEN"
```

**Status Codes:**
- `200`: Summary retrieved successfully
- `401`: Authentication required
- `422`: Invalid query parameters

---

### Get Algorithm Status

#### `GET /api/v1/algorithm_status`

Retrieve detailed status and performance metrics of the clustering algorithm.

**Authentication:** Required (JWT or API Key)

**Response Schema:**
```json
{
  "is_ready": true,
  "current_dynamic_threshold": 0.73,
  "clustering_quality": 0.92,
  "global_stability": 0.85,
  "adaptation_rate": 1.2,
  "total_points_processed": 50000,
  "average_processing_time_ms": 0.22,
  "max_processing_time_ms": 45.1,
  "memory_usage_mb": 12.4,
  "uptime_seconds": 86400.5,
  "throughput_points_per_second": 6309.2,
  "error_rate": 0.001,
  "drift_detected": false,
  "active_clusters": 8,
  "last_drift_detection": "2025-01-15T09:15:00Z",
  "performance_metrics": {
    "p50_latency_ms": 0.15,
    "p95_latency_ms": 0.35,
    "p99_latency_ms": 1.2
  }
}
```

**Example:**
```bash
curl -X GET "http://localhost:8000/api/v1/algorithm_status" \
     -H "Authorization: Bearer $TOKEN"
```

**Status Codes:**
- `200`: Status retrieved successfully
- `401`: Authentication required
- `503`: Algorithm not ready

---

### Batch Processing

#### `POST /api/v1/process_batch`

Process large batches of data points with optimized throughput.

**Authentication:** Required (JWT or API Key)

**Request Schema:**
```json
{
  "batches": [
    {
      "batch_id": "batch_001",
      "points": [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]
    },
    {
      "batch_id": "batch_002", 
      "points": [[5.0, 6.0, 7.0], [5.1, 6.1, 7.1]]
    }
  ],
  "processing_options": {
    "parallel": true,
    "max_concurrent": 4,
    "timeout_per_batch": 60000
  }
}
```

**Response Schema:**
```json
{
  "batch_results": [
    {
      "batch_id": "batch_001",
      "status": "success",
      "clusters": [...],
      "outliers": [...],
      "processing_time_ms": 25.4
    }
  ],
  "total_batches": 2,
  "successful_batches": 2,
  "failed_batches": 0,
  "total_processing_time_ms": 45.8,
  "request_id": "req_batch_123"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/process_batch" \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "batches": [
         {"batch_id": "b1", "points": [[1,2,3], [1.1,2.1,3.1]]},
         {"batch_id": "b2", "points": [[5,6,7], [5.1,6.1,7.1]]}
       ]
     }'
```

## 🔄 WebSocket Streaming

### Stream Endpoint

#### `WS /ws/stream`

Establish a WebSocket connection for real-time data point streaming and clustering updates.

**Authentication:** Query parameter (JWT token or API key)

**Connection URL:**
```
ws://localhost:8000/ws/stream?token=your-jwt-token
# or
ws://localhost:8000/ws/stream?api_key=your-api-key
```

### Message Types

#### Client → Server Messages

**Send Single Point:**
```json
{
  "type": "point",
  "data": [1.0, 2.0, 3.0],
  "metadata": {
    "source": "sensor_01",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

**Send Batch:**
```json
{
  "type": "batch",
  "data": [
    [1.0, 2.0, 3.0],
    [1.1, 2.1, 3.1]
  ],
  "batch_id": "batch_stream_001"
}
```

**Request Status:**
```json
{
  "type": "status_request"
}
```

#### Server → Client Messages

**Cluster Update:**
```json
{
  "type": "cluster_update",
  "data": {
    "cluster_id": 1,
    "center": [1.05, 2.05, 3.05],
    "size": 15,
    "quality": 0.92
  },
  "timestamp": "2025-01-15T10:30:00Z"
}
```

**Algorithm Status:**
```json
{
  "type": "algorithm_status",
  "data": {
    "clustering_quality": 0.92,
    "active_clusters": 8,
    "drift_detected": false
  },
  "timestamp": "2025-01-15T10:30:00Z"
}
```

**Error Message:**
```json
{
  "type": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid data point format",
    "details": "Point must be array of numbers"
  },
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### WebSocket Examples

#### JavaScript
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream?token=' + jwtToken);

ws.onopen = () => {
  console.log('WebSocket connected');
  
  // Send a data point
  ws.send(JSON.stringify({
    type: 'point',
    data: [1.0, 2.0, 3.0]
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

#### Python
```python
import asyncio
import websockets
import json

async def stream_client():
    uri = "ws://localhost:8000/ws/stream?token=" + jwt_token
    
    async with websockets.connect(uri) as websocket:
        # Send a data point
        await websocket.send(json.dumps({
            "type": "point",
            "data": [1.0, 2.0, 3.0]
        }))
        
        # Listen for responses
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data}")

asyncio.run(stream_client())
```

## 📊 Data Models

### Core Data Types

#### Point
```typescript
type Point = number[];  // [x, y, z, ...]
```

#### Cluster
```json
{
  "id": 1,
  "center": [1.05, 2.05, 3.05],
  "points": [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]],
  "size": 2,
  "quality": 0.95,
  "created_at": "2025-01-15T10:30:00Z",
  "last_updated": "2025-01-15T10:30:00Z"
}
```

#### Processing Result
```json
{
  "clusters": [Cluster],
  "outliers": [Point],
  "processing_time_ms": 15.2,
  "algorithm_quality": 0.92,
  "request_id": "req_12345678",
  "total_points": 10
}
```

#### Algorithm Status
```json
{
  "is_ready": true,
  "current_dynamic_threshold": 0.73,
  "clustering_quality": 0.92,
  "global_stability": 0.85,
  "adaptation_rate": 1.2,
  "total_points_processed": 50000,
  "average_processing_time_ms": 0.22,
  "memory_usage_mb": 12.4,
  "uptime_seconds": 86400.5,
  "throughput_points_per_second": 6309.2,
  "error_rate": 0.001,
  "drift_detected": false
}
```

### Validation Rules

#### Data Points
- Must be arrays of numbers
- Minimum 1 dimension, maximum 1000 dimensions
- Numbers must be finite (no NaN, Infinity)
- Maximum 10,000 points per request

#### Clustering Configuration
- `threshold`: 0.1 ≤ value ≤ 1.0
- `max_clusters`: 1 ≤ value ≤ 100
- `learning_rate`: 0.01 ≤ value ≤ 0.2

## ❌ Error Handling

### Standard Error Response
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request data",
    "details": {
      "field": "points",
      "issue": "Must be array of arrays"
    }
  },
  "request_id": "req_12345678",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|------------|-------------|
| `VALIDATION_ERROR` | 400/422 | Request validation failed |
| `AUTHENTICATION_ERROR` | 401 | Authentication required or failed |
| `AUTHORIZATION_ERROR` | 403 | Insufficient permissions |
| `RATE_LIMIT_ERROR` | 429 | Rate limit exceeded |
| `PROCESSING_ERROR` | 500 | Algorithm processing failed |
| `RESOURCE_ERROR` | 503 | Insufficient system resources |
| `CONFIGURATION_ERROR` | 500 | Invalid system configuration |

### Rate Limiting Errors
```json
{
  "error": {
    "code": "RATE_LIMIT_ERROR",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 100,
      "window": "60s",
      "retry_after": 45
    }
  }
}
```

### Validation Errors
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": [
      {
        "field": "points.0",
        "message": "Point must contain at least 1 dimension"
      },
      {
        "field": "threshold",
        "message": "Value must be between 0.1 and 1.0"
      }
    ]
  }
}
```

## 🚦 Rate Limiting

### Default Limits

| Endpoint | Requests/Minute | Requests/Hour |
|----------|----------------|---------------|
| `/health` | 600 | 10,000 |
| `/api/v1/process_points` | 100 | 2,000 |
| `/api/v1/clusters_summary` | 300 | 5,000 |
| `/api/v1/algorithm_status` | 600 | 10,000 |
| `/auth/login` | 10 | 100 |

### Rate Limit Headers

**Response Headers:**
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 85
X-RateLimit-Reset: 1642248000
X-RateLimit-Window: 60
```

### Rate Limit Exceeded Response
```http
HTTP/1.1 429 Too Many Requests
Retry-After: 30
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1642248030

{
  "error": {
    "code": "RATE_LIMIT_ERROR",
    "message": "Rate limit exceeded",
    "retry_after": 30
  }
}
```

## 💡 Examples

### Complete Integration Example

#### Python Client
```python
import requests
import json
import time

class NCSClient:
    def __init__(self, base_url, api_key=None):
        self.base_url = base_url
        self.headers = {'X-API-Key': api_key} if api_key else {}
    
    def authenticate(self, username, password):
        response = requests.post(f'{self.base_url}/auth/login', data={
            'username': username,
            'password': password
        })
        token = response.json()['access_token']
        self.headers['Authorization'] = f'Bearer {token}'
        return token
    
    def process_points(self, points):
        response = requests.post(
            f'{self.base_url}/api/v1/process_points',
            headers=self.headers,
            json={'points': points}
        )
        return response.json()
    
    def get_status(self):
        response = requests.get(
            f'{self.base_url}/api/v1/algorithm_status',
            headers=self.headers
        )
        return response.json()

# Usage
client = NCSClient('http://localhost:8000')
client.authenticate('admin', 'admin123')

# Process some data
points = [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [5.0, 6.0, 7.0]]
result = client.process_points(points)

print(f"Found {len(result['clusters'])} clusters")
print(f"Processing time: {result['processing_time_ms']}ms")

# Check algorithm status
status = client.get_status()
print(f"Algorithm quality: {status['clustering_quality']}")
```

#### JavaScript Client
```javascript
class NCSClient {
  constructor(baseUrl, apiKey = null) {
    this.baseUrl = baseUrl;
    this.headers = apiKey ? { 'X-API-Key': apiKey } : {};
  }
  
  async authenticate(username, password) {
    const response = await fetch(`${this.baseUrl}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: `username=${username}&password=${password}`
    });
    
    const data = await response.json();
    this.headers['Authorization'] = `Bearer ${data.access_token}`;
    return data.access_token;
  }
  
  async processPoints(points) {
    const response = await fetch(`${this.baseUrl}/api/v1/process_points`, {
      method: 'POST',
      headers: { ...this.headers, 'Content-Type': 'application/json' },
      body: JSON.stringify({ points })
    });
    
    return response.json();
  }
  
  async getStatus() {
    const response = await fetch(`${this.baseUrl}/api/v1/algorithm_status`, {
      headers: this.headers
    });
    
    return response.json();
  }
}

// Usage
const client = new NCSClient('http://localhost:8000');
await client.authenticate('admin', 'admin123');

const points = [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [5.0, 6.0, 7.0]];
const result = await client.processPoints(points);

console.log(`Found ${result.clusters.length} clusters`);
console.log(`Processing time: ${result.processing_time_ms}ms`);
```

### Streaming Example

```javascript
class StreamingNCSClient {
  constructor(baseUrl, token) {
    this.wsUrl = baseUrl.replace('http', 'ws') + '/ws/stream';
    this.token = token;
    this.ws = null;
  }
  
  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(`${this.wsUrl}?token=${this.token}`);
      
      this.ws.onopen = () => resolve();
      this.ws.onerror = (error) => reject(error);
      
      this.ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        this.handleMessage(message);
      };
    });
  }
  
  sendPoint(point) {
    this.ws.send(JSON.stringify({
      type: 'point',
      data: point
    }));
  }
  
  handleMessage(message) {
    switch (message.type) {
      case 'cluster_update':
        console.log('Cluster updated:', message.data);
        break;
      case 'algorithm_status':
        console.log('Algorithm status:', message.data);
        break;
      case 'error':
        console.error('Error:', message.error);
        break;
    }
  }
}

// Usage
const streamClient = new StreamingNCSClient('ws://localhost:8000', jwtToken);
await streamClient.connect();

// Send data points in real-time
setInterval(() => {
  const randomPoint = [Math.random(), Math.random(), Math.random()];
  streamClient.sendPoint(randomPoint);
}, 1000);
```

---

For more examples and advanced usage patterns, see the [examples documentation](examples/) directory.