# NeuroCluster Streamer API - Quick Start Guide

Get up and running with the NeuroCluster Streamer API in just 5 minutes! This guide will walk you through the essential steps to start processing your data with our high-performance clustering algorithm.

## 🎯 What You'll Learn

- How to set up and authenticate with the NCS API
- Process your first data points
- Understand the clustering results
- Monitor algorithm performance
- Handle common errors

## 📋 Prerequisites

- **Python 3.11+** or **Node.js 14+** (depending on your preferred SDK)
- **API access** (development server or production endpoint)
- **Basic understanding** of data clustering concepts

## 🚀 Step 1: Choose Your Setup Method

### Option A: Using Docker (Recommended for Testing)

```bash
# Clone and start the API locally
git clone https://github.com/your-org/ncs-api.git
cd ncs-api

# Start with Docker Compose
docker-compose up -d

# Verify it's running
curl http://localhost:8000/health
```

### Option B: Using Hosted API

If you have access to a hosted instance:

```bash
# Test connectivity
curl https://api.yourdomain.com/health

# Should return:
# {
#   "status": "healthy",
#   "timestamp": "2025-01-15T10:30:00Z",
#   "version": "1.0.0",
#   "algorithm_ready": true
# }
```

## 🔐 Step 2: Authentication

Choose your authentication method:

### Method A: API Key (Recommended for Services)

```bash
# Get your API key from the dashboard or admin
export NCS_API_KEY="ncs_1234567890abcdef1234567890abcdef"

# Test authentication
curl -H "X-API-Key: $NCS_API_KEY" http://localhost:8000/api/v1/algorithm_status
```

### Method B: Username/Password (Interactive Use)

```bash
# Get JWT token
curl -X POST "http://localhost:8000/auth/login" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=admin&password=admin123"

# Response contains your token:
# {
#   "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
#   "token_type": "bearer",
#   "expires_in": 1800
# }

export NCS_TOKEN="your-jwt-token-here"
```

## 📊 Step 3: Process Your First Data Points

Now let's process some sample data:

### Using cURL (Any Language)

```bash
# Process a simple dataset
curl -X POST "http://localhost:8000/api/v1/process_points" \
     -H "Authorization: Bearer $NCS_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "points": [
         [1.0, 2.0, 3.0],
         [1.1, 2.1, 3.1],
         [1.2, 2.2, 3.2],
         [5.0, 6.0, 7.0],
         [5.1, 6.1, 7.1],
         [10.0, 11.0, 12.0]
       ]
     }'
```

### Using Python SDK

```python
# Install the SDK first
# pip install ncs-python-sdk

from ncs_client import NCSClient

# Initialize client
client = NCSClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"  # or use JWT token
)

# Your data points (can be any number of dimensions)
points = [
    [1.0, 2.0, 3.0],    # Point 1: close to next two points
    [1.1, 2.1, 3.1],    # Point 2: close to point 1 and 3
    [1.2, 2.2, 3.2],    # Point 3: close to previous points
    [5.0, 6.0, 7.0],    # Point 4: separate cluster
    [5.1, 6.1, 7.1],    # Point 5: close to point 4
    [10.0, 11.0, 12.0]  # Point 6: outlier/separate
]

# Process the points
result = client.process_points(points)

# Print results
print(f"✅ Processing completed!")
print(f"📊 Found {len(result.clusters)} clusters")
print(f"🎯 Processing time: {result.processing_time_ms:.2f}ms")
print(f"📈 Algorithm quality: {result.algorithm_quality:.3f}")

# Show cluster details
for i, cluster in enumerate(result.clusters):
    print(f"\n🔵 Cluster {cluster.id}:")
    print(f"   Center: {cluster.center}")
    print(f"   Size: {cluster.size} points")
    print(f"   Quality: {cluster.quality:.3f}")
    print(f"   Points: {cluster.points}")

# Show outliers
if result.outliers:
    print(f"\n🔴 Outliers ({len(result.outliers)}):")
    for outlier in result.outliers:
        print(f"   {outlier}")
```

### Using JavaScript SDK

```javascript
// Install: npm install ncs-javascript-sdk

import { NCSClient } from 'ncs-javascript-sdk';

// Initialize client
const client = new NCSClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

async function quickStart() {
  try {
    // Your data points
    const points = [
      [1.0, 2.0, 3.0],
      [1.1, 2.1, 3.1], 
      [1.2, 2.2, 3.2],
      [5.0, 6.0, 7.0],
      [5.1, 6.1, 7.1],
      [10.0, 11.0, 12.0]
    ];

    // Process the points
    const result = await client.processPoints(points);

    console.log('✅ Processing completed!');
    console.log(`📊 Found ${result.clusters.length} clusters`);
    console.log(`🎯 Processing time: ${result.processingTimeMs.toFixed(2)}ms`);
    console.log(`📈 Algorithm quality: ${result.algorithmQuality.toFixed(3)}`);

    // Show cluster details
    result.clusters.forEach((cluster, i) => {
      console.log(`\n🔵 Cluster ${cluster.id}:`);
      console.log(`   Center: [${cluster.center.join(', ')}]`);
      console.log(`   Size: ${cluster.size} points`);
      console.log(`   Quality: ${cluster.quality.toFixed(3)}`);
    });

    // Show outliers
    if (result.outliers.length > 0) {
      console.log(`\n🔴 Outliers (${result.outliers.length}):`);
      result.outliers.forEach(outlier => {
        console.log(`   [${outlier.join(', ')}]`);
      });
    }

  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

quickStart();
```

## 📋 Step 4: Understanding the Results

The API returns a structured response with these key components:

### Clusters
Each cluster contains:
- **`id`**: Unique cluster identifier
- **`center`**: The computed center point of the cluster
- **`points`**: All data points assigned to this cluster
- **`size`**: Number of points in the cluster
- **`quality`**: Quality score (0.0-1.0, higher is better)

### Outliers
Points that don't fit well into any cluster are returned as outliers.

### Metadata
- **`processing_time_ms`**: How long the processing took
- **`algorithm_quality`**: Overall clustering quality score
- **`request_id`**: Unique identifier for this request

### Example Response

```json
{
  "clusters": [
    {
      "id": 1,
      "center": [1.1, 2.1, 3.1],
      "points": [
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [1.2, 2.2, 3.2]
      ],
      "size": 3,
      "quality": 0.92
    },
    {
      "id": 2,
      "center": [5.05, 6.05, 7.05],
      "points": [
        [5.0, 6.0, 7.0],
        [5.1, 6.1, 7.1]
      ],
      "size": 2,
      "quality": 0.89
    }
  ],
  "outliers": [
    [10.0, 11.0, 12.0]
  ],
  "processing_time_ms": 1.2,
  "algorithm_quality": 0.91,
  "request_id": "req_abc123"
}
```

## 🔍 Step 5: Monitor Algorithm Performance

Check the current status of the clustering algorithm:

### Python

```python
# Get algorithm status
status = client.get_algorithm_status()

print(f"🟢 Algorithm ready: {status.is_ready}")
print(f"📊 Active clusters: {status.active_clusters}")
print(f"🎯 Clustering quality: {status.clustering_quality:.3f}")
print(f"💾 Memory usage: {status.memory_usage_mb:.1f}MB")
print(f"⚡ Average processing time: {status.average_processing_time_ms:.2f}ms")
print(f"📈 Throughput: {status.throughput_points_per_second:.0f} points/sec")
```

### JavaScript

```javascript
const status = await client.getAlgorithmStatus();

console.log(`🟢 Algorithm ready: ${status.isReady}`);
console.log(`📊 Active clusters: ${status.activeClusters}`);
console.log(`🎯 Clustering quality: ${status.clusteringQuality.toFixed(3)}`);
console.log(`💾 Memory usage: ${status.memoryUsageMb.toFixed(1)}MB`);
console.log(`⚡ Average processing time: ${status.averageProcessingTimeMs.toFixed(2)}ms`);
```

### cURL

```bash
curl -H "Authorization: Bearer $NCS_TOKEN" \
     http://localhost:8000/api/v1/algorithm_status | jq
```

## 🛠️ Step 6: Handle Common Scenarios

### Checking API Health

```python
# Python
health = client.health_check()
print(f"Status: {health.status}")
print(f"Version: {health.version}")
print(f"Uptime: {health.uptime_seconds}s")
```

```javascript
// JavaScript
const health = await client.healthCheck();
console.log(`Status: ${health.status}`);
console.log(`Version: ${health.version}`);
```

### Error Handling

```python
# Python
from ncs_client import NCSError, AuthenticationError, RateLimitError

try:
    result = client.process_points(points)
except AuthenticationError:
    print("❌ Authentication failed - check your credentials")
except RateLimitError as e:
    print(f"⏸️ Rate limited - retry after {e.retry_after} seconds")
except NCSError as e:
    print(f"🚫 API error: {e.message}")
```

```javascript
// JavaScript
import { AuthenticationError, RateLimitError, ValidationError } from 'ncs-javascript-sdk';

try {
  const result = await client.processPoints(points);
} catch (error) {
  if (error instanceof AuthenticationError) {
    console.error('❌ Authentication failed - check your credentials');
  } else if (error instanceof RateLimitError) {
    console.error(`⏸️ Rate limited - retry after ${error.retryAfter}s`);
  } else if (error instanceof ValidationError) {
    console.error(`🚫 Validation error: ${error.message}`);
  } else {
    console.error(`💥 Unexpected error: ${error.message}`);
  }
}
```

## 📝 Step 7: Try Different Data Types

### 2D Data (Good for Visualization)

```python
# Example: Customer segmentation data
customers = [
    [25, 50000],    # Age, Income
    [30, 55000],
    [28, 52000],
    [45, 80000],
    [42, 75000],
    [50, 85000],
    [22, 25000],
    [24, 28000]
]

result = client.process_points(customers)
print(f"Customer segments found: {len(result.clusters)}")
```

### High-Dimensional Data

```python
# Example: Feature vectors from machine learning
import random

# Generate random 10-dimensional data
high_dim_data = [
    [random.random() for _ in range(10)]
    for _ in range(100)
]

result = client.process_points(high_dim_data)
print(f"Clusters in high-dimensional space: {len(result.clusters)}")
```

### Time Series Features

```python
# Example: Extracted features from time series
time_series_features = [
    [2.1, 0.5, 1.2],  # [mean, std, trend]
    [2.0, 0.4, 1.1],
    [5.2, 1.1, -0.2],
    [5.1, 1.0, -0.1],
    [8.5, 2.1, 0.8]
]

result = client.process_points(time_series_features)
print(f"Time series patterns: {len(result.clusters)}")
```

## 🎉 Congratulations!

You've successfully:
- ✅ Set up the NCS API
- ✅ Authenticated with the service
- ✅ Processed your first data points
- ✅ Understood the clustering results
- ✅ Monitored algorithm performance
- ✅ Handled common errors

## 🚀 Next Steps

Now that you've got the basics down, you're ready to:

1. **[Advanced Usage](advanced_usage.md)** - Learn streaming, batch processing, and optimization
2. **[Production Setup](production_setup.md)** - Deploy for production workloads
3. **[API Reference](../API_REFERENCE.md)** - Explore all available endpoints
4. **[Security Guide](../SECURITY_GUIDE.md)** - Implement proper security measures

## 🤔 Common Questions

### Q: How many points can I process at once?
**A:** The API supports up to 10,000 points per request. For larger datasets, use batch processing or streaming.

### Q: What data formats are supported?
**A:** Points should be arrays of numbers. Each point can have 1-1000 dimensions. Missing values are not supported.

### Q: How fast is the processing?
**A:** Typical performance is >6,300 points/second with sub-millisecond latency per point.

### Q: Can I customize the algorithm parameters?
**A:** Yes! See the [Advanced Usage guide](advanced_usage.md) for parameter tuning.

### Q: Is real-time streaming supported?
**A:** Absolutely! Check out the streaming examples in the [Advanced Usage guide](advanced_usage.md).

## 🆘 Need Help?

- 📖 **Documentation**: [Complete API docs](../API_REFERENCE.md)
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-org/ncs-api/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-org/ncs-api/discussions)
- 📧 **Support**: support@yourdomain.com

---

**Happy clustering! 🎯** Your data insights journey starts here.