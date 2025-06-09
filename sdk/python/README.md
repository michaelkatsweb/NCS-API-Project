# NeuroCluster Streamer Python SDK

Official Python client library for the NeuroCluster Streamer API.

## Features

- **Synchronous and Asynchronous** client support
- **Comprehensive Error Handling** with retry logic
- **Authentication** via JWT tokens and API keys
- **Type Safety** with full type hints
- **Streaming Support** for real-time data processing
- **Batch Processing** for high-throughput scenarios
- **Configurable Timeouts** and connection management
- **Built-in Logging** and debugging capabilities

## Installation

```bash
pip install ncs-python-sdk
```

### Development Installation

```bash
git clone https://github.com/your-org/ncs-api.git
cd ncs-api/sdk/python
pip install -e .
```

## Quick Start

### Basic Usage

```python
from ncs_client import NCSClient

# Initialize client
client = NCSClient(
    base_url="https://api.yourdomain.com",
    api_key="your-api-key"
)

# Process data points
points = [
    [1.0, 2.0, 3.0],
    [1.1, 2.1, 3.1],
    [5.0, 6.0, 7.0]
]

result = client.process_points(points)
print(f"Processed {len(result.clusters)} clusters")
```

### JWT Authentication

```python
from ncs_client import NCSClient

# Authenticate with username/password
client = NCSClient(base_url="https://api.yourdomain.com")
client.authenticate(username="user", password="password")

# Use the client
status = client.get_algorithm_status()
print(f"Algorithm quality: {status.clustering_quality}")
```

### Async Usage

```python
import asyncio
from ncs_client import AsyncNCSClient

async def main():
    async with AsyncNCSClient(
        base_url="https://api.yourdomain.com",
        api_key="your-api-key"
    ) as client:
        result = await client.process_points([[1, 2, 3]])
        print(result)

asyncio.run(main())
```

### Streaming Data Processing

```python
from ncs_client import NCSClient

client = NCSClient(
    base_url="https://api.yourdomain.com",
    api_key="your-api-key"
)

# Stream processing with callback
def on_cluster_update(cluster_data):
    print(f"Cluster updated: {cluster_data}")

# Start streaming
stream = client.start_streaming(callback=on_cluster_update)

# Send points to stream
for point in data_stream:
    stream.send_point(point)

# Stop streaming
stream.stop()
```

### Batch Processing

```python
from ncs_client import NCSClient
import pandas as pd

client = NCSClient(
    base_url="https://api.yourdomain.com",
    api_key="your-api-key"
)

# Load large dataset
df = pd.read_csv('large_dataset.csv')
points = df[['x', 'y', 'z']].values.tolist()

# Process in batches
batch_size = 1000
results = []

for i in range(0, len(points), batch_size):
    batch = points[i:i + batch_size]
    result = client.process_points_batch(
        batch,
        batch_options={'timeout': 30}
    )
    results.append(result)
    print(f"Processed batch {i//batch_size + 1}")

# Combine results
final_result = client.combine_batch_results(results)
```

## Configuration

### Environment Variables

```bash
export NCS_API_URL="https://api.yourdomain.com"
export NCS_API_KEY="your-api-key"
export NCS_TIMEOUT=30
export NCS_MAX_RETRIES=3
```

### Configuration File

```python
# config.py
NCS_CONFIG = {
    'base_url': 'https://api.yourdomain.com',
    'api_key': 'your-api-key',
    'timeout': 30,
    'max_retries': 3,
    'retry_delay': 1.0,
    'verify_ssl': True,
    'log_level': 'INFO'
}

# Usage
from ncs_client import NCSClient
client = NCSClient.from_config(NCS_CONFIG)
```

## Error Handling

```python
from ncs_client import NCSClient, NCSError, AuthenticationError, RateLimitError

client = NCSClient(base_url="https://api.yourdomain.com")

try:
    result = client.process_points(points)
except AuthenticationError:
    print("Authentication failed - check credentials")
except RateLimitError as e:
    print(f"Rate limited - retry after {e.retry_after} seconds")
except NCSError as e:
    print(f"API error: {e.message}")
    print(f"Error code: {e.error_code}")
    print(f"Request ID: {e.request_id}")
```

## Advanced Features

### Custom Headers and Parameters

```python
client = NCSClient(
    base_url="https://api.yourdomain.com",
    api_key="your-api-key",
    headers={'X-Custom-Header': 'value'},
    default_params={'version': 'v2'}
)
```

### Connection Pooling

```python
from ncs_client import NCSClient

# Configure connection pooling
client = NCSClient(
    base_url="https://api.yourdomain.com",
    api_key="your-api-key",
    pool_connections=10,
    pool_maxsize=20,
    pool_block=True
)
```

### Logging Configuration

```python
import logging
from ncs_client import NCSClient

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

client = NCSClient(
    base_url="https://api.yourdomain.com",
    api_key="your-api-key",
    log_level='DEBUG'
)
```

## API Reference

### NCSClient

Main synchronous client for the NCS API.

#### Methods

- `process_points(points, options=None)` - Process data points
- `get_clusters_summary(filters=None)` - Get cluster summary
- `get_algorithm_status()` - Get algorithm status
- `authenticate(username, password)` - Authenticate with JWT
- `health_check()` - Check API health

### AsyncNCSClient

Asynchronous client for high-performance applications.

#### Methods

- `async process_points(points, options=None)` - Process data points
- `async get_clusters_summary(filters=None)` - Get cluster summary
- `async get_algorithm_status()` - Get algorithm status
- `async authenticate(username, password)` - Authenticate with JWT

### Data Models

#### Point
```python
from typing import List, Union
Point = List[Union[int, float]]  # [x, y, z, ...]
```

#### ProcessingResult
```python
@dataclass
class ProcessingResult:
    clusters: List[Cluster]
    outliers: List[Point]
    processing_time_ms: float
    algorithm_quality: float
    request_id: str
```

#### Cluster
```python
@dataclass
class Cluster:
    id: int
    center: Point
    points: List[Point]
    size: int
    quality: float
```

## Examples

See the `examples/` directory for complete working examples:

- `basic_usage.py` - Simple point processing
- `streaming_example.py` - Real-time streaming
- `batch_processing.py` - Large dataset processing
- `error_handling.py` - Comprehensive error handling
- `async_example.py` - Asynchronous operations

## Testing

```bash
# Install test dependencies
pip install -e ".[test]"

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=ncs_client --cov-report=html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: https://docs.ncs-api.com
- **Issues**: https://github.com/your-org/ncs-api/issues
- **Email**: support@yourdomain.com

## Changelog

### v1.0.0
- Initial release
- Synchronous and asynchronous clients
- JWT and API key authentication
- Streaming and batch processing support
- Comprehensive error handling