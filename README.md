# NCS-API-Project: High-Performance Clustering Algorithm

A high-performance clustering and data processing API with real-time streaming capabilities, designed for processing large datasets with exceptional speed and accuracy.

## 🚀 Features

- **High Performance**: Process 6,300+ points per second
- **Superior Quality**: 91.8% clustering quality score
- **Real-time Streaming**: WebSocket support for live data processing
- **Multiple SDKs**: Python and JavaScript client libraries
- **Production Ready**: Comprehensive monitoring, health checks, and security
- **Scalable**: Designed for horizontal scaling and cloud deployment

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Processing Speed | 6,300+ points/second |
| Clustering Quality | 91.8% accuracy |
| Latency | < 50ms average response time |
| Memory Usage | < 512MB for 10K points |

## 🛠️ Quick Start

### Prerequisites

- Python 3.9+
- 4GB RAM minimum
- Redis (optional, for caching)

### Installation

```bash
# Clone the repository
git clone https://github.com/michaelkatsweb/NCS-API-Project.git
cd NCS-API-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Docker Setup

```bash
# Build and run with Docker
docker build -t ncs-api .
docker run -p 8000:8000 ncs-api
```

### Docker Compose (with Redis)

```bash
docker-compose up -d
```

## 📚 API Documentation

Once running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json
- **Health Check**: http://localhost:8000/health

## 🔧 Configuration

Create a `.env` file:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Algorithm Settings
MAX_CLUSTERS=100
QUALITY_THRESHOLD=0.85
BATCH_SIZE=1000

# Redis (optional)
REDIS_URL=redis://localhost:6379

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO
```

## 🎯 API Endpoints

### Core Clustering API

```bash
# Process clustering
POST /api/v1/cluster/process
Content-Type: application/json

{
  "data": [[1, 2], [3, 4], [5, 6]],
  "algorithm": "ncs",
  "params": {
    "n_clusters": 3,
    "quality_threshold": 0.9
  }
}

# Get job status
GET /api/v1/cluster/status/{job_id}

# Stream real-time results
WS /ws/stream/cluster
```

### System Endpoints

```bash
# Health check
GET /health

# Metrics
GET /metrics

# API version
GET /api/v1/version
```

## 📦 Client SDKs

### Python SDK

```python
from ncs_client import NCSClient

client = NCSClient("http://localhost:8000")
result = await client.cluster(data=[[1, 2], [3, 4]], algorithm="ncs")
print(f"Clusters: {result.clusters}")
```

### JavaScript SDK

```javascript
import { NCSClient } from 'ncs-client-js';

const client = new NCSClient('http://localhost:8000');
const result = await client.cluster({
  data: [[1, 2], [3, 4]],
  algorithm: 'ncs'
});
console.log('Clusters:', result.clusters);
```

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Load testing
python benchmarks/load_test.py

# Integration tests
pytest tests/integration/
```

## 📈 Performance Benchmarking

```bash
# Run performance benchmarks
python benchmarks/performance_test.py

# Custom benchmark
python benchmarks/custom_benchmark.py --points 10000 --iterations 100
```

## 🚀 Deployment

### Production Setup

```bash
# Install production dependencies
pip install -r requirements-prod.txt

# Run with gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker

# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Kubernetes Deployment

```bash
kubectl apply -f k8s/
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `WORKERS` | Number of workers | `1` |
| `MAX_CLUSTERS` | Maximum clusters allowed | `100` |
| `REDIS_URL` | Redis connection URL | `None` |

## 📊 Monitoring

The API includes built-in monitoring:

- **Prometheus metrics** at `/metrics`
- **Health checks** at `/health` and `/ready`
- **Structured logging** with correlation IDs
- **Performance tracking** for all endpoints

### Key Metrics

- `ncs_requests_total` - Total requests processed
- `ncs_processing_duration_seconds` - Processing time histogram
- `ncs_active_connections` - Active WebSocket connections
- `ncs_cluster_quality_score` - Clustering quality metrics

## 🔒 Security

- **Rate limiting** (100 requests/minute by default)
- **API key authentication** (optional)
- **CORS protection**
- **Input validation** and sanitization
- **Request size limits**

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
flake8 src/
black src/
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full Documentation](docs/README.md)
- **Issues**: [GitHub Issues](https://github.com/michaelkatsweb/NCS-API-Project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/michaelkatsweb/NCS-API-Project/discussions)

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Clustering algorithms based on research in computational geometry
- Performance optimizations inspired by high-frequency trading systems

---

**Made with ❤️ by the NCS Team**
