# Contributing to NeuroCluster Streamer API

We welcome contributions to the NeuroCluster Streamer API! This guide will help you get started with contributing to the project, whether you're fixing bugs, adding features, or improving documentation.

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [Getting Started](#-getting-started)
- [Development Environment](#-development-environment)
- [Contribution Types](#-contribution-types)
- [Development Workflow](#-development-workflow)
- [Code Standards](#-code-standards)
- [Testing Requirements](#-testing-requirements)
- [Documentation Guidelines](#-documentation-guidelines)
- [Pull Request Process](#-pull-request-process)
- [Release Process](#-release-process)
- [Community Guidelines](#-community-guidelines)

## 🤝 Code of Conduct

This project adheres to a Code of Conduct that we expect all contributors to follow. Please read and understand our community standards:

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Expected Behavior

- **Be respectful** and inclusive in discussions
- **Be constructive** when giving feedback
- **Be collaborative** and help others learn
- **Be patient** with newcomers to the project
- **Be professional** in all communications

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing private information without consent
- Spam, promotional content, or off-topic discussions
- Any conduct that would be inappropriate in a professional setting

### Reporting Issues

If you experience or witness unacceptable behavior, please report it to conduct@yourdomain.com. All reports will be handled confidentially.

## 🚀 Getting Started

### Prerequisites

Before contributing, make sure you have:

- **Python 3.11+** installed
- **Git** for version control
- **Docker & Docker Compose** (for containerized development)
- **PostgreSQL 13+** and **Redis 6+** (for local development)
- Basic understanding of **FastAPI**, **SQLAlchemy**, and **AsyncIO**

### Quick Setup

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/ncs-api.git
   cd ncs-api
   
   # Add upstream remote
   git remote add upstream https://github.com/your-org/ncs-api.git
   ```

2. **Set Up Development Environment**
   ```bash
   # Run setup script
   ./scripts/setup.sh development
   
   # Or manual setup
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Start Development Services**
   ```bash
   # Using Docker Compose (recommended)
   docker-compose up -d postgres redis
   
   # Or install locally (see DEPLOYMENT_GUIDE.md)
   ```

4. **Verify Setup**
   ```bash
   # Run tests
   pytest tests/test_api.py -v
   
   # Start development server
   uvicorn main_secure:app --reload
   
   # Check health
   curl http://localhost:8000/health
   ```

## 🛠️ Development Environment

### Recommended Development Setup

#### IDE Configuration

**VS Code (Recommended)**
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "python.sortImports.args": ["--profile", "black"],
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/node_modules": true,
        "**/.pytest_cache": true
    }
}
```

**PyCharm**
- Configure Python interpreter to use `venv/bin/python`
- Enable Black formatter with line length 88
- Enable isort with Black profile
- Configure pytest as test runner

#### Environment Configuration

```bash
# .env.development
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=dev-secret-key-not-for-production
LOG_LEVEL=DEBUG

# Database (Docker Compose)
DATABASE_URL=postgresql://ncs_user:ncs_password@localhost:5432/ncs_dev
REDIS_URL=redis://localhost:6379/0

# Development features
ENABLE_DETAILED_METRICS=true
ENABLE_DEBUG_TOOLBAR=true
RELOAD_ON_CHANGE=true
```

### Development Tools

#### Essential Tools

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Code formatting
black .                    # Format code
isort .                    # Sort imports

# Code quality
flake8 .                   # Linting
mypy .                     # Type checking
bandit -r .               # Security scanning

# Testing
pytest                     # Run tests
pytest --cov=.            # Coverage report
pytest -x                 # Stop on first failure
pytest -v                 # Verbose output

# Pre-commit hooks (recommended)
pre-commit install
```

#### Performance Tools

```bash
# Performance profiling
python -m cProfile -o profile.prof main_secure.py
snakeviz profile.prof     # Visualize profile

# Memory profiling
pip install memory-profiler
python -m memory_profiler script.py

# Load testing
pip install locust
locust -f tests/load_test.py
```

### Docker Development

**Development Compose Override**
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  ncs-api:
    build:
      context: .
      dockerfile: docker/Dockerfile.dev
    volumes:
      - .:/app
      - /app/venv
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
    ports:
      - "8000:8000"
      - "5678:5678"  # Debugger port
```

## 🎯 Contribution Types

We welcome various types of contributions:

### 🐛 Bug Fixes

- Fix incorrect behavior or errors
- Improve error handling and edge cases
- Performance optimizations
- Security vulnerability fixes

### ✨ New Features

- Algorithm improvements and optimizations
- New API endpoints or functionality
- Integration with external services
- Performance enhancements

### 📚 Documentation

- API documentation improvements
- Code comments and docstrings
- Tutorial and example updates
- README and guide enhancements

### 🧪 Testing

- Unit test coverage improvements
- Integration test additions
- Performance test development
- Test infrastructure improvements

### 🔧 Infrastructure

- CI/CD pipeline improvements
- Docker and Kubernetes enhancements
- Monitoring and logging improvements
- Development tool upgrades

## 🔄 Development Workflow

### Branch Strategy

We use **Git Flow** with the following branches:

- **`main`**: Production-ready code
- **`develop`**: Integration branch for features
- **`feature/*`**: New features and enhancements
- **`bugfix/*`**: Bug fixes
- **`hotfix/*`**: Critical production fixes
- **`release/*`**: Release preparation

### Feature Development Workflow

1. **Create Feature Branch**
   ```bash
   # Sync with upstream
   git checkout develop
   git pull upstream develop
   
   # Create feature branch
   git checkout -b feature/your-feature-name
   ```

2. **Development Cycle**
   ```bash
   # Make changes
   git add .
   git commit -m "feat: add new clustering algorithm optimization"
   
   # Run tests frequently
   pytest tests/
   
   # Push to your fork
   git push origin feature/your-feature-name
   ```

3. **Stay Updated**
   ```bash
   # Regularly sync with upstream
   git fetch upstream
   git rebase upstream/develop
   ```

4. **Prepare for PR**
   ```bash
   # Final checks
   black .
   isort .
   flake8 .
   mypy .
   pytest --cov=.
   
   # Update documentation if needed
   # Add/update tests
   # Update CHANGELOG.md
   ```

### Commit Message Convention

We follow **Conventional Commits** specification:

```bash
# Format
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes

**Examples:**
```bash
feat(api): add batch processing endpoint
fix(auth): resolve JWT token expiration issue
docs(readme): update installation instructions
test(algorithm): add unit tests for clustering logic
perf(database): optimize connection pooling
ci(github): add security scanning workflow
```

## 📏 Code Standards

### Python Code Style

We follow **PEP 8** with some modifications:

```python
# Line length: 88 characters (Black default)
# Use Black for formatting
# Use isort for import sorting

# Example function
async def process_data_points(
    points: List[List[float]],
    algorithm_config: Optional[AlgorithmConfig] = None,
    timeout: float = 30.0
) -> ProcessingResult:
    """
    Process data points through the clustering algorithm.
    
    Args:
        points: List of data points to process
        algorithm_config: Optional algorithm configuration overrides
        timeout: Maximum processing time in seconds
        
    Returns:
        ProcessingResult containing clusters and metadata
        
    Raises:
        ValidationError: If input data is invalid
        ProcessingError: If algorithm processing fails
        TimeoutError: If processing exceeds timeout
    """
    logger.info(
        "Processing data points",
        point_count=len(points),
        timeout=timeout
    )
    
    # Validate input
    if not points:
        raise ValidationError("Points list cannot be empty")
    
    try:
        # Process points
        result = await algorithm.process_points(
            points=points,
            config=algorithm_config or get_default_config(),
            timeout=timeout
        )
        
        logger.info(
            "Processing completed",
            cluster_count=len(result.clusters),
            outlier_count=len(result.outliers),
            processing_time=result.processing_time_ms
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "Processing failed",
            error=str(e),
            error_type=type(e).__name__
        )
        raise ProcessingError(f"Failed to process points: {e}") from e
```

### Code Quality Standards

#### Type Hints
```python
# Use type hints for all functions
from typing import List, Dict, Optional, Union, Any

def calculate_similarity(
    point1: List[float],
    point2: List[float]
) -> float:
    """Calculate Euclidean distance between two points."""
    return sum((a - b) ** 2 for a, b in zip(point1, point2)) ** 0.5
```

#### Error Handling
```python
# Use specific exception types
from app.exceptions import ValidationError, ProcessingError

def validate_point(point: List[float]) -> None:
    """Validate a data point."""
    if not isinstance(point, list):
        raise ValidationError("Point must be a list")
    
    if not point:
        raise ValidationError("Point cannot be empty")
    
    if not all(isinstance(x, (int, float)) for x in point):
        raise ValidationError("Point coordinates must be numbers")
```

#### Logging
```python
# Use structured logging
import structlog

logger = structlog.get_logger(__name__)

def process_batch(batch_id: str, points: List[List[float]]) -> None:
    """Process a batch of points."""
    logger.info(
        "Starting batch processing",
        batch_id=batch_id,
        point_count=len(points)
    )
    
    try:
        # Processing logic
        result = process_points(points)
        
        logger.info(
            "Batch processing completed",
            batch_id=batch_id,
            cluster_count=len(result.clusters)
        )
        
    except Exception as e:
        logger.error(
            "Batch processing failed",
            batch_id=batch_id,
            error=str(e),
            exc_info=True
        )
        raise
```

### API Design Standards

#### Endpoint Design
```python
# RESTful endpoints with clear naming
from fastapi import APIRouter, Depends, HTTPException, status
from app.models import ProcessPointsRequest, ProcessingResult

router = APIRouter(prefix="/api/v1", tags=["clustering"])

@router.post(
    "/process_points",
    response_model=ProcessingResult,
    status_code=status.HTTP_200_OK,
    summary="Process data points",
    description="Process a batch of data points through the clustering algorithm"
)
async def process_points(
    request: ProcessPointsRequest,
    current_user: User = Depends(get_current_user)
) -> ProcessingResult:
    """Process data points endpoint."""
    # Implementation
```

#### Request/Response Models
```python
# Use Pydantic models for validation
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class ProcessPointsRequest(BaseModel):
    """Request model for processing data points."""
    
    points: List[List[float]] = Field(
        ...,
        min_items=1,
        max_items=10000,
        description="Array of data points to process"
    )
    batch_mode: Optional[bool] = Field(
        default=False,
        description="Enable batch processing mode"
    )
    timeout: Optional[int] = Field(
        default=30000,
        ge=1000,
        le=300000,
        description="Processing timeout in milliseconds"
    )
    
    @validator('points')
    def validate_points(cls, v):
        """Validate data points."""
        for i, point in enumerate(v):
            if len(point) == 0:
                raise ValueError(f"Point {i} cannot be empty")
            if len(point) > 1000:
                raise ValueError(f"Point {i} has too many dimensions")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "points": [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]],
                "batch_mode": False,
                "timeout": 30000
            }
        }
```

## 🧪 Testing Requirements

### Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration and fixtures
├── test_api.py              # API endpoint tests
├── test_auth.py             # Authentication tests
├── test_algorithm.py        # Algorithm unit tests
├── test_security.py         # Security tests
├── test_performance.py      # Performance tests
├── integration/             # Integration tests
│   ├── test_database.py
│   └── test_external_apis.py
└── fixtures/                # Test data
    ├── test_data.json
    └── mock_responses.json
```

### Testing Standards

#### Unit Tests
```python
# tests/test_algorithm.py
import pytest
import numpy as np
from unittest.mock import Mock, patch

from app.algorithm import NCSAlgorithm, ClusteringConfig
from app.exceptions import ValidationError, ProcessingError

class TestNCSAlgorithm:
    """Unit tests for NCS algorithm."""
    
    @pytest.fixture
    def algorithm(self):
        """Create algorithm instance for testing."""
        config = ClusteringConfig(
            base_threshold=0.71,
            learning_rate=0.06,
            max_clusters=30
        )
        return NCSAlgorithm(config)
    
    @pytest.fixture
    def sample_points(self):
        """Sample data points for testing."""
        return [
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [5.0, 6.0, 7.0]
        ]
    
    def test_process_single_point(self, algorithm, sample_points):
        """Test processing a single data point."""
        point = sample_points[0]
        result = algorithm.process_point(point)
        
        assert result is not None
        assert len(result.clusters) >= 0
        assert isinstance(result.processing_time_ms, float)
        assert result.processing_time_ms > 0
    
    def test_process_empty_points_raises_error(self, algorithm):
        """Test that empty points list raises ValidationError."""
        with pytest.raises(ValidationError, match="Points list cannot be empty"):
            algorithm.process_points([])
    
    def test_process_invalid_point_format(self, algorithm):
        """Test that invalid point format raises ValidationError."""
        invalid_points = [["a", "b", "c"]]  # Non-numeric
        
        with pytest.raises(ValidationError):
            algorithm.process_points(invalid_points)
    
    @patch('app.algorithm.calculate_distance')
    def test_process_points_with_mock(self, mock_distance, algorithm, sample_points):
        """Test processing with mocked distance calculation."""
        mock_distance.return_value = 0.5
        
        result = algorithm.process_points(sample_points)
        
        assert mock_distance.called
        assert len(result.clusters) > 0
    
    def test_algorithm_performance(self, algorithm):
        """Test algorithm performance with larger dataset."""
        # Generate larger test dataset
        points = np.random.rand(1000, 3).tolist()
        
        import time
        start_time = time.time()
        result = algorithm.process_points(points)
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000  # Convert to ms
        
        # Performance assertions
        assert processing_time < 5000  # Should complete in under 5 seconds
        assert len(result.clusters) > 0
        assert result.algorithm_quality > 0.5
```

#### API Tests
```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from main_secure import app
from tests.conftest import create_test_user, get_test_token

class TestProcessPointsAPI:
    """API tests for process_points endpoint."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self, test_client):
        """Get authentication headers for testing."""
        token = get_test_token(test_client)
        return {"Authorization": f"Bearer {token}"}
    
    def test_process_points_success(self, test_client, auth_headers):
        """Test successful point processing."""
        data = {
            "points": [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]],
            "batch_mode": False
        }
        
        response = test_client.post(
            "/api/v1/process_points",
            json=data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        
        result = response.json()
        assert "clusters" in result
        assert "outliers" in result
        assert "processing_time_ms" in result
        assert "algorithm_quality" in result
        assert isinstance(result["clusters"], list)
    
    def test_process_points_validation_error(self, test_client, auth_headers):
        """Test validation error handling."""
        data = {
            "points": []  # Empty points list
        }
        
        response = test_client.post(
            "/api/v1/process_points",
            json=data,
            headers=auth_headers
        )
        
        assert response.status_code == 422
        
        error = response.json()
        assert "detail" in error
    
    def test_process_points_without_auth(self, test_client):
        """Test endpoint requires authentication."""
        data = {
            "points": [[1.0, 2.0, 3.0]]
        }
        
        response = test_client.post(
            "/api/v1/process_points",
            json=data
        )
        
        assert response.status_code == 401
    
    @patch('app.algorithm.NCSAlgorithm.process_points')
    def test_process_points_algorithm_error(self, mock_process, test_client, auth_headers):
        """Test algorithm error handling."""
        from app.exceptions import ProcessingError
        mock_process.side_effect = ProcessingError("Algorithm failed")
        
        data = {
            "points": [[1.0, 2.0, 3.0]]
        }
        
        response = test_client.post(
            "/api/v1/process_points",
            json=data,
            headers=auth_headers
        )
        
        assert response.status_code == 500
        assert "Algorithm failed" in response.json()["detail"]
```

#### Performance Tests
```python
# tests/test_performance.py
import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient

class TestPerformance:
    """Performance tests for the API."""
    
    def test_single_request_latency(self, test_client, auth_headers):
        """Test single request latency."""
        data = {"points": [[1.0, 2.0, 3.0]]}
        
        # Measure latency
        start_time = time.time()
        response = test_client.post(
            "/api/v1/process_points",
            json=data,
            headers=auth_headers
        )
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert latency_ms < 100  # Should be under 100ms
    
    def test_throughput(self, test_client, auth_headers):
        """Test API throughput."""
        data = {"points": [[1.0, 2.0, 3.0]]}
        
        def make_request():
            response = test_client.post(
                "/api/v1/process_points",
                json=data,
                headers=auth_headers
            )
            return response.status_code == 200
        
        # Measure throughput
        start_time = time.time()
        num_requests = 100
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda _: make_request(), range(num_requests)))
        
        end_time = time.time()
        
        success_rate = sum(results) / len(results)
        throughput = num_requests / (end_time - start_time)
        
        assert success_rate > 0.95  # 95% success rate
        assert throughput > 50     # 50 requests per second
```

### Test Configuration

```python
# tests/conftest.py
import pytest
import asyncio
from typing import Generator
from fastapi.testclient import TestClient
import tempfile
import os

from main_secure import app
from app.database import get_db, create_tables
from app.auth import create_access_token

# Test database setup
@pytest.fixture(scope="session")
def test_db():
    """Create test database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        test_db_url = f"sqlite:///{temp_db.name}"
        
        # Override database URL for testing
        os.environ["DATABASE_URL"] = test_db_url
        
        # Create tables
        create_tables()
        
        yield test_db_url
        
        # Cleanup
        os.unlink(temp_db.name)

@pytest.fixture
def test_client(test_db) -> Generator[TestClient, None, None]:
    """Create test client."""
    with TestClient(app) as client:
        yield client

@pytest.fixture
def test_user():
    """Create test user data."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123"
    }

@pytest.fixture
def auth_headers(test_user):
    """Create authentication headers."""
    token = create_access_token({"sub": test_user["username"]})
    return {"Authorization": f"Bearer {token}"}

# Test data fixtures
@pytest.fixture
def sample_points():
    """Sample data points for testing."""
    return [
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [5.0, 6.0, 7.0],
        [5.1, 6.1, 7.1]
    ]

@pytest.fixture
def large_dataset():
    """Large dataset for performance testing."""
    import numpy as np
    return np.random.rand(1000, 3).tolist()

# Async test support
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
```

### Coverage Requirements

- **Minimum Coverage**: 85% overall
- **Critical Components**: 95% coverage required
  - Authentication and security modules
  - Core algorithm logic
  - API endpoints
- **Documentation**: All public functions must have docstrings
- **Type Hints**: All functions must have complete type annotations

```bash
# Run tests with coverage
pytest --cov=app --cov-report=html --cov-report=term-missing

# Check coverage requirements
coverage report --fail-under=85
```

## 📚 Documentation Guidelines

### Code Documentation

#### Docstring Format
```python
def calculate_cluster_quality(
    cluster: Cluster,
    points: List[List[float]]
) -> float:
    """
    Calculate the quality score for a cluster.
    
    The quality score is based on intra-cluster cohesion and
    inter-cluster separation, using silhouette analysis.
    
    Args:
        cluster: The cluster to evaluate
        points: All data points in the dataset
        
    Returns:
        Quality score between 0.0 and 1.0, where higher is better
        
    Raises:
        ValueError: If cluster is empty or points list is invalid
        
    Example:
        >>> cluster = Cluster(id=1, center=[1.0, 1.0], points=[[1,1], [1,2]])
        >>> quality = calculate_cluster_quality(cluster, all_points)
        >>> print(f"Quality: {quality:.3f}")
        Quality: 0.857
        
    Note:
        This function uses O(n²) algorithm and may be slow for large datasets.
        Consider using approximate methods for datasets > 10,000 points.
    """
```

#### API Documentation
```python
from fastapi import APIRouter, Depends
from app.models import ProcessPointsRequest, ProcessingResult

router = APIRouter()

@router.post(
    "/process_points",
    response_model=ProcessingResult,
    summary="Process data points",
    description="""
    Process a batch of data points through the clustering algorithm.
    
    This endpoint accepts an array of numerical data points and returns
    the clustering results including identified clusters and outliers.
    
    **Performance Notes:**
    - Typical processing time: <1ms per point
    - Maximum batch size: 10,000 points
    - Timeout: Configurable, default 30 seconds
    
    **Algorithm Details:**
    - Uses adaptive threshold clustering
    - Automatically determines optimal number of clusters
    - Provides quality metrics for results
    """,
    responses={
        200: {
            "description": "Successful processing",
            "content": {
                "application/json": {
                    "example": {
                        "clusters": [
                            {
                                "id": 1,
                                "center": [1.05, 2.05, 3.05],
                                "points": [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]],
                                "size": 2,
                                "quality": 0.95
                            }
                        ],
                        "outliers": [[5.0, 6.0, 7.0]],
                        "processing_time_ms": 15.2,
                        "algorithm_quality": 0.92
                    }
                }
            }
        },
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Processing error"}
    }
)
async def process_points(request: ProcessPointsRequest) -> ProcessingResult:
    """Process data points endpoint."""
    # Implementation
```

### README Updates

When adding new features, update relevant README sections:

- **Features**: Add new capabilities
- **API Endpoints**: Document new endpoints
- **Configuration**: Add new environment variables
- **Examples**: Provide usage examples

### Changelog Maintenance

Update `CHANGELOG.md` for all notable changes:

```markdown
## [Unreleased]

### Added
- New batch processing endpoint for improved throughput
- Support for custom algorithm parameters per request
- Enhanced error reporting with detailed context

### Changed
- Improved clustering algorithm performance by 15%
- Updated authentication to support API key rotation

### Fixed
- Memory leak in long-running processes
- Race condition in concurrent request handling

### Security
- Added rate limiting per user and endpoint
- Enhanced input validation to prevent injection attacks
```

## 🔄 Pull Request Process

### Before Submitting

1. **Code Quality Checklist**
   ```bash
   # Format code
   black .
   isort .
   
   # Check code quality
   flake8 .
   mypy .
   bandit -r .
   
   # Run tests
   pytest --cov=app --cov-fail-under=85
   
   # Check documentation
   # Update docstrings and README if needed
   ```

2. **Performance Testing**
   ```bash
   # Run performance tests
   pytest tests/test_performance.py -v
   
   # Check for regressions
   python scripts/benchmark_compare.py
   ```

3. **Security Review**
   ```bash
   # Security scanning
   bandit -r app/
   safety check
   
   # Check for secrets
   git log --patch | grep -E "(password|token|key|secret)" || echo "No secrets found"
   ```

### Pull Request Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to change)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Security enhancement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Performance tests passed
- [ ] Manual testing completed

## Code Quality
- [ ] Code formatted with Black
- [ ] Imports sorted with isort
- [ ] Linting passed (flake8)
- [ ] Type checking passed (mypy)
- [ ] Security scan passed (bandit)
- [ ] Test coverage ≥85%

## Documentation
- [ ] Code comments added/updated
- [ ] Docstrings added/updated
- [ ] README updated (if applicable)
- [ ] CHANGELOG.md updated
- [ ] API documentation updated (if applicable)

## Security Considerations
- [ ] No sensitive data exposed
- [ ] Input validation implemented
- [ ] Authentication/authorization considered
- [ ] Security tests added (if applicable)

## Performance Impact
- [ ] Performance tests passed
- [ ] No significant performance regression
- [ ] Memory usage considered
- [ ] Database queries optimized (if applicable)

## Screenshots (if applicable)
Add screenshots or gifs for UI changes.

## Additional Notes
Any additional information, context, or considerations.

## Checklist
- [ ] PR title follows conventional commit format
- [ ] Branch is up to date with target branch
- [ ] All CI checks pass
- [ ] Code review requested from appropriate team members
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: At least one maintainer review required
3. **Testing**: All tests must pass
4. **Documentation**: Review for completeness and accuracy
5. **Security**: Security-focused review for sensitive changes
6. **Performance**: Performance impact assessment

### Merge Requirements

- [ ] All CI checks pass
- [ ] At least one approved review from maintainer
- [ ] No merge conflicts
- [ ] Branch is up to date with target
- [ ] All conversations resolved

## 🚀 Release Process

### Versioning

We follow **Semantic Versioning** (SemVer):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Workflow

1. **Prepare Release**
   ```bash
   # Create release branch
   git checkout develop
   git pull upstream develop
   git checkout -b release/v1.2.0
   
   # Update version numbers
   # Update CHANGELOG.md
   # Final testing
   ```

2. **Release Checklist**
   - [ ] Version bumped in all relevant files
   - [ ] CHANGELOG.md updated with release date
   - [ ] All tests pass
   - [ ] Documentation updated
   - [ ] Security review completed
   - [ ] Performance benchmarks pass

3. **Merge and Tag**
   ```bash
   # Merge to main
   git checkout main
   git merge release/v1.2.0
   git tag v1.2.0
   git push upstream main --tags
   
   # Merge back to develop
   git checkout develop
   git merge main
   git push upstream develop
   ```

4. **Post-Release**
   - GitHub release created with changelog
   - Docker images built and pushed
   - Documentation deployed
   - Announcement posted

## 👥 Community Guidelines

### Communication Channels

- **GitHub Discussions**: General questions and community chat
- **GitHub Issues**: Bug reports and feature requests
- **Discord/Slack**: Real-time development discussion (if available)
- **Email**: conduct@yourdomain.com for Code of Conduct issues

### Getting Help

1. **Check Documentation**: README, docs/, and API reference
2. **Search Issues**: Look for existing solutions
3. **Ask Questions**: Use GitHub Discussions for general help
4. **Report Bugs**: Use GitHub Issues with full details

### Recognition

We recognize contributors through:

- **Contributors File**: All contributors listed in CONTRIBUTORS.md
- **Release Notes**: Notable contributions mentioned in releases
- **GitHub**: Contributor recognition on project page
- **Social Media**: Highlighting significant contributions

### Maintainer Responsibilities

Current maintainers:
- Review pull requests within 3 business days
- Provide constructive feedback
- Maintain project roadmap and vision
- Ensure code quality and security standards
- Foster inclusive community environment

### Becoming a Maintainer

Path to maintainership:
1. **Consistent Contributions**: Regular, high-quality contributions
2. **Community Involvement**: Help other contributors and users
3. **Technical Expertise**: Deep understanding of project architecture
4. **Review Participation**: Provide helpful code reviews
5. **Nomination**: Existing maintainer nomination and team approval

---

## 🙏 Thank You

Thank you for contributing to the NeuroCluster Streamer API! Your contributions help make this project better for everyone. Whether you're fixing bugs, adding features, improving documentation, or helping other users, every contribution is valuable and appreciated.

### Questions?

If you have any questions about contributing, please:
- Open a GitHub Discussion
- Review our documentation
- Contact us at dev@yourdomain.com

**Happy coding! 🚀**