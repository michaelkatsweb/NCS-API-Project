---
name: 🐛 Bug Report
about: Create a detailed bug report to help us improve the NCS API
title: '[BUG] Brief description of the issue'
labels: ['bug', 'needs-triage']
assignees: []
---

## 🐛 Bug Description

**Describe the bug clearly and concisely:**
A clear and concise description of what the bug is.

**Expected behavior:**
A clear and concise description of what you expected to happen.

**Actual behavior:**
A clear and concise description of what actually happened.

## 🔍 Steps to Reproduce

Please provide detailed steps to reproduce the behavior:

1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Minimal reproducible example:**
```python
# Please provide a minimal code example that reproduces the issue
# Include any relevant API calls, data, or configuration
```

## 📊 Environment Information

**NCS API Version:**
- Version: [e.g., v1.0.0]
- Deployment: [e.g., Docker, Kubernetes, local development]

**Client Information:**
- SDK: [e.g., Python SDK v1.0.0, JavaScript SDK v1.0.0, Direct HTTP calls]
- Programming Language: [e.g., Python 3.11, Node.js 18]
- Operating System: [e.g., Ubuntu 22.04, macOS 13.0, Windows 11]

**Runtime Environment:**
- CPU Architecture: [e.g., x86_64, ARM64]
- Memory: [e.g., 8GB RAM]
- Container Runtime: [e.g., Docker 24.0.0, containerd 1.6.0]
- Kubernetes Version: [e.g., v1.28.0] (if applicable)

**Dependencies:**
- FastAPI version: [e.g., 0.104.0]
- Python version: [e.g., 3.11.5]
- Other relevant dependencies: [list if applicable]

## 📋 API Request Details

**Endpoint affected:**
- Endpoint: [e.g., POST /api/v1/process_points]
- HTTP Method: [e.g., POST, GET, PUT, DELETE]

**Request details:**
```json
{
  "// Include the request payload": "if applicable",
  "// Remove any sensitive information": "like API keys or personal data"
}
```

**Request headers:**
```
Content-Type: application/json
Authorization: Bearer [REDACTED]
X-API-Key: [REDACTED]
```

**Query parameters:**
```
?param1=value1&param2=value2
```

## 📝 Response Information

**HTTP Status Code:** [e.g., 500, 400, 422]

**Response headers:**
```
Content-Type: application/json
X-Request-ID: 123e4567-e89b-12d3-a456-426614174000
```

**Response body:**
```json
{
  "error": "Description of the error response",
  "detail": "Additional error details",
  "request_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

**Error logs:** (if available)
```
[2025-01-15 10:30:45] ERROR: Detailed error message from logs
[2025-01-15 10:30:45] TRACE: Stack trace if available
```

## 💾 Data Information

**Input data characteristics:**
- Data size: [e.g., 1000 points, 50MB file]
- Data format: [e.g., JSON array, CSV file, streaming data]
- Data source: [e.g., file upload, API call, database]

**Sample data** (please anonymize sensitive information):
```json
{
  "points": [
    {"x": 1.0, "y": 2.0, "z": 3.0},
    {"x": 1.1, "y": 2.1, "z": 3.1}
  ],
  "parameters": {
    "algorithm": "ncs_v8",
    "threshold": 0.5
  }
}
```

## 🔄 Frequency and Impact

**How often does this bug occur?**
- [ ] Always (100%)
- [ ] Frequently (>50%)
- [ ] Sometimes (10-50%)
- [ ] Rarely (<10%)
- [ ] Once

**Impact level:**
- [ ] 🔴 Critical - Service completely unusable
- [ ] 🟠 High - Major functionality broken
- [ ] 🟡 Medium - Some functionality affected
- [ ] 🟢 Low - Minor issue or cosmetic

**Affected components:**
- [ ] Authentication/Authorization
- [ ] Core clustering algorithm
- [ ] Data processing pipeline
- [ ] API endpoints
- [ ] Database operations
- [ ] Caching (Redis)
- [ ] Monitoring/Metrics
- [ ] Documentation
- [ ] SDK (Python)
- [ ] SDK (JavaScript)

## 🛠️ Troubleshooting Attempted

**What troubleshooting steps have you already tried?**
- [ ] Checked API documentation
- [ ] Verified authentication credentials
- [ ] Tested with different input data
- [ ] Checked server logs
- [ ] Reviewed error messages
- [ ] Tested with different client/SDK version
- [ ] Reproduced in different environment

**Configuration details:**
```yaml
# Include relevant configuration files (remove sensitive data)
# For example:
environment: production
debug: false
log_level: INFO
database_url: postgresql://[REDACTED]
redis_url: redis://[REDACTED]
```

## 📎 Additional Context

**Screenshots/Videos:**
[If applicable, add screenshots or videos to help explain the problem]

**Related issues:**
- Link to any related issues or discussions
- Reference to similar problems in the past

**Workarounds:**
- Any temporary workarounds you've discovered
- Alternative approaches that work

**Additional information:**
Any other context about the problem here, such as:
- Network configuration details
- Security policies that might be relevant
- Recent changes to your environment
- Specific business requirements or constraints

## ✅ Checklist

Please confirm you have completed the following:

- [ ] I have searched for existing issues and this is not a duplicate
- [ ] I have provided a clear and descriptive title
- [ ] I have included all relevant environment information
- [ ] I have provided steps to reproduce the issue
- [ ] I have included relevant code examples or data samples
- [ ] I have removed any sensitive information (API keys, passwords, personal data)
- [ ] I have checked the [documentation](https://your-org.github.io/ncs-api) for solutions
- [ ] I have tried basic troubleshooting steps

## 🏷️ Labels to Add

**For maintainers - please add appropriate labels:**
- Component: `api`, `algorithm`, `sdk-python`, `sdk-javascript`, `docs`, `infra`
- Priority: `priority/low`, `priority/medium`, `priority/high`, `priority/critical`
- Type: `bug`, `performance`, `security`, `regression`
- Area: `authentication`, `clustering`, `database`, `monitoring`, `deployment`

---

**Thank you for taking the time to report this bug! 🙏**

The NCS API development team will review this issue and provide updates. For urgent issues, please contact our support team directly.