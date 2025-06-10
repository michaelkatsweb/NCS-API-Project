"""
Authentication and authorization tests for NeuroCluster Streamer API.

This module tests all authentication mechanisms including:
- JWT token authentication and validation
- API key authentication
- Role-based access control (RBAC)
- Token refresh and expiration
- Security headers and CORS
- Rate limiting and abuse prevention
"""

import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import jwt

from auth import (
    create_access_token,
    create_refresh_token,
    verify_token,
    hash_password,
    verify_password,
    get_current_user,
)
from config import settings
from . import TEST_USERS, API_ENDPOINTS


class TestJWTAuthentication:
    """Test JWT token authentication functionality."""

    def test_create_access_token(self):
        """Test JWT access token creation."""
        user_data = {
            "sub": "test_user_123",
            "email": "test@example.com",
            "role": "user",
            "scopes": ["read", "write"],
        }

        token = create_access_token(user_data)

        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are long

        # Verify token can be decoded
        decoded = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )

        assert decoded["sub"] == user_data["sub"]
        assert decoded["email"] == user_data["email"]
        assert decoded["role"] == user_data["role"]
        assert decoded["scopes"] == user_data["scopes"]
        assert "exp" in decoded
        assert "iat" in decoded

    def test_create_refresh_token(self):
        """Test JWT refresh token creation."""
        user_data = {"sub": "test_user_123", "email": "test@example.com"}

        token = create_refresh_token(user_data)

        assert isinstance(token, str)

        # Verify token
        decoded = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )

        assert decoded["sub"] == user_data["sub"]
        assert decoded["token_type"] == "refresh"

    def test_verify_valid_token(self):
        """Test verification of valid JWT token."""
        user_data = {
            "sub": "test_user_123",
            "email": "test@example.com",
            "role": "user",
            "scopes": ["read", "write"],
        }

        token = create_access_token(user_data)
        verified_data = verify_token(token)

        assert verified_data is not None
        assert verified_data["sub"] == user_data["sub"]
        assert verified_data["email"] == user_data["email"]

    def test_verify_expired_token(self):
        """Test verification of expired JWT token."""
        user_data = {
            "sub": "test_user_123",
            "email": "test@example.com",
            "role": "user",
        }

        # Create token with very short expiration
        expired_token = create_access_token(
            user_data, expires_delta=timedelta(seconds=-1)
        )

        # Should return None for expired token
        verified_data = verify_token(expired_token)
        assert verified_data is None

    def test_verify_invalid_signature(self):
        """Test verification of token with invalid signature."""
        user_data = {"sub": "test_user_123", "email": "test@example.com"}

        # Create token with different secret
        invalid_token = jwt.encode(
            user_data, "wrong_secret_key", algorithm=settings.JWT_ALGORITHM
        )

        # Should return None for invalid signature
        verified_data = verify_token(invalid_token)
        assert verified_data is None

    def test_verify_malformed_token(self):
        """Test verification of malformed JWT token."""
        malformed_tokens = ["not.a.jwt.token", "invalid_token", "", None]

        for token in malformed_tokens:
            verified_data = verify_token(token)
            assert verified_data is None


class TestPasswordHashing:
    """Test password hashing and verification."""

    def test_hash_password(self):
        """Test password hashing."""
        password = "test_password_123"
        hashed = hash_password(password)

        assert isinstance(hashed, str)
        assert hashed != password  # Should be hashed
        assert len(hashed) > 50  # Bcrypt hashes are long
        assert hashed.startswith("$2b$")  # Bcrypt format

    def test_verify_correct_password(self):
        """Test verification of correct password."""
        password = "test_password_123"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_incorrect_password(self):
        """Test verification of incorrect password."""
        password = "test_password_123"
        wrong_password = "wrong_password"
        hashed = hash_password(password)

        assert verify_password(wrong_password, hashed) is False

    def test_verify_empty_password(self):
        """Test verification with empty password."""
        hashed = hash_password("test_password")

        assert verify_password("", hashed) is False
        assert verify_password(None, hashed) is False


class TestAuthenticationEndpoints:
    """Test authentication endpoints."""

    def test_login_success(self, test_client: TestClient):
        """Test successful login."""
        login_data = {
            "username": TEST_USERS["user"]["email"],
            "password": TEST_USERS["user"]["password"],
        }

        with patch("auth.authenticate_user") as mock_auth:
            mock_auth.return_value = {
                "user_id": TEST_USERS["user"]["user_id"],
                "email": TEST_USERS["user"]["email"],
                "role": TEST_USERS["user"]["role"],
                "scopes": TEST_USERS["user"]["scopes"],
            }

            response = test_client.post(API_ENDPOINTS["auth_login"], data=login_data)

        assert response.status_code == 200
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data

    def test_login_invalid_credentials(self, test_client: TestClient):
        """Test login with invalid credentials."""
        login_data = {"username": "invalid@example.com", "password": "wrong_password"}

        with patch("auth.authenticate_user") as mock_auth:
            mock_auth.return_value = None  # Failed authentication

            response = test_client.post(API_ENDPOINTS["auth_login"], data=login_data)

        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "access_token" not in data

    def test_login_missing_credentials(self, test_client: TestClient):
        """Test login with missing credentials."""
        incomplete_data = {
            "username": "test@example.com"
            # Missing password
        }

        response = test_client.post(API_ENDPOINTS["auth_login"], data=incomplete_data)

        assert response.status_code == 422  # Validation error

    def test_refresh_token_success(self, test_client: TestClient):
        """Test successful token refresh."""
        user_data = {
            "sub": TEST_USERS["user"]["user_id"],
            "email": TEST_USERS["user"]["email"],
            "role": TEST_USERS["user"]["role"],
        }

        refresh_token = create_refresh_token(user_data)

        response = test_client.post(
            API_ENDPOINTS["auth_refresh"], json={"refresh_token": refresh_token}
        )

        assert response.status_code == 200
        data = response.json()

        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"

    def test_refresh_token_invalid(self, test_client: TestClient):
        """Test token refresh with invalid token."""
        response = test_client.post(
            API_ENDPOINTS["auth_refresh"], json={"refresh_token": "invalid_token"}
        )

        assert response.status_code == 401

    def test_refresh_token_expired(self, test_client: TestClient):
        """Test token refresh with expired token."""
        user_data = {"sub": "test_user", "email": "test@example.com"}

        # Create expired refresh token
        expired_token = create_refresh_token(
            user_data, expires_delta=timedelta(seconds=-1)
        )

        response = test_client.post(
            API_ENDPOINTS["auth_refresh"], json={"refresh_token": expired_token}
        )

        assert response.status_code == 401


class TestAPIKeyAuthentication:
    """Test API key authentication."""

    def test_valid_api_key(self, test_client: TestClient):
        """Test authentication with valid API key."""
        api_key = "test_api_key_1"  # From test configuration

        response = test_client.get(
            API_ENDPOINTS["health"], headers={"X-API-Key": api_key}
        )

        assert response.status_code == 200

    def test_invalid_api_key(self, test_client: TestClient):
        """Test authentication with invalid API key."""
        invalid_key = "invalid_api_key"

        response = test_client.get(
            "/process-point", headers={"X-API-Key": invalid_key}  # Protected endpoint
        )

        assert response.status_code == 401

    def test_missing_api_key(self, test_client: TestClient):
        """Test access to protected endpoint without API key."""
        response = test_client.get("/process-point")

        assert response.status_code == 401

    def test_api_key_and_jwt_precedence(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test precedence when both API key and JWT are provided."""
        headers = {**user_headers, "X-API-Key": "test_api_key_1"}

        response = test_client.get(API_ENDPOINTS["health"], headers=headers)

        assert response.status_code == 200


class TestRoleBasedAccessControl:
    """Test role-based access control."""

    def test_admin_access_all_endpoints(
        self, test_client: TestClient, admin_headers: Dict
    ):
        """Test admin access to all endpoints."""
        admin_endpoints = [
            "/health",
            "/statistics",
            "/metrics",
            "/admin/users",
            "/admin/config",
        ]

        for endpoint in admin_endpoints:
            response = test_client.get(endpoint, headers=admin_headers)
            assert response.status_code in [200, 404]  # 404 if endpoint doesn't exist

    def test_user_access_restricted_endpoints(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test regular user access to restricted endpoints."""
        restricted_endpoints = ["/admin/users", "/admin/config", "/admin/logs"]

        for endpoint in restricted_endpoints:
            response = test_client.get(endpoint, headers=user_headers)
            assert response.status_code == 403  # Forbidden

    def test_readonly_user_write_operations(
        self, test_client: TestClient, readonly_headers: Dict
    ):
        """Test readonly user attempting write operations."""
        write_endpoints = [
            ("POST", "/process-point"),
            ("POST", "/process-batch"),
            ("DELETE", "/session/123"),
        ]

        for method, endpoint in write_endpoints:
            if method == "POST":
                response = test_client.post(
                    endpoint, json={"test": "data"}, headers=readonly_headers
                )
            elif method == "DELETE":
                response = test_client.delete(endpoint, headers=readonly_headers)

            assert response.status_code == 403  # Forbidden

    def test_scope_based_access_control(self, test_client: TestClient):
        """Test scope-based access control."""
        # Create token with limited scopes
        limited_user_data = {
            "sub": "limited_user",
            "email": "limited@example.com",
            "role": "user",
            "scopes": ["read"],  # Only read scope
        }

        token = create_access_token(limited_user_data)
        headers = {"Authorization": f"Bearer {token}"}

        # Should be able to read
        response = test_client.get("/health", headers=headers)
        assert response.status_code == 200

        # Should not be able to write
        response = test_client.post(
            "/process-point", json={"test": "data"}, headers=headers
        )
        assert response.status_code == 403


class TestTokenValidation:
    """Test JWT token validation middleware."""

    def test_valid_token_access(self, test_client: TestClient, user_headers: Dict):
        """Test access with valid JWT token."""
        response = test_client.get("/process-point", headers=user_headers)

        # Should not be unauthorized (might be 422 for missing data)
        assert response.status_code != 401

    def test_invalid_token_access(self, test_client: TestClient):
        """Test access with invalid JWT token."""
        invalid_headers = {"Authorization": "Bearer invalid_token"}

        response = test_client.get("/process-point", headers=invalid_headers)

        assert response.status_code == 401

    def test_expired_token_access(self, test_client: TestClient):
        """Test access with expired JWT token."""
        user_data = {
            "sub": "test_user",
            "email": "test@example.com",
            "role": "user",
            "scopes": ["read", "write"],
        }

        # Create expired token
        expired_token = create_access_token(
            user_data, expires_delta=timedelta(seconds=-1)
        )
        expired_headers = {"Authorization": f"Bearer {expired_token}"}

        response = test_client.get("/process-point", headers=expired_headers)

        assert response.status_code == 401

    def test_malformed_authorization_header(self, test_client: TestClient):
        """Test access with malformed authorization header."""
        malformed_headers = [
            {"Authorization": "InvalidFormat token"},
            {"Authorization": "Bearer"},  # Missing token
            {"Authorization": "token_without_bearer"},
            {"Authorization": ""},
        ]

        for headers in malformed_headers:
            response = test_client.get("/process-point", headers=headers)
            assert response.status_code == 401


class TestSecurityHeaders:
    """Test security headers and CORS."""

    def test_security_headers_present(self, test_client: TestClient):
        """Test presence of security headers."""
        response = test_client.get(API_ENDPOINTS["health"])

        assert response.status_code == 200

        # Check for security headers
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
        ]

        for header in expected_headers:
            assert header in response.headers

    def test_cors_headers(self, test_client: TestClient):
        """Test CORS headers for cross-origin requests."""
        # Preflight request
        response = test_client.options(
            API_ENDPOINTS["health"],
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization",
            },
        )

        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
        assert "Access-Control-Allow-Headers" in response.headers

    def test_content_security_policy(self, test_client: TestClient):
        """Test Content Security Policy header."""
        response = test_client.get(API_ENDPOINTS["health"])

        if "Content-Security-Policy" in response.headers:
            csp = response.headers["Content-Security-Policy"]
            assert "default-src" in csp


class TestRateLimiting:
    """Test rate limiting and abuse prevention."""

    def test_rate_limit_per_user(self, test_client: TestClient, user_headers: Dict):
        """Test rate limiting per authenticated user."""
        responses = []

        # Make many requests rapidly
        for i in range(100):
            response = test_client.get(API_ENDPOINTS["health"], headers=user_headers)
            responses.append(response.status_code)

            # Stop if we hit rate limit
            if response.status_code == 429:
                break

        # Should eventually get rate limited
        assert 429 in responses

    def test_rate_limit_headers(self, test_client: TestClient, user_headers: Dict):
        """Test rate limit headers in response."""
        response = test_client.get(API_ENDPOINTS["health"], headers=user_headers)

        # Check for rate limit headers
        rate_limit_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ]

        for header in rate_limit_headers:
            # Headers might not be present if rate limiting is not configured
            if header in response.headers:
                assert isinstance(response.headers[header], str)

    def test_rate_limit_different_endpoints(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test rate limiting across different endpoints."""
        endpoints = [API_ENDPOINTS["health"], "/statistics", "/clusters"]

        # Test that rate limits might be different for different endpoints
        for endpoint in endpoints:
            response = test_client.get(endpoint, headers=user_headers)
            # Should be able to access different endpoints
            assert response.status_code in [200, 404, 422]


class TestAuthenticationAuditLog:
    """Test authentication event logging."""

    def test_successful_login_logged(self, test_client: TestClient, captured_logs):
        """Test that successful logins are logged."""
        login_data = {
            "username": TEST_USERS["user"]["email"],
            "password": TEST_USERS["user"]["password"],
        }

        with patch("auth.authenticate_user") as mock_auth:
            mock_auth.return_value = {
                "user_id": TEST_USERS["user"]["user_id"],
                "email": TEST_USERS["user"]["email"],
                "role": TEST_USERS["user"]["role"],
            }

            response = test_client.post(API_ENDPOINTS["auth_login"], data=login_data)

        assert response.status_code == 200

        # Check that login was logged
        log_content = captured_logs.getvalue()
        assert "login" in log_content.lower() or "authentication" in log_content.lower()

    def test_failed_login_logged(self, test_client: TestClient, captured_logs):
        """Test that failed logins are logged."""
        login_data = {"username": "invalid@example.com", "password": "wrong_password"}

        with patch("auth.authenticate_user") as mock_auth:
            mock_auth.return_value = None

            response = test_client.post(API_ENDPOINTS["auth_login"], data=login_data)

        assert response.status_code == 401

        # Check that failed login was logged
        log_content = captured_logs.getvalue()
        assert "failed" in log_content.lower() or "unauthorized" in log_content.lower()

    def test_token_refresh_logged(self, test_client: TestClient, captured_logs):
        """Test that token refresh events are logged."""
        user_data = {
            "sub": TEST_USERS["user"]["user_id"],
            "email": TEST_USERS["user"]["email"],
        }

        refresh_token = create_refresh_token(user_data)

        response = test_client.post(
            API_ENDPOINTS["auth_refresh"], json={"refresh_token": refresh_token}
        )

        assert response.status_code == 200

        # Check that refresh was logged
        log_content = captured_logs.getvalue()
        assert "refresh" in log_content.lower() or "token" in log_content.lower()


@pytest.mark.security
class TestSecurityVulnerabilities:
    """Test protection against common security vulnerabilities."""

    def test_sql_injection_protection(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test protection against SQL injection."""
        malicious_input = {
            "point_id": "'; DROP TABLE data_points; --",
            "features": [1.0, 2.0, 3.0],
            "session_id": str(uuid.uuid4()),
        }

        response = test_client.post(
            "/process-point", json=malicious_input, headers=user_headers
        )

        # Should handle gracefully without SQL injection
        assert response.status_code in [200, 400, 422]

    def test_xss_protection(self, test_client: TestClient, user_headers: Dict):
        """Test protection against XSS attacks."""
        malicious_input = {
            "point_id": "<script>alert('xss')</script>",
            "features": [1.0, 2.0, 3.0],
            "session_id": str(uuid.uuid4()),
        }

        response = test_client.post(
            "/process-point", json=malicious_input, headers=user_headers
        )

        # Check response doesn't contain unescaped script
        if response.status_code == 200:
            response_text = response.text
            assert "<script>" not in response_text

    def test_jwt_algorithm_confusion(self, test_client: TestClient):
        """Test protection against JWT algorithm confusion attacks."""
        # Try to create token with 'none' algorithm
        payload = {"sub": "test_user", "exp": datetime.utcnow() + timedelta(hours=1)}

        malicious_token = jwt.encode(payload, "", algorithm="none")
        headers = {"Authorization": f"Bearer {malicious_token}"}

        response = test_client.get("/process-point", headers=headers)

        # Should reject tokens with 'none' algorithm
        assert response.status_code == 401

    def test_timing_attack_protection(self, test_client: TestClient):
        """Test protection against timing attacks on login."""
        import time

        # Test with valid vs invalid usernames
        valid_user = TEST_USERS["user"]["email"]
        invalid_user = "nonexistent@example.com"
        password = "wrong_password"

        # Measure timing for valid user
        start_time = time.time()
        response1 = test_client.post(
            API_ENDPOINTS["auth_login"],
            data={"username": valid_user, "password": password},
        )
        time1 = time.time() - start_time

        # Measure timing for invalid user
        start_time = time.time()
        response2 = test_client.post(
            API_ENDPOINTS["auth_login"],
            data={"username": invalid_user, "password": password},
        )
        time2 = time.time() - start_time

        # Both should fail
        assert response1.status_code == 401
        assert response2.status_code == 401

        # Timing difference should be minimal (less than 100ms)
        timing_difference = abs(time1 - time2)
        assert timing_difference < 0.1
