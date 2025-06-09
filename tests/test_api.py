"""
API endpoint tests for NeuroCluster Streamer API.

This module tests all API endpoints including:
- Health check endpoints
- Data processing endpoints
- Cluster management endpoints
- Statistics and metrics endpoints
- Session management endpoints
- Error handling and edge cases
"""

import json
import time
import uuid
from typing import Dict, Any
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from . import SAMPLE_DATA_POINTS, SAMPLE_CLUSTERING_CONFIG, API_ENDPOINTS

class TestHealthEndpoints:
    """Test health check and status endpoints."""
    
    def test_basic_health_check(self, test_client: TestClient):
        """Test basic health endpoint."""
        response = test_client.get(API_ENDPOINTS["health"])
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
    
    def test_detailed_health_check(self, test_client: TestClient):
        """Test detailed health endpoint with component checks."""
        response = test_client.get("/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "overall_status" in data
        assert "components" in data
        assert "timestamp" in data
        
        # Check component health data
        components = data["components"]
        expected_components = ["database", "algorithm", "system_resources"]
        
        for component in expected_components:
            if component in components:
                assert "status" in components[component]
                assert "duration_ms" in components[component]
    
    def test_health_check_with_unhealthy_component(self, test_client: TestClient):
        """Test health check when components are unhealthy."""
        with patch('monitoring.health.get_health_checker') as mock_health:
            mock_checker = MagicMock()
            mock_checker.check_health.return_value = {
                "status": "unhealthy",
                "message": "Database connection failed",
                "checks": {
                    "database": {
                        "status": "unhealthy",
                        "message": "Connection timeout"
                    }
                }
            }
            mock_health.return_value = mock_checker
            
            response = test_client.get(API_ENDPOINTS["health"])
            
            # Should still return 200 but with unhealthy status
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"

class TestProcessingEndpoints:
    """Test data processing endpoints."""
    
    def test_process_single_point_success(self, test_client: TestClient, user_headers: Dict):
        """Test successful single point processing."""
        point_data = {
            "point_id": "test_point_1",
            "features": [1.0, 2.0, 3.0],
            "session_id": str(uuid.uuid4())
        }
        
        response = test_client.post(
            API_ENDPOINTS["process_point"],
            json=point_data,
            headers=user_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "result" in data
        assert "processing_time_ms" in data["result"]
        assert "cluster_id" in data["result"]
        assert "confidence" in data["result"]
    
    def test_process_single_point_invalid_data(self, test_client: TestClient, user_headers: Dict):
        """Test single point processing with invalid data."""
        invalid_data = {
            "point_id": "",  # Invalid empty ID
            "features": [],  # Invalid empty features
            "session_id": "invalid_uuid"  # Invalid UUID
        }
        
        response = test_client.post(
            API_ENDPOINTS["process_point"],
            json=invalid_data,
            headers=user_headers
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_process_single_point_missing_auth(self, test_client: TestClient):
        """Test single point processing without authentication."""
        point_data = {
            "point_id": "test_point_1",
            "features": [1.0, 2.0, 3.0],
            "session_id": str(uuid.uuid4())
        }
        
        response = test_client.post(
            API_ENDPOINTS["process_point"],
            json=point_data
        )
        
        assert response.status_code == 401  # Unauthorized
    
    def test_process_batch_success(self, test_client: TestClient, user_headers: Dict):
        """Test successful batch processing."""
        session_id = str(uuid.uuid4())
        batch_data = {
            "session_id": session_id,
            "data_points": SAMPLE_DATA_POINTS,
            "clustering_config": SAMPLE_CLUSTERING_CONFIG
        }
        
        response = test_client.post(
            API_ENDPOINTS["process_batch"],
            json=batch_data,
            headers=user_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "session_id" in data
        assert "results" in data
        assert len(data["results"]) == len(SAMPLE_DATA_POINTS)
        
        # Check each result
        for result in data["results"]:
            assert "point_id" in result
            assert "cluster_id" in result
            assert "confidence" in result
            assert "processing_time_ms" in result
    
    def test_process_batch_large_dataset(self, test_client: TestClient, user_headers: Dict, large_data_batch):
        """Test batch processing with large dataset."""
        session_id = str(uuid.uuid4())
        batch_data = {
            "session_id": session_id,
            "data_points": large_data_batch[:100],  # Limit for test performance
            "clustering_config": SAMPLE_CLUSTERING_CONFIG
        }
        
        start_time = time.time()
        response = test_client.post(
            API_ENDPOINTS["process_batch"],
            json=batch_data,
            headers=user_headers
        )
        processing_time = time.time() - start_time
        
        assert response.status_code == 200
        assert processing_time < 30  # Should complete within 30 seconds
        
        data = response.json()
        assert len(data["results"]) == 100
    
    def test_process_batch_concurrent_sessions(self, test_client: TestClient, user_headers: Dict):
        """Test concurrent batch processing for different sessions."""
        import threading
        
        results = []
        
        def process_batch(session_num):
            session_id = str(uuid.uuid4())
            batch_data = {
                "session_id": session_id,
                "data_points": SAMPLE_DATA_POINTS,
                "clustering_config": SAMPLE_CLUSTERING_CONFIG
            }
            
            response = test_client.post(
                API_ENDPOINTS["process_batch"],
                json=batch_data,
                headers=user_headers
            )
            results.append((session_num, response.status_code, response.json()))
        
        # Run 3 concurrent sessions
        threads = []
        for i in range(3):
            thread = threading.Thread(target=process_batch, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all sessions succeeded
        assert len(results) == 3
        for session_num, status_code, data in results:
            assert status_code == 200
            assert data["success"] is True
    
    def test_process_batch_algorithm_error(self, test_client: TestClient, user_headers: Dict, simulate_algorithm_error):
        """Test batch processing with algorithm error."""
        with patch('app.dependencies.get_algorithm_instance') as mock_algo:
            mock_instance = MagicMock()
            mock_instance.process_point.side_effect = simulate_algorithm_error
            mock_algo.return_value = mock_instance
            
            batch_data = {
                "session_id": str(uuid.uuid4()),
                "data_points": SAMPLE_DATA_POINTS[:1],  # Single point to trigger error
                "clustering_config": SAMPLE_CLUSTERING_CONFIG
            }
            
            response = test_client.post(
                API_ENDPOINTS["process_batch"],
                json=batch_data,
                headers=user_headers
            )
            
            assert response.status_code == 500
            data = response.json()
            assert data["success"] is False
            assert "error" in data

class TestClusterEndpoints:
    """Test cluster management endpoints."""
    
    def test_get_clusters_success(self, test_client: TestClient, user_headers: Dict, seed_test_data):
        """Test successful cluster retrieval."""
        session_id = seed_test_data["session"].id
        
        response = test_client.get(
            f"/clusters?session_id={session_id}",
            headers=user_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "clusters" in data
        assert len(data["clusters"]) >= 1
        
        # Check cluster structure
        cluster = data["clusters"][0]
        assert "id" in cluster
        assert "centroid" in cluster
        assert "size" in cluster
        assert "health" in cluster
    
    def test_get_clusters_pagination(self, test_client: TestClient, user_headers: Dict, seed_test_data):
        """Test cluster retrieval with pagination."""
        session_id = seed_test_data["session"].id
        
        response = test_client.get(
            f"/clusters?session_id={session_id}&skip=0&limit=2",
            headers=user_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "clusters" in data
        assert "pagination" in data
        assert data["pagination"]["skip"] == 0
        assert data["pagination"]["limit"] == 2
    
    def test_get_cluster_details(self, test_client: TestClient, user_headers: Dict, seed_test_data):
        """Test detailed cluster information retrieval."""
        cluster_id = seed_test_data["cluster"].id
        
        response = test_client.get(
            f"/clusters/{cluster_id}",
            headers=user_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "cluster" in data
        cluster = data["cluster"]
        assert cluster["id"] == str(cluster_id)
        assert "statistics" in cluster
        assert "data_points" in cluster
    
    def test_get_cluster_not_found(self, test_client: TestClient, user_headers: Dict):
        """Test cluster retrieval with invalid ID."""
        invalid_id = str(uuid.uuid4())
        
        response = test_client.get(
            f"/clusters/{invalid_id}",
            headers=user_headers
        )
        
        assert response.status_code == 404
    
    def test_get_clusters_unauthorized(self, test_client: TestClient):
        """Test cluster retrieval without authentication."""
        response = test_client.get("/clusters")
        
        assert response.status_code == 401

class TestSessionEndpoints:
    """Test session management endpoints."""
    
    def test_get_session_success(self, test_client: TestClient, user_headers: Dict, seed_test_data):
        """Test successful session retrieval."""
        session_id = seed_test_data["session"].id
        
        response = test_client.get(
            API_ENDPOINTS["get_session"].format(session_id=session_id),
            headers=user_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "session" in data
        session = data["session"]
        assert session["id"] == str(session_id)
        assert "statistics" in data
        assert "clusters" in data
    
    def test_get_session_not_found(self, test_client: TestClient, user_headers: Dict):
        """Test session retrieval with invalid ID."""
        invalid_id = str(uuid.uuid4())
        
        response = test_client.get(
            API_ENDPOINTS["get_session"].format(session_id=invalid_id),
            headers=user_headers
        )
        
        assert response.status_code == 404
    
    def test_delete_session_success(self, test_client: TestClient, admin_headers: Dict, seed_test_data):
        """Test successful session deletion."""
        session_id = seed_test_data["session"].id
        
        response = test_client.delete(
            f"/session/{session_id}",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_delete_session_insufficient_permissions(self, test_client: TestClient, user_headers: Dict, seed_test_data):
        """Test session deletion with insufficient permissions."""
        session_id = seed_test_data["session"].id
        
        response = test_client.delete(
            f"/session/{session_id}",
            headers=user_headers
        )
        
        assert response.status_code == 403  # Forbidden

class TestStatisticsEndpoints:
    """Test statistics and metrics endpoints."""
    
    def test_get_statistics_success(self, test_client: TestClient, user_headers: Dict, seed_test_data):
        """Test successful statistics retrieval."""
        session_id = seed_test_data["session"].id
        
        response = test_client.get(
            f"{API_ENDPOINTS['get_statistics']}?session_id={session_id}",
            headers=user_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "statistics" in data
        stats = data["statistics"]
        assert "total_points" in stats
        assert "active_clusters" in stats
        assert "processing_performance" in stats
    
    def test_get_global_statistics(self, test_client: TestClient, admin_headers: Dict):
        """Test global statistics retrieval."""
        response = test_client.get(
            API_ENDPOINTS["get_statistics"],
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "statistics" in data
        assert "global_metrics" in data["statistics"]
    
    def test_get_metrics_prometheus_format(self, test_client: TestClient, admin_headers: Dict):
        """Test metrics endpoint in Prometheus format."""
        response = test_client.get(
            API_ENDPOINTS["get_metrics"],
            headers=admin_headers
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        # Check for Prometheus metrics format
        content = response.text
        assert "# HELP" in content or "# TYPE" in content

class TestDataExportEndpoints:
    """Test data export endpoints."""
    
    def test_export_session_data_csv(self, test_client: TestClient, user_headers: Dict, seed_test_data):
        """Test session data export in CSV format."""
        session_id = seed_test_data["session"].id
        
        response = test_client.get(
            f"/export/session/{session_id}?format=csv",
            headers=user_headers
        )
        
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        
        # Check CSV content
        content = response.text
        assert "point_id" in content
        assert "features" in content
        assert "cluster_id" in content
    
    def test_export_session_data_json(self, test_client: TestClient, user_headers: Dict, seed_test_data):
        """Test session data export in JSON format."""
        session_id = seed_test_data["session"].id
        
        response = test_client.get(
            f"/export/session/{session_id}?format=json",
            headers=user_headers
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert "session_info" in data
        assert "data_points" in data
        assert "clusters" in data
    
    def test_export_clusters_only(self, test_client: TestClient, user_headers: Dict, seed_test_data):
        """Test exporting only cluster information."""
        session_id = seed_test_data["session"].id
        
        response = test_client.get(
            f"/export/clusters/{session_id}",
            headers=user_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "clusters" in data
        assert "cluster_statistics" in data

class TestErrorHandling:
    """Test API error handling."""
    
    def test_invalid_json_payload(self, test_client: TestClient, user_headers: Dict):
        """Test handling of invalid JSON payload."""
        response = test_client.post(
            API_ENDPOINTS["process_point"],
            data="invalid json",
            headers={**user_headers, "Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_required_fields(self, test_client: TestClient, user_headers: Dict):
        """Test handling of missing required fields."""
        incomplete_data = {
            "point_id": "test_point"
            # Missing features and session_id
        }
        
        response = test_client.post(
            API_ENDPOINTS["process_point"],
            json=incomplete_data,
            headers=user_headers
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_database_error_handling(self, test_client: TestClient, user_headers: Dict, simulate_database_error):
        """Test handling of database errors."""
        with patch('database.crud.session_crud.create') as mock_create:
            mock_create.side_effect = simulate_database_error
            
            response = test_client.post(
                API_ENDPOINTS["process_point"],
                json={
                    "point_id": "test_point",
                    "features": [1.0, 2.0, 3.0],
                    "session_id": str(uuid.uuid4())
                },
                headers=user_headers
            )
            
            assert response.status_code == 500
            data = response.json()
            assert data["success"] is False
    
    def test_rate_limiting(self, test_client: TestClient, user_headers: Dict):
        """Test API rate limiting."""
        # Make many rapid requests
        responses = []
        for i in range(50):  # Exceed typical rate limit
            response = test_client.get(
                API_ENDPOINTS["health"],
                headers=user_headers
            )
            responses.append(response.status_code)
        
        # Should eventually get rate limited
        assert 429 in responses  # Too Many Requests
    
    def test_request_timeout(self, test_client: TestClient, user_headers: Dict):
        """Test request timeout handling."""
        with patch('NCS_V8.NCSClusteringAlgorithm.process_point') as mock_process:
            # Simulate slow processing
            import time
            def slow_process(*args, **kwargs):
                time.sleep(10)  # Longer than timeout
                return {"cluster_id": "test"}
            
            mock_process.side_effect = slow_process
            
            response = test_client.post(
                API_ENDPOINTS["process_point"],
                json={
                    "point_id": "test_point",
                    "features": [1.0, 2.0, 3.0],
                    "session_id": str(uuid.uuid4())
                },
                headers=user_headers,
                timeout=5  # 5 second timeout
            )
            
            # Should handle timeout gracefully
            assert response.status_code in [408, 500]  # Timeout or Server Error

class TestContentNegotiation:
    """Test content negotiation and response formats."""
    
    def test_json_response_default(self, test_client: TestClient, user_headers: Dict):
        """Test default JSON response format."""
        response = test_client.get(
            API_ENDPOINTS["health"],
            headers=user_headers
        )
        
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
    
    def test_accept_header_json(self, test_client: TestClient, user_headers: Dict):
        """Test JSON response with Accept header."""
        headers = {**user_headers, "Accept": "application/json"}
        
        response = test_client.get(
            API_ENDPOINTS["health"],
            headers=headers
        )
        
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
    
    def test_unsupported_media_type(self, test_client: TestClient, user_headers: Dict):
        """Test unsupported media type handling."""
        headers = {**user_headers, "Content-Type": "application/xml"}
        
        response = test_client.post(
            API_ENDPOINTS["process_point"],
            data="<xml>invalid</xml>",
            headers=headers
        )
        
        assert response.status_code == 415  # Unsupported Media Type

@pytest.mark.performance
class TestPerformanceRequirements:
    """Test API performance requirements."""
    
    def test_health_check_response_time(self, test_client: TestClient):
        """Test health check response time requirement."""
        start_time = time.time()
        response = test_client.get(API_ENDPOINTS["health"])
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 0.1  # Should respond in under 100ms
    
    def test_single_point_processing_time(self, test_client: TestClient, user_headers: Dict):
        """Test single point processing time requirement."""
        point_data = {
            "point_id": "perf_test_point",
            "features": [1.0, 2.0, 3.0],
            "session_id": str(uuid.uuid4())
        }
        
        start_time = time.time()
        response = test_client.post(
            API_ENDPOINTS["process_point"],
            json=point_data,
            headers=user_headers
        )
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0  # Should process in under 1 second