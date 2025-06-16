"""
Enhanced performance tests for NeuroCluster Streamer API.

This module provides comprehensive performance testing including:
- Throughput and latency benchmarks
- Load testing with concurrent users
- Stress testing under extreme conditions
- Memory and resource utilization testing
- Algorithm performance validation
- Scalability testing across different data sizes
- Performance regression detection
"""

import asyncio
import gc
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import psutil
import pytest
from fastapi.testclient import TestClient

from . import PERFORMANCE_TEST_CONFIG, SAMPLE_DATA_POINTS


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""

    operation_name: str
    total_operations: int
    total_duration_seconds: float
    throughput_ops_per_second: float
    latency_percentiles: Dict[str, float] = field(default_factory=dict)
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_count: int = 0
    success_rate: float = 1.0
    timestamps: List[float] = field(default_factory=list)

    def __post_init__(self):
        """Calculate derived metrics."""
        if self.total_operations > 0:
            self.success_rate = (
                self.total_operations - self.error_count
            ) / self.total_operations
            self.avg_latency_ms = (
                self.total_duration_seconds / self.total_operations
            ) * 1000


@dataclass
class LoadTestResult:
    """Container for load test results."""

    test_name: str
    concurrent_users: int
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    throughput_rps: float
    avg_response_time_ms: float
    percentile_95_ms: float
    percentile_99_ms: float
    error_rate: float
    memory_peak_mb: float
    cpu_peak_percent: float


class PerformanceMonitor:
    """Monitor system resources during performance tests."""

    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.start_time = None

    def start_monitoring(self):
        """Start monitoring system resources."""
        self.monitoring = True
        self.start_time = time.time()
        self.metrics = []

        def monitor_loop():
            while self.monitoring:
                timestamp = time.time() - self.start_time
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()

                self.metrics.append(
                    {
                        "timestamp": timestamp,
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_used_mb": memory.used / (1024 * 1024),
                    }
                )

                time.sleep(0.5)  # Sample every 500ms

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring and return metrics."""
        self.monitoring = False
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(timeout=1)

        if not self.metrics:
            return {
                "peak_cpu": 0,
                "peak_memory_mb": 0,
                "avg_cpu": 0,
                "avg_memory_mb": 0,
            }

        return {
            "peak_cpu": max(m["cpu_percent"] for m in self.metrics),
            "peak_memory_mb": max(m["memory_used_mb"] for m in self.metrics),
            "avg_cpu": statistics.mean(m["cpu_percent"] for m in self.metrics),
            "avg_memory_mb": statistics.mean(m["memory_used_mb"] for m in self.metrics),
            "samples": len(self.metrics),
        }


class PerformanceTester:
    """Main performance testing class."""

    def __init__(self, test_client: TestClient):
        self.client = test_client
        self.monitor = PerformanceMonitor()

    def measure_operation(
        self, operation_func, operation_name: str, iterations: int = 100
    ) -> PerformanceMetrics:
        """Measure performance of a single operation."""
        latencies = []
        errors = 0

        # Warm up
        for _ in range(5):
            try:
                operation_func()
            except:
                pass

        # Force garbage collection before measurement
        gc.collect()

        self.monitor.start_monitoring()
        start_time = time.time()

        for i in range(iterations):
            op_start = time.time()
            try:
                operation_func()
                latency = (time.time() - op_start) * 1000  # Convert to ms
                latencies.append(latency)
            except Exception as e:
                errors += 1
                latencies.append(0)  # Record 0 for failed operations

        end_time = time.time()
        resource_metrics = self.monitor.stop_monitoring()

        total_duration = end_time - start_time
        successful_ops = iterations - errors

        # Calculate percentiles
        valid_latencies = [l for l in latencies if l > 0]
        percentiles = {}
        if valid_latencies:
            percentiles = {
                "p50": statistics.median(valid_latencies),
                "p90": self._percentile(valid_latencies, 90),
                "p95": self._percentile(valid_latencies, 95),
                "p99": self._percentile(valid_latencies, 99),
                "min": min(valid_latencies),
                "max": max(valid_latencies),
            }

        return PerformanceMetrics(
            operation_name=operation_name,
            total_operations=successful_ops,
            total_duration_seconds=total_duration,
            throughput_ops_per_second=(
                successful_ops / total_duration if total_duration > 0 else 0
            ),
            latency_percentiles=percentiles,
            memory_usage_mb=resource_metrics["peak_memory_mb"],
            cpu_usage_percent=resource_metrics["peak_cpu"],
            error_count=errors,
            timestamps=latencies,
        )

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    def load_test(
        self,
        operation_func,
        test_name: str,
        concurrent_users: int,
        duration_seconds: int = 60,
    ) -> LoadTestResult:
        """Perform load testing with concurrent users."""
        results = {
            "response_times": [],
            "errors": 0,
            "successful_requests": 0,
            "total_requests": 0,
        }

        stop_time = time.time() + duration_seconds
        self.monitor.start_monitoring()

        def user_simulation(user_id: int):
            """Simulate a single user's behavior."""
            user_results = {"requests": 0, "errors": 0, "response_times": []}

            while time.time() < stop_time:
                request_start = time.time()
                try:
                    operation_func()
                    response_time = (time.time() - request_start) * 1000
                    user_results["response_times"].append(response_time)
                    user_results["requests"] += 1
                except Exception:
                    user_results["errors"] += 1

                # Small delay between requests (realistic user behavior)
                time.sleep(0.1)

            return user_results

        # Run concurrent users
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [
                executor.submit(user_simulation, user_id)
                for user_id in range(concurrent_users)
            ]

            # Collect results from all users
            for future in as_completed(futures):
                user_result = future.result()
                results["total_requests"] += user_result["requests"]
                results["errors"] += user_result["errors"]
                results["successful_requests"] += user_result["requests"]
                results["response_times"].extend(user_result["response_times"])

        resource_metrics = self.monitor.stop_monitoring()

        # Calculate metrics
        total_requests = results["total_requests"]
        successful_requests = results["successful_requests"]
        response_times = results["response_times"]

        return LoadTestResult(
            test_name=test_name,
            concurrent_users=concurrent_users,
            duration_seconds=duration_seconds,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=results["errors"],
            throughput_rps=(
                total_requests / duration_seconds if duration_seconds > 0 else 0
            ),
            avg_response_time_ms=(
                statistics.mean(response_times) if response_times else 0
            ),
            percentile_95_ms=self._percentile(response_times, 95),
            percentile_99_ms=self._percentile(response_times, 99),
            error_rate=results["errors"] / total_requests if total_requests > 0 else 0,
            memory_peak_mb=resource_metrics["peak_memory_mb"],
            cpu_peak_percent=resource_metrics["peak_cpu"],
        )


@pytest.mark.performance
class TestSingleOperationPerformance:
    """Test performance of individual API operations."""

    def test_health_check_performance(self, test_client: TestClient):
        """Test health check endpoint performance."""
        tester = PerformanceTester(test_client)

        def health_check():
            response = test_client.get("/health")
            assert response.status_code == 200

        metrics = tester.measure_operation(
            health_check, "health_check", iterations=1000
        )

        # Assert performance requirements
        assert (
            metrics.throughput_ops_per_second > 500
        )  # Should handle 500+ health checks per second
        assert metrics.latency_percentiles["p95"] < 100  # 95th percentile under 100ms
        assert metrics.success_rate > 0.99  # 99%+ success rate

        print(f"Health Check Performance:")
        print(f"  Throughput: {metrics.throughput_ops_per_second:.1f} ops/sec")
        print(f"  P95 Latency: {metrics.latency_percentiles['p95']:.1f}ms")
        print(f"  Success Rate: {metrics.success_rate:.3f}")

    def test_single_point_processing_performance(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test single data point processing performance."""
        tester = PerformanceTester(test_client)

        def process_point():
            data = {
                "point_id": f"perf_point_{time.time()}",
                "features": [1.0, 2.0, 3.0],
                "session_id": "perf_session_123",
            }
            response = test_client.post(
                "/process-point", json=data, headers=user_headers
            )
            assert response.status_code == 200

        metrics = tester.measure_operation(
            process_point, "process_point", iterations=500
        )

        # Assert performance requirements
        assert metrics.throughput_ops_per_second > 50  # Target: 50+ points per second
        assert (
            metrics.latency_percentiles["p95"] < 1000
        )  # 95th percentile under 1 second
        assert metrics.success_rate > 0.95  # 95%+ success rate

        print(f"Single Point Processing Performance:")
        print(f"  Throughput: {metrics.throughput_ops_per_second:.1f} points/sec")
        print(f"  P95 Latency: {metrics.latency_percentiles['p95']:.1f}ms")
        print(f"  Memory Usage: {metrics.memory_usage_mb:.1f}MB")

    def test_batch_processing_performance(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test batch processing performance with different sizes."""
        tester = PerformanceTester(test_client)

        batch_sizes = [10, 50, 100, 500]
        results = {}

        for batch_size in batch_sizes:

            def process_batch():
                data = {
                    "session_id": f"batch_session_{batch_size}",
                    "data_points": [
                        {
                            "id": f"batch_point_{i}",
                            "features": [i * 0.1, i * 0.2, i * 0.3],
                        }
                        for i in range(batch_size)
                    ],
                    "clustering_config": {
                        "similarity_threshold": 0.85,
                        "min_cluster_size": 3,
                    },
                }
                response = test_client.post(
                    "/process-batch", json=data, headers=user_headers
                )
                assert response.status_code == 200

            metrics = tester.measure_operation(
                process_batch, f"batch_processing_{batch_size}", iterations=20
            )

            results[batch_size] = metrics

            # Calculate points per second
            points_per_second = batch_size * metrics.throughput_ops_per_second

            print(f"Batch Size {batch_size}:")
            print(
                f"  Batch Throughput: {metrics.throughput_ops_per_second:.1f} batches/sec"
            )
            print(f"  Points Throughput: {points_per_second:.1f} points/sec")
            print(f"  P95 Latency: {metrics.latency_percentiles['p95']:.1f}ms")

        # Test that larger batches are more efficient per point
        small_batch_efficiency = results[10].throughput_ops_per_second * 10
        large_batch_efficiency = results[100].throughput_ops_per_second * 100

        # Large batches should be at least 2x more efficient per point
        assert large_batch_efficiency > small_batch_efficiency * 2


@pytest.mark.performance
class TestConcurrencyPerformance:
    """Test performance under concurrent load."""

    def test_concurrent_health_checks(self, test_client: TestClient):
        """Test health check performance under concurrent load."""
        tester = PerformanceTester(test_client)

        def health_check():
            response = test_client.get("/health")
            assert response.status_code == 200

        concurrent_users = [1, 5, 10, 20]

        for users in concurrent_users:
            result = tester.load_test(
                health_check, f"health_concurrent_{users}", users, duration_seconds=30
            )

            print(f"Concurrent Health Checks ({users} users):")
            print(f"  Throughput: {result.throughput_rps:.1f} req/sec")
            print(f"  Avg Response Time: {result.avg_response_time_ms:.1f}ms")
            print(f"  Error Rate: {result.error_rate:.3f}")

            # Assert performance requirements
            assert result.error_rate < 0.01  # Less than 1% error rate
            assert result.avg_response_time_ms < 200  # Average response under 200ms

    def test_concurrent_data_processing(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test data processing under concurrent load."""
        tester = PerformanceTester(test_client)

        def process_point():
            data = {
                "point_id": f"concurrent_point_{threading.current_thread().ident}_{time.time()}",
                "features": [1.0, 2.0, 3.0],
                "session_id": f"concurrent_session_{threading.current_thread().ident}",
            }
            response = test_client.post(
                "/process-point", json=data, headers=user_headers
            )
            return response.status_code == 200

        concurrent_users = [1, 5, 10]

        for users in concurrent_users:
            result = tester.load_test(
                process_point,
                f"processing_concurrent_{users}",
                users,
                duration_seconds=60,
            )

            print(f"Concurrent Processing ({users} users):")
            print(f"  Throughput: {result.throughput_rps:.1f} req/sec")
            print(f"  P95 Response Time: {result.percentile_95_ms:.1f}ms")
            print(f"  Success Rate: {1 - result.error_rate:.3f}")
            print(f"  Peak Memory: {result.memory_peak_mb:.1f}MB")

            # Assert requirements
            assert result.error_rate < 0.05  # Less than 5% error rate
            assert result.percentile_95_ms < 2000  # P95 under 2 seconds

    def test_mixed_workload_performance(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Test performance with mixed API operations."""
        tester = PerformanceTester(test_client)

        operations = [
            lambda: test_client.get("/health"),
            lambda: test_client.get("/statistics", headers=user_headers),
            lambda: test_client.post(
                "/process-point",
                json={
                    "point_id": f"mixed_point_{time.time()}",
                    "features": [1.0, 2.0, 3.0],
                    "session_id": "mixed_session",
                },
                headers=user_headers,
            ),
        ]

        def mixed_operations():
            import random

            operation = random.choice(operations)
            response = operation()
            return response.status_code < 500

        result = tester.load_test(
            mixed_operations, "mixed_workload", 10, duration_seconds=60
        )

        print(f"Mixed Workload Performance:")
        print(f"  Total Requests: {result.total_requests}")
        print(f"  Throughput: {result.throughput_rps:.1f} req/sec")
        print(f"  Error Rate: {result.error_rate:.3f}")
        print(f"  Peak CPU: {result.cpu_peak_percent:.1f}%")


@pytest.mark.performance
@pytest.mark.slow
class TestStressAndScalabilityTesting:
    """Stress testing and scalability validation."""

    def test_memory_stress_test(self, test_client: TestClient, user_headers: Dict):
        """Test system behavior under memory stress."""
        tester = PerformanceTester(test_client)

        # Process increasingly large batches to stress memory
        batch_sizes = [100, 500, 1000, 2000]
        memory_usage = []

        for batch_size in batch_sizes:

            def large_batch_processing():
                data = {
                    "session_id": f"stress_session_{batch_size}",
                    "data_points": [
                        {
                            "id": f"stress_point_{i}",
                            "features": [i * 0.001] * 100,  # Large feature vectors
                        }
                        for i in range(batch_size)
                    ],
                    "clustering_config": {"similarity_threshold": 0.85},
                }
                response = test_client.post(
                    "/process-batch", json=data, headers=user_headers
                )
                return response.status_code == 200

            metrics = tester.measure_operation(
                large_batch_processing, f"memory_stress_{batch_size}", iterations=5
            )

            memory_usage.append(
                {
                    "batch_size": batch_size,
                    "memory_mb": metrics.memory_usage_mb,
                    "success_rate": metrics.success_rate,
                }
            )

            print(f"Memory Stress Test (batch size {batch_size}):")
            print(f"  Peak Memory: {metrics.memory_usage_mb:.1f}MB")
            print(f"  Success Rate: {metrics.success_rate:.3f}")

            # Should handle at least moderate stress
            if batch_size <= 1000:
                assert metrics.success_rate > 0.8  # 80%+ success for reasonable loads

        # Memory usage should grow somewhat linearly with batch size
        # (not exponentially, which would indicate memory leaks)
        memory_growth = memory_usage[-1]["memory_mb"] / memory_usage[0]["memory_mb"]
        batch_growth = batch_sizes[-1] / batch_sizes[0]

        # Memory growth should not be much more than batch size growth
        assert memory_growth < batch_growth * 2  # Allow 2x factor for overhead

    def test_throughput_scalability(self, test_client: TestClient, user_headers: Dict):
        """Test throughput scalability with increasing load."""
        tester = PerformanceTester(test_client)

        def simple_processing():
            data = {
                "point_id": f"scale_point_{time.time()}_{threading.current_thread().ident}",
                "features": [1.0, 2.0, 3.0],
                "session_id": f"scale_session_{threading.current_thread().ident}",
            }
            response = test_client.post(
                "/process-point", json=data, headers=user_headers
            )
            return response.status_code == 200

        user_counts = [1, 2, 5, 10, 15, 20]
        throughput_results = []

        for users in user_counts:
            result = tester.load_test(
                simple_processing, f"scalability_{users}", users, duration_seconds=30
            )

            throughput_results.append(
                {
                    "users": users,
                    "throughput": result.throughput_rps,
                    "error_rate": result.error_rate,
                    "avg_response_time": result.avg_response_time_ms,
                }
            )

            print(f"Scalability Test ({users} users):")
            print(f"  Throughput: {result.throughput_rps:.1f} req/sec")
            print(f"  Error Rate: {result.error_rate:.3f}")
            print(f"  Avg Response Time: {result.avg_response_time_ms:.1f}ms")

        # Test scalability characteristics
        single_user_throughput = throughput_results[0]["throughput"]
        multi_user_throughput = throughput_results[2]["throughput"]  # 5 users

        # Should achieve some degree of scalability
        scalability_factor = multi_user_throughput / single_user_throughput
        assert (
            scalability_factor > 2.0
        )  # Should at least double throughput with 5x users

        # Error rate should not increase dramatically with load
        low_load_errors = throughput_results[1]["error_rate"]  # 2 users
        high_load_errors = throughput_results[-2]["error_rate"]  # 15 users

        assert high_load_errors - low_load_errors < 0.1  # Error rate increase < 10%

    def test_long_duration_stability(self, test_client: TestClient, user_headers: Dict):
        """Test system stability over extended periods."""
        tester = PerformanceTester(test_client)

        def continuous_processing():
            data = {
                "point_id": f"stability_point_{time.time()}",
                "features": [1.0, 2.0, 3.0],
                "session_id": "stability_session",
            }
            response = test_client.post(
                "/process-point", json=data, headers=user_headers
            )
            return response.status_code == 200

        # Run for 5 minutes with moderate load
        result = tester.load_test(
            continuous_processing, "stability_test", 5, duration_seconds=300
        )

        print(f"Long Duration Stability Test (5 minutes):")
        print(f"  Total Requests: {result.total_requests}")
        print(f"  Average Throughput: {result.throughput_rps:.1f} req/sec")
        print(f"  Final Error Rate: {result.error_rate:.3f}")
        print(f"  Peak Memory: {result.memory_peak_mb:.1f}MB")

        # System should remain stable over time
        assert result.error_rate < 0.02  # Less than 2% error rate
        assert result.successful_requests > 1000  # Should process substantial volume
        assert result.memory_peak_mb < 1024  # Should not consume excessive memory


@pytest.mark.performance
class TestPerformanceRegression:
    """Test for performance regressions."""

    def test_baseline_performance_metrics(
        self, test_client: TestClient, user_headers: Dict
    ):
        """Establish baseline performance metrics for regression testing."""
        tester = PerformanceTester(test_client)

        # Define baseline tests
        baseline_tests = [
            {
                "name": "health_check",
                "operation": lambda: test_client.get("/health"),
                "iterations": 100,
                "expected_throughput": 200,  # req/sec
                "expected_p95_latency": 50,  # ms
            },
            {
                "name": "single_point_processing",
                "operation": lambda: test_client.post(
                    "/process-point",
                    json={
                        "point_id": f"baseline_{time.time()}",
                        "features": [1.0, 2.0, 3.0],
                        "session_id": "baseline_session",
                    },
                    headers=user_headers,
                ),
                "iterations": 50,
                "expected_throughput": 20,  # req/sec
                "expected_p95_latency": 500,  # ms
            },
        ]

        baseline_results = {}

        for test in baseline_tests:
            metrics = tester.measure_operation(
                test["operation"], test["name"], test["iterations"]
            )

            baseline_results[test["name"]] = {
                "throughput": metrics.throughput_ops_per_second,
                "p95_latency": metrics.latency_percentiles.get("p95", 0),
                "success_rate": metrics.success_rate,
                "memory_usage": metrics.memory_usage_mb,
            }

            print(f"Baseline {test['name']}:")
            print(f"  Throughput: {metrics.throughput_ops_per_second:.1f} ops/sec")
            print(f"  P95 Latency: {metrics.latency_percentiles.get('p95', 0):.1f}ms")
            print(f"  Memory: {metrics.memory_usage_mb:.1f}MB")

            # Assert meets minimum baseline requirements
            assert (
                metrics.throughput_ops_per_second >= test["expected_throughput"] * 0.8
            )  # 20% tolerance
            assert (
                metrics.latency_percentiles.get("p95", float("inf"))
                <= test["expected_p95_latency"] * 1.2
            )
            assert metrics.success_rate >= 0.95

        # Store baseline results for future regression testing
        baseline_file = "performance_baseline.json"
        try:
            with open(baseline_file, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "version": "1.0.0",
                        "results": baseline_results,
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass  # Don't fail test if we can't write baseline


@pytest.mark.performance
class TestResourceUtilization:
    """Test resource utilization and efficiency."""

    def test_memory_efficiency(self, test_client: TestClient, user_headers: Dict):
        """Test memory usage efficiency."""
        tester = PerformanceTester(test_client)

        # Test memory usage with different data sizes
        data_sizes = [10, 100, 1000]
        memory_results = []

        for size in data_sizes:

            def memory_test():
                data = {
                    "session_id": f"memory_test_{size}",
                    "data_points": [
                        {
                            "id": f"mem_point_{i}",
                            "features": [float(i)] * 10,  # 10D features
                        }
                        for i in range(size)
                    ],
                    "clustering_config": {"similarity_threshold": 0.85},
                }
                response = test_client.post(
                    "/process-batch", json=data, headers=user_headers
                )
                return response.status_code == 200

            metrics = tester.measure_operation(
                memory_test, f"memory_efficiency_{size}", iterations=3
            )

            memory_per_point = metrics.memory_usage_mb / size if size > 0 else 0
            memory_results.append(
                {
                    "data_size": size,
                    "total_memory_mb": metrics.memory_usage_mb,
                    "memory_per_point_kb": memory_per_point * 1024,
                }
            )

            print(f"Memory Efficiency (size {size}):")
            print(f"  Total Memory: {metrics.memory_usage_mb:.1f}MB")
            print(f"  Memory per Point: {memory_per_point * 1024:.1f}KB")

            # Memory per point should be reasonable
            assert memory_per_point * 1024 < 100  # Less than 100KB per point

        # Memory efficiency should not degrade significantly with size
        small_efficiency = memory_results[0]["memory_per_point_kb"]
        large_efficiency = memory_results[-1]["memory_per_point_kb"]

        # Large batches should be more memory efficient per point
        assert (
            large_efficiency <= small_efficiency * 2
        )  # Allow some overhead but not excessive

    def test_cpu_utilization(self, test_client: TestClient, user_headers: Dict):
        """Test CPU utilization patterns."""
        tester = PerformanceTester(test_client)

        def cpu_intensive_processing():
            # Process multiple points to create CPU load
            for i in range(10):
                data = {
                    "point_id": f"cpu_point_{i}_{time.time()}",
                    "features": [float(j) for j in range(50)],  # 50D features
                    "session_id": "cpu_test_session",
                }
                response = test_client.post(
                    "/process-point", json=data, headers=user_headers
                )
                assert response.status_code == 200

        metrics = tester.measure_operation(
            cpu_intensive_processing, "cpu_utilization", iterations=20
        )

        print(f"CPU Utilization Test:")
        print(f"  Peak CPU: {metrics.cpu_usage_percent:.1f}%")
        print(f"  Throughput: {metrics.throughput_ops_per_second:.1f} ops/sec")
        print(
            f"  CPU Efficiency: {metrics.throughput_ops_per_second / max(metrics.cpu_usage_percent, 1):.2f} ops/sec per CPU%"
        )

        # CPU usage should be reasonable
        assert metrics.cpu_usage_percent < 95  # Should not max out CPU completely

        # Should achieve reasonable CPU efficiency
        cpu_efficiency = metrics.throughput_ops_per_second / max(
            metrics.cpu_usage_percent, 1
        )
        assert (
            cpu_efficiency > 0.1
        )  # At least 0.1 operations per second per CPU percent


def generate_performance_report(results: List[PerformanceMetrics]) -> str:
    """Generate a comprehensive performance report."""
    report = ["Performance Test Report", "=" * 50, ""]

    for metrics in results:
        report.extend(
            [
                f"Test: {metrics.operation_name}",
                f"  Operations: {metrics.total_operations}",
                f"  Duration: {metrics.total_duration_seconds:.2f}s",
                f"  Throughput: {metrics.throughput_ops_per_second:.1f} ops/sec",
                f"  Success Rate: {metrics.success_rate:.3f}",
                f"  Latency P95: {metrics.latency_percentiles.get('p95', 0):.1f}ms",
                f"  Memory Usage: {metrics.memory_usage_mb:.1f}MB",
                f"  CPU Usage: {metrics.cpu_usage_percent:.1f}%",
                "",
            ]
        )

    return "\n".join(report)


if __name__ == "__main__":
    """Run performance tests standalone."""
    print("NeuroCluster Streamer API Performance Tests")
    print("Note: Run with pytest for full test suite integration")
