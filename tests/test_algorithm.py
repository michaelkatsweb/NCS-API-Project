"""
Algorithm tests for NeuroCluster Streamer (NCS) clustering algorithm.

This module tests the core NCS algorithm functionality including:
- Data point processing and clustering
- Cluster formation and management
- Outlier detection and handling
- Performance metrics and statistics
- Memory management and optimization
- Edge cases and error conditions
"""

import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from NCS_V8 import NCSClusteringAlgorithm

from . import SAMPLE_CLUSTERING_CONFIG, SAMPLE_DATA_POINTS


class TestAlgorithmInitialization:
    """Test algorithm initialization and configuration."""

    def test_default_initialization(self):
        """Test algorithm initialization with default parameters."""
        algorithm = NCSClusteringAlgorithm()

        assert algorithm.similarity_threshold == 0.85
        assert algorithm.min_cluster_size == 3
        assert algorithm.max_clusters == 1000
        assert algorithm.outlier_threshold == 0.75
        assert algorithm.adaptive_threshold is True

    def test_custom_initialization(self):
        """Test algorithm initialization with custom parameters."""
        config = {
            "similarity_threshold": 0.9,
            "min_cluster_size": 5,
            "max_clusters": 500,
            "outlier_threshold": 0.8,
            "adaptive_threshold": False,
        }

        algorithm = NCSClusteringAlgorithm(**config)

        assert algorithm.similarity_threshold == 0.9
        assert algorithm.min_cluster_size == 5
        assert algorithm.max_clusters == 500
        assert algorithm.outlier_threshold == 0.8
        assert algorithm.adaptive_threshold is False

    def test_invalid_parameters(self):
        """Test algorithm initialization with invalid parameters."""
        with pytest.raises((ValueError, TypeError)):
            NCSClusteringAlgorithm(similarity_threshold=-0.5)  # Negative threshold

        with pytest.raises((ValueError, TypeError)):
            NCSClusteringAlgorithm(min_cluster_size=0)  # Zero cluster size

        with pytest.raises((ValueError, TypeError)):
            NCSClusteringAlgorithm(max_clusters=-1)  # Negative max clusters


class TestDataPointProcessing:
    """Test individual data point processing."""

    def test_process_single_point_new_cluster(self, test_algorithm):
        """Test processing first data point (creates new cluster)."""
        point = {"id": "point_1", "features": [1.0, 2.0, 3.0]}

        result = test_algorithm.process_point(
            point_id=point["id"], features=point["features"]
        )

        assert "cluster_id" in result
        assert "is_outlier" in result
        assert "confidence" in result
        assert "processing_time_ms" in result

        assert result["is_outlier"] is False  # First point shouldn't be outlier
        assert result["confidence"] > 0
        assert result["processing_time_ms"] > 0

    def test_process_similar_points_same_cluster(self, test_algorithm):
        """Test that similar points are assigned to the same cluster."""
        # Process first point
        point1 = {"id": "point_1", "features": [1.0, 2.0, 3.0]}
        result1 = test_algorithm.process_point(
            point_id=point1["id"], features=point1["features"]
        )

        # Process similar point
        point2 = {"id": "point_2", "features": [1.1, 2.1, 3.1]}
        result2 = test_algorithm.process_point(
            point_id=point2["id"], features=point2["features"]
        )

        # Should be in same cluster
        assert result1["cluster_id"] == result2["cluster_id"]
        assert result2["is_outlier"] is False
        assert result2["confidence"] > 0.5  # High confidence for similar points

    def test_process_dissimilar_points_different_clusters(self, test_algorithm):
        """Test that dissimilar points create different clusters."""
        # Process first point
        point1 = {"id": "point_1", "features": [1.0, 2.0, 3.0]}
        result1 = test_algorithm.process_point(
            point_id=point1["id"], features=point1["features"]
        )

        # Process dissimilar point
        point2 = {"id": "point_2", "features": [10.0, 20.0, 30.0]}
        result2 = test_algorithm.process_point(
            point_id=point2["id"], features=point2["features"]
        )

        # Should be in different clusters
        assert result1["cluster_id"] != result2["cluster_id"]
        assert result2["is_outlier"] is False  # Still forms valid cluster

    def test_process_outlier_point(self, test_algorithm):
        """Test outlier detection."""
        # Create a small cluster first
        cluster_points = [
            {"id": "point_1", "features": [1.0, 2.0, 3.0]},
            {"id": "point_2", "features": [1.1, 2.1, 3.1]},
            {"id": "point_3", "features": [1.2, 2.2, 3.2]},
        ]

        for point in cluster_points:
            test_algorithm.process_point(
                point_id=point["id"], features=point["features"]
            )

        # Process outlier point
        outlier_point = {"id": "outlier", "features": [100.0, 200.0, 300.0]}
        result = test_algorithm.process_point(
            point_id=outlier_point["id"], features=outlier_point["features"]
        )

        # Should be detected as outlier if far enough
        # (depending on outlier threshold and algorithm parameters)
        assert "is_outlier" in result
        assert "outlier_score" in result

    def test_process_point_invalid_features(self, test_algorithm):
        """Test processing point with invalid features."""
        invalid_cases = [
            {"id": "test", "features": []},  # Empty features
            {"id": "test", "features": [1.0, "invalid"]},  # Non-numeric features
            {"id": "test", "features": None},  # None features
        ]

        for case in invalid_cases:
            with pytest.raises((ValueError, TypeError)):
                test_algorithm.process_point(
                    point_id=case["id"], features=case["features"]
                )

    def test_process_point_duplicate_id(self, test_algorithm):
        """Test processing point with duplicate ID."""
        point = {"id": "duplicate_point", "features": [1.0, 2.0, 3.0]}

        # Process first time
        result1 = test_algorithm.process_point(
            point_id=point["id"], features=point["features"]
        )

        # Process again with same ID
        result2 = test_algorithm.process_point(
            point_id=point["id"], features=[1.5, 2.5, 3.5]  # Different features
        )

        # Should handle gracefully (update or reject)
        assert result2 is not None


class TestBatchProcessing:
    """Test batch processing functionality."""

    def test_process_batch_success(self, test_algorithm):
        """Test successful batch processing."""
        batch_results = []

        for point in SAMPLE_DATA_POINTS:
            result = test_algorithm.process_point(
                point_id=point["id"], features=point["features"]
            )
            batch_results.append(result)

        assert len(batch_results) == len(SAMPLE_DATA_POINTS)

        # All results should have required fields
        for result in batch_results:
            assert "cluster_id" in result
            assert "is_outlier" in result
            assert "confidence" in result
            assert "processing_time_ms" in result

    def test_process_batch_performance(self, test_algorithm):
        """Test batch processing performance."""
        # Generate larger dataset
        large_batch = []
        for i in range(1000):
            features = [
                1.0 + (i % 10) * 0.1,
                2.0 + (i % 10) * 0.1,
                3.0 + (i % 10) * 0.1,
            ]
            large_batch.append({"id": f"point_{i}", "features": features})

        start_time = time.time()

        for point in large_batch:
            test_algorithm.process_point(
                point_id=point["id"], features=point["features"]
            )

        processing_time = time.time() - start_time
        throughput = len(large_batch) / processing_time

        # Should achieve reasonable throughput (target: >1000 points/sec)
        assert throughput > 100  # Minimum acceptable throughput

    def test_concurrent_processing(self, test_algorithm):
        """Test concurrent point processing."""
        import threading

        results = []
        errors = []

        def process_points(start_idx, count):
            try:
                for i in range(start_idx, start_idx + count):
                    result = test_algorithm.process_point(
                        point_id=f"concurrent_point_{i}",
                        features=[i * 0.1, i * 0.2, i * 0.3],
                    )
                    results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_points, args=(i * 20, 20))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0  # No errors should occur
        assert len(results) == 100  # All points processed


class TestClusterManagement:
    """Test cluster creation, management, and evolution."""

    def test_cluster_creation(self, test_algorithm):
        """Test automatic cluster creation."""
        initial_clusters = len(test_algorithm.get_clusters())

        # Process points to create clusters
        test_algorithm.process_point("point_1", [1.0, 2.0, 3.0])
        test_algorithm.process_point("point_2", [10.0, 20.0, 30.0])

        clusters = test_algorithm.get_clusters()
        assert len(clusters) > initial_clusters

    def test_cluster_growth(self, test_algorithm):
        """Test cluster growth as similar points are added."""
        # Create initial cluster
        test_algorithm.process_point("point_1", [1.0, 2.0, 3.0])

        clusters_before = test_algorithm.get_clusters()
        initial_cluster_id = None
        for cluster in clusters_before:
            if cluster["size"] > 0:
                initial_cluster_id = cluster["id"]
                initial_size = cluster["size"]
                break

        # Add similar points
        test_algorithm.process_point("point_2", [1.1, 2.1, 3.1])
        test_algorithm.process_point("point_3", [1.2, 2.2, 3.2])

        clusters_after = test_algorithm.get_clusters()

        # Find the same cluster and check growth
        for cluster in clusters_after:
            if cluster["id"] == initial_cluster_id:
                assert cluster["size"] > initial_size
                break

    def test_cluster_merging(self, test_algorithm):
        """Test cluster merging when clusters become similar."""
        # This test depends on algorithm implementation details
        # Create two separate small clusters
        test_algorithm.process_point("point_1", [1.0, 2.0, 3.0])
        test_algorithm.process_point("point_2", [5.0, 6.0, 7.0])

        initial_cluster_count = len(
            [c for c in test_algorithm.get_clusters() if c["size"] > 0]
        )

        # Add points that bridge the clusters
        for i in range(10):
            features = [1.0 + i * 0.4, 2.0 + i * 0.4, 3.0 + i * 0.4]
            test_algorithm.process_point(f"bridge_point_{i}", features)

        final_cluster_count = len(
            [c for c in test_algorithm.get_clusters() if c["size"] > 0]
        )

        # Clusters might merge (depends on algorithm implementation)
        assert final_cluster_count <= initial_cluster_count + 1

    def test_cluster_health_monitoring(self, test_algorithm):
        """Test cluster health monitoring."""
        # Create cluster with several points
        for i in range(10):
            test_algorithm.process_point(
                f"health_point_{i}", [1.0 + i * 0.1, 2.0 + i * 0.1, 3.0 + i * 0.1]
            )

        clusters = test_algorithm.get_clusters()

        for cluster in clusters:
            if cluster["size"] > 0:
                assert "health" in cluster
                assert cluster["health"] in ["healthy", "degraded", "unhealthy"]

                # Healthy clusters should have reasonable properties
                if cluster["health"] == "healthy":
                    assert cluster["size"] >= test_algorithm.min_cluster_size
                    assert "centroid" in cluster
                    assert len(cluster["centroid"]) > 0


class TestOutlierDetection:
    """Test outlier detection functionality."""

    def test_outlier_detection_threshold(self, test_algorithm):
        """Test outlier detection based on threshold."""
        # Create a tight cluster
        cluster_points = [
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [1.2, 2.2, 3.2],
            [1.3, 2.3, 3.3],
        ]

        for i, features in enumerate(cluster_points):
            test_algorithm.process_point(f"cluster_point_{i}", features)

        # Process clear outlier
        outlier_result = test_algorithm.process_point(
            "outlier_point", [100.0, 200.0, 300.0]
        )

        # Check outlier detection
        assert "is_outlier" in outlier_result
        assert "outlier_score" in outlier_result

        if outlier_result["is_outlier"]:
            assert outlier_result["outlier_score"] > test_algorithm.outlier_threshold

    def test_adaptive_outlier_threshold(self):
        """Test adaptive outlier threshold adjustment."""
        algorithm = NCSClusteringAlgorithm(adaptive_threshold=True)

        # Process several normal points
        normal_points = [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]]

        for i, features in enumerate(normal_points):
            algorithm.process_point(f"normal_{i}", features)

        # Get initial threshold
        initial_threshold = algorithm.outlier_threshold

        # Process some outliers
        outlier_points = [[50.0, 60.0, 70.0], [100.0, 110.0, 120.0]]

        for i, features in enumerate(outlier_points):
            algorithm.process_point(f"outlier_{i}", features)

        # Threshold might adapt
        final_threshold = algorithm.outlier_threshold

        # With adaptive threshold, it might change
        # (depends on algorithm implementation)
        assert isinstance(final_threshold, (int, float))

    def test_outlier_statistics(self, test_algorithm):
        """Test outlier statistics tracking."""
        # Process mix of normal points and outliers
        points = [
            {"features": [1.0, 2.0, 3.0], "expected_outlier": False},
            {"features": [1.1, 2.1, 3.1], "expected_outlier": False},
            {"features": [100.0, 200.0, 300.0], "expected_outlier": True},
            {"features": [1.2, 2.2, 3.2], "expected_outlier": False},
        ]

        for i, point in enumerate(points):
            test_algorithm.process_point(f"test_point_{i}", point["features"])

        stats = test_algorithm.get_statistics()

        assert "outliers_detected" in stats
        assert "outlier_percentage" in stats
        assert stats["outliers_detected"] >= 0
        assert 0 <= stats["outlier_percentage"] <= 100


class TestPerformanceMetrics:
    """Test performance monitoring and metrics."""

    def test_processing_time_tracking(self, test_algorithm):
        """Test processing time measurement."""
        result = test_algorithm.process_point("timing_test", [1.0, 2.0, 3.0])

        assert "processing_time_ms" in result
        assert result["processing_time_ms"] > 0
        assert result["processing_time_ms"] < 1000  # Should be fast

    def test_throughput_measurement(self, test_algorithm):
        """Test throughput measurement."""
        start_time = time.time()

        # Process batch of points
        for i in range(100):
            test_algorithm.process_point(
                f"throughput_point_{i}", [i * 0.1, i * 0.2, i * 0.3]
            )

        end_time = time.time()
        duration = end_time - start_time
        throughput = 100 / duration

        stats = test_algorithm.get_statistics()

        # Check throughput tracking
        if "throughput_points_per_sec" in stats:
            assert stats["throughput_points_per_sec"] > 0

    def test_memory_usage_tracking(self, test_algorithm):
        """Test memory usage monitoring."""
        # Process many points to increase memory usage
        for i in range(1000):
            test_algorithm.process_point(
                f"memory_test_{i}", [i * 0.01, i * 0.02, i * 0.03]
            )

        memory_usage = test_algorithm.get_memory_usage()

        assert isinstance(memory_usage, (int, float))
        assert memory_usage > 0

    def test_statistics_comprehensive(self, test_algorithm):
        """Test comprehensive statistics collection."""
        # Process varied data
        for i in range(50):
            features = [1.0 + (i % 5) * 2.0, 2.0 + (i % 5) * 2.0, 3.0 + (i % 5) * 2.0]
            test_algorithm.process_point(f"stats_point_{i}", features)

        stats = test_algorithm.get_statistics()

        # Check required statistics
        required_stats = [
            "total_points_processed",
            "active_clusters",
            "total_clusters_created",
            "outliers_detected",
            "avg_processing_time_ms",
            "memory_usage_mb",
        ]

        for stat in required_stats:
            assert stat in stats
            assert isinstance(stats[stat], (int, float))


class TestMemoryManagement:
    """Test memory management and optimization."""

    def test_bounded_memory_usage(self, test_algorithm):
        """Test that memory usage remains bounded."""
        initial_memory = test_algorithm.get_memory_usage()

        # Process large number of points
        for i in range(5000):
            test_algorithm.process_point(
                f"memory_point_{i}", [i * 0.001, i * 0.002, i * 0.003]
            )

        final_memory = test_algorithm.get_memory_usage()
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (not unbounded)
        assert memory_growth < 500  # Less than 500MB growth

    def test_cluster_cleanup(self, test_algorithm):
        """Test cleanup of inactive clusters."""
        # Create many small clusters
        for i in range(100):
            # Spread out points to create separate clusters
            test_algorithm.process_point(
                f"cleanup_point_{i}", [i * 10.0, i * 10.0, i * 10.0]
            )

        initial_cluster_count = len(test_algorithm.get_clusters())

        # Algorithm should eventually clean up small/inactive clusters
        # (depending on implementation)
        assert initial_cluster_count > 0

    def test_data_point_cleanup(self, test_algorithm):
        """Test cleanup of old data points."""
        # Process many points
        for i in range(1000):
            test_algorithm.process_point(f"old_point_{i}", [1.0, 2.0, 3.0])

        stats_before = test_algorithm.get_statistics()

        # Force cleanup if implemented
        if hasattr(test_algorithm, "cleanup_old_data"):
            test_algorithm.cleanup_old_data()

            stats_after = test_algorithm.get_statistics()

            # Memory usage might decrease after cleanup
            memory_before = stats_before.get("memory_usage_mb", 0)
            memory_after = stats_after.get("memory_usage_mb", 0)

            assert memory_after <= memory_before


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataset(self, test_algorithm):
        """Test algorithm behavior with no data."""
        stats = test_algorithm.get_statistics()

        assert stats["total_points_processed"] == 0
        assert stats["active_clusters"] == 0
        assert len(test_algorithm.get_clusters()) == 0

    def test_single_point_dataset(self, test_algorithm):
        """Test algorithm with only one data point."""
        test_algorithm.process_point("single_point", [1.0, 2.0, 3.0])

        stats = test_algorithm.get_statistics()
        clusters = test_algorithm.get_clusters()

        assert stats["total_points_processed"] == 1
        assert len(clusters) >= 1

    def test_identical_points(self, test_algorithm):
        """Test processing multiple identical points."""
        identical_features = [1.0, 2.0, 3.0]

        results = []
        for i in range(10):
            result = test_algorithm.process_point(
                f"identical_{i}", identical_features.copy()
            )
            results.append(result)

        # All should be in same cluster
        cluster_ids = [r["cluster_id"] for r in results]
        assert len(set(cluster_ids)) == 1  # Only one unique cluster ID

    def test_high_dimensional_data(self, test_algorithm):
        """Test algorithm with high-dimensional data."""
        high_dim_features = [i * 0.1 for i in range(100)]  # 100 dimensions

        result = test_algorithm.process_point("high_dim", high_dim_features)

        assert "cluster_id" in result
        assert "confidence" in result

    def test_extreme_values(self, test_algorithm):
        """Test algorithm with extreme numerical values."""
        extreme_cases = [
            [1e-10, 1e-10, 1e-10],  # Very small values
            [1e10, 1e10, 1e10],  # Very large values
            [0.0, 0.0, 0.0],  # Zero values
            [-1e5, -1e5, -1e5],  # Large negative values
        ]

        for i, features in enumerate(extreme_cases):
            result = test_algorithm.process_point(f"extreme_{i}", features)

            assert result is not None
            assert "cluster_id" in result

    def test_nan_and_infinite_values(self, test_algorithm):
        """Test algorithm handling of NaN and infinite values."""
        invalid_cases = [
            [float("nan"), 1.0, 2.0],  # NaN values
            [float("inf"), 1.0, 2.0],  # Infinite values
            [1.0, float("-inf"), 2.0],  # Negative infinite
        ]

        for i, features in enumerate(invalid_cases):
            with pytest.raises((ValueError, TypeError)):
                test_algorithm.process_point(f"invalid_{i}", features)


@pytest.mark.performance
class TestAlgorithmPerformance:
    """Test algorithm performance requirements."""

    def test_target_throughput(self, test_algorithm):
        """Test algorithm meets target throughput."""
        start_time = time.time()

        # Process 1000 points
        for i in range(1000):
            test_algorithm.process_point(
                f"perf_point_{i}", [i % 10, (i % 10) + 1, (i % 10) + 2]
            )

        end_time = time.time()
        duration = end_time - start_time
        throughput = 1000 / duration

        # Should achieve target throughput (>1000 points/sec)
        assert throughput > 1000

    def test_memory_efficiency(self, test_algorithm):
        """Test memory efficiency with large datasets."""
        initial_memory = test_algorithm.get_memory_usage()

        # Process 10,000 points
        for i in range(10000):
            test_algorithm.process_point(
                f"memory_perf_{i}", [i % 100, (i % 100) + 1, (i % 100) + 2]
            )

        final_memory = test_algorithm.get_memory_usage()
        memory_per_point = (final_memory - initial_memory) / 10000

        # Should use reasonable memory per point (<1KB per point)
        assert memory_per_point < 1.0  # Less than 1MB per 1000 points

    def test_processing_time_consistency(self, test_algorithm):
        """Test consistent processing times."""
        processing_times = []

        for i in range(100):
            start_time = time.time()
            test_algorithm.process_point(f"timing_{i}", [i, i + 1, i + 2])
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            processing_times.append(processing_time)

        # Calculate statistics
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)

        # Processing should be consistent
        assert avg_time < 10  # Average under 10ms
        assert max_time < 50  # No single operation over 50ms
