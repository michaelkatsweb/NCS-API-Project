"""
NCS Clustering Algorithm Implementation
High-performance clustering with multiple algorithm support
"""

import asyncio
import concurrent.futures
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Clustering result container"""

    clusters: Optional[np.ndarray]
    centroids: Optional[np.ndarray]
    quality_score: float
    n_clusters: int
    algorithm_info: Dict[str, Any]
    processing_time_ms: float


class NCSClusteringAlgorithm:
    """
    Neural Clustering System - High-performance clustering algorithm
    Supports multiple clustering algorithms with quality assessment
    """

    def __init__(self):
        self.version = "1.0.0"
        self.scaler = StandardScaler()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def get_version(self) -> str:
        """Get algorithm version"""
        return self.version

    async def cluster_ncs(
        self,
        data: np.ndarray,
        n_clusters: Optional[int] = None,
        quality_threshold: float = 0.85,
        max_iterations: int = 100,
        random_seed: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Main NCS clustering algorithm
        Proprietary high-performance clustering with automatic cluster detection
        """
        start_time = time.time()

        logger.info(f"Starting NCS clustering for {len(data)} points")

        # Input validation
        if len(data) < 2:
            raise ValueError("Need at least 2 data points for clustering")

        if data.shape[1] < 1:
            raise ValueError("Data must have at least 1 dimension")

        # Normalize data
        data_normalized = self._normalize_data(data)

        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = await self._find_optimal_clusters(
                data_normalized, max_clusters=min(20, len(data) // 2)
            )

        # Run NCS algorithm
        clusters, centroids, quality = await self._run_ncs_algorithm(
            data_normalized, n_clusters, quality_threshold, max_iterations, random_seed
        )

        # Denormalize centroids
        if centroids is not None:
            centroids = self.scaler.inverse_transform(centroids)

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"NCS clustering completed in {processing_time:.2f}ms with quality {quality:.3f}"
        )

        return {
            "clusters": clusters,
            "centroids": centroids,
            "quality_score": quality,
            "n_clusters": n_clusters,
            "algorithm": "ncs",
            "processing_time_ms": processing_time,
        }

    async def cluster_kmeans(
        self,
        data: np.ndarray,
        n_clusters: int,
        max_iterations: int = 300,
        random_seed: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Standard K-Means clustering"""
        start_time = time.time()

        if n_clusters < 1:
            raise ValueError("Number of clusters must be at least 1")

        if n_clusters > len(data):
            raise ValueError("Number of clusters cannot exceed number of data points")

        # Normalize data
        data_normalized = self._normalize_data(data)

        # Run K-Means in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._run_kmeans,
            data_normalized,
            n_clusters,
            max_iterations,
            random_seed,
        )

        clusters, centroids = result

        # Calculate quality score
        quality = self._calculate_silhouette_score(data_normalized, clusters)

        # Denormalize centroids
        centroids = self.scaler.inverse_transform(centroids)

        processing_time = (time.time() - start_time) * 1000

        logger.info(f"K-Means clustering completed in {processing_time:.2f}ms")

        return {
            "clusters": clusters,
            "centroids": centroids,
            "quality_score": quality,
            "n_clusters": n_clusters,
            "algorithm": "kmeans",
            "processing_time_ms": processing_time,
        }

    async def cluster_dbscan(
        self, data: np.ndarray, eps: float = 0.5, min_samples: int = 5, **kwargs
    ) -> Dict[str, Any]:
        """DBSCAN clustering algorithm"""
        start_time = time.time()

        # Normalize data
        data_normalized = self._normalize_data(data)

        # Run DBSCAN in thread pool
        loop = asyncio.get_event_loop()
        clusters = await loop.run_in_executor(
            self.executor, self._run_dbscan, data_normalized, eps, min_samples
        )

        # Calculate cluster statistics
        unique_clusters = np.unique(clusters)
        n_clusters = len(unique_clusters[unique_clusters != -1])  # Exclude noise (-1)

        # Calculate quality score (only for non-noise points)
        non_noise_mask = clusters != -1
        if np.sum(non_noise_mask) > 1 and n_clusters > 1:
            quality = self._calculate_silhouette_score(
                data_normalized[non_noise_mask], clusters[non_noise_mask]
            )
        else:
            quality = 0.0

        # DBSCAN doesn't have centroids, calculate them manually
        centroids = None
        if n_clusters > 0:
            centroids = []
            for cluster_id in unique_clusters:
                if cluster_id != -1:  # Skip noise
                    cluster_points = data_normalized[clusters == cluster_id]
                    centroid = np.mean(cluster_points, axis=0)
                    centroids.append(centroid)

            if centroids:
                centroids = np.array(centroids)
                centroids = self.scaler.inverse_transform(centroids)

        processing_time = (time.time() - start_time) * 1000

        logger.info(f"DBSCAN clustering completed in {processing_time:.2f}ms")

        return {
            "clusters": clusters,
            "centroids": centroids,
            "quality_score": quality,
            "n_clusters": n_clusters,
            "algorithm": "dbscan",
            "processing_time_ms": processing_time,
        }

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using StandardScaler"""
        return self.scaler.fit_transform(data)

    async def _find_optimal_clusters(
        self, data: np.ndarray, max_clusters: int = 20
    ) -> int:
        """Find optimal number of clusters using multiple metrics"""

        if len(data) <= max_clusters:
            return min(3, len(data) // 2)

        # Test different cluster numbers
        scores = []
        cluster_range = range(2, min(max_clusters + 1, len(data)))

        for k in cluster_range:
            try:
                # Quick K-means run
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=50)
                labels = kmeans.fit_predict(data)

                # Calculate multiple quality metrics
                silhouette = silhouette_score(data, labels)
                calinski = calinski_harabasz_score(data, labels)

                # Weighted score (silhouette is more important)
                combined_score = 0.7 * silhouette + 0.3 * (calinski / 1000)
                scores.append((k, combined_score))

            except Exception as e:
                logger.warning(f"Error calculating scores for k={k}: {e}")
                scores.append((k, 0.0))

        # Find best k
        if scores:
            best_k = max(scores, key=lambda x: x[1])[0]
            logger.info(f"Optimal cluster count determined: {best_k}")
            return best_k

        # Fallback
        return min(5, len(data) // 10)

    async def _run_ncs_algorithm(
        self,
        data: np.ndarray,
        n_clusters: int,
        quality_threshold: float,
        max_iterations: int,
        random_seed: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the proprietary NCS algorithm
        This is a simplified implementation - in production this would be more sophisticated
        """

        # For now, use an enhanced K-Means with quality optimization
        best_clusters = None
        best_centroids = None
        best_quality = 0.0

        # Multiple runs with different initializations
        num_runs = min(10, max_iterations // 10)

        for run in range(num_runs):
            seed = random_seed + run if random_seed else None

            # Run K-Means
            clusters, centroids = self._run_kmeans(
                data, n_clusters, max_iterations // num_runs, seed
            )

            # Calculate quality
            quality = self._calculate_silhouette_score(data, clusters)

            # Keep best result
            if quality > best_quality:
                best_quality = quality
                best_clusters = clusters
                best_centroids = centroids

                # Early stopping if quality threshold reached
                if quality >= quality_threshold:
                    logger.info(f"Quality threshold reached in run {run + 1}")
                    break

        return best_clusters, best_centroids, best_quality

    def _run_kmeans(
        self,
        data: np.ndarray,
        n_clusters: int,
        max_iterations: int,
        random_seed: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run K-Means clustering"""
        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iterations,
            random_state=random_seed,
            n_init=1,  # Single initialization for speed
        )

        clusters = kmeans.fit_predict(data)
        centroids = kmeans.cluster_centers_

        return clusters, centroids

    def _run_dbscan(self, data: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
        """Run DBSCAN clustering"""
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(data)

        return clusters

    def _calculate_silhouette_score(
        self, data: np.ndarray, clusters: np.ndarray
    ) -> float:
        """Calculate silhouette score for quality assessment"""
        try:
            # Check if we have enough clusters and points
            unique_clusters = np.unique(clusters)

            # Remove noise points for DBSCAN (-1 labels)
            valid_mask = clusters != -1
            if np.sum(valid_mask) < 2:
                return 0.0

            valid_data = data[valid_mask]
            valid_clusters = clusters[valid_mask]

            unique_valid = np.unique(valid_clusters)
            if len(unique_valid) < 2:
                return 0.0

            score = silhouette_score(valid_data, valid_clusters)
            return max(0.0, score)  # Ensure non-negative

        except Exception as e:
            logger.warning(f"Error calculating silhouette score: {e}")
            return 0.0

    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about available algorithms"""
        return {
            "ncs": {
                "name": "Neural Clustering System",
                "description": "Proprietary high-performance clustering algorithm",
                "supports_auto_clusters": True,
                "quality_optimization": True,
            },
            "kmeans": {
                "name": "K-Means",
                "description": "Classic centroid-based clustering",
                "supports_auto_clusters": False,
                "quality_optimization": False,
            },
            "dbscan": {
                "name": "DBSCAN",
                "description": "Density-based clustering with noise detection",
                "supports_auto_clusters": True,
                "quality_optimization": False,
            },
        }

    async def benchmark_performance(
        self,
        data_sizes: List[int] = [100, 1000, 5000, 10000],
        algorithms: List[str] = ["ncs", "kmeans"],
    ) -> Dict[str, List[Dict]]:
        """Benchmark clustering performance"""
        results = {alg: [] for alg in algorithms}

        for size in data_sizes:
            # Generate random test data
            test_data = np.random.rand(size, 2)

            for algorithm in algorithms:
                try:
                    start_time = time.time()

                    if algorithm == "ncs":
                        result = await self.cluster_ncs(test_data, n_clusters=5)
                    elif algorithm == "kmeans":
                        result = await self.cluster_kmeans(test_data, n_clusters=5)

                    processing_time = time.time() - start_time

                    results[algorithm].append(
                        {
                            "data_size": size,
                            "processing_time_seconds": processing_time,
                            "points_per_second": size / processing_time,
                            "quality_score": result["quality_score"],
                        }
                    )

                except Exception as e:
                    logger.error(
                        f"Benchmark error for {algorithm} with size {size}: {e}"
                    )
                    results[algorithm].append({"data_size": size, "error": str(e)})

        return results

    def __del__(self):
        """Cleanup executor on destruction"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
