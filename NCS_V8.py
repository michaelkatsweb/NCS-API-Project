"""
NeuroCluster Streamer V8 - High-Performance Adaptive Clustering Algorithm
========================================================================

A vectorized streaming clustering algorithm with adaptive intelligence capabilities.
Achieves >6,300 points/second processing with sub-millisecond latency.

Based on research by Michael Katsaros (2025)
"NeuroCluster Streamer: A High-Performance Adaptive Clustering Algorithm
for Real-Time Data Streams"

Key Features:
- Vectorized NumPy operations for maximum performance
- Dynamic threshold adaptation based on data characteristics
- Multi-layer outlier detection (geometric, statistical, temporal)
- Intelligent cluster health monitoring and management
- Concept drift detection and adaptation
- Memory-efficient bounded collections architecture

Author: Based on NCS Algorithm by Katsaros Michael
Year: 2025
"""

import numpy as np
import time
import warnings
from collections import deque
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
import threading
import gc


@dataclass
class ClusterInfo:
    """Information about a cluster."""

    centroid: np.ndarray
    confidence: float
    age: float
    updates: int
    last_update: int
    health_score: float


class NeuroClusterStreamer:
    """
    High-performance streaming clustering algorithm with adaptive intelligence.

    The algorithm uses vectorized operations, dynamic thresholds, and multi-layer
    outlier detection to achieve superior performance and clustering quality.
    """

    def __init__(
        self,
        base_threshold: float = 0.71,
        learning_rate: float = 0.06,
        decay_rate: float = 0.002,
        min_confidence: float = 0.2,
        merge_threshold: float = 0.9,
        outlier_threshold: float = 0.2,
        stability_window: int = 100,
        validation_window: int = 15,
        performance_mode: bool = True,
        max_clusters: int = 30,
        max_batch_size: int = 10000,
        max_point_dimensions: int = 1000,
        latency_warning_threshold_ms: float = 0.2,
        memory_warning_threshold_mb: float = 100.0,
    ):
        """
        Initialize NeuroCluster Streamer with optimized parameters.

        Args:
            base_threshold: Base similarity threshold for cluster assignment
            learning_rate: Learning rate for centroid updates
            decay_rate: Confidence decay rate over time
            min_confidence: Minimum confidence for cluster retention
            merge_threshold: Threshold for merging similar clusters
            outlier_threshold: Threshold for outlier detection
            stability_window: Window size for stability calculations
            validation_window: Window size for validation metrics
            performance_mode: Enable performance optimizations
            max_clusters: Maximum number of clusters
            max_batch_size: Maximum batch size for processing
            max_point_dimensions: Maximum dimensions per point
            latency_warning_threshold_ms: Warning threshold for processing time
            memory_warning_threshold_mb: Warning threshold for memory usage
        """
        # Core algorithm parameters
        self.base_threshold = base_threshold
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.min_confidence = min_confidence
        self.merge_threshold = merge_threshold
        self.outlier_threshold = outlier_threshold

        # Performance parameters
        self.stability_window = stability_window
        self.validation_window = validation_window
        self.performance_mode = performance_mode
        self.max_clusters = max_clusters
        self.max_batch_size = max_batch_size
        self.max_point_dimensions = max_point_dimensions

        # Warning thresholds
        self.latency_warning_threshold_ms = latency_warning_threshold_ms
        self.memory_warning_threshold_mb = memory_warning_threshold_mb

        # Algorithm state
        self.clusters: List[Dict[str, Any]] = []
        self.time_step = 0
        self.total_points_processed = 0

        # Bounded tracking collections for memory efficiency
        self.similarity_history = deque(maxlen=stability_window)
        self.distance_history = deque(maxlen=stability_window)
        self.quality_history = deque(maxlen=50)
        self.outlier_buffer = deque(maxlen=25)

        # Performance tracking
        self.processing_times = deque(maxlen=50)
        self.error_count = 0
        self.warning_count = 0

        # Adaptive state
        self.current_dynamic_threshold = base_threshold
        self.global_stability = 1.0
        self.clustering_quality = 1.0
        self.adaptation_rate = 1.0

        # Caching for performance (when performance_mode is True)
        self._threshold_cache = {}
        self._cluster_array_cache = None
        self._cache_dirty = True
        self._last_cache_time = 0

        # Thread safety
        self._lock = threading.RLock()

        # Suppress NumPy warnings for performance
        if self.performance_mode:
            warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Initialize internal state
        self._initialize_state()

    def _initialize_state(self):
        """Initialize internal algorithm state."""
        # Pre-allocate arrays for performance
        if self.performance_mode:
            self._temp_similarities = np.zeros(self.max_clusters, dtype=np.float32)
            self._temp_distances = np.zeros(self.max_clusters, dtype=np.float32)

        # Initialize statistics
        self._stats = {
            "total_points_processed": 0,
            "clusters_created": 0,
            "clusters_merged": 0,
            "outliers_detected": 0,
            "avg_processing_time_ms": 0.0,
            "max_processing_time_ms": 0.0,
            "memory_usage_estimate_mb": 0.0,
        }

    def process_data_point(
        self, point: Union[List[float], np.ndarray]
    ) -> Tuple[int, bool, float]:
        """
        Process a single data point through the NCS algorithm.

        Args:
            point: Input data point as list or numpy array

        Returns:
            Tuple of (cluster_id, is_outlier, outlier_score)
            - cluster_id: Assigned cluster ID (-1 for outliers)
            - is_outlier: Boolean indicating if point is an outlier
            - outlier_score: Outlier confidence score [0-1]
        """
        start_time = time.perf_counter()

        try:
            with self._lock:
                # Convert and validate input
                x_array = self._validate_and_convert_point(point)

                # Increment time step
                self.time_step += 1
                self.total_points_processed += 1

                # Multi-layer outlier detection
                outlier_score = self._enhanced_outlier_detection(x_array)
                if outlier_score > self.outlier_threshold:
                    self.outlier_buffer.append(x_array.copy())
                    self._stats["outliers_detected"] += 1
                    processing_time = (time.perf_counter() - start_time) * 1000
                    self._record_processing_time(processing_time)
                    return -1, True, outlier_score

                # Find best cluster using vectorized operations
                best_cluster_idx, best_similarity = self._vectorized_find_best_cluster(
                    x_array
                )

                # Get dynamic threshold
                dynamic_threshold = self._get_dynamic_threshold()

                # Cluster assignment decision
                if (
                    best_cluster_idx is not None
                    and best_similarity >= dynamic_threshold
                ):
                    # Update existing cluster
                    self._adaptive_cluster_update(best_cluster_idx, x_array)
                    cluster_id = best_cluster_idx
                    is_outlier = False
                else:
                    # Check if we can create a new cluster
                    if self._can_create_cluster():
                        cluster_id = self._create_new_cluster(x_array)
                        is_outlier = False
                        outlier_score = 0.0
                    else:
                        # Add to outlier buffer
                        self.outlier_buffer.append(x_array.copy())
                        self._stats["outliers_detected"] += 1
                        cluster_id = -1
                        is_outlier = True

                # Periodic maintenance
                if self.time_step % 50 == 0:
                    self._enhanced_maintenance()

                # Record performance metrics
                processing_time = (time.perf_counter() - start_time) * 1000
                self._record_processing_time(processing_time)

                return cluster_id, is_outlier, outlier_score

        except Exception as e:
            self.error_count += 1
            processing_time = (time.perf_counter() - start_time) * 1000
            self._record_processing_time(processing_time)
            # Return outlier classification for error cases
            return -1, True, 1.0

    def _validate_and_convert_point(
        self, point: Union[List[float], np.ndarray]
    ) -> np.ndarray:
        """Validate and convert input point to numpy array."""
        if isinstance(point, list):
            if len(point) == 0:
                raise ValueError("Empty point provided")
            if len(point) > self.max_point_dimensions:
                raise ValueError(
                    f"Point has {len(point)} dimensions, max allowed: {self.max_point_dimensions}"
                )
            x_array = np.array(point, dtype=np.float32)
        elif isinstance(point, np.ndarray):
            if point.size == 0:
                raise ValueError("Empty point provided")
            if point.size > self.max_point_dimensions:
                raise ValueError(
                    f"Point has {point.size} dimensions, max allowed: {self.max_point_dimensions}"
                )
            x_array = point.astype(np.float32)
        else:
            raise ValueError(f"Point must be list or numpy array, got {type(point)}")

        # Check for invalid values
        if not np.isfinite(x_array).all():
            raise ValueError("Point contains invalid values (NaN or infinity)")

        return x_array

    def _enhanced_outlier_detection(self, point: np.ndarray) -> float:
        """
        Multi-layer outlier detection combining geometric, statistical, and temporal analysis.

        Args:
            point: Input data point

        Returns:
            Outlier score [0-1], higher values indicate higher outlier probability
        """
        if len(self.clusters) == 0:
            return 0.0  # No clusters yet, not an outlier

        # Layer 1: Geometric distance analysis
        geometric_score = self._geometric_outlier_detection(point)

        # Layer 2: Statistical anomaly detection
        statistical_score = self._statistical_outlier_detection(point)

        # Layer 3: Temporal coherence validation
        temporal_score = self._temporal_outlier_detection(point)

        # Combine scores with weighted average
        outlier_score = (
            0.4 * geometric_score + 0.3 * statistical_score + 0.3 * temporal_score
        )

        return np.clip(outlier_score, 0.0, 1.0)

    def _geometric_outlier_detection(self, point: np.ndarray) -> float:
        """Layer 1: Geometric distance-based outlier detection."""
        if len(self.clusters) == 0:
            return 0.0

        # Compute minimum distance to any cluster centroid
        min_distance = float("inf")
        for cluster in self.clusters:
            distance = np.linalg.norm(point - cluster["centroid"])
            min_distance = min(min_distance, distance)

        # Adaptive distance threshold based on historical distances
        if len(self.distance_history) > 10:
            distance_mean = np.mean(list(self.distance_history))
            distance_std = np.std(list(self.distance_history))
            adaptive_threshold = distance_mean + 2.0 * distance_std
        else:
            adaptive_threshold = 2.0  # Default threshold

        # Record distance for future use
        self.distance_history.append(min_distance)

        # Compute geometric outlier score
        if min_distance > adaptive_threshold:
            return min(1.0, min_distance / adaptive_threshold - 1.0)
        else:
            return 0.0

    def _statistical_outlier_detection(self, point: np.ndarray) -> float:
        """Layer 2: Statistical anomaly detection using z-score analysis."""
        if len(self.distance_history) < 10:
            return 0.0

        # Compute minimum distance to clusters
        min_distance = float("inf")
        for cluster in self.clusters:
            distance = np.linalg.norm(point - cluster["centroid"])
            min_distance = min(min_distance, distance)

        # Z-score analysis
        distance_mean = np.mean(list(self.distance_history))
        distance_std = np.std(list(self.distance_history))

        if distance_std > 1e-6:  # Avoid division by zero
            z_score = abs(min_distance - distance_mean) / distance_std
            if z_score > 2.5:  # Statistical outlier threshold
                return min(1.0, (z_score - 2.5) / 2.5)

        return 0.0

    def _temporal_outlier_detection(self, point: np.ndarray) -> float:
        """Layer 3: Temporal coherence validation."""
        if len(self.outlier_buffer) < 3:
            return 0.0

        # Analyze recent points for coherence
        recent_points = list(self.outlier_buffer)[-5:]  # Last 5 points

        if len(recent_points) < 3:
            return 0.0

        # Compute centroid of recent points
        recent_centroid = np.mean(recent_points, axis=0)

        # Compute variance within recent points
        distances_to_centroid = [
            np.linalg.norm(p - recent_centroid) for p in recent_points
        ]
        coherence = 1.0 / (1.0 + np.std(distances_to_centroid))

        # Low coherence indicates temporal inconsistency (potential outlier)
        if coherence < 0.3:
            return min(1.0, (0.3 - coherence) / 0.3)

        return 0.0

    def _vectorized_find_best_cluster(
        self, point: np.ndarray
    ) -> Tuple[Optional[int], float]:
        """
        Find best cluster using vectorized operations for maximum performance.

        Args:
            point: Input data point

        Returns:
            Tuple of (best_cluster_index, best_similarity)
        """
        if len(self.clusters) == 0:
            return None, 0.0

        # Build cluster matrix for vectorized operations
        cluster_matrix = self._get_cluster_matrix()

        if cluster_matrix is None or cluster_matrix.shape[0] == 0:
            return None, 0.0

        # Vectorized similarity computation
        similarities = self._compute_vectorized_similarities(point, cluster_matrix)

        # Find best cluster
        if len(similarities) > 0:
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]

            # Record similarity for adaptive threshold computation
            self.similarity_history.append(best_similarity)

            return int(best_idx), float(best_similarity)

        return None, 0.0

    def _get_cluster_matrix(self) -> Optional[np.ndarray]:
        """Get cluster centroids as a matrix for vectorized operations."""
        if len(self.clusters) == 0:
            return None

        # Use caching for performance
        if (
            self.performance_mode
            and self._cluster_array_cache is not None
            and not self._cache_dirty
        ):
            return self._cluster_array_cache

        try:
            # Build matrix from cluster centroids
            centroids = []
            for cluster in self.clusters:
                centroids.append(cluster["centroid"])

            if len(centroids) > 0:
                cluster_matrix = np.vstack(centroids)

                # Cache the result
                if self.performance_mode:
                    self._cluster_array_cache = cluster_matrix
                    self._cache_dirty = False

                return cluster_matrix
        except Exception:
            # Handle dimension mismatch or other errors
            pass

        return None

    def _compute_vectorized_similarities(
        self, point: np.ndarray, cluster_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarities using vectorized operations.

        Args:
            point: Input data point
            cluster_matrix: Matrix of cluster centroids

        Returns:
            Array of similarity scores
        """
        try:
            # Compute dot products
            dot_products = np.dot(cluster_matrix, point)

            # Compute norms
            point_norm = np.linalg.norm(point)
            centroid_norms = np.linalg.norm(cluster_matrix, axis=1)

            # Avoid division by zero
            valid_mask = (centroid_norms > 1e-10) & (point_norm > 1e-10)

            similarities = np.zeros(len(cluster_matrix))
            if np.any(valid_mask) and point_norm > 1e-10:
                similarities[valid_mask] = dot_products[valid_mask] / (
                    point_norm * centroid_norms[valid_mask]
                )

                # Clip to valid range
                similarities = np.clip(similarities, -1.0, 1.0)

            return similarities

        except Exception:
            # Fallback to zeros if computation fails
            return np.zeros(len(cluster_matrix))

    def _get_dynamic_threshold(self) -> float:
        """
        Compute dynamic threshold based on data characteristics and system state.

        Returns:
            Adaptive threshold value
        """
        # Use cached value if available and fresh
        current_time = self.time_step
        cache_key = current_time // 10  # Cache for 10 time steps

        if self.performance_mode and cache_key in self._threshold_cache:
            return self._threshold_cache[cache_key]

        # Compute dynamic threshold
        if len(self.similarity_history) < 10:
            threshold = self.base_threshold
        else:
            similarities = np.array(list(self.similarity_history))

            # Base threshold from 75th percentile of similarities
            p75_similarity = np.percentile(similarities, 75)

            # Stability bonus/penalty
            similarity_std = np.std(similarities)
            stability_factor = max(0.5, 1.0 - similarity_std)

            # Quality bonus
            quality_bonus = self.clustering_quality * 0.1

            # Global stability bonus
            stability_bonus = self.global_stability * 0.05

            # Compute adaptive threshold
            threshold = (
                p75_similarity * stability_factor + quality_bonus + stability_bonus
            )

            # Constrain to reasonable bounds
            threshold = np.clip(threshold, 0.3, 0.95)

        # Cache the result
        if self.performance_mode:
            self._threshold_cache[cache_key] = threshold

            # Clean old cache entries
            if len(self._threshold_cache) > 10:
                old_keys = [
                    k for k in self._threshold_cache.keys() if k < cache_key - 5
                ]
                for old_key in old_keys:
                    del self._threshold_cache[old_key]

        self.current_dynamic_threshold = threshold
        return threshold

    def _adaptive_cluster_update(self, cluster_idx: int, point: np.ndarray):
        """
        Update cluster centroid using adaptive learning rate.

        Args:
            cluster_idx: Index of cluster to update
            point: New data point
        """
        if cluster_idx >= len(self.clusters):
            return

        cluster = self.clusters[cluster_idx]

        # Compute adaptive learning rate
        age_factor = 1.0 / (1.0 + cluster["age"] * 0.01)
        stability_factor = min(1.2, 1.0 + cluster["confidence"] * 0.2)
        adaptive_lr = self.learning_rate * age_factor * stability_factor

        # Temporal smoothing for noise reduction
        smoothed_point = 0.7 * point + 0.3 * cluster["centroid"]

        # Update centroid with adaptive learning rate
        cluster["centroid"] = (1.0 - adaptive_lr) * cluster[
            "centroid"
        ] + adaptive_lr * smoothed_point

        # Update cluster metadata
        cluster["confidence"] += 0.3
        cluster["age"] += 0.1
        cluster["updates"] += 1
        cluster["last_update"] = self.time_step

        # Recompute health score
        cluster["health_score"] = self._compute_cluster_health(cluster)

        # Mark cache as dirty
        self._cache_dirty = True

    def _can_create_cluster(self) -> bool:
        """Check if a new cluster can be created."""
        return len(self.clusters) < self.max_clusters

    def _create_new_cluster(self, point: np.ndarray) -> int:
        """
        Create a new cluster with the given point as centroid.

        Args:
            point: Data point to use as cluster centroid

        Returns:
            Index of newly created cluster
        """
        new_cluster = {
            "centroid": point.copy(),
            "confidence": 1.0,
            "age": 0.0,
            "updates": 1,
            "last_update": self.time_step,
            "health_score": 1.0,
        }

        self.clusters.append(new_cluster)
        self._stats["clusters_created"] += 1

        # Mark cache as dirty
        self._cache_dirty = True

        return len(self.clusters) - 1

    def _compute_cluster_health(self, cluster: Dict[str, Any]) -> float:
        """
        Compute comprehensive health score for a cluster.

        Args:
            cluster: Cluster dictionary

        Returns:
            Health score [0-1]
        """
        # Age factor (mature clusters are more stable)
        age_factor = min(1.2, 1.0 + cluster["age"] / 200.0)

        # Consistency factor (regular updates indicate relevance)
        consistency_factor = min(
            1.1, 0.8 + (cluster["updates"] / (cluster["age"] + 1)) * 0.3
        )

        # Recency factor (recent updates indicate ongoing relevance)
        recency = max(0.5, 1.0 - (self.time_step - cluster["last_update"]) / 50.0)

        # Combine factors
        health_score = cluster["confidence"] * age_factor * consistency_factor * recency

        return min(1.0, health_score)

    def _enhanced_maintenance(self):
        """Perform enhanced maintenance operations for optimal performance."""
        # Remove weak clusters
        self._remove_weak_clusters()

        # Apply confidence decay
        self._apply_confidence_decay()

        # Process outlier buffer
        self._process_outlier_buffer()

        # Merge similar clusters
        self._merge_similar_clusters()

        # Update global metrics
        self._update_global_metrics()

        # Memory cleanup
        if self.performance_mode and self.time_step % 200 == 0:
            self._memory_cleanup()

    def _remove_weak_clusters(self):
        """Remove clusters with low health scores."""
        initial_count = len(self.clusters)

        # Filter clusters based on health and confidence
        self.clusters = [
            cluster
            for cluster in self.clusters
            if (
                cluster["confidence"] >= self.min_confidence
                and cluster["health_score"] >= 0.3
            )
        ]

        # Mark cache as dirty if clusters were removed
        if len(self.clusters) != initial_count:
            self._cache_dirty = True

    def _apply_confidence_decay(self):
        """Apply confidence decay to all clusters."""
        for cluster in self.clusters:
            # Age-based decay
            decay_factor = 1.0 - self.decay_rate * (1.0 + cluster["age"] * 0.01)
            cluster["confidence"] *= decay_factor
            cluster["age"] += 1.0

            # Recompute health score
            cluster["health_score"] = self._compute_cluster_health(cluster)

    def _process_outlier_buffer(self):
        """Process points in outlier buffer for potential cluster creation or assignment."""
        if len(self.outlier_buffer) < 3:
            return

        # Analyze outlier buffer for potential clusters
        outlier_points = list(self.outlier_buffer)

        # Simple clustering of outliers using distance-based approach
        potential_centroids = []
        used_points = set()

        for i, point1 in enumerate(outlier_points):
            if i in used_points:
                continue

            nearby_points = [point1]
            used_points.add(i)

            for j, point2 in enumerate(outlier_points):
                if j <= i or j in used_points:
                    continue

                distance = np.linalg.norm(point1 - point2)
                if distance < 1.5:  # Proximity threshold
                    nearby_points.append(point2)
                    used_points.add(j)

            if len(nearby_points) >= 3 and self._can_create_cluster():
                # Create cluster from outlier group
                centroid = np.mean(nearby_points, axis=0)
                self._create_new_cluster(centroid)

        # Clear processed outliers
        self.outlier_buffer.clear()

    def _merge_similar_clusters(self):
        """Merge clusters that are too similar."""
        if len(self.clusters) < 2:
            return

        merged_pairs = []

        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                if (i, j) in merged_pairs or (j, i) in merged_pairs:
                    continue

                cluster1 = self.clusters[i]
                cluster2 = self.clusters[j]

                # Compute merge compatibility
                merge_score = self._compute_merge_compatibility(cluster1, cluster2)

                if merge_score >= self.merge_threshold:
                    # Merge clusters
                    self._merge_clusters(i, j)
                    merged_pairs.append((i, j))
                    self._stats["clusters_merged"] += 1

        # Remove merged clusters (process in reverse order to maintain indices)
        for i, j in reversed(merged_pairs):
            if j < len(self.clusters):
                del self.clusters[j]

        # Mark cache as dirty if clusters were merged
        if merged_pairs:
            self._cache_dirty = True

    def _compute_merge_compatibility(self, cluster1: Dict, cluster2: Dict) -> float:
        """
        Compute compatibility score for merging two clusters.

        Args:
            cluster1: First cluster
            cluster2: Second cluster

        Returns:
            Merge compatibility score [0-1]
        """
        # Geometric similarity
        centroid1 = cluster1["centroid"]
        centroid2 = cluster2["centroid"]

        # Cosine similarity between centroids
        dot_product = np.dot(centroid1, centroid2)
        norm1 = np.linalg.norm(centroid1)
        norm2 = np.linalg.norm(centroid2)

        if norm1 > 1e-10 and norm2 > 1e-10:
            geometric_sim = dot_product / (norm1 * norm2)
        else:
            geometric_sim = 0.0

        # Quality compatibility
        conf_diff = abs(cluster1["confidence"] - cluster2["confidence"])
        quality_compat = 1.0 - min(
            1.0, conf_diff / max(cluster1["confidence"], cluster2["confidence"])
        )

        # Health compatibility
        health_diff = abs(cluster1["health_score"] - cluster2["health_score"])
        health_compat = 1.0 - min(1.0, health_diff)

        # Temporal compatibility
        age_diff = abs(cluster1["age"] - cluster2["age"])
        temporal_compat = 1.0 - min(1.0, age_diff / 100.0)

        # Weighted combination
        merge_score = (
            0.4 * max(0, geometric_sim)
            + 0.25 * quality_compat
            + 0.2 * health_compat
            + 0.15 * temporal_compat
        )

        return merge_score

    def _merge_clusters(self, idx1: int, idx2: int):
        """
        Merge two clusters.

        Args:
            idx1: Index of first cluster (will be kept)
            idx2: Index of second cluster (will be removed)
        """
        if idx1 >= len(self.clusters) or idx2 >= len(self.clusters):
            return

        cluster1 = self.clusters[idx1]
        cluster2 = self.clusters[idx2]

        # Weighted average of centroids based on confidence
        total_confidence = cluster1["confidence"] + cluster2["confidence"]
        if total_confidence > 0:
            weight1 = cluster1["confidence"] / total_confidence
            weight2 = cluster2["confidence"] / total_confidence

            cluster1["centroid"] = (
                weight1 * cluster1["centroid"] + weight2 * cluster2["centroid"]
            )
            cluster1["confidence"] = (
                total_confidence * 0.8
            )  # Slight penalty for merging
            cluster1["age"] = (cluster1["age"] + cluster2["age"]) / 2
            cluster1["updates"] += cluster2["updates"]
            cluster1["last_update"] = max(
                cluster1["last_update"], cluster2["last_update"]
            )
            cluster1["health_score"] = self._compute_cluster_health(cluster1)

    def _update_global_metrics(self):
        """Update global algorithm metrics."""
        if len(self.clusters) > 0:
            # Clustering quality based on similarity statistics
            if len(self.similarity_history) > 10:
                similarities = np.array(list(self.similarity_history))
                mean_sim = np.mean(similarities)
                std_sim = np.std(similarities)
                self.clustering_quality = min(1.0, mean_sim * (1.0 / (1.0 + std_sim)))

            # Global stability based on cluster count and quality stability
            cluster_count_stability = min(
                1.0, 1.0 / (1.0 + abs(len(self.clusters) - 4) * 0.1)
            )

            if len(self.quality_history) > 10:
                quality_std = np.std(list(self.quality_history))
                quality_stability = 1.0 / (1.0 + quality_std)
            else:
                quality_stability = 1.0

            self.global_stability = cluster_count_stability * quality_stability

            # Adaptation rate based on stability
            self.adaptation_rate = 1.0 + self.global_stability * 0.4

            # Record quality in history
            self.quality_history.append(self.clustering_quality)

    def _memory_cleanup(self):
        """Perform memory cleanup operations."""
        # Trigger garbage collection
        gc.collect()

        # Clear old cache entries
        if hasattr(self, "_threshold_cache"):
            self._threshold_cache.clear()

        # Reset cache
        self._cluster_array_cache = None
        self._cache_dirty = True

    def _record_processing_time(self, processing_time_ms: float):
        """Record processing time for performance monitoring."""
        self.processing_times.append(processing_time_ms)

        # Update statistics
        self._stats["avg_processing_time_ms"] = np.mean(list(self.processing_times))
        self._stats["max_processing_time_ms"] = max(
            self._stats["max_processing_time_ms"], processing_time_ms
        )

        # Check for performance warnings
        if processing_time_ms > self.latency_warning_threshold_ms:
            self.warning_count += 1

    def get_clusters(self) -> List[Tuple[np.ndarray, float, float, int, float]]:
        """
        Get current clusters with metadata.

        Returns:
            List of tuples (centroid, stability, age, updates, quality)
        """
        with self._lock:
            clusters_info = []
            for cluster in self.clusters:
                centroid = cluster["centroid"].copy()
                stability = cluster["confidence"] * min(
                    1.2, 1.0 + cluster["age"] * 0.01
                )
                age = cluster["age"]
                updates = cluster["updates"]
                quality = cluster["health_score"]

                clusters_info.append((centroid, stability, age, updates, quality))

            return clusters_info

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive algorithm statistics.

        Returns:
            Dictionary containing performance and algorithm metrics
        """
        with self._lock:
            # Update memory usage estimate
            estimated_memory = (
                len(self.clusters) * 1000  # Rough estimate per cluster
                + len(self.similarity_history) * 8
                + len(self.distance_history) * 8
                + len(self.outlier_buffer) * 100
            ) / (
                1024 * 1024
            )  # Convert to MB

            self._stats["memory_usage_estimate_mb"] = estimated_memory
            self._stats["total_points_processed"] = self.total_points_processed

            stats = {
                "num_clusters": len(self.clusters),
                "total_points_processed": self.total_points_processed,
                "clustering_quality": self.clustering_quality,
                "global_stability": self.global_stability,
                "adaptation_rate": self.adaptation_rate,
                "current_dynamic_threshold": self.current_dynamic_threshold,
                "avg_processing_time_ms": self._stats["avg_processing_time_ms"],
                "max_processing_time_ms": self._stats["max_processing_time_ms"],
                "memory_usage_estimate_mb": self._stats["memory_usage_estimate_mb"],
                "error_count": self.error_count,
                "warning_count": self.warning_count,
                "clusters_created": self._stats["clusters_created"],
                "clusters_merged": self._stats["clusters_merged"],
                "outliers_detected": self._stats["outliers_detected"],
                "time_step": self.time_step,
            }

            return stats

    def reset(self):
        """Reset algorithm state while preserving configuration."""
        with self._lock:
            self.clusters.clear()
            self.time_step = 0
            self.total_points_processed = 0

            self.similarity_history.clear()
            self.distance_history.clear()
            self.quality_history.clear()
            self.outlier_buffer.clear()
            self.processing_times.clear()

            self.error_count = 0
            self.warning_count = 0

            self.current_dynamic_threshold = self.base_threshold
            self.global_stability = 1.0
            self.clustering_quality = 1.0
            self.adaptation_rate = 1.0

            # Clear caches
            if hasattr(self, "_threshold_cache"):
                self._threshold_cache.clear()
            self._cluster_array_cache = None
            self._cache_dirty = True

            # Reset statistics
            self._initialize_state()


# Compatibility alias for existing code
NeuroClusterStreamerV8 = NeuroClusterStreamer


if __name__ == "__main__":
    """Test the NeuroCluster Streamer algorithm."""
    print("🧠 Testing NeuroCluster Streamer V8...")

    # Initialize algorithm
    ncs = NeuroClusterStreamer(
        base_threshold=0.71, learning_rate=0.06, performance_mode=True
    )

    # Generate test data
    np.random.seed(42)

    # Test data with cluster structure
    cluster1_points = np.random.normal([0, 0], 0.5, (50, 2))
    cluster2_points = np.random.normal([5, 5], 0.5, (50, 2))
    outlier_points = np.random.uniform(-3, 8, (10, 2))

    all_points = np.vstack([cluster1_points, cluster2_points, outlier_points])
    np.random.shuffle(all_points)

    # Process points and measure performance
    start_time = time.time()
    results = []

    for point in all_points:
        cluster_id, is_outlier, outlier_score = ncs.process_data_point(point)
        results.append((cluster_id, is_outlier, outlier_score))

    processing_time = time.time() - start_time

    # Print results
    print(f"✅ Processed {len(all_points)} points in {processing_time:.3f} seconds")
    print(f"📊 Processing rate: {len(all_points) / processing_time:.0f} points/second")

    # Get algorithm statistics
    stats = ncs.get_statistics()
    print(f"🔬 Algorithm Statistics:")
    print(f"   Clusters found: {stats['num_clusters']}")
    print(f"   Clustering quality: {stats['clustering_quality']:.3f}")
    print(f"   Global stability: {stats['global_stability']:.3f}")
    print(f"   Avg processing time: {stats['avg_processing_time_ms']:.3f}ms")
    print(f"   Memory usage: {stats['memory_usage_estimate_mb']:.1f}MB")
    print(f"   Outliers detected: {stats['outliers_detected']}")

    # Get cluster information
    clusters = ncs.get_clusters()
    print(f"🎯 Cluster Details:")
    for i, (centroid, stability, age, updates, quality) in enumerate(clusters):
        print(
            f"   Cluster {i}: centroid={centroid[:2]}, stability={stability:.3f}, quality={quality:.3f}"
        )

    print("🎉 NeuroCluster Streamer V8 test completed successfully!")
