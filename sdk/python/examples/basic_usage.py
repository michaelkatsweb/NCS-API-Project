#!/usr/bin/env python3
"""
NeuroCluster Streamer Python SDK - Basic Usage Example
=====================================================
Demonstrates basic usage patterns for the NCS Python SDK

This example shows:
- Client initialization and authentication
- Basic data point processing
- Error handling patterns
- Health checking
- Configuration management

Author: NCS API Development Team
Year: 2025
"""

import os
import sys
import logging
from typing import List
import random
import time

# Add the parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncs_client import (
    NCSClient,
    AsyncNCSClient,
    NCSError,
    AuthenticationError,
    RateLimitError,
    configure_logging,
    create_client_from_env,
)


def main():
    """Main example function demonstrating basic SDK usage."""

    # Configure logging for better visibility
    configure_logging("INFO")
    logger = logging.getLogger(__name__)

    print("🚀 NeuroCluster Streamer Python SDK - Basic Usage Example")
    print("=" * 60)

    # =============================================================================
    # Example 1: Client Initialization
    # =============================================================================
    print("\n📡 Example 1: Client Initialization")

    # Method 1: Direct initialization with API key
    if os.getenv("NCS_API_KEY"):
        client = NCSClient(
            base_url="https://api.yourdomain.com",
            api_key=os.getenv("NCS_API_KEY"),
            timeout=30.0,
            log_level="INFO",
        )
        print("✅ Client initialized with API key")

    # Method 2: Environment-based initialization
    elif os.getenv("NCS_API_URL"):
        try:
            client = create_client_from_env()
            print("✅ Client initialized from environment variables")
        except ValueError as e:
            print(f"❌ Environment setup error: {e}")
            print("💡 Set NCS_API_URL and NCS_API_KEY environment variables")
            return

    # Method 3: Manual configuration for demo
    else:
        print("⚠️  No environment variables found. Using demo configuration.")
        client = NCSClient(
            base_url="https://demo.ncs-api.com",
            api_key="demo-api-key-12345",
            timeout=30.0,
        )
        print("✅ Client initialized with demo configuration")

    # =============================================================================
    # Example 2: Health Check
    # =============================================================================
    print("\n🏥 Example 2: API Health Check")

    try:
        health = client.health_check()
        print(f"✅ API Status: {health.status}")
        print(f"   Version: {health.version}")
        print(f"   Algorithm Ready: {health.algorithm_ready}")
        print(f"   Uptime: {health.uptime_seconds:.1f} seconds")

        if health.status != "healthy":
            print("⚠️  API is not fully healthy - proceeding with caution")

    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("💡 Check your API URL and network connection")
        return

    # =============================================================================
    # Example 3: Generate Sample Data
    # =============================================================================
    print("\n📊 Example 3: Generating Sample Data")

    # Generate random 3D points for clustering
    def generate_sample_points(num_points: int = 20) -> List[List[float]]:
        """Generate random 3D points for demonstration."""
        points = []

        # Create clusters around specific centers
        centers = [[0, 0, 0], [5, 5, 5], [10, 0, 10]]

        for i in range(num_points):
            center = centers[i % len(centers)]
            # Add noise around the center
            point = [
                center[0] + random.gauss(0, 1),
                center[1] + random.gauss(0, 1),
                center[2] + random.gauss(0, 1),
            ]
            points.append(point)

        return points

    sample_points = generate_sample_points(15)
    print(f"✅ Generated {len(sample_points)} sample 3D points")
    print(f"   First few points: {sample_points[:3]}")

    # =============================================================================
    # Example 4: Basic Point Processing
    # =============================================================================
    print("\n⚙️  Example 4: Processing Data Points")

    try:
        # Process the points
        start_time = time.time()
        result = client.process_points(sample_points)
        processing_time = time.time() - start_time

        print(f"✅ Processing completed in {processing_time:.3f} seconds")
        print(f"   Server processing time: {result.processing_time_ms:.2f} ms")
        print(f"   Algorithm quality: {result.algorithm_quality:.3f}")
        print(f"   Request ID: {result.request_id}")
        print(f"   Total points processed: {result.total_points}")

        # Display cluster information
        print(f"\n📈 Clustering Results:")
        print(f"   Number of clusters: {len(result.clusters)}")
        print(f"   Number of outliers: {len(result.outliers)}")

        for i, cluster in enumerate(result.clusters):
            print(
                f"   Cluster {cluster.id}: {cluster.size} points, quality={cluster.quality:.3f}"
            )
            print(
                f"      Center: [{cluster.center[0]:.2f}, {cluster.center[1]:.2f}, {cluster.center[2]:.2f}]"
            )

        if result.outliers:
            print(f"   Outliers: {len(result.outliers)} points detected")

    except ValidationError as e:
        print(f"❌ Validation error: {e.message}")
        print("💡 Check your data format - points should be lists of numbers")

    except ProcessingError as e:
        print(f"❌ Processing error: {e.message}")
        print("💡 The algorithm may be overloaded or encountering issues")

    except Exception as e:
        print(f"❌ Unexpected error during processing: {e}")

    # =============================================================================
    # Example 5: Algorithm Status
    # =============================================================================
    print("\n📊 Example 5: Algorithm Status")

    try:
        status = client.get_algorithm_status()
        print(f"✅ Algorithm Status Retrieved:")
        print(f"   Ready: {status.is_ready}")
        print(f"   Active clusters: {status.active_clusters}")
        print(f"   Total points processed: {status.total_points_processed}")
        print(f"   Clustering quality: {status.clustering_quality:.3f}")
        print(f"   Memory usage: {status.memory_usage_mb:.2f} MB")
        print(f"   Error count: {status.error_count}")

        if status.error_count > 0:
            print("⚠️  Algorithm has encountered some errors")

    except Exception as e:
        print(f"❌ Failed to get algorithm status: {e}")

    # =============================================================================
    # Example 6: Cluster Summary
    # =============================================================================
    print("\n📋 Example 6: Cluster Summary")

    try:
        summary = client.get_clusters_summary()
        print(f"✅ Cluster Summary:")
        print(f"   Total clusters: {summary.get('total_clusters', 0)}")
        print(f"   Average cluster size: {summary.get('average_cluster_size', 0):.1f}")
        print(f"   Total points: {summary.get('total_points', 0)}")

        # Display top clusters if available
        if "top_clusters" in summary:
            print(f"   Top clusters by size:")
            for cluster in summary["top_clusters"][:3]:
                print(f"      Cluster {cluster['id']}: {cluster['size']} points")

    except Exception as e:
        print(f"❌ Failed to get cluster summary: {e}")

    # =============================================================================
    # Example 7: Error Handling Patterns
    # =============================================================================
    print("\n🛡️  Example 7: Error Handling Patterns")

    # Demonstrate different error scenarios
    try:
        # Try processing invalid data
        invalid_points = [["not", "a", "number"], [1, 2]]  # Mixed types and dimensions
        result = client.process_points(invalid_points)

    except ValidationError as e:
        print(f"✅ Caught validation error as expected: {e.message}")
        print(f"   Error code: {e.error_code}")
        print(f"   Request ID: {e.request_id}")

    except RateLimitError as e:
        print(f"⚠️  Rate limit exceeded: {e.message}")
        print(f"   Retry after: {e.retry_after} seconds")

    except AuthenticationError as e:
        print(f"❌ Authentication failed: {e.message}")
        print("💡 Check your API key or JWT token")

    except NCSError as e:
        print(f"❌ NCS API error: {e.message}")
        print(f"   Status code: {e.status_code}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    # =============================================================================
    # Example 8: Context Manager Usage
    # =============================================================================
    print("\n🔒 Example 8: Context Manager Usage")

    # Using context manager for automatic cleanup
    try:
        with NCSClient(
            base_url="https://api.yourdomain.com",
            api_key=os.getenv("NCS_API_KEY", "demo-key"),
            timeout=15.0,
        ) as context_client:

            # Generate small dataset
            small_dataset = generate_sample_points(5)
            result = context_client.process_points(small_dataset)

            print(f"✅ Context manager processing successful")
            print(f"   Processed {len(small_dataset)} points")
            print(f"   Found {len(result.clusters)} clusters")

        print("✅ Context manager automatically closed the client")

    except Exception as e:
        print(f"❌ Context manager example failed: {e}")

    # =============================================================================
    # Example 9: Configuration Management
    # =============================================================================
    print("\n⚙️  Example 9: Configuration Management")

    # Show how to create clients with different configurations
    configs = [
        {"name": "High timeout config", "config": {"timeout": 60.0, "max_retries": 5}},
        {"name": "Debug config", "config": {"log_level": "DEBUG", "verify_ssl": False}},
        {
            "name": "Production config",
            "config": {"timeout": 30.0, "max_retries": 3, "verify_ssl": True},
        },
    ]

    for config_example in configs:
        try:
            test_config = {
                "base_url": "https://demo.ncs-api.com",
                "api_key": "demo-key",
                **config_example["config"],
            }

            test_client = NCSClient.from_config(test_config)
            print(f"✅ Created client with {config_example['name']}")

            # Test the configuration
            health = test_client.health_check()
            print(f"   Health check successful: {health.status}")

            test_client.close()

        except Exception as e:
            print(f"❌ {config_example['name']} failed: {e}")

    # =============================================================================
    # Cleanup and Summary
    # =============================================================================
    print("\n🎯 Example Summary")
    print("=" * 60)
    print("✅ Basic client initialization and configuration")
    print("✅ API health checking")
    print("✅ Data point processing and clustering")
    print("✅ Algorithm status monitoring")
    print("✅ Comprehensive error handling")
    print("✅ Context manager usage")
    print("✅ Configuration management")

    # Close the main client
    client.close()
    print("\n🔒 Client connection closed")
    print("\n🎉 Basic usage example completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Example interrupted by user")
    except Exception as e:
        print(f"\n💥 Example failed with error: {e}")
        import traceback

        traceback.print_exc()
