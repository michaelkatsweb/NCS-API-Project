#!/usr/bin/env python3
"""
NeuroCluster Streamer Python SDK - Streaming Example
===================================================
Demonstrates real-time streaming capabilities of the NCS Python SDK

This example shows:
- Async client usage with streaming
- WebSocket connections for real-time data
- Concurrent processing patterns
- Stream processing with callbacks
- Real-time cluster updates

Author: NCS API Development Team
Year: 2025
"""

import asyncio
import json
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List

# Add the parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncs_client import (
    AsyncNCSClient,
    NCSError,
    ProcessingResult,
    async_client_context,
    configure_logging,
)


class DataStreamSimulator:
    """Simulates a real-time data stream for demonstration."""

    def __init__(self, points_per_second: int = 10):
        self.points_per_second = points_per_second
        self.running = False
        self.cluster_centers = [[0, 0, 0], [5, 5, 5], [10, 0, 10], [-5, 5, -5]]

    async def generate_stream(self):
        """Generate a continuous stream of data points."""
        self.running = True

        while self.running:
            # Generate points around random cluster centers
            center = random.choice(self.cluster_centers)
            point = [
                center[0] + random.gauss(0, 1.5),
                center[1] + random.gauss(0, 1.5),
                center[2] + random.gauss(0, 1.5),
            ]

            yield point

            # Control the rate of point generation
            await asyncio.sleep(1.0 / self.points_per_second)

    def stop(self):
        """Stop the data stream."""
        self.running = False


class ClusterMonitor:
    """Monitors and displays cluster updates in real-time."""

    def __init__(self):
        self.cluster_history = []
        self.total_points_processed = 0
        self.start_time = time.time()

    async def on_cluster_update(self, result: ProcessingResult):
        """Handle cluster update events."""
        self.total_points_processed += result.total_points
        self.cluster_history.append(result)

        # Display update
        elapsed = time.time() - self.start_time
        print(f"\n📊 Cluster Update at {elapsed:.1f}s:")
        print(f"   Clusters: {len(result.clusters)}")
        print(f"   Outliers: {len(result.outliers)}")
        print(f"   Quality: {result.algorithm_quality:.3f}")
        print(f"   Processing time: {result.processing_time_ms:.1f}ms")
        print(f"   Total processed: {self.total_points_processed}")

        # Show cluster details
        for cluster in result.clusters:
            print(
                f"      Cluster {cluster.id}: {cluster.size} points, "
                f"center=[{cluster.center[0]:.1f}, {cluster.center[1]:.1f}, {cluster.center[2]:.1f}]"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        if not self.cluster_history:
            return {}

        elapsed = time.time() - self.start_time

        return {
            "total_updates": len(self.cluster_history),
            "total_points": self.total_points_processed,
            "elapsed_time": elapsed,
            "points_per_second": (
                self.total_points_processed / elapsed if elapsed > 0 else 0
            ),
            "average_clusters": sum(len(r.clusters) for r in self.cluster_history)
            / len(self.cluster_history),
            "average_quality": sum(r.algorithm_quality for r in self.cluster_history)
            / len(self.cluster_history),
        }


async def example_basic_streaming():
    """Example 1: Basic streaming with WebSocket connection."""
    print("\n🌊 Example 1: Basic WebSocket Streaming")
    print("-" * 50)

    async with async_client_context(
        base_url=os.getenv("NCS_API_URL", "wss://demo.ncs-api.com"),
        api_key=os.getenv("NCS_API_KEY", "demo-key"),
    ) as client:
        # Message handler for incoming cluster updates
        async def handle_stream_message(data: Dict[str, Any]):
            if data.get("type") == "cluster_update":
                clusters = data.get("clusters", [])
                print(f"📡 Received cluster update: {len(clusters)} clusters")

                for cluster in clusters[:3]:  # Show first 3 clusters
                    print(f"   Cluster {cluster['id']}: {cluster['size']} points")

        try:
            # Start streaming connection
            stream = await client.start_streaming(
                on_message=handle_stream_message, connection_id="basic_stream"
            )

            print("✅ Streaming connection established")

            # Send some test points
            test_points = [
                [1.0, 2.0, 3.0],
                [1.1, 2.1, 3.1],
                [5.0, 6.0, 7.0],
                [5.1, 6.1, 7.1],
            ]

            for i, point in enumerate(test_points):
                await stream.send_point(point)
                print(f"📤 Sent point {i+1}: {point}")
                await asyncio.sleep(1)

            # Wait for responses
            await asyncio.sleep(3)

            # Stop streaming
            await client.stop_streaming("basic_stream")
            print("✅ Streaming connection closed")

        except Exception as e:
            print(f"❌ Streaming error: {e}")


async def example_concurrent_processing():
    """Example 2: Concurrent processing of multiple data streams."""
    print("\n⚡ Example 2: Concurrent Stream Processing")
    print("-" * 50)

    async with async_client_context(
        base_url=os.getenv("NCS_API_URL", "https://demo.ncs-api.com"),
        api_key=os.getenv("NCS_API_KEY", "demo-key"),
        max_connections=10,
    ) as client:
        # Generate multiple batches of points concurrently
        def generate_batch(batch_id: int, size: int = 50) -> List[List[float]]:
            """Generate a batch of points around a specific area."""
            center = [batch_id * 3, batch_id * 3, 0]
            return [
                [
                    center[0] + random.gauss(0, 1),
                    center[1] + random.gauss(0, 1),
                    center[2] + random.gauss(0, 1),
                ]
                for _ in range(size)
            ]

        # Create multiple batches
        batches = [generate_batch(i, 30) for i in range(5)]

        print(
            f"📦 Generated {len(batches)} batches with {sum(len(b) for b in batches)} total points"
        )

        try:
            # Process all batches concurrently
            start_time = time.time()
            results = await client.process_points_concurrent(batches, max_concurrent=3)
            processing_time = time.time() - start_time

            print(f"✅ Concurrent processing completed in {processing_time:.2f} seconds")
            print(f"📊 Results summary:")

            total_clusters = sum(len(result.clusters) for result in results)
            total_outliers = sum(len(result.outliers) for result in results)
            avg_quality = sum(result.algorithm_quality for result in results) / len(
                results
            )

            print(f"   Total clusters: {total_clusters}")
            print(f"   Total outliers: {total_outliers}")
            print(f"   Average quality: {avg_quality:.3f}")
            print(
                f"   Processing rate: {sum(len(b) for b in batches) / processing_time:.1f} points/sec"
            )

        except Exception as e:
            print(f"❌ Concurrent processing error: {e}")


async def example_stream_processing():
    """Example 3: Real-time stream processing with monitoring."""
    print("\n📈 Example 3: Real-time Stream Processing")
    print("-" * 50)

    # Initialize components
    simulator = DataStreamSimulator(points_per_second=5)
    monitor = ClusterMonitor()

    async with async_client_context(
        base_url=os.getenv("NCS_API_URL", "https://demo.ncs-api.com"),
        api_key=os.getenv("NCS_API_KEY", "demo-key"),
    ) as client:
        print("🚀 Starting real-time stream processing...")

        # Stream processor function
        async def process_data_stream():
            """Process continuous data stream in batches."""
            batch = []
            batch_size = 10

            async for point in simulator.generate_stream():
                batch.append(point)

                if len(batch) >= batch_size:
                    try:
                        # Process the batch
                        result = await client.process_points(batch)
                        await monitor.on_cluster_update(result)

                        # Clear the batch
                        batch = []

                    except Exception as e:
                        print(f"❌ Processing error: {e}")
                        await asyncio.sleep(1)  # Brief pause on error

        # Statistics reporter
        async def report_statistics():
            """Periodically report processing statistics."""
            while simulator.running:
                await asyncio.sleep(10)  # Report every 10 seconds

                stats = monitor.get_statistics()
                if stats:
                    print(f"\n📊 Processing Statistics:")
                    print(f"   Updates: {stats['total_updates']}")
                    print(f"   Points processed: {stats['total_points']}")
                    print(f"   Rate: {stats['points_per_second']:.1f} points/sec")
                    print(f"   Avg clusters: {stats['average_clusters']:.1f}")
                    print(f"   Avg quality: {stats['average_quality']:.3f}")

        try:
            # Start processing and monitoring
            processing_task = asyncio.create_task(process_data_stream())
            stats_task = asyncio.create_task(report_statistics())

            # Run for 30 seconds
            await asyncio.sleep(30)

            # Stop the simulation
            simulator.stop()

            # Wait for tasks to complete
            await processing_task
            stats_task.cancel()

            # Final statistics
            final_stats = monitor.get_statistics()
            print(f"\n🎯 Final Statistics:")
            print(
                f"   Total processing time: {final_stats['elapsed_time']:.1f} seconds"
            )
            print(f"   Total points: {final_stats['total_points']}")
            print(
                f"   Average throughput: {final_stats['points_per_second']:.1f} points/sec"
            )
            print(f"   Total cluster updates: {final_stats['total_updates']}")

        except Exception as e:
            print(f"❌ Stream processing error: {e}")
            simulator.stop()


async def example_adaptive_batching():
    """Example 4: Adaptive batching based on processing performance."""
    print("\n🎛️  Example 4: Adaptive Batching")
    print("-" * 50)

    async with async_client_context(
        base_url=os.getenv("NCS_API_URL", "https://demo.ncs-api.com"),
        api_key=os.getenv("NCS_API_KEY", "demo-key"),
    ) as client:
        # Adaptive batch processor
        class AdaptiveBatchProcessor:
            def __init__(self):
                self.batch_size = 20  # Start with moderate batch size
                self.min_batch_size = 5
                self.max_batch_size = 100
                self.target_processing_time = 500  # Target 500ms processing time
                self.performance_history = []

            def adjust_batch_size(self, processing_time_ms: float):
                """Adjust batch size based on processing performance."""
                self.performance_history.append(processing_time_ms)

                # Keep only recent history
                if len(self.performance_history) > 10:
                    self.performance_history = self.performance_history[-10:]

                avg_time = sum(self.performance_history) / len(self.performance_history)

                if avg_time > self.target_processing_time * 1.2:  # Too slow
                    self.batch_size = max(
                        self.min_batch_size, int(self.batch_size * 0.8)
                    )
                    print(
                        f"📉 Reducing batch size to {self.batch_size} (avg time: {avg_time:.1f}ms)"
                    )

                elif avg_time < self.target_processing_time * 0.8:  # Too fast
                    self.batch_size = min(
                        self.max_batch_size, int(self.batch_size * 1.2)
                    )
                    print(
                        f"📈 Increasing batch size to {self.batch_size} (avg time: {avg_time:.1f}ms)"
                    )

        processor = AdaptiveBatchProcessor()
        total_processed = 0

        # Generate data stream
        def generate_continuous_data():
            """Generate continuous stream of points."""
            while True:
                center = random.choice([[0, 0, 0], [10, 10, 10], [-5, 5, 0]])
                yield [
                    center[0] + random.gauss(0, 2),
                    center[1] + random.gauss(0, 2),
                    center[2] + random.gauss(0, 2),
                ]

        data_stream = generate_continuous_data()

        print("🔄 Starting adaptive batching demonstration...")

        try:
            for iteration in range(15):  # Run 15 iterations
                # Collect batch
                batch = [next(data_stream) for _ in range(processor.batch_size)]

                # Process with timing
                start_time = time.time()
                result = await client.process_points(batch)
                actual_time = (time.time() - start_time) * 1000  # Convert to ms

                total_processed += len(batch)

                print(f"📊 Iteration {iteration + 1}:")
                print(f"   Batch size: {len(batch)}")
                print(
                    f"   Processing time: {actual_time:.1f}ms (server: {result.processing_time_ms:.1f}ms)"
                )
                print(f"   Clusters found: {len(result.clusters)}")
                print(f"   Quality: {result.algorithm_quality:.3f}")

                # Adjust batch size for next iteration
                processor.adjust_batch_size(actual_time)

                await asyncio.sleep(0.5)  # Brief pause between iterations

            print(f"\n✅ Adaptive batching completed:")
            print(f"   Total points processed: {total_processed}")
            print(f"   Final optimal batch size: {processor.batch_size}")
            print(
                f"   Average processing time: {sum(processor.performance_history)/len(processor.performance_history):.1f}ms"
            )

        except Exception as e:
            print(f"❌ Adaptive batching error: {e}")


async def main():
    """Main async function running all streaming examples."""
    configure_logging("INFO")

    print("🌊 NeuroCluster Streamer Python SDK - Streaming Examples")
    print("=" * 65)

    try:
        # Run streaming examples
        await example_basic_streaming()
        await asyncio.sleep(2)

        await example_concurrent_processing()
        await asyncio.sleep(2)

        await example_stream_processing()
        await asyncio.sleep(2)

        await example_adaptive_batching()

        print("\n🎉 All streaming examples completed successfully!")

    except Exception as e:
        print(f"\n💥 Streaming examples failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n💥 Examples failed with error: {e}")
