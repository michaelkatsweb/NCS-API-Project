#!/usr/bin/env python3
"""
NeuroCluster Streamer Python SDK - Batch Processing Example
==========================================================
Demonstrates high-throughput batch processing with the NCS Python SDK

This example shows:
- Large dataset processing strategies
- Memory-efficient batch processing
- Progress tracking and monitoring
- Error recovery and retry logic
- Performance optimization techniques
- Result aggregation and analysis

Author: NCS API Development Team
Year: 2025
"""

import asyncio
import csv
import json
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

# Add the parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncs_client import (
    AsyncNCSClient,
    NCSClient,
    NCSError,
    ProcessingResult,
    RateLimitError,
    async_client_context,
    configure_logging,
)


@dataclass
class BatchProcessingStats:
    """Statistics for batch processing operations."""

    total_points: int = 0
    total_batches: int = 0
    successful_batches: int = 0
    failed_batches: int = 0
    total_clusters: int = 0
    total_outliers: int = 0
    total_processing_time: float = 0.0
    average_quality: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def elapsed_time(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def points_per_second(self) -> float:
        if self.elapsed_time > 0:
            return self.total_points / self.elapsed_time
        return 0.0

    @property
    def success_rate(self) -> float:
        if self.total_batches > 0:
            return self.successful_batches / self.total_batches
        return 0.0


class LargeDatasetGenerator:
    """Generates large synthetic datasets for batch processing demonstration."""

    def __init__(self, num_clusters: int = 5, noise_level: float = 1.5):
        self.num_clusters = num_clusters
        self.noise_level = noise_level

        # Generate cluster centers in 3D space
        self.cluster_centers = []
        for i in range(num_clusters):
            center = [
                random.uniform(-20, 20),
                random.uniform(-20, 20),
                random.uniform(-20, 20),
            ]
            self.cluster_centers.append(center)

    def generate_points(self, num_points: int) -> List[List[float]]:
        """Generate a specified number of data points."""
        points = []

        for _ in range(num_points):
            # Choose random cluster center
            center = random.choice(self.cluster_centers)

            # Add noise around the center
            point = [
                center[0] + random.gauss(0, self.noise_level),
                center[1] + random.gauss(0, self.noise_level),
                center[2] + random.gauss(0, self.noise_level),
            ]
            points.append(point)

        return points

    def generate_batches(
        self, total_points: int, batch_size: int
    ) -> Iterator[List[List[float]]]:
        """Generate data in batches."""
        remaining = total_points

        while remaining > 0:
            current_batch_size = min(batch_size, remaining)
            yield self.generate_points(current_batch_size)
            remaining -= current_batch_size

    def save_to_csv(self, filename: str, num_points: int):
        """Save generated data to CSV file."""
        points = self.generate_points(num_points)

        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["x", "y", "z"])  # Header

            for point in points:
                writer.writerow(point)

        print(f"💾 Saved {num_points} points to {filename}")


class BatchProgressTracker:
    """Tracks and displays progress for batch processing operations."""

    def __init__(self, total_batches: int):
        self.total_batches = total_batches
        self.completed_batches = 0
        self.stats = BatchProcessingStats()
        self.lock = threading.Lock()

    def update_progress(self, result: ProcessingResult, processing_time: float):
        """Update progress with a completed batch."""
        with self.lock:
            self.completed_batches += 1
            self.stats.successful_batches += 1
            self.stats.total_clusters += len(result.clusters)
            self.stats.total_outliers += len(result.outliers)
            self.stats.total_points += result.total_points
            self.stats.total_processing_time += processing_time

            # Update quality (running average)
            total_quality = (
                self.stats.average_quality * (self.stats.successful_batches - 1)
                + result.algorithm_quality
            )
            self.stats.average_quality = total_quality / self.stats.successful_batches

            self._display_progress()

    def update_failure(self, error: Exception):
        """Update progress with a failed batch."""
        with self.lock:
            self.completed_batches += 1
            self.stats.failed_batches += 1
            self._display_progress()
            print(f"❌ Batch failed: {error}")

    def _display_progress(self):
        """Display current progress."""
        percentage = (self.completed_batches / self.total_batches) * 100

        print(
            f"\r📊 Progress: {self.completed_batches}/{self.total_batches} "
            f"({percentage:.1f}%) | "
            f"Success: {self.stats.success_rate:.1%} | "
            f"Points: {self.stats.total_points} | "
            f"Clusters: {self.stats.total_clusters}",
            end="",
            flush=True,
        )

        if self.completed_batches == self.total_batches:
            print()  # New line when complete


def example_synchronous_batch_processing():
    """Example 1: Synchronous batch processing with progress tracking."""
    print("\n📦 Example 1: Synchronous Batch Processing")
    print("-" * 50)

    # Configuration
    total_points = 5000
    batch_size = 250

    # Initialize components
    generator = LargeDatasetGenerator(num_clusters=4)
    batches = list(generator.generate_batches(total_points, batch_size))
    tracker = BatchProgressTracker(len(batches))

    print(
        f"🎯 Processing {total_points} points in {len(batches)} batches of {batch_size}"
    )

    # Initialize client
    client = NCSClient(
        base_url=os.getenv("NCS_API_URL", "https://demo.ncs-api.com"),
        api_key=os.getenv("NCS_API_KEY", "demo-key"),
        timeout=60.0,  # Longer timeout for large batches
        max_retries=3,
    )

    tracker.stats.start_time = time.time()

    try:
        # Process each batch
        for i, batch in enumerate(batches):
            try:
                start_time = time.time()
                result = client.process_points_batch(
                    batch, batch_options={"timeout": 45}
                )
                processing_time = time.time() - start_time

                tracker.update_progress(result, processing_time)

            except RateLimitError as e:
                print(f"\n⏳ Rate limited, waiting {e.retry_after} seconds...")
                time.sleep(e.retry_after)

                # Retry the batch
                try:
                    start_time = time.time()
                    result = client.process_points_batch(batch)
                    processing_time = time.time() - start_time
                    tracker.update_progress(result, processing_time)
                except Exception as retry_error:
                    tracker.update_failure(retry_error)

            except Exception as e:
                tracker.update_failure(e)

    finally:
        tracker.stats.end_time = time.time()
        client.close()

    # Display final results
    print(f"\n✅ Synchronous batch processing completed!")
    print_batch_statistics(tracker.stats)


async def example_async_batch_processing():
    """Example 2: Asynchronous batch processing with concurrency control."""
    print("\n⚡ Example 2: Asynchronous Batch Processing")
    print("-" * 50)

    # Configuration
    total_points = 8000
    batch_size = 400
    max_concurrent = 5  # Limit concurrent requests

    # Initialize components
    generator = LargeDatasetGenerator(num_clusters=6)
    batches = list(generator.generate_batches(total_points, batch_size))
    tracker = BatchProgressTracker(len(batches))

    print(f"🎯 Processing {total_points} points in {len(batches)} batches")
    print(f"⚡ Max concurrent requests: {max_concurrent}")

    async with async_client_context(
        base_url=os.getenv("NCS_API_URL", "https://demo.ncs-api.com"),
        api_key=os.getenv("NCS_API_KEY", "demo-key"),
        timeout=60.0,
        max_connections=max_concurrent * 2,
    ) as client:
        # Semaphore to control concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch_async(batch: List[List[float]], batch_id: int):
            """Process a single batch asynchronously."""
            async with semaphore:
                try:
                    start_time = time.time()
                    result = await client.process_points(batch)
                    processing_time = time.time() - start_time

                    tracker.update_progress(result, processing_time)
                    return result

                except RateLimitError as e:
                    await asyncio.sleep(e.retry_after)
                    # Retry once
                    start_time = time.time()
                    result = await client.process_points(batch)
                    processing_time = time.time() - start_time
                    tracker.update_progress(result, processing_time)
                    return result

                except Exception as e:
                    tracker.update_failure(e)
                    return None

        tracker.stats.start_time = time.time()

        # Create tasks for all batches
        tasks = [process_batch_async(batch, i) for i, batch in enumerate(batches)]

        # Process all batches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        tracker.stats.end_time = time.time()

        # Count successful results
        successful_results = [r for r in results if isinstance(r, ProcessingResult)]

        print(f"\n✅ Async batch processing completed!")
        print(f"📊 Successful batches: {len(successful_results)}/{len(batches)}")
        print_batch_statistics(tracker.stats)


def example_csv_file_processing():
    """Example 3: Processing data from CSV files."""
    print("\n📄 Example 3: CSV File Processing")
    print("-" * 50)

    # Generate a sample CSV file
    csv_filename = "sample_dataset.csv"
    generator = LargeDatasetGenerator(num_clusters=3)
    generator.save_to_csv(csv_filename, 2000)

    def read_csv_in_batches(
        filename: str, batch_size: int
    ) -> Iterator[List[List[float]]]:
        """Read CSV file in batches."""
        batch = []

        with open(filename, "r") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                try:
                    point = [float(row["x"]), float(row["y"]), float(row["z"])]
                    batch.append(point)

                    if len(batch) >= batch_size:
                        yield batch
                        batch = []

                except (ValueError, KeyError) as e:
                    print(f"⚠️  Skipping invalid row: {row} - {e}")

            # Yield remaining points
            if batch:
                yield batch

    # Process CSV file
    batch_size = 300
    client = NCSClient(
        base_url=os.getenv("NCS_API_URL", "https://demo.ncs-api.com"),
        api_key=os.getenv("NCS_API_KEY", "demo-key"),
    )

    print(f"📊 Processing CSV file '{csv_filename}' in batches of {batch_size}")

    total_processed = 0
    total_clusters = 0
    processing_results = []

    try:
        start_time = time.time()

        for batch_num, batch in enumerate(
            read_csv_in_batches(csv_filename, batch_size)
        ):
            try:
                result = client.process_points(batch)
                total_processed += len(batch)
                total_clusters += len(result.clusters)
                processing_results.append(result)

                print(
                    f"✅ Batch {batch_num + 1}: {len(batch)} points → "
                    f"{len(result.clusters)} clusters (quality: {result.algorithm_quality:.3f})"
                )

            except Exception as e:
                print(f"❌ Batch {batch_num + 1} failed: {e}")

        elapsed_time = time.time() - start_time

        print(f"\n📊 CSV Processing Results:")
        print(f"   Total points processed: {total_processed}")
        print(f"   Total clusters found: {total_clusters}")
        print(f"   Processing time: {elapsed_time:.2f} seconds")
        print(f"   Throughput: {total_processed / elapsed_time:.1f} points/sec")

        if processing_results:
            avg_quality = sum(r.algorithm_quality for r in processing_results) / len(
                processing_results
            )
            print(f"   Average quality: {avg_quality:.3f}")

    finally:
        client.close()
        # Clean up CSV file
        try:
            os.remove(csv_filename)
            print(f"🗑️  Cleaned up {csv_filename}")
        except OSError:
            pass


async def example_memory_efficient_processing():
    """Example 4: Memory-efficient processing of very large datasets."""
    print("\n🧠 Example 4: Memory-Efficient Large Dataset Processing")
    print("-" * 50)

    # Configuration for very large dataset
    total_points = 50000  # Large dataset
    batch_size = 500
    max_concurrent = 3  # Conservative concurrency for memory efficiency

    print(f"🎯 Memory-efficient processing of {total_points} points")
    print(f"📦 Batch size: {batch_size}")
    print(f"⚡ Concurrency: {max_concurrent}")

    async with async_client_context(
        base_url=os.getenv("NCS_API_URL", "https://demo.ncs-api.com"),
        api_key=os.getenv("NCS_API_KEY", "demo-key"),
    ) as client:
        # Generator for memory efficiency (doesn't store all data in memory)
        def memory_efficient_generator():
            """Generate batches on-demand to save memory."""
            generator = LargeDatasetGenerator(num_clusters=8)
            remaining = total_points

            while remaining > 0:
                current_batch_size = min(batch_size, remaining)
                yield generator.generate_points(current_batch_size)
                remaining -= current_batch_size

        # Process with controlled memory usage
        semaphore = asyncio.Semaphore(max_concurrent)
        stats = BatchProcessingStats()
        stats.start_time = time.time()

        async def process_with_memory_control(batch: List[List[float]], batch_id: int):
            """Process batch with memory management."""
            async with semaphore:
                try:
                    result = await client.process_points(batch)

                    # Update stats atomically
                    stats.successful_batches += 1
                    stats.total_points += len(batch)
                    stats.total_clusters += len(result.clusters)

                    # Memory-efficient: don't store full results
                    del batch  # Explicitly free memory

                    if batch_id % 10 == 0:  # Progress every 10 batches
                        print(
                            f"📊 Processed batch {batch_id + 1}: "
                            f"{stats.total_points} points, "
                            f"{stats.total_clusters} clusters"
                        )

                    return True

                except Exception as e:
                    stats.failed_batches += 1
                    if batch_id % 20 == 0:  # Less frequent error reporting
                        print(f"❌ Batch {batch_id + 1} failed: {e}")
                    return False

        # Process all batches
        tasks = []
        for batch_id, batch in enumerate(memory_efficient_generator()):
            task = process_with_memory_control(batch, batch_id)
            tasks.append(task)

            # Process in chunks to control memory usage
            if len(tasks) >= max_concurrent * 2:
                await asyncio.gather(*tasks)
                tasks = []

        # Process remaining tasks
        if tasks:
            await asyncio.gather(*tasks)

        stats.end_time = time.time()

        print(f"\n✅ Memory-efficient processing completed!")
        print(f"📊 Final Statistics:")
        print(f"   Total points: {stats.total_points}")
        print(f"   Successful batches: {stats.successful_batches}")
        print(f"   Failed batches: {stats.failed_batches}")
        print(f"   Success rate: {stats.success_rate:.1%}")
        print(f"   Total clusters: {stats.total_clusters}")
        print(f"   Processing time: {stats.elapsed_time:.2f} seconds")
        print(f"   Throughput: {stats.points_per_second:.1f} points/sec")


def print_batch_statistics(stats: BatchProcessingStats):
    """Print detailed batch processing statistics."""
    print(f"\n📊 Batch Processing Statistics:")
    print(f"   Total points processed: {stats.total_points:,}")
    print(f"   Total batches: {stats.total_batches}")
    print(f"   Successful batches: {stats.successful_batches}")
    print(f"   Failed batches: {stats.failed_batches}")
    print(f"   Success rate: {stats.success_rate:.1%}")
    print(f"   Total clusters found: {stats.total_clusters}")
    print(f"   Total outliers: {stats.total_outliers}")
    print(f"   Average quality: {stats.average_quality:.3f}")
    print(f"   Total processing time: {stats.elapsed_time:.2f} seconds")
    print(f"   Throughput: {stats.points_per_second:.1f} points/second")

    if stats.successful_batches > 0:
        avg_batch_time = stats.total_processing_time / stats.successful_batches
        print(f"   Average batch processing time: {avg_batch_time:.3f} seconds")


async def main():
    """Main function running all batch processing examples."""
    configure_logging("INFO")

    print("📦 NeuroCluster Streamer Python SDK - Batch Processing Examples")
    print("=" * 70)

    try:
        # Run batch processing examples
        example_synchronous_batch_processing()
        await asyncio.sleep(1)

        await example_async_batch_processing()
        await asyncio.sleep(1)

        example_csv_file_processing()
        await asyncio.sleep(1)

        await example_memory_efficient_processing()

        print("\n🎉 All batch processing examples completed successfully!")
        print("\n💡 Key Takeaways:")
        print("   ✅ Async processing significantly improves throughput")
        print("   ✅ Concurrency control prevents overwhelming the API")
        print("   ✅ Memory-efficient patterns enable large dataset processing")
        print("   ✅ Error handling and retry logic ensure reliability")
        print("   ✅ Progress tracking helps monitor long-running operations")

    except Exception as e:
        print(f"\n💥 Batch processing examples failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n💥 Examples failed with error: {e}")
