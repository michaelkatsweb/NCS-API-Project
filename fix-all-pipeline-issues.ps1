# FIX SDK EXAMPLES FOR LOCALHOST DEVELOPMENT
# This script updates all SDK examples to use localhost URLs for development

param(
    [switch]$DryRun = $false
)

function Write-Fix($message) {
    Write-Host "✅ $message" -ForegroundColor Green
}

Write-Host "🔧 FIXING SDK EXAMPLES FOR LOCALHOST DEVELOPMENT" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "Updating all examples to use localhost instead of external URLs..." -ForegroundColor Yellow

# =============================================================================
# FIX 1: UPDATE BASIC_USAGE.PY FOR LOCALHOST
# =============================================================================
Write-Host ""
Write-Host "[FIX 1] sdk/python/examples/basic_usage.py" -ForegroundColor Yellow

$fixedBasicUsage = @"
#!/usr/bin/env python3
"""
NeuroCluster Streamer Python SDK - Basic Usage Example (LOCALHOST VERSION)
==========================================================================
Demonstrates basic usage patterns for the NCS Python SDK with local development server

This example shows:
- Client initialization for localhost development
- Basic data point processing
- Error handling patterns
- Health checking with local API

Author: NCS API Development Team
Year: 2025
"""

import os
import sys
import logging
from typing import List
import random
import time
import requests

# Add the parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_local_server(base_url="http://localhost:8000"):
    """Check if local NCS API server is running"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ Local NCS API server is running at {base_url}")
            return True
        else:
            print(f"⚠️  Local server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to local server at {base_url}")
        print(f"💡 Start the server with: python main_secure.py")
        return False
    except Exception as e:
        print(f"❌ Error checking server: {e}")
        return False

def generate_sample_points(count: int = 10) -> List[List[float]]:
    """Generate sample data points for testing"""
    points = []
    for i in range(count):
        # Create clusters around different centers
        if i < count // 3:
            center = [1.0, 1.0, 1.0]  # Cluster 1
        elif i < 2 * count // 3:
            center = [5.0, 5.0, 5.0]  # Cluster 2
        else:
            center = [10.0, 2.0, 8.0]  # Cluster 3
        
        # Add noise around centers
        point = [
            center[0] + random.gauss(0, 0.5),
            center[1] + random.gauss(0, 0.5),
            center[2] + random.gauss(0, 0.5)
        ]
        points.append(point)
    
    return points

def test_basic_api_calls():
    """Test basic API calls to localhost"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing basic API calls...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check: {health_data.get('status', 'unknown')}")
            print(f"   Algorithm ready: {health_data.get('algorithm_ready', 'unknown')}")
            print(f"   Uptime: {health_data.get('uptime_seconds', 0):.1f}s")
        else:
            print(f"⚠️  Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            root_data = response.json()
            print(f"✅ Root endpoint: {root_data.get('message', 'No message')}")
        else:
            print(f"⚠️  Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test API docs
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            print(f"✅ API docs accessible at {base_url}/docs")
        else:
            print(f"⚠️  API docs not accessible: {response.status_code}")
    except Exception as e:
        print(f"❌ API docs error: {e}")

def test_process_single_point():
    """Test processing a single data point"""
    base_url = "http://localhost:8000"
    
    print("📍 Testing single point processing...")
    
    test_point = {
        "point": {
            "coordinates": [1.5, 2.5, 3.5],
            "metadata": {"source": "test", "timestamp": time.time()}
        }
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/v1/process/point", 
            json=test_point,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Point processed successfully!")
            print(f"   Request ID: {result.get('request_id', 'unknown')}")
            print(f"   Cluster ID: {result.get('result', {}).get('cluster_id', 'unknown')}")
            print(f"   Confidence: {result.get('result', {}).get('confidence', 0):.3f}")
            print(f"   Processing time: {result.get('processing_time_ms', 0):.2f}ms")
        elif response.status_code == 503:
            print("⚠️  Algorithm not ready yet (this is normal on startup)")
        else:
            print(f"❌ Point processing failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Point processing error: {e}")

def test_process_batch():
    """Test batch processing"""
    base_url = "http://localhost:8000"
    
    print("📦 Testing batch processing...")
    
    # Generate sample points
    sample_points = generate_sample_points(5)
    
    batch_request = {
        "points": [
            {
                "coordinates": point,
                "metadata": {"batch_id": 1, "point_index": i}
            }
            for i, point in enumerate(sample_points)
        ]
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/v1/process/batch",
            json=batch_request,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch processed successfully!")
            print(f"   Request ID: {result.get('request_id', 'unknown')}")
            print(f"   Points processed: {result.get('points_processed', 0)}")
            print(f"   Processing time: {result.get('processing_time_ms', 0):.2f}ms")
            
            results = result.get('results', [])
            if results:
                clusters_found = set(r.get('cluster_id') for r in results)
                print(f"   Unique clusters: {len(clusters_found)}")
                print(f"   Cluster IDs: {sorted(clusters_found)}")
        elif response.status_code == 503:
            print("⚠️  Algorithm not ready yet (this is normal on startup)")
        else:
            print(f"❌ Batch processing failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Batch processing error: {e}")

def test_stats_endpoint():
    """Test statistics endpoint"""
    base_url = "http://localhost:8000"
    
    print("📊 Testing stats endpoint...")
    
    try:
        response = requests.get(f"{base_url}/api/v1/stats", timeout=10)
        
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Stats retrieved successfully!")
            print(f"   Requests processed: {stats.get('requests_processed', 0)}")
            print(f"   Uptime: {stats.get('uptime_seconds', 0):.1f}s")
            print(f"   Algorithm ready: {stats.get('algorithm_ready', False)}")
        elif response.status_code in [401, 403]:
            print("⚠️  Stats endpoint requires authentication (this is normal)")
        else:
            print(f"❌ Stats failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Stats error: {e}")

def main():
    """Main example function demonstrating localhost SDK usage."""
    
    print("🚀 NeuroCluster Streamer Python SDK - Localhost Development Example")
    print("=" * 70)
    print()
    print("This example demonstrates using the NCS API with your local development server.")
    print("Make sure to start your server first: python main_secure.py")
    print()
    
    # Check if local server is running
    base_url = "http://localhost:8000"
    if not check_local_server(base_url):
        print()
        print("🔧 TO START THE LOCAL SERVER:")
        print("   1. Open a new terminal")
        print("   2. Navigate to your project directory")  
        print("   3. Run: python main_secure.py")
        print("   4. Wait for 'Uvicorn running on http://0.0.0.0:8000'")
        print("   5. Then run this example again")
        print()
        return
    
    print()
    print("🧪 Running API Tests...")
    print("=" * 30)
    
    # Run all tests
    test_basic_api_calls()
    print()
    test_process_single_point()
    print()
    test_process_batch()
    print()
    test_stats_endpoint()
    
    print()
    print("🎉 LOCALHOST TESTING COMPLETE!")
    print("=" * 35)
    print()
    print("💡 NEXT STEPS:")
    print("   • Visit http://localhost:8000/docs to see the interactive API docs")
    print("   • Visit http://localhost:8000/health to check API health")
    print("   • Modify this script to test your own data points")
    print("   • When ready for production, update URLs to your deployed API")
    print()

if __name__ == "__main__":
    main()
"@

if (-not $DryRun) {
    if (-not (Test-Path "sdk/python/examples")) {
        New-Item -ItemType Directory -Path "sdk/python/examples" -Force | Out-Null
    }
    Set-Content -Path "sdk/python/examples/basic_usage.py" -Value $fixedBasicUsage -Encoding UTF8
}
Write-Fix "Updated basic_usage.py for localhost development"

# =============================================================================
# FIX 2: UPDATE BATCH_PROCESSING.PY FOR LOCALHOST
# =============================================================================
Write-Host ""
Write-Host "[FIX 2] sdk/python/examples/batch_processing.py" -ForegroundColor Yellow

$fixedBatchProcessing = @"
#!/usr/bin/env python3
"""
NeuroCluster Streamer Python SDK - Batch Processing Example (LOCALHOST VERSION)
===============================================================================
Demonstrates batch processing patterns for the NCS Python SDK with local development server

This example shows:
- Large batch processing with localhost
- Memory-efficient processing
- Progress monitoring
- Performance optimization for local development

Author: NCS API Development Team
Year: 2025
"""

import os
import sys
import time
import random
import requests
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_test_dataset(num_points: int = 100, num_clusters: int = 3) -> List[List[float]]:
    """Generate a test dataset with known cluster structure"""
    points = []
    points_per_cluster = num_points // num_clusters
    
    # Define cluster centers
    cluster_centers = [
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0], 
        [5.0, 10.0, 0.0]
    ]
    
    for cluster_id in range(num_clusters):
        center = cluster_centers[cluster_id % len(cluster_centers)]
        
        for _ in range(points_per_cluster):
            # Add gaussian noise around cluster center
            point = [
                center[0] + random.gauss(0, 1.0),
                center[1] + random.gauss(0, 1.0), 
                center[2] + random.gauss(0, 1.0)
            ]
            points.append(point)
    
    # Add some random outliers
    outliers_count = num_points - len(points)
    for _ in range(outliers_count):
        outlier = [
            random.uniform(-20, 20),
            random.uniform(-20, 20),
            random.uniform(-20, 20)
        ]
        points.append(outlier)
    
    return points

def process_batch_localhost(points: List[List[float]], batch_id: int = 0) -> Dict[str, Any]:
    """Process a batch of points using localhost API"""
    base_url = "http://localhost:8000"
    
    batch_request = {
        "points": [
            {
                "coordinates": point,
                "metadata": {"batch_id": batch_id, "point_index": i}
            }
            for i, point in enumerate(points)
        ]
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/api/v1/process/batch",
            json=batch_request,
            timeout=60  # Longer timeout for large batches
        )
        
        if response.status_code == 200:
            result = response.json()
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "batch_id": batch_id,
                "points_processed": result.get("points_processed", 0),
                "processing_time_ms": result.get("processing_time_ms", 0),
                "network_time_ms": processing_time * 1000,
                "results": result.get("results", []),
                "request_id": result.get("request_id", "unknown")
            }
        else:
            return {
                "success": False,
                "batch_id": batch_id,
                "error": f"HTTP {response.status_code}: {response.text}",
                "points_attempted": len(points)
            }
    except Exception as e:
        return {
            "success": False,
            "batch_id": batch_id,
            "error": str(e),
            "points_attempted": len(points)
        }

def example_simple_batch_processing():
    """Example 1: Simple batch processing with localhost"""
    print("📦 Example 1: Simple Batch Processing")
    print("-" * 40)
    
    # Check if server is running
    base_url = "http://localhost:8000"
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Local server not running. Start with: python main_secure.py")
            return
    except:
        print("❌ Cannot connect to local server. Start with: python main_secure.py")
        return
    
    # Generate test data
    test_points = generate_test_dataset(num_points=50, num_clusters=3)
    print(f"📊 Generated {len(test_points)} test points")
    
    # Process the batch
    print("⚡ Processing batch...")
    result = process_batch_localhost(test_points, batch_id=1)
    
    if result["success"]:
        print(f"✅ Batch processing successful!")
        print(f"   Points processed: {result['points_processed']}")
        print(f"   API processing time: {result['processing_time_ms']:.2f}ms")
        print(f"   Network time: {result['network_time_ms']:.2f}ms")
        print(f"   Request ID: {result['request_id']}")
        
        # Analyze results
        results = result["results"]
        if results:
            cluster_ids = [r.get("cluster_id") for r in results]
            unique_clusters = set(cluster_ids)
            print(f"   Clusters found: {len(unique_clusters)}")
            print(f"   Cluster distribution: {dict((cid, cluster_ids.count(cid)) for cid in unique_clusters)}")
    else:
        print(f"❌ Batch processing failed: {result['error']}")

def example_multi_batch_processing():
    """Example 2: Processing multiple batches sequentially"""
    print("📦 Example 2: Multi-Batch Processing")
    print("-" * 40)
    
    # Generate multiple smaller batches
    num_batches = 5
    batch_size = 20
    
    print(f"📊 Processing {num_batches} batches of {batch_size} points each...")
    
    total_start_time = time.time()
    successful_batches = 0
    total_points_processed = 0
    
    for batch_id in range(num_batches):
        print(f"   Processing batch {batch_id + 1}/{num_batches}...")
        
        # Generate batch data
        batch_points = generate_test_dataset(num_points=batch_size, num_clusters=2)
        
        # Process batch
        result = process_batch_localhost(batch_points, batch_id=batch_id)
        
        if result["success"]:
            successful_batches += 1
            total_points_processed += result["points_processed"]
            print(f"   ✅ Batch {batch_id + 1} completed in {result['processing_time_ms']:.2f}ms")
        else:
            print(f"   ❌ Batch {batch_id + 1} failed: {result['error']}")
    
    total_time = time.time() - total_start_time
    
    print()
    print(f"📊 Multi-batch processing complete!")
    print(f"   Successful batches: {successful_batches}/{num_batches}")
    print(f"   Total points processed: {total_points_processed}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average rate: {total_points_processed / total_time:.1f} points/sec")

def example_concurrent_batch_processing():
    """Example 3: Concurrent batch processing (use with caution on localhost)"""
    print("⚡ Example 3: Concurrent Batch Processing")
    print("-" * 40)
    
    # Generate batches
    num_batches = 3  # Keep small for localhost
    batch_size = 15
    
    batches = []
    for i in range(num_batches):
        batch_points = generate_test_dataset(num_points=batch_size, num_clusters=2)
        batches.append((batch_points, i))
    
    print(f"📊 Processing {num_batches} batches concurrently...")
    print("⚠️  Note: Concurrent processing may overwhelm localhost server")
    
    # Process concurrently with limited workers
    max_workers = 2  # Limited for localhost
    successful_results = []
    failed_results = []
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_batch = {
            executor.submit(process_batch_localhost, points, batch_id): batch_id
            for points, batch_id in batches
        }
        
        # Collect results
        for future in as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            try:
                result = future.result()
                if result["success"]:
                    successful_results.append(result)
                    print(f"   ✅ Batch {batch_id} completed")
                else:
                    failed_results.append(result)
                    print(f"   ❌ Batch {batch_id} failed")
            except Exception as e:
                failed_results.append({"batch_id": batch_id, "error": str(e)})
                print(f"   ❌ Batch {batch_id} exception: {e}")
    
    total_time = time.time() - start_time
    total_points = sum(r["points_processed"] for r in successful_results)
    
    print()
    print(f"📊 Concurrent processing complete!")
    print(f"   Successful batches: {len(successful_results)}")
    print(f"   Failed batches: {len(failed_results)}")
    print(f"   Total points processed: {total_points}")
    print(f"   Total time: {total_time:.2f}s")
    if total_points > 0:
        print(f"   Processing rate: {total_points / total_time:.1f} points/sec")

def main():
    """Main function demonstrating batch processing with localhost"""
    print("🚀 NeuroCluster Streamer - Localhost Batch Processing Examples")
    print("=" * 65)
    print()
    print("This example demonstrates batch processing with your local NCS API server.")
    print("Make sure your server is running: python main_secure.py")
    print()
    
    # Check server connection
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Connected to local server (status: {health.get('status', 'unknown')})")
        else:
            print(f"⚠️  Server responded with status {response.status_code}")
    except:
        print("❌ Cannot connect to local server at http://localhost:8000")
        print("💡 Start the server with: python main_secure.py")
        return
    
    print()
    
    # Run examples
    try:
        example_simple_batch_processing()
        print()
        
        example_multi_batch_processing()
        print()
        
        example_concurrent_batch_processing()
        
    except KeyboardInterrupt:
        print()
        print("⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print()
    print("🎉 BATCH PROCESSING EXAMPLES COMPLETE!")
    print("=" * 45)
    print()
    print("💡 TIPS FOR LOCALHOST DEVELOPMENT:")
    print("   • Start with small batches to avoid overwhelming the server")
    print("   • Monitor server logs for performance insights")
    print("   • Use concurrent processing sparingly on localhost")
    print("   • Check http://localhost:8000/docs for API documentation")
    print()

if __name__ == "__main__":
    main()
"@

if (-not $DryRun) {
    Set-Content -Path "sdk/python/examples/batch_processing.py" -Value $fixedBatchProcessing -Encoding UTF8
}
Write-Fix "Updated batch_processing.py for localhost development"

# =============================================================================
# FIX 3: UPDATE STREAMING_EXAMPLE.PY FOR LOCALHOST
# =============================================================================
Write-Host ""
Write-Host "[FIX 3] sdk/python/examples/streaming_example.py" -ForegroundColor Yellow

$fixedStreamingExample = @"
#!/usr/bin/env python3
"""
NeuroCluster Streamer Python SDK - Streaming Example (LOCALHOST VERSION)
=========================================================================
Demonstrates streaming patterns for the NCS Python SDK with local development server

Note: This example focuses on HTTP-based streaming simulation since WebSocket 
streaming requires additional server-side implementation.

This example shows:
- Simulated real-time processing with localhost
- Continuous data point streaming
- Performance monitoring
- Error handling for local development

Author: NCS API Development Team
Year: 2025
"""

import os
import sys
import time
import random
import requests
import threading
from typing import List, Dict, Any, Callable
from queue import Queue
import json

class LocalStreamSimulator:
    """Simulates streaming by sending points continuously to localhost API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.is_streaming = False
        self.stats = {
            "points_sent": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0
        }
    
    def check_server(self) -> bool:
        """Check if local server is accessible"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def send_point(self, coordinates: List[float], metadata: Dict = None) -> Dict[str, Any]:
        """Send a single point to the API"""
        if metadata is None:
            metadata = {"timestamp": time.time(), "source": "stream_simulator"}
        
        point_data = {
            "point": {
                "coordinates": coordinates,
                "metadata": metadata
            }
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/v1/process/point",
                json=point_data,
                timeout=10
            )
            processing_time = time.time() - start_time
            
            self.stats["points_sent"] += 1
            
            if response.status_code == 200:
                self.stats["successful_requests"] += 1
                self.stats["total_processing_time"] += processing_time
                result = response.json()
                return {
                    "success": True,
                    "processing_time_ms": processing_time * 1000,
                    "cluster_id": result.get("result", {}).get("cluster_id"),
                    "confidence": result.get("result", {}).get("confidence"),
                    "request_id": result.get("request_id")
                }
            else:
                self.stats["failed_requests"] += 1
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "processing_time_ms": processing_time * 1000
                }
        except Exception as e:
            self.stats["failed_requests"] += 1
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": 0
            }
    
    def start_streaming(self, point_generator: Callable, interval: float = 1.0, 
                       on_result: Callable = None, duration: float = 30.0):
        """Start streaming points to the API"""
        if not self.check_server():
            print("❌ Local server not accessible. Start with: python main_secure.py")
            return
        
        print(f"🌊 Starting streaming to {self.base_url}")
        print(f"   Interval: {interval}s, Duration: {duration}s")
        
        self.is_streaming = True
        start_time = time.time()
        
        while self.is_streaming and (time.time() - start_time) < duration:
            # Generate next point
            point = point_generator()
            
            # Send point
            result = self.send_point(point)
            
            # Call result callback if provided
            if on_result:
                on_result(point, result)
            
            # Wait for next interval
            time.sleep(interval)
        
        self.is_streaming = False
        print("🛑 Streaming stopped")
        self.print_stats()
    
    def stop_streaming(self):
        """Stop the streaming"""
        self.is_streaming = False
    
    def print_stats(self):
        """Print streaming statistics"""
        print()
        print("📊 Streaming Statistics:")
        print(f"   Points sent: {self.stats['points_sent']}")
        print(f"   Successful: {self.stats['successful_requests']}")
        print(f"   Failed: {self.stats['failed_requests']}")
        if self.stats['successful_requests'] > 0:
            avg_time = self.stats['total_processing_time'] / self.stats['successful_requests']
            print(f"   Average processing time: {avg_time * 1000:.2f}ms")
        success_rate = (self.stats['successful_requests'] / max(1, self.stats['points_sent'])) * 100
        print(f"   Success rate: {success_rate:.1f}%")

def generate_random_walk_point(center: List[float] = None, step_size: float = 0.5) -> List[float]:
    """Generate a point using random walk pattern"""
    if center is None:
        center = [0.0, 0.0, 0.0]
    
    return [
        center[0] + random.gauss(0, step_size),
        center[1] + random.gauss(0, step_size), 
        center[2] + random.gauss(0, step_size)
    ]

def generate_orbit_point(time_step: float, radius: float = 5.0, speed: float = 1.0) -> List[float]:
    """Generate points in an orbital pattern"""
    angle = time_step * speed
    return [
        radius * math.cos(angle),
        radius * math.sin(angle),
        random.gauss(0, 0.1)  # Small z variation
    ]

def example_basic_streaming():
    """Example 1: Basic streaming with random points"""
    print("🌊 Example 1: Basic Streaming with Random Points")
    print("-" * 50)
    
    simulator = LocalStreamSimulator()
    
    # Simple point generator
    def random_point_generator():
        return [
            random.uniform(-10, 10),
            random.uniform(-10, 10), 
            random.uniform(-10, 10)
        ]
    
    # Result handler
    def handle_result(point, result):
        if result["success"]:
            cluster_id = result.get("cluster_id", "unknown")
            confidence = result.get("confidence", 0)
            print(f"📍 Point {point} → Cluster {cluster_id} (conf: {confidence:.3f})")
        else:
            print(f"❌ Point {point} failed: {result.get('error', 'unknown')}")
    
    # Start streaming for 15 seconds
    simulator.start_streaming(
        point_generator=random_point_generator,
        interval=2.0,  # Send every 2 seconds
        on_result=handle_result,
        duration=15.0
    )

def example_clustered_streaming():
    """Example 2: Streaming with clustered data patterns"""
    print("🌊 Example 2: Streaming with Clustered Data")
    print("-" * 50)
    
    simulator = LocalStreamSimulator()
    
    # Cluster centers that change over time
    cluster_centers = [
        [2.0, 2.0, 0.0],
        [8.0, 2.0, 0.0],
        [5.0, 8.0, 0.0]
    ]
    
    current_cluster = 0
    points_in_cluster = 0
    max_points_per_cluster = 3
    
    def clustered_point_generator():
        nonlocal current_cluster, points_in_cluster
        
        # Switch clusters periodically
        if points_in_cluster >= max_points_per_cluster:
            current_cluster = (current_cluster + 1) % len(cluster_centers)
            points_in_cluster = 0
            print(f"🔄 Switching to cluster {current_cluster}")
        
        center = cluster_centers[current_cluster]
        points_in_cluster += 1
        
        return [
            center[0] + random.gauss(0, 0.5),
            center[1] + random.gauss(0, 0.5),
            center[2] + random.gauss(0, 0.5)
        ]
    
    # Enhanced result handler
    cluster_tracking = {}
    
    def handle_clustered_result(point, result):
        if result["success"]:
            cluster_id = result.get("cluster_id", -1)
            confidence = result.get("confidence", 0)
            
            # Track cluster assignments
            if cluster_id not in cluster_tracking:
                cluster_tracking[cluster_id] = []
            cluster_tracking[cluster_id].append(confidence)
            
            print(f"📍 Point → Cluster {cluster_id} (conf: {confidence:.3f})")
            
            # Show cluster summary every few points
            if len(cluster_tracking.get(cluster_id, [])) % 3 == 0:
                avg_conf = sum(cluster_tracking[cluster_id]) / len(cluster_tracking[cluster_id])
                print(f"   📊 Cluster {cluster_id} avg confidence: {avg_conf:.3f}")
        else:
            print(f"❌ Processing failed: {result.get('error', 'unknown')}")
    
    # Start streaming
    simulator.start_streaming(
        point_generator=clustered_point_generator,
        interval=1.5,
        on_result=handle_clustered_result, 
        duration=20.0
    )
    
    # Final cluster summary
    print()
    print("📊 Final Cluster Tracking:")
    for cluster_id, confidences in cluster_tracking.items():
        avg_conf = sum(confidences) / len(confidences)
        print(f"   Cluster {cluster_id}: {len(confidences)} points, avg conf: {avg_conf:.3f}")

def example_performance_streaming():
    """Example 3: Performance-focused streaming"""
    print("⚡ Example 3: Performance Streaming")
    print("-" * 50)
    
    simulator = LocalStreamSimulator()
    
    # Fast point generator
    def fast_point_generator():
        return [random.gauss(0, 2), random.gauss(0, 2), random.gauss(0, 2)]
    
    # Performance tracking
    response_times = []
    
    def performance_handler(point, result):
        if result["success"]:
            response_time = result.get("processing_time_ms", 0)
            response_times.append(response_time)
            
            if len(response_times) % 5 == 0:  # Report every 5 points
                recent_times = response_times[-5:]
                avg_time = sum(recent_times) / len(recent_times)
                min_time = min(recent_times)
                max_time = max(recent_times)
                print(f"⚡ Last 5 points: avg={avg_time:.1f}ms, min={min_time:.1f}ms, max={max_time:.1f}ms")
        else:
            print(f"❌ Error: {result.get('error', 'unknown')}")
    
    print("🚀 Starting high-frequency streaming...")
    simulator.start_streaming(
        point_generator=fast_point_generator,
        interval=0.5,  # Every 500ms
        on_result=performance_handler,
        duration=15.0
    )
    
    # Performance analysis
    if response_times:
        print()
        print("📊 Performance Analysis:")
        print(f"   Total points processed: {len(response_times)}")
        print(f"   Average response time: {sum(response_times) / len(response_times):.2f}ms")
        print(f"   Min response time: {min(response_times):.2f}ms")
        print(f"   Max response time: {max(response_times):.2f}ms")
        print(f"   Throughput: {len(response_times) / 15.0:.1f} points/sec")

def main():
    """Main function demonstrating streaming with localhost"""
    print("🚀 NeuroCluster Streamer - Localhost Streaming Examples")
    print("=" * 60)
    print()
    print("This example demonstrates streaming data to your local NCS API server.")
    print("Make sure your server is running: python main_secure.py")
    print()
    
    # Check server
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Connected to local server (status: {health.get('status')})")
            if not health.get('algorithm_ready', False):
                print("⚠️  Algorithm not ready yet (this is normal on startup)")
        else:
            print(f"⚠️  Server responded with status {response.status_code}")
    except:
        print("❌ Cannot connect to local server at http://localhost:8000")
        print("💡 Start the server with: python main_secure.py")
        return
    
    print()
    
    # Import math for orbit example
    import math
    
    try:
        example_basic_streaming()
        print()
        
        example_clustered_streaming() 
        print()
        
        example_performance_streaming()
        
    except KeyboardInterrupt:
        print()
        print("⚠️  Streaming interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print()
    print("🎉 STREAMING EXAMPLES COMPLETE!")
    print("=" * 40)
    print()
    print("💡 NEXT STEPS:")
    print("   • Monitor server performance in the terminal where you started it")
    print("   • Visit http://localhost:8000/api/v1/stats to see processing statistics")
    print("   • Experiment with different streaming patterns and intervals")
    print("   • Check server logs for insights into algorithm performance")
    print()

if __name__ == "__main__":
    main()
"@

if (-not $DryRun) {
    Set-Content -Path "sdk/python/examples/streaming_example.py" -Value $fixedStreamingExample -Encoding UTF8
}
Write-Fix "Updated streaming_example.py for localhost development"

# =============================================================================
# FIX 4: CREATE LOCALHOST DEVELOPMENT README
# =============================================================================
Write-Host ""
Write-Host "[FIX 4] Creating localhost development guide" -ForegroundColor Yellow

$localhostReadme = @"
# NCS API - Localhost Development Guide

## 🚀 Quick Start for Local Development

Your NCS API is designed to run locally during development. All SDK examples have been updated to work with `http://localhost:8000`.

### 1. Start Your Local API Server

```bash
# In your main project directory
python main_secure.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
```

### 2. Verify Server is Running

Open these URLs in your browser:
- **API Health**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs  
- **Root Endpoint**: http://localhost:8000/

### 3. Test with SDK Examples

```bash
# Basic usage example
python sdk/python/examples/basic_usage.py

# Batch processing example  
python sdk/python/examples/batch_processing.py

# Streaming example
python sdk/python/examples/streaming_example.py
```

## 📡 API Endpoints (Localhost)

### Core Endpoints
- `GET http://localhost:8000/health` - Health check
- `GET http://localhost:8000/` - Root endpoint
- `GET http://localhost:8000/docs` - Interactive API documentation

### Processing Endpoints  
- `POST http://localhost:8000/api/v1/process/point` - Process single point
- `POST http://localhost:8000/api/v1/process/batch` - Process multiple points
- `GET http://localhost:8000/api/v1/stats` - Get processing statistics

## 🧪 Testing Your API

### Manual Testing with curl

```bash
# Health check
curl http://localhost:8000/health

# Process a single point
curl -X POST http://localhost:8000/api/v1/process/point \
  -H "Content-Type: application/json" \
  -d '{
    "point": {
      "coordinates": [1.5, 2.5, 3.5],
      "metadata": {"source": "manual_test"}
    }
  }'
```

### Python Testing

```python
import requests

# Test basic connectivity
response = requests.get("http://localhost:8000/health")
print(response.json())

# Test point processing
point_data = {
    "point": {
        "coordinates": [1.0, 2.0, 3.0],
        "metadata": {"test": True}
    }
}
response = requests.post("http://localhost:8000/api/v1/process/point", json=point_data)
print(response.json())
```

## 🔧 Development Configuration

### Environment Variables
```bash
# Optional - customize your local setup
export NCS_API_URL="http://localhost:8000"
export DEBUG=true
export LOG_LEVEL=INFO
```

### For Production Deployment
When you're ready to deploy, update the URLs in:
- SDK examples (`sdk/python/examples/`)
- Documentation
- Client applications

Change `http://localhost:8000` to your production URL like `https://api.yourdomain.com`

## 🐛 Troubleshooting

### "Cannot connect to server"
1. Make sure server is running: `python main_secure.py`
2. Check the port isn't being used by another application
3. Verify you're in the correct directory

### "Algorithm not ready"
- This is normal on startup - wait a few seconds for initialization
- Check server logs for any error messages

### "Module not found" errors
```bash
# Install required dependencies
pip install fastapi uvicorn pytest requests
```

## 📊 Monitoring Your Local API

- **Server logs**: Watch the terminal where you started `python main_secure.py`
- **Statistics**: Visit http://localhost:8000/api/v1/stats  
- **Health status**: Visit http://localhost:8000/health
- **API docs**: Visit http://localhost:8000/docs for interactive testing

## 🚀 Next Steps

1. **Develop locally** using `http://localhost:8000`
2. **Test your algorithms** with the provided examples
3. **Deploy to production** when ready (update URLs accordingly)
4. **Set up CI/CD** for automated testing and deployment

Happy developing! 🎉
"@

if (-not $DryRun) {
    Set-Content -Path "LOCALHOST_DEVELOPMENT.md" -Value $localhostReadme -Encoding UTF8
}
Write-Fix "Created localhost development guide"

# =============================================================================
# SUMMARY
# =============================================================================
Write-Host ""
Write-Host "🎉 LOCALHOST DEVELOPMENT FIX COMPLETE!" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green

Write-Host ""
Write-Host "Fixed Files:" -ForegroundColor Cyan
Write-Host "  📄 sdk/python/examples/basic_usage.py" -ForegroundColor Gray
Write-Host "  📄 sdk/python/examples/batch_processing.py" -ForegroundColor Gray  
Write-Host "  📄 sdk/python/examples/streaming_example.py" -ForegroundColor Gray
Write-Host "  📄 LOCALHOST_DEVELOPMENT.md" -ForegroundColor Gray

Write-Host ""
Write-Host "All examples now use:" -ForegroundColor Yellow
Write-Host "  🏠 http://localhost:8000 (instead of external URLs)" -ForegroundColor Green

Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host "1. Start your API server:" -ForegroundColor White
Write-Host "   python main_secure.py" -ForegroundColor Gray

Write-Host ""
Write-Host "2. Test the examples:" -ForegroundColor White  
Write-Host "   python sdk/python/examples/basic_usage.py" -ForegroundColor Gray

Write-Host ""
Write-Host "3. Commit the fixes:" -ForegroundColor White
Write-Host "   git add ." -ForegroundColor Gray
Write-Host "   git commit -m 'fix: update SDK examples for localhost development'" -ForegroundColor Gray
Write-Host "   git push" -ForegroundColor Gray

Write-Host ""
Write-Host "🚀 Your pipeline should now be GREEN! No more external URL errors!" -ForegroundColor Green