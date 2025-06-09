# NeuroCluster Streamer API - Advanced Usage Guide

Master the advanced features of the NCS API for production-scale deployments, real-time streaming, performance optimization, and enterprise integration patterns.

## 📋 Table of Contents

- [Real-Time Streaming](#-real-time-streaming)
- [Batch Processing](#-batch-processing)
- [Algorithm Optimization](#-algorithm-optimization)
- [Performance Tuning](#-performance-tuning)
- [Production Integration](#-production-integration)
- [Monitoring & Observability](#-monitoring--observability)
- [Error Handling & Resilience](#-error-handling--resilience)
- [Security Best Practices](#-security-best-practices)
- [Real-World Use Cases](#-real-world-use-cases)

## 🌊 Real-Time Streaming

Process continuous data streams with WebSocket connections for ultra-low latency clustering.

### Python Streaming Client

```python
import asyncio
import json
import time
from ncs_client import NCSClient

class StreamProcessor:
    def __init__(self, api_url, api_key):
        self.client = NCSClient(base_url=api_url, api_key=api_key)
        self.stream = None
        self.cluster_cache = {}
        
    async def start_streaming(self):
        """Initialize streaming connection with comprehensive callbacks."""
        
        def on_cluster_update(data):
            """Handle real-time cluster updates."""
            if data['type'] == 'cluster_update':
                cluster = data['data']
                cluster_id = cluster['cluster_id']
                
                print(f"🔄 Cluster {cluster_id} updated:")
                print(f"   Size: {cluster['size']} points")
                print(f"   Quality: {cluster['quality']:.3f}")
                print(f"   Drift: {cluster.get('drift_coefficient', 0):.3f}")
                
                # Update local cache
                self.cluster_cache[cluster_id] = cluster
                
            elif data['type'] == 'algorithm_status':
                status = data['data']
                print(f"📊 Algorithm status: Quality={status['clustering_quality']:.3f}")
                
        def on_error(error):
            """Handle streaming errors with reconnection logic."""
            print(f"❌ Stream error: {error}")
            
        def on_close():
            """Handle connection close."""
            print("🔌 Stream connection closed")
            
        # Start streaming with callbacks
        self.stream = await self.client.start_streaming({
            'onMessage': on_cluster_update,
            'onError': on_error,
            'onClose': on_close,
            'reconnectAttempts': 5
        })
        
        print("🌊 Streaming connection established")
        
    async def process_data_stream(self, data_generator):
        """Process continuous data stream."""
        
        async for point in data_generator:
            try:
                # Send point to streaming endpoint
                await self.stream.send_point(point)
                
                # Optional: Send metadata
                await self.stream.send_point(point, metadata={
                    'timestamp': time.time(),
                    'source': 'sensor_001',
                    'quality': 'high'
                })
                
            except Exception as e:
                print(f"⚠️ Failed to send point {point}: {e}")
                
    async def batch_stream(self, points):
        """Send batch of points to stream."""
        await self.stream.send_batch(points)
        
    async def stop_streaming(self):
        """Gracefully close streaming connection."""
        if self.stream:
            await self.stream.stop()
            print("🛑 Streaming stopped")

# Usage example
async def streaming_example():
    processor = StreamProcessor(
        api_url="https://api.yourdomain.com",
        api_key="your-api-key"
    )
    
    # Start streaming
    await processor.start_streaming()
    
    # Simulate continuous data
    async def data_generator():
        for i in range(1000):
            # Simulate sensor data
            yield [
                i + random.random(),
                i * 2 + random.random(),
                i / 2 + random.random()
            ]
            await asyncio.sleep(0.1)  # 10 points per second
    
    # Process stream
    await processor.process_data_stream(data_generator())
    
    # Stop when done
    await processor.stop_streaming()

# Run the example
asyncio.run(streaming_example())
```

### JavaScript Streaming Client

```javascript
import { NCSClient } from 'ncs-javascript-sdk';

class AdvancedStreamProcessor {
  constructor(apiUrl, apiKey) {
    this.client = new NCSClient({
      baseUrl: apiUrl,
      apiKey: apiKey
    });
    this.stream = null;
    this.clusterState = new Map();
    this.metrics = {
      pointsProcessed: 0,
      clustersUpdated: 0,
      averageLatency: 0
    };
  }

  async startStreaming() {
    const streamOptions = {
      connectionId: 'main-processor',
      reconnectAttempts: 10,
      
      onMessage: (data) => {
        this.handleStreamMessage(data);
      },
      
      onError: (error) => {
        console.error('🚨 Stream error:', error);
        this.handleStreamError(error);
      },
      
      onReconnect: (attempt) => {
        console.log(`🔄 Reconnecting... attempt ${attempt}`);
      },
      
      onClose: () => {
        console.log('🔌 Stream closed');
        this.handleStreamClose();
      }
    };

    this.stream = await this.client.startStreaming(streamOptions);
    console.log('🌊 Advanced streaming started');
  }

  handleStreamMessage(data) {
    switch (data.type) {
      case 'cluster_update':
        this.handleClusterUpdate(data.data);
        break;
        
      case 'algorithm_status':
        this.handleAlgorithmStatus(data.data);
        break;
        
      case 'drift_detection':
        this.handleDriftDetection(data.data);
        break;
        
      case 'performance_metrics':
        this.updateMetrics(data.data);
        break;
    }
  }

  handleClusterUpdate(cluster) {
    const clusterId = cluster.cluster_id;
    const previousState = this.clusterState.get(clusterId);
    
    // Detect significant changes
    if (previousState) {
      const sizeDelta = cluster.size - previousState.size;
      const qualityDelta = cluster.quality - previousState.quality;
      
      if (Math.abs(sizeDelta) > 5) {
        console.log(`📈 Cluster ${clusterId} size changed by ${sizeDelta}`);
      }
      
      if (Math.abs(qualityDelta) > 0.1) {
        console.log(`📊 Cluster ${clusterId} quality changed by ${qualityDelta.toFixed(3)}`);
      }
    }
    
    // Update state
    this.clusterState.set(clusterId, cluster);
    this.metrics.clustersUpdated++;
  }

  handleDriftDetection(driftData) {
    console.log('🌀 Concept drift detected:', {
      severity: driftData.severity,
      affectedClusters: driftData.affected_clusters,
      recommendation: driftData.recommendation
    });
    
    // Automatic handling based on severity
    if (driftData.severity === 'high') {
      this.requestAlgorithmReset();
    }
  }

  async requestAlgorithmReset() {
    try {
      await this.client.resetAlgorithm();
      console.log('🔄 Algorithm reset requested due to high drift');
    } catch (error) {
      console.error('❌ Failed to reset algorithm:', error);
    }
  }

  // High-throughput streaming
  async streamHighVolume(dataSource, options = {}) {
    const batchSize = options.batchSize || 100;
    const maxConcurrent = options.maxConcurrent || 5;
    const buffer = [];
    
    let activeBatches = 0;
    
    for await (const point of dataSource) {
      buffer.push(point);
      
      if (buffer.length >= batchSize) {
        // Wait if too many concurrent batches
        while (activeBatches >= maxConcurrent) {
          await new Promise(resolve => setTimeout(resolve, 10));
        }
        
        // Send batch asynchronously
        activeBatches++;
        this.stream.sendBatch([...buffer])
          .then(() => {
            activeBatches--;
            this.metrics.pointsProcessed += buffer.length;
          })
          .catch(error => {
            activeBatches--;
            console.error('❌ Batch failed:', error);
          });
        
        buffer.length = 0; // Clear buffer
      }
    }
    
    // Send remaining points
    if (buffer.length > 0) {
      await this.stream.sendBatch(buffer);
    }
  }
}

// Usage example
async function advancedStreamingExample() {
  const processor = new AdvancedStreamProcessor(
    'https://api.yourdomain.com',
    'your-api-key'
  );
  
  await processor.startStreaming();
  
  // Simulate high-volume data source
  async function* generateHighVolumeData() {
    for (let i = 0; i < 10000; i++) {
      yield [
        Math.random() * 100,
        Math.random() * 100,
        Math.random() * 100
      ];
      
      // Throttle to prevent overwhelming
      if (i % 100 === 0) {
        await new Promise(resolve => setTimeout(resolve, 10));
      }
    }
  }
  
  // Process high-volume stream
  await processor.streamHighVolume(generateHighVolumeData(), {
    batchSize: 50,
    maxConcurrent: 3
  });
}
```

## 🔄 Batch Processing

Efficiently process large datasets with optimized batch operations.

### Intelligent Batch Processing

```python
import asyncio
import numpy as np
from typing import List, Generator, Optional
from ncs_client import AsyncNCSClient

class BatchProcessor:
    def __init__(self, client: AsyncNCSClient, config: dict = None):
        self.client = client
        self.config = {
            'batch_size': 1000,
            'max_concurrent': 5,
            'retry_attempts': 3,
            'timeout_per_batch': 30,
            'memory_limit_mb': 512,
            **config or {}
        }
        self.stats = {
            'batches_processed': 0,
            'total_points': 0,
            'total_time': 0,
            'failed_batches': 0
        }
    
    async def process_large_dataset(
        self, 
        dataset: List[List[float]], 
        progress_callback: Optional[callable] = None
    ) -> dict:
        """Process large dataset with intelligent batching."""
        
        # Calculate optimal batch size based on data characteristics
        optimal_batch_size = self._calculate_optimal_batch_size(dataset)
        batches = self._create_batches(dataset, optimal_batch_size)
        
        print(f"📊 Processing {len(dataset)} points in {len(batches)} batches")
        print(f"🎯 Optimal batch size: {optimal_batch_size}")
        
        # Process batches with concurrency control
        results = []
        semaphore = asyncio.Semaphore(self.config['max_concurrent'])
        
        async def process_batch_with_semaphore(batch_id, batch_points):
            async with semaphore:
                return await self._process_single_batch(batch_id, batch_points)
        
        # Create tasks for all batches
        tasks = [
            process_batch_with_semaphore(i, batch)
            for i, batch in enumerate(batches)
        ]
        
        # Process with progress tracking
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await task
                results.append(result)
                
                if progress_callback:
                    progress = (i + 1) / len(tasks)
                    progress_callback(progress, result)
                    
            except Exception as e:
                print(f"❌ Batch {i} failed: {e}")
                self.stats['failed_batches'] += 1
        
        # Combine results
        combined_result = self._combine_batch_results(results)
        
        print(f"✅ Batch processing complete:")
        print(f"   Total points: {self.stats['total_points']}")
        print(f"   Success rate: {(len(results) / len(batches)) * 100:.1f}%")
        print(f"   Average time per batch: {self.stats['total_time'] / len(results):.2f}s")
        
        return combined_result
    
    def _calculate_optimal_batch_size(self, dataset: List[List[float]]) -> int:
        """Calculate optimal batch size based on data characteristics."""
        
        # Consider data dimensions
        if not dataset:
            return self.config['batch_size']
            
        dimensions = len(dataset[0])
        data_size = len(dataset)
        
        # Estimate memory per point (rough calculation)
        bytes_per_point = dimensions * 8 + 64  # 8 bytes per float + overhead
        memory_limit_bytes = self.config['memory_limit_mb'] * 1024 * 1024
        
        # Calculate batch size based on memory limit
        memory_based_batch_size = memory_limit_bytes // bytes_per_point
        
        # Consider processing complexity (higher dimensions = smaller batches)
        complexity_factor = max(1, dimensions / 10)
        complexity_adjusted_size = int(memory_based_batch_size / complexity_factor)
        
        # Apply bounds
        optimal_size = max(
            100,  # Minimum batch size
            min(
                complexity_adjusted_size,
                self.config['batch_size'],
                data_size  # Don't exceed total data size
            )
        )
        
        return optimal_size
    
    def _create_batches(
        self, 
        dataset: List[List[float]], 
        batch_size: int
    ) -> List[List[List[float]]]:
        """Create balanced batches from dataset."""
        
        batches = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    async def _process_single_batch(
        self, 
        batch_id: int, 
        batch_points: List[List[float]]
    ) -> dict:
        """Process a single batch with retry logic."""
        
        for attempt in range(self.config['retry_attempts']):
            try:
                start_time = asyncio.get_event_loop().time()
                
                result = await self.client.process_points(
                    batch_points,
                    options={
                        'batch_mode': True,
                        'timeout': self.config['timeout_per_batch'] * 1000
                    }
                )
                
                end_time = asyncio.get_event_loop().time()
                processing_time = end_time - start_time
                
                # Update statistics
                self.stats['batches_processed'] += 1
                self.stats['total_points'] += len(batch_points)
                self.stats['total_time'] += processing_time
                
                print(f"✅ Batch {batch_id}: {len(batch_points)} points, "
                      f"{len(result.clusters)} clusters, "
                      f"{processing_time:.2f}s")
                
                return {
                    'batch_id': batch_id,
                    'result': result,
                    'processing_time': processing_time,
                    'point_count': len(batch_points)
                }
                
            except asyncio.TimeoutError:
                print(f"⏰ Batch {batch_id} timeout (attempt {attempt + 1})")
                if attempt == self.config['retry_attempts'] - 1:
                    raise
                    
            except Exception as e:
                print(f"❌ Batch {batch_id} error (attempt {attempt + 1}): {e}")
                if attempt == self.config['retry_attempts'] - 1:
                    raise
                    
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
    
    def _combine_batch_results(self, batch_results: List[dict]) -> dict:
        """Combine results from multiple batches."""
        
        all_clusters = []
        all_outliers = []
        total_processing_time = 0
        total_points = 0
        
        for batch_result in batch_results:
            if batch_result:
                result = batch_result['result']
                all_clusters.extend(result.clusters)
                all_outliers.extend(result.outliers)
                total_processing_time += batch_result['processing_time']
                total_points += batch_result['point_count']
        
        # Calculate combined metrics
        average_quality = np.mean([c.quality for c in all_clusters]) if all_clusters else 0
        
        return {
            'clusters': all_clusters,
            'outliers': all_outliers,
            'total_processing_time': total_processing_time,
            'total_points': total_points,
            'average_quality': average_quality,
            'batch_count': len(batch_results)
        }

# Usage example
async def batch_processing_example():
    # Create large synthetic dataset
    dataset = np.random.rand(50000, 5).tolist()  # 50K points, 5 dimensions
    
    client = AsyncNCSClient(
        base_url="https://api.yourdomain.com",
        api_key="your-api-key"
    )
    
    processor = BatchProcessor(client, config={
        'batch_size': 2000,
        'max_concurrent': 3,
        'memory_limit_mb': 256
    })
    
    def progress_callback(progress, batch_result):
        print(f"📈 Progress: {progress * 100:.1f}% complete")
    
    result = await processor.process_large_dataset(
        dataset, 
        progress_callback=progress_callback
    )
    
    print(f"🎉 Final results:")
    print(f"   Total clusters: {len(result['clusters'])}")
    print(f"   Total outliers: {len(result['outliers'])}")
    print(f"   Average quality: {result['average_quality']:.3f}")
```

### Parallel Processing with Multiple Workers

```python
import asyncio
import aiohttp
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

class ParallelBatchProcessor:
    def __init__(self, api_url: str, api_key: str, num_workers: int = None):
        self.api_url = api_url
        self.api_key = api_key
        self.num_workers = num_workers or mp.cpu_count()
        
    async def process_with_workers(self, dataset: List[List[float]]) -> dict:
        """Process dataset using multiple worker processes."""
        
        # Split dataset among workers
        chunk_size = len(dataset) // self.num_workers
        chunks = [
            dataset[i:i + chunk_size]
            for i in range(0, len(dataset), chunk_size)
        ]
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            loop = asyncio.get_event_loop()
            
            tasks = [
                loop.run_in_executor(
                    executor,
                    self._process_chunk_sync,
                    chunk,
                    worker_id
                )
                for worker_id, chunk in enumerate(chunks)
            ]
            
            results = await asyncio.gather(*tasks)
        
        # Combine results from all workers
        return self._combine_worker_results(results)
    
    def _process_chunk_sync(self, chunk: List[List[float]], worker_id: int) -> dict:
        """Synchronous processing function for worker processes."""
        
        import requests
        import time
        
        print(f"🔄 Worker {worker_id} processing {len(chunk)} points")
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {'points': chunk}
        
        start_time = time.time()
        response = requests.post(
            f"{self.api_url}/api/v1/process_points",
            json=data,
            headers=headers,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            processing_time = time.time() - start_time
            
            print(f"✅ Worker {worker_id} completed in {processing_time:.2f}s")
            
            return {
                'worker_id': worker_id,
                'result': result,
                'processing_time': processing_time,
                'success': True
            }
        else:
            print(f"❌ Worker {worker_id} failed: {response.status_code}")
            return {
                'worker_id': worker_id,
                'error': response.text,
                'success': False
            }
```

## ⚙️ Algorithm Optimization

Fine-tune the clustering algorithm for your specific use case.

### Custom Algorithm Configuration

```python
from ncs_client import NCSClient

class AlgorithmTuner:
    def __init__(self, client: NCSClient):
        self.client = client
        self.default_config = self._get_default_config()
        
    def _get_default_config(self) -> dict:
        """Get current algorithm configuration."""
        status = self.client.get_algorithm_status()
        return {
            'base_threshold': status.current_dynamic_threshold,
            'learning_rate': 0.06,  # Default value
            'max_clusters': 30,
            'memory_limit_mb': 50
        }
    
    def optimize_for_use_case(self, use_case: str) -> dict:
        """Get optimized configuration for specific use cases."""
        
        configs = {
            'real_time_low_latency': {
                'base_threshold': 0.65,      # Lower threshold for faster processing
                'learning_rate': 0.08,       # Higher learning rate for quick adaptation
                'max_clusters': 20,          # Fewer clusters for speed
                'memory_limit_mb': 32,       # Lower memory usage
                'batch_size': 100,           # Smaller batches
                'enable_drift_detection': False  # Disable for speed
            },
            
            'high_accuracy_research': {
                'base_threshold': 0.75,      # Higher threshold for quality
                'learning_rate': 0.04,       # Lower learning rate for stability
                'max_clusters': 50,          # More clusters allowed
                'memory_limit_mb': 128,      # More memory for accuracy
                'batch_size': 2000,          # Larger batches
                'enable_drift_detection': True,
                'quality_check_interval': 100
            },
            
            'large_scale_production': {
                'base_threshold': 0.70,      # Balanced threshold
                'learning_rate': 0.06,       # Standard learning rate
                'max_clusters': 40,          # Good balance
                'memory_limit_mb': 64,       # Reasonable memory usage
                'batch_size': 1000,          # Standard batch size
                'enable_drift_detection': True,
                'auto_scaling': True
            },
            
            'streaming_iot': {
                'base_threshold': 0.68,      # Slightly lower for variety
                'learning_rate': 0.10,       # High adaptation for changing patterns
                'max_clusters': 25,          # Moderate cluster count
                'memory_limit_mb': 48,       # IoT-friendly memory usage
                'drift_sensitivity': 0.3,   # High sensitivity to changes
                'outlier_threshold': 0.85    # IoT data often has outliers
            }
        }
        
        if use_case not in configs:
            raise ValueError(f"Unknown use case: {use_case}")
            
        return configs[use_case]
    
    def benchmark_configurations(
        self, 
        test_dataset: List[List[float]], 
        configurations: List[dict]
    ) -> dict:
        """Benchmark different configurations against test dataset."""
        
        results = {}
        
        for i, config in enumerate(configurations):
            print(f"🧪 Testing configuration {i + 1}/{len(configurations)}")
            
            try:
                # Apply configuration
                self._apply_configuration(config)
                
                # Run benchmark
                start_time = time.time()
                result = self.client.process_points(test_dataset)
                end_time = time.time()
                
                # Calculate metrics
                metrics = {
                    'processing_time': end_time - start_time,
                    'cluster_count': len(result.clusters),
                    'outlier_count': len(result.outliers),
                    'algorithm_quality': result.algorithm_quality,
                    'average_cluster_quality': np.mean([c.quality for c in result.clusters]),
                    'throughput': len(test_dataset) / (end_time - start_time)
                }
                
                results[f"config_{i}"] = {
                    'configuration': config,
                    'metrics': metrics
                }
                
                print(f"✅ Config {i + 1}: Quality={metrics['algorithm_quality']:.3f}, "
                      f"Time={metrics['processing_time']:.2f}s")
                
            except Exception as e:
                print(f"❌ Config {i + 1} failed: {e}")
                results[f"config_{i}"] = {
                    'configuration': config,
                    'error': str(e)
                }
        
        # Find best configuration
        best_config = self._find_best_configuration(results)
        
        return {
            'all_results': results,
            'best_configuration': best_config,
            'recommendation': self._generate_recommendation(best_config)
        }
    
    def _apply_configuration(self, config: dict):
        """Apply configuration to the algorithm."""
        # Note: In a real implementation, this would call an API endpoint
        # to update algorithm parameters
        print(f"🔧 Applying configuration: {config}")
    
    def adaptive_tuning(
        self, 
        dataset: List[List[float]], 
        target_metrics: dict
    ) -> dict:
        """Automatically tune parameters to achieve target metrics."""
        
        print("🎯 Starting adaptive tuning...")
        
        # Define parameter search space
        parameter_space = {
            'base_threshold': [0.60, 0.65, 0.70, 0.75, 0.80],
            'learning_rate': [0.02, 0.04, 0.06, 0.08, 0.10],
            'max_clusters': [15, 20, 25, 30, 35, 40]
        }
        
        best_config = None
        best_score = float('-inf')
        
        # Simple grid search (in production, use more sophisticated optimization)
        for threshold in parameter_space['base_threshold']:
            for lr in parameter_space['learning_rate']:
                for max_clusters in parameter_space['max_clusters']:
                    
                    config = {
                        'base_threshold': threshold,
                        'learning_rate': lr,
                        'max_clusters': max_clusters
                    }
                    
                    try:
                        # Test configuration
                        self._apply_configuration(config)
                        result = self.client.process_points(dataset[:1000])  # Use subset for speed
                        
                        # Calculate score based on target metrics
                        score = self._calculate_config_score(result, target_metrics)
                        
                        if score > best_score:
                            best_score = score
                            best_config = config
                            
                        print(f"⚡ Config score: {score:.3f} (threshold={threshold}, "
                              f"lr={lr}, max_clusters={max_clusters})")
                        
                    except Exception as e:
                        print(f"❌ Config failed: {e}")
                        continue
        
        print(f"🏆 Best configuration found: {best_config}")
        print(f"🎯 Score: {best_score:.3f}")
        
        return {
            'best_configuration': best_config,
            'score': best_score,
            'target_metrics': target_metrics
        }

# Usage example
def algorithm_optimization_example():
    client = NCSClient(
        base_url="https://api.yourdomain.com",
        api_key="your-api-key"
    )
    
    tuner = AlgorithmTuner(client)
    
    # Get optimized config for use case
    iot_config = tuner.optimize_for_use_case('streaming_iot')
    print("IoT optimized config:", iot_config)
    
    # Generate test dataset
    test_data = np.random.rand(5000, 4).tolist()
    
    # Benchmark different configurations
    configurations = [
        tuner.optimize_for_use_case('real_time_low_latency'),
        tuner.optimize_for_use_case('high_accuracy_research'),
        tuner.optimize_for_use_case('large_scale_production')
    ]
    
    benchmark_results = tuner.benchmark_configurations(test_data, configurations)
    print("\nBenchmark results:", benchmark_results['best_configuration'])
    
    # Adaptive tuning for specific targets
    target_metrics = {
        'min_quality': 0.85,
        'max_processing_time': 2.0,
        'min_throughput': 1000
    }
    
    adaptive_result = tuner.adaptive_tuning(test_data, target_metrics)
    print("\nAdaptive tuning result:", adaptive_result['best_configuration'])
```

## 🚀 Performance Tuning

Optimize for maximum throughput and minimum latency.

### Connection Pool Optimization

```python
import asyncio
import aiohttp
from aiohttp import ClientSession, ClientTimeout, TCPConnector

class HighPerformanceClient:
    def __init__(self, base_url: str, api_key: str, config: dict = None):
        self.base_url = base_url
        self.api_key = api_key
        self.config = {
            'max_connections': 100,
            'max_connections_per_host': 30,
            'connection_timeout': 30,
            'read_timeout': 60,
            'keepalive_timeout': 30,
            'enable_tcp_nodelay': True,
            **config or {}
        }
        self.session = None
        
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def start(self):
        """Initialize optimized HTTP session."""
        
        # Configure TCP connector for high performance
        connector = TCPConnector(
            limit=self.config['max_connections'],
            limit_per_host=self.config['max_connections_per_host'],
            ttl_dns_cache=300,
            use_dns_cache=True,
            tcp_nodelay=self.config['enable_tcp_nodelay'],
            keepalive_timeout=self.config['keepalive_timeout']
        )
        
        # Configure timeouts
        timeout = ClientTimeout(
            total=None,  # No total timeout
            connect=self.config['connection_timeout'],
            sock_read=self.config['read_timeout']
        )
        
        # Create session with optimization
        self.session = ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'Connection': 'keep-alive'
            }
        )
        
        print(f"🚀 High-performance client initialized")
        print(f"   Max connections: {self.config['max_connections']}")
        print(f"   Per-host limit: {self.config['max_connections_per_host']}")
    
    async def close(self):
        """Clean shutdown of HTTP session."""
        if self.session:
            await self.session.close()
            print("🔌 HTTP session closed")
    
    async def process_points_optimized(
        self, 
        points: List[List[float]],
        **kwargs
    ) -> dict:
        """Optimized point processing with connection reuse."""
        
        data = {'points': points, **kwargs}
        
        async with self.session.post(
            f"{self.base_url}/api/v1/process_points",
            json=data
        ) as response:
            
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"HTTP {response.status}: {error_text}")
    
    async def benchmark_throughput(
        self, 
        test_points: List[List[float]],
        duration_seconds: int = 60,
        concurrent_requests: int = 10
    ) -> dict:
        """Benchmark maximum throughput."""
        
        start_time = asyncio.get_event_loop().time()
        end_time = start_time + duration_seconds
        
        results = {
            'requests_completed': 0,
            'total_points_processed': 0,
            'errors': 0,
            'latencies': []
        }
        
        async def worker():
            """Individual worker for concurrent requests."""
            while asyncio.get_event_loop().time() < end_time:
                try:
                    request_start = asyncio.get_event_loop().time()
                    
                    await self.process_points_optimized(test_points)
                    
                    request_end = asyncio.get_event_loop().time()
                    latency = (request_end - request_start) * 1000  # ms
                    
                    results['requests_completed'] += 1
                    results['total_points_processed'] += len(test_points)
                    results['latencies'].append(latency)
                    
                except Exception as e:
                    results['errors'] += 1
                    print(f"❌ Request failed: {e}")
        
        # Start concurrent workers
        workers = [asyncio.create_task(worker()) for _ in range(concurrent_requests)]
        
        print(f"🏁 Starting throughput benchmark...")
        print(f"   Duration: {duration_seconds}s")
        print(f"   Concurrent requests: {concurrent_requests}")
        print(f"   Points per request: {len(test_points)}")
        
        # Wait for completion
        await asyncio.sleep(duration_seconds)
        
        # Cancel remaining workers
        for worker_task in workers:
            worker_task.cancel()
        
        # Calculate metrics
        actual_duration = asyncio.get_event_loop().time() - start_time
        throughput_rps = results['requests_completed'] / actual_duration
        throughput_pps = results['total_points_processed'] / actual_duration
        
        avg_latency = np.mean(results['latencies']) if results['latencies'] else 0
        p95_latency = np.percentile(results['latencies'], 95) if results['latencies'] else 0
        
        benchmark_results = {
            'duration_seconds': actual_duration,
            'requests_per_second': throughput_rps,
            'points_per_second': throughput_pps,
            'total_requests': results['requests_completed'],
            'total_points': results['total_points_processed'],
            'error_rate': results['errors'] / max(1, results['requests_completed']),
            'average_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency
        }
        
        print(f"\n📊 Benchmark Results:")
        print(f"   Throughput: {throughput_rps:.1f} req/s, {throughput_pps:.1f} points/s")
        print(f"   Latency: {avg_latency:.1f}ms avg, {p95_latency:.1f}ms p95")
        print(f"   Error rate: {benchmark_results['error_rate']:.3f}")
        
        return benchmark_results

# Usage example
async def performance_tuning_example():
    # Test with different configurations
    configs = [
        {'max_connections': 50, 'max_connections_per_host': 10},
        {'max_connections': 100, 'max_connections_per_host': 20},
        {'max_connections': 200, 'max_connections_per_host': 50}
    ]
    
    test_points = [[random.random() for _ in range(5)] for _ in range(100)]
    
    for i, config in enumerate(configs):
        print(f"\n🧪 Testing configuration {i + 1}")
        
        async with HighPerformanceClient(
            "https://api.yourdomain.com",
            "your-api-key",
            config
        ) as client:
            
            results = await client.benchmark_throughput(
                test_points=test_points,
                duration_seconds=30,
                concurrent_requests=config['max_connections_per_host']
            )
            
            print(f"Config {i + 1} results: {results['points_per_second']:.1f} points/s")
```

## 🏢 Production Integration

Enterprise-grade integration patterns and best practices.

### Service Integration Pattern

```python
import logging
import asyncio
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

class ServiceHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class ClusteringService:
    """Production-ready clustering service wrapper."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self.health_status = ServiceHealth.UNHEALTHY
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception
        )
        self.metrics_collector = MetricsCollector()
        self.logger = self._setup_logging()
        
    async def start(self):
        """Initialize service."""
        try:
            self.client = AsyncNCSClient(
                base_url=self.config['api_url'],
                api_key=self.config['api_key'],
                timeout=self.config.get('timeout', 30),
                max_retries=self.config.get('max_retries', 3)
            )
            
            # Health check
            await self._health_check()
            self.health_status = ServiceHealth.HEALTHY
            
            self.logger.info("Clustering service started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start clustering service: {e}")
            self.health_status = ServiceHealth.UNHEALTHY
            raise
    
    async def process_data(
        self, 
        data: List[List[float]], 
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Process data with full error handling and monitoring."""
        
        request_id = metadata.get('request_id') if metadata else None
        
        # Record request metrics
        self.metrics_collector.increment('requests_total')
        start_time = time.time()
        
        try:
            # Circuit breaker protection
            async with self.circuit_breaker:
                result = await self.client.process_points(data)
                
            # Record success metrics
            processing_time = time.time() - start_time
            self.metrics_collector.record('processing_time_seconds', processing_time)
            self.metrics_collector.record('points_processed_total', len(data))
            self.metrics_collector.increment('requests_successful')
            
            self.logger.info(
                "Data processing completed",
                extra={
                    'request_id': request_id,
                    'points_count': len(data),
                    'clusters_found': len(result.clusters),
                    'processing_time_ms': result.processing_time_ms
                }
            )
            
            return {
                'success': True,
                'result': result,
                'processing_time': processing_time,
                'request_id': request_id
            }
            
        except CircuitBreakerOpenException:
            self.metrics_collector.increment('requests_circuit_breaker')
            self.logger.warning("Circuit breaker open, request rejected")
            
            return {
                'success': False,
                'error': 'Service temporarily unavailable',
                'error_type': 'circuit_breaker',
                'request_id': request_id
            }
            
        except Exception as e:
            # Record error metrics
            processing_time = time.time() - start_time
            self.metrics_collector.increment('requests_failed')
            self.metrics_collector.record('error_processing_time_seconds', processing_time)
            
            self.logger.error(
                "Data processing failed",
                extra={
                    'request_id': request_id,
                    'error': str(e),
                    'error_type': type(e).__name__
                },
                exc_info=True
            )
            
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'request_id': request_id
            }
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get comprehensive service health status."""
        
        try:
            # Check API health
            api_health = await self.client.health_check()
            
            # Check circuit breaker status
            circuit_status = self.circuit_breaker.current_state
            
            # Get metrics
            metrics = self.metrics_collector.get_summary()
            
            overall_health = ServiceHealth.HEALTHY
            
            if circuit_status != 'closed':
                overall_health = ServiceHealth.DEGRADED
                
            if not api_health.algorithm_ready:
                overall_health = ServiceHealth.UNHEALTHY
            
            return {
                'status': overall_health.value,
                'api_health': {
                    'status': api_health.status,
                    'algorithm_ready': api_health.algorithm_ready,
                    'uptime_seconds': api_health.uptime_seconds
                },
                'circuit_breaker': {
                    'state': circuit_status,
                    'failure_count': self.circuit_breaker.failure_count
                },
                'metrics': metrics,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'status': ServiceHealth.UNHEALTHY.value,
                'error': str(e),
                'timestamp': time.time()
            }

# Circuit breaker implementation
class CircuitBreaker:
    def __init__(self, failure_threshold: int, recovery_timeout: int, expected_exception: type):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.current_state = 'closed'  # closed, open, half_open
        
    async def __aenter__(self):
        if self.current_state == 'open':
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise CircuitBreakerOpenException("Circuit breaker is open")
            else:
                self.current_state = 'half_open'
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success
            if self.current_state == 'half_open':
                self.current_state = 'closed'
                self.failure_count = 0
        elif issubclass(exc_type, self.expected_exception):
            # Failure
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.current_state = 'open'
                
        return False  # Don't suppress exceptions

class CircuitBreakerOpenException(Exception):
    pass

# Metrics collector
class MetricsCollector:
    def __init__(self):
        self.counters = {}
        self.histograms = {}
        
    def increment(self, metric_name: str, value: int = 1):
        self.counters[metric_name] = self.counters.get(metric_name, 0) + value
        
    def record(self, metric_name: str, value: float):
        if metric_name not in self.histograms:
            self.histograms[metric_name] = []
        self.histograms[metric_name].append(value)
        
    def get_summary(self) -> Dict[str, Any]:
        summary = {'counters': self.counters}
        
        for name, values in self.histograms.items():
            if values:
                summary[name] = {
                    'count': len(values),
                    'sum': sum(values),
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
                
        return summary
```

This completes the advanced usage guide with comprehensive examples covering streaming, batch processing, algorithm optimization, performance tuning, and production integration patterns. The guide provides practical, production-ready code examples that demonstrate enterprise-grade usage of the NCS API.

**Ready to proceed to the next folder or need any modifications to these guides?**