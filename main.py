"""
NCS API - High-Performance Clustering Algorithm
Main FastAPI application with real-time streaming capabilities
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
import os
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
import numpy as np
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Configuration
from src.config import get_settings
from src.clustering import NCSClusteringAlgorithm
from src.models import ClusterRequest, ClusterResponse, JobStatus
from src.websocket_manager import ConnectionManager
from src.monitoring import setup_logging, get_metrics_registry
from src.cache import get_cache_client

# Initialize settings
settings = get_settings()

# Setup logging
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Global variables
clustering_algorithm = None
connection_manager = ConnectionManager()
job_store: Dict[str, Dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting NCS API...")
    global clustering_algorithm
    clustering_algorithm = NCSClusteringAlgorithm()
    logger.info("NCS API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down NCS API...")
    await connection_manager.disconnect_all()
    logger.info("NCS API shutdown complete")

# FastAPI app initialization
app = FastAPI(
    title="NCS Clustering API",
    description="High-performance clustering and data processing API with real-time streaming",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Middleware
app.add_middleware(SlowAPIMiddleware)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.allowed_hosts
)

# Prometheus metrics
if settings.enable_metrics:
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app)

# API Key authentication (optional)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: Optional[str] = Depends(api_key_header)):
    if settings.require_api_key and not api_key:
        raise HTTPException(status_code=401, detail="API Key required")
    if settings.require_api_key and api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

# Pydantic models
class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    uptime_seconds: float

class ClusteringParams(BaseModel):
    n_clusters: Optional[int] = Field(None, ge=1, le=100)
    quality_threshold: float = Field(0.85, ge=0.0, le=1.0)
    max_iterations: int = Field(100, ge=1, le=1000)
    random_seed: Optional[int] = None

class ClusterRequest(BaseModel):
    data: List[List[float]] = Field(..., min_items=1, max_items=100000)
    algorithm: str = Field("ncs", regex="^(ncs|kmeans|dbscan)$")
    params: ClusteringParams = Field(default_factory=ClusteringParams)
    
    @validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError("Data cannot be empty")
        
        # Check dimensions are consistent
        if len(v) > 1:
            first_dim = len(v[0])
            for row in v[1:]:
                if len(row) != first_dim:
                    raise ValueError("All data points must have same dimensions")
        
        return v

class ClusterResponse(BaseModel):
    job_id: str
    status: str
    clusters: Optional[List[int]] = None
    centroids: Optional[List[List[float]]] = None
    quality_score: Optional[float] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float = Field(ge=0.0, le=1.0)
    result: Optional[ClusterResponse] = None
    created_at: datetime
    updated_at: datetime

# Global state
start_time = time.time()

# Health check endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        uptime_seconds=time.time() - start_time
    )

@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check for Kubernetes"""
    if clustering_algorithm is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}

# API version endpoint
@app.get("/api/v1/version", tags=["Info"])
async def get_version():
    """Get API version information"""
    return {
        "version": "1.0.0",
        "algorithm_version": clustering_algorithm.get_version() if clustering_algorithm else "unknown",
        "build_time": "2025-01-10T12:00:00Z"
    }

# Main clustering endpoint
@app.post("/api/v1/cluster/process", response_model=ClusterResponse, tags=["Clustering"])
@limiter.limit("100/minute")
async def process_clustering(
    request: ClusterRequest, 
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(get_api_key)
):
    """Process clustering request"""
    start_time = time.time()
    job_id = str(uuid.uuid4())
    
    logger.info(f"Processing clustering request {job_id} with {len(request.data)} points")
    
    try:
        # Create job entry
        job_store[job_id] = {
            "status": "processing",
            "progress": 0.0,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Convert data to numpy array
        data_array = np.array(request.data)
        
        # Process clustering
        if request.algorithm == "ncs":
            result = await clustering_algorithm.cluster_ncs(
                data_array, 
                **request.params.dict(exclude_none=True)
            )
        elif request.algorithm == "kmeans":
            result = await clustering_algorithm.cluster_kmeans(
                data_array, 
                **request.params.dict(exclude_none=True)
            )
        elif request.algorithm == "dbscan":
            result = await clustering_algorithm.cluster_dbscan(
                data_array, 
                **request.params.dict(exclude_none=True)
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported algorithm")
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update job status
        job_store[job_id].update({
            "status": "completed",
            "progress": 1.0,
            "updated_at": datetime.utcnow()
        })
        
        response = ClusterResponse(
            job_id=job_id,
            status="completed",
            clusters=result["clusters"].tolist() if result["clusters"] is not None else None,
            centroids=result["centroids"].tolist() if result["centroids"] is not None else None,
            quality_score=result["quality_score"],
            processing_time_ms=processing_time
        )
        
        # Store result
        job_store[job_id]["result"] = response
        
        # Notify WebSocket connections
        await connection_manager.broadcast({
            "type": "job_completed",
            "job_id": job_id,
            "result": response.dict()
        })
        
        logger.info(f"Completed clustering request {job_id} in {processing_time:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Error processing clustering request {job_id}: {str(e)}")
        
        # Update job status with error
        job_store[job_id].update({
            "status": "failed",
            "updated_at": datetime.utcnow(),
            "error": str(e)
        })
        
        raise HTTPException(status_code=500, detail=str(e))

# Job status endpoint
@app.get("/api/v1/cluster/status/{job_id}", response_model=JobStatusResponse, tags=["Clustering"])
async def get_job_status(job_id: str, api_key: Optional[str] = Depends(get_api_key)):
    """Get clustering job status"""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_store[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        result=job.get("result"),
        created_at=job["created_at"],
        updated_at=job["updated_at"]
    )

# List jobs endpoint
@app.get("/api/v1/cluster/jobs", tags=["Clustering"])
async def list_jobs(
    limit: int = Field(50, ge=1, le=1000),
    offset: int = Field(0, ge=0),
    api_key: Optional[str] = Depends(get_api_key)
):
    """List clustering jobs"""
    jobs = list(job_store.items())
    total = len(jobs)
    
    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x[1]["created_at"], reverse=True)
    
    # Apply pagination
    paginated_jobs = jobs[offset:offset + limit]
    
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "created_at": job["created_at"],
                "updated_at": job["updated_at"]
            }
            for job_id, job in paginated_jobs
        ],
        "total": total,
        "limit": limit,
        "offset": offset
    }

# WebSocket endpoint for real-time updates
@app.websocket("/ws/stream/cluster")
async def websocket_cluster_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time clustering updates"""
    await connection_manager.connect(websocket)
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket message: {data}")
            
            # Echo back for now (can be extended for real-time processing)
            await websocket.send_text(f"Echo: {data}")
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")

# Metrics endpoint (if not using Prometheus middleware)
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get application metrics"""
    return {
        "active_jobs": len([j for j in job_store.values() if j["status"] == "processing"]),
        "total_jobs": len(job_store),
        "completed_jobs": len([j for j in job_store.values() if j["status"] == "completed"]),
        "failed_jobs": len([j for j in job_store.values() if j["status"] == "failed"]),
        "active_websocket_connections": connection_manager.get_connection_count(),
        "uptime_seconds": time.time() - start_time
    }

# Algorithm info endpoint
@app.get("/api/v1/algorithms", tags=["Info"])
async def get_algorithms():
    """Get available clustering algorithms"""
    return {
        "algorithms": [
            {
                "name": "ncs",
                "description": "Neural Clustering System - High-performance proprietary algorithm",
                "parameters": {
                    "n_clusters": "Number of clusters (optional)",
                    "quality_threshold": "Minimum quality threshold (0.0-1.0)",
                    "max_iterations": "Maximum iterations (1-1000)",
                    "random_seed": "Random seed for reproducibility"
                }
            },
            {
                "name": "kmeans",
                "description": "K-Means clustering algorithm",
                "parameters": {
                    "n_clusters": "Number of clusters (required)",
                    "max_iterations": "Maximum iterations (1-1000)",
                    "random_seed": "Random seed for reproducibility"
                }
            },
            {
                "name": "dbscan",
                "description": "DBSCAN density-based clustering",
                "parameters": {
                    "eps": "Maximum distance between points",
                    "min_samples": "Minimum samples in neighborhood"
                }
            }
        ]
    }

# Exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )