"""
CRUD operations for NeuroCluster Streamer API database models.

This module provides data access layer operations for all database models,
including specialized queries for clustering analysis and performance monitoring.
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple, Union
from decimal import Decimal

from sqlalchemy import and_, or_, func, desc, asc, text, select
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.dialects.postgresql import insert

from .models import (
    DataPointRecord, ClusterRecord, ProcessingSession, PerformanceMetric,
    AuditLog, UserActivity, SystemConfiguration
)
from app.exceptions import DatabaseException, ValidationException

logger = logging.getLogger(__name__)

class BaseCRUD:
    """Base CRUD operations for all models."""
    
    def __init__(self, model):
        self.model = model
        
    def create(self, db: Session, *, obj_in: Dict[str, Any]) -> Any:
        """Create a new record."""
        try:
            db_obj = self.model(**obj_in)
            db.add(db_obj)
            db.flush()
            db.refresh(db_obj)
            return db_obj
        except IntegrityError as e:
            db.rollback()
            raise ValidationException(f"Integrity constraint violation: {e}")
        except SQLAlchemyError as e:
            db.rollback()
            raise DatabaseException(f"Database error creating {self.model.__name__}: {e}")
            
    def get(self, db: Session, id: Union[str, uuid.UUID]) -> Optional[Any]:
        """Get record by ID."""
        try:
            return db.query(self.model).filter(self.model.id == id).first()
        except SQLAlchemyError as e:
            raise DatabaseException(f"Database error retrieving {self.model.__name__}: {e}")
            
    def get_multi(
        self, 
        db: Session, 
        *, 
        skip: int = 0, 
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """Get multiple records with optional filtering."""
        try:
            query = db.query(self.model)
            
            if filters:
                for key, value in filters.items():
                    if hasattr(self.model, key):
                        query = query.filter(getattr(self.model, key) == value)
                        
            return query.offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            raise DatabaseException(f"Database error retrieving {self.model.__name__} records: {e}")
            
    def update(
        self, 
        db: Session, 
        *, 
        db_obj: Any, 
        obj_in: Dict[str, Any]
    ) -> Any:
        """Update an existing record."""
        try:
            for field, value in obj_in.items():
                if hasattr(db_obj, field):
                    setattr(db_obj, field, value)
                    
            db.flush()
            db.refresh(db_obj)
            return db_obj
        except SQLAlchemyError as e:
            db.rollback()
            raise DatabaseException(f"Database error updating {self.model.__name__}: {e}")
            
    def delete(self, db: Session, *, id: Union[str, uuid.UUID]) -> bool:
        """Delete a record by ID."""
        try:
            obj = db.query(self.model).filter(self.model.id == id).first()
            if obj:
                db.delete(obj)
                db.flush()
                return True
            return False
        except SQLAlchemyError as e:
            db.rollback()
            raise DatabaseException(f"Database error deleting {self.model.__name__}: {e}")

class DataPointCRUD(BaseCRUD):
    """CRUD operations for DataPointRecord."""
    
    def __init__(self):
        super().__init__(DataPointRecord)
        
    def create_batch(self, db: Session, *, data_points: List[Dict[str, Any]]) -> List[DataPointRecord]:
        """Create multiple data points in a single transaction."""
        try:
            db_objects = []
            for point_data in data_points:
                db_obj = DataPointRecord(**point_data)
                db_objects.append(db_obj)
                
            db.add_all(db_objects)
            db.flush()
            
            for obj in db_objects:
                db.refresh(obj)
                
            return db_objects
        except SQLAlchemyError as e:
            db.rollback()
            raise DatabaseException(f"Database error creating batch data points: {e}")
            
    def get_by_session(
        self, 
        db: Session, 
        session_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 1000,
        include_outliers: bool = True
    ) -> List[DataPointRecord]:
        """Get data points for a specific session."""
        try:
            query = db.query(DataPointRecord).filter(
                DataPointRecord.session_id == session_id
            )
            
            if not include_outliers:
                query = query.filter(DataPointRecord.is_outlier == False)
                
            return query.order_by(DataPointRecord.processing_order)\
                       .offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            raise DatabaseException(f"Database error retrieving session data points: {e}")
            
    def get_by_cluster(
        self, 
        db: Session, 
        cluster_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 1000
    ) -> List[DataPointRecord]:
        """Get all data points assigned to a specific cluster."""
        try:
            return db.query(DataPointRecord)\
                    .filter(DataPointRecord.cluster_id == cluster_id)\
                    .order_by(DataPointRecord.processing_order)\
                    .offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            raise DatabaseException(f"Database error retrieving cluster data points: {e}")
            
    def get_outliers(
        self,
        db: Session,
        session_id: Optional[uuid.UUID] = None,
        *,
        min_score: float = 0.5,
        skip: int = 0,
        limit: int = 1000
    ) -> List[DataPointRecord]:
        """Get outlier data points with optional filtering."""
        try:
            query = db.query(DataPointRecord).filter(
                and_(
                    DataPointRecord.is_outlier == True,
                    DataPointRecord.outlier_score >= min_score
                )
            )
            
            if session_id:
                query = query.filter(DataPointRecord.session_id == session_id)
                
            return query.order_by(desc(DataPointRecord.outlier_score))\
                       .offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            raise DatabaseException(f"Database error retrieving outliers: {e}")
            
    def get_statistics(self, db: Session, session_id: uuid.UUID) -> Dict[str, Any]:
        """Get statistical summary for data points in a session."""
        try:
            stats = db.query(
                func.count(DataPointRecord.id).label('total_points'),
                func.count().filter(DataPointRecord.is_outlier == True).label('outlier_count'),
                func.count().filter(DataPointRecord.cluster_id.isnot(None)).label('clustered_count'),
                func.avg(DataPointRecord.confidence_score).label('avg_confidence'),
                func.avg(DataPointRecord.processing_time_ms).label('avg_processing_time'),
                func.max(DataPointRecord.processing_time_ms).label('max_processing_time'),
                func.min(DataPointRecord.processing_time_ms).label('min_processing_time')
            ).filter(DataPointRecord.session_id == session_id).first()
            
            return {
                'total_points': stats.total_points or 0,
                'outlier_count': stats.outlier_count or 0,
                'clustered_count': stats.clustered_count or 0,
                'outlier_percentage': (stats.outlier_count / stats.total_points * 100) if stats.total_points else 0,
                'clustering_percentage': (stats.clustered_count / stats.total_points * 100) if stats.total_points else 0,
                'avg_confidence': float(stats.avg_confidence) if stats.avg_confidence else 0,
                'avg_processing_time_ms': float(stats.avg_processing_time) if stats.avg_processing_time else 0,
                'max_processing_time_ms': float(stats.max_processing_time) if stats.max_processing_time else 0,
                'min_processing_time_ms': float(stats.min_processing_time) if stats.min_processing_time else 0
            }
        except SQLAlchemyError as e:
            raise DatabaseException(f"Database error calculating statistics: {e}")

class ClusterCRUD(BaseCRUD):
    """CRUD operations for ClusterRecord."""
    
    def __init__(self):
        super().__init__(ClusterRecord)
        
    def get_active_clusters(
        self, 
        db: Session, 
        session_id: uuid.UUID,
        *,
        health_status: Optional[str] = None
    ) -> List[ClusterRecord]:
        """Get active clusters for a session."""
        try:
            query = db.query(ClusterRecord).filter(
                and_(
                    ClusterRecord.session_id == session_id,
                    ClusterRecord.is_active == True
                )
            )
            
            if health_status:
                query = query.filter(ClusterRecord.health_status == health_status)
                
            return query.order_by(desc(ClusterRecord.point_count)).all()
        except SQLAlchemyError as e:
            raise DatabaseException(f"Database error retrieving active clusters: {e}")
            
    def get_cluster_evolution(
        self, 
        db: Session, 
        cluster_id: uuid.UUID
    ) -> List[ClusterRecord]:
        """Get cluster evolution history (parent-child relationships)."""
        try:
            # Get the cluster and its ancestry
            cluster = db.query(ClusterRecord).filter(ClusterRecord.id == cluster_id).first()
            if not cluster:
                return []
                
            # Build the family tree
            family_tree = []
            
            # Get ancestors
            current = cluster
            while current.parent_cluster_id:
                parent = db.query(ClusterRecord).filter(
                    ClusterRecord.id == current.parent_cluster_id
                ).first()
                if parent:
                    family_tree.insert(0, parent)
                    current = parent
                else:
                    break
                    
            # Add the cluster itself
            family_tree.append(cluster)
            
            # Get descendants (simplified - direct children only)
            children = db.query(ClusterRecord).filter(
                ClusterRecord.parent_cluster_id == cluster_id
            ).all()
            family_tree.extend(children)
            
            return family_tree
        except SQLAlchemyError as e:
            raise DatabaseException(f"Database error retrieving cluster evolution: {e}")
            
    def get_merge_candidates(
        self, 
        db: Session, 
        session_id: uuid.UUID,
        *,
        max_distance: float = 0.5
    ) -> List[Tuple[ClusterRecord, ClusterRecord]]:
        """Find clusters that are candidates for merging."""
        try:
            # This is a simplified implementation
            # In practice, you'd use more sophisticated distance calculations
            candidates = db.query(ClusterRecord).filter(
                and_(
                    ClusterRecord.session_id == session_id,
                    ClusterRecord.is_active == True,
                    ClusterRecord.merge_candidate == True
                )
            ).all()
            
            merge_pairs = []
            for i, cluster1 in enumerate(candidates):
                for cluster2 in candidates[i+1:]:
                    # Simple centroid distance check
                    # In practice, use proper distance calculation
                    if cluster1.separation and cluster2.separation:
                        avg_separation = (cluster1.separation + cluster2.separation) / 2
                        if avg_separation < max_distance:
                            merge_pairs.append((cluster1, cluster2))
                            
            return merge_pairs
        except SQLAlchemyError as e:
            raise DatabaseException(f"Database error finding merge candidates: {e}")
            
    def update_cluster_stats(
        self, 
        db: Session, 
        cluster_id: uuid.UUID
    ) -> ClusterRecord:
        """Update cluster statistics based on its data points."""
        try:
            cluster = db.query(ClusterRecord).filter(ClusterRecord.id == cluster_id).first()
            if not cluster:
                raise ValidationException(f"Cluster {cluster_id} not found")
                
            # Calculate updated statistics
            stats = db.query(
                func.count(DataPointRecord.id).label('point_count'),
                func.avg(DataPointRecord.confidence_score).label('avg_confidence'),
                func.avg(DataPointRecord.silhouette_score).label('avg_silhouette'),
                func.avg(DataPointRecord.stability_score).label('avg_stability')
            ).filter(DataPointRecord.cluster_id == cluster_id).first()
            
            # Update cluster properties
            cluster.point_count = stats.point_count or 0
            cluster.silhouette_avg = float(stats.avg_silhouette) if stats.avg_silhouette else None
            cluster.stability_score = float(stats.avg_stability) if stats.avg_stability else None
            cluster.last_updated = datetime.utcnow()
            
            # Update health status based on statistics
            if cluster.point_count < cluster.min_points:
                cluster.health_status = 'critical'
            elif cluster.silhouette_avg and cluster.silhouette_avg < 0.3:
                cluster.health_status = 'warning'
            else:
                cluster.health_status = 'healthy'
                
            db.flush()
            db.refresh(cluster)
            return cluster
        except SQLAlchemyError as e:
            db.rollback()
            raise DatabaseException(f"Database error updating cluster stats: {e}")

class SessionCRUD(BaseCRUD):
    """CRUD operations for ProcessingSession."""
    
    def __init__(self):
        super().__init__(ProcessingSession)
        
    def get_by_user(
        self, 
        db: Session, 
        user_id: str,
        *,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[ProcessingSession]:
        """Get sessions for a specific user."""
        try:
            query = db.query(ProcessingSession).filter(ProcessingSession.user_id == user_id)
            
            if status:
                query = query.filter(ProcessingSession.status == status)
                
            return query.order_by(desc(ProcessingSession.start_time))\
                       .offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            raise DatabaseException(f"Database error retrieving user sessions: {e}")
            
    def get_active_sessions(self, db: Session) -> List[ProcessingSession]:
        """Get all currently active sessions."""
        try:
            return db.query(ProcessingSession)\
                    .filter(ProcessingSession.status == 'active')\
                    .order_by(ProcessingSession.start_time).all()
        except SQLAlchemyError as e:
            raise DatabaseException(f"Database error retrieving active sessions: {e}")
            
    def complete_session(
        self, 
        db: Session, 
        session_id: uuid.UUID,
        *,
        status: str = 'completed',
        final_stats: Optional[Dict[str, Any]] = None
    ) -> ProcessingSession:
        """Mark a session as completed and update final statistics."""
        try:
            session = db.query(ProcessingSession).filter(
                ProcessingSession.id == session_id
            ).first()
            
            if not session:
                raise ValidationException(f"Session {session_id} not found")
                
            session.status = status
            session.end_time = datetime.utcnow()
            
            if session.start_time:
                session.total_duration_seconds = (
                    session.end_time - session.start_time
                ).total_seconds()
                
            if final_stats:
                for key, value in final_stats.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                        
            db.flush()
            db.refresh(session)
            return session
        except SQLAlchemyError as e:
            db.rollback()
            raise DatabaseException(f"Database error completing session: {e}")
            
    def get_session_summary(self, db: Session, session_id: uuid.UUID) -> Dict[str, Any]:
        """Get comprehensive session summary with related data."""
        try:
            session = db.query(ProcessingSession)\
                       .options(
                           selectinload(ProcessingSession.data_points),
                           selectinload(ProcessingSession.clusters),
                           selectinload(ProcessingSession.performance_metrics)
                       )\
                       .filter(ProcessingSession.id == session_id).first()
                       
            if not session:
                raise ValidationException(f"Session {session_id} not found")
                
            # Build comprehensive summary
            summary = {
                'session': {
                    'id': str(session.id),
                    'name': session.session_name,
                    'status': session.status,
                    'start_time': session.start_time.isoformat() if session.start_time else None,
                    'end_time': session.end_time.isoformat() if session.end_time else None,
                    'duration_seconds': session.total_duration_seconds,
                    'user_id': session.user_id
                },
                'data_summary': {
                    'total_points': session.total_points_processed,
                    'clustered_points': session.total_points_clustered,
                    'outliers': session.total_outliers_detected,
                    'clusters_created': session.unique_clusters_created,
                    'noise_ratio': session.noise_ratio
                },
                'performance': {
                    'avg_processing_time_ms': session.avg_processing_time_ms,
                    'throughput_points_per_sec': session.throughput_points_per_sec,
                    'peak_memory_mb': session.peak_memory_usage_mb,
                    'avg_cpu_percent': session.avg_cpu_usage_percent
                },
                'quality': {
                    'silhouette_score': session.overall_silhouette_score,
                    'cluster_purity': session.cluster_purity
                },
                'errors': {
                    'error_count': session.error_count,
                    'warning_count': session.warning_count,
                    'last_error': session.last_error_message
                }
            }
            
            return summary
        except SQLAlchemyError as e:
            raise DatabaseException(f"Database error retrieving session summary: {e}")

class MetricsCRUD(BaseCRUD):
    """CRUD operations for PerformanceMetric."""
    
    def __init__(self):
        super().__init__(PerformanceMetric)
        
    def record_metric(
        self,
        db: Session,
        session_id: uuid.UUID,
        metric_name: str,
        metric_category: str,
        value: Union[float, str, Dict[str, Any]],
        **kwargs
    ) -> PerformanceMetric:
        """Record a performance metric."""
        try:
            metric_data = {
                'session_id': session_id,
                'metric_name': metric_name,
                'metric_category': metric_category,
                **kwargs
            }
            
            # Set appropriate value field based on type
            if isinstance(value, (int, float, Decimal)):
                metric_data['numeric_value'] = Decimal(str(value))
            elif isinstance(value, str):
                metric_data['string_value'] = value
            elif isinstance(value, dict):
                metric_data['json_value'] = value
            else:
                metric_data['string_value'] = str(value)
                
            return self.create(db, obj_in=metric_data)
        except Exception as e:
            raise DatabaseException(f"Error recording metric: {e}")
            
    def get_metrics_by_category(
        self,
        db: Session,
        session_id: uuid.UUID,
        category: str,
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[PerformanceMetric]:
        """Get metrics by category with optional time filtering."""
        try:
            query = db.query(PerformanceMetric).filter(
                and_(
                    PerformanceMetric.session_id == session_id,
                    PerformanceMetric.metric_category == category
                )
            )
            
            if start_time:
                query = query.filter(PerformanceMetric.measurement_timestamp >= start_time)
            if end_time:
                query = query.filter(PerformanceMetric.measurement_timestamp <= end_time)
                
            return query.order_by(PerformanceMetric.measurement_timestamp).all()
        except SQLAlchemyError as e:
            raise DatabaseException(f"Database error retrieving metrics: {e}")
            
    def get_metric_trends(
        self,
        db: Session,
        session_id: uuid.UUID,
        metric_name: str,
        *,
        window_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """Get metric trends over time windows."""
        try:
            start_time = datetime.utcnow() - timedelta(minutes=window_minutes)
            
            metrics = db.query(PerformanceMetric).filter(
                and_(
                    PerformanceMetric.session_id == session_id,
                    PerformanceMetric.metric_name == metric_name,
                    PerformanceMetric.measurement_timestamp >= start_time,
                    PerformanceMetric.numeric_value.isnot(None)
                )
            ).order_by(PerformanceMetric.measurement_timestamp).all()
            
            trends = []
            for metric in metrics:
                trends.append({
                    'timestamp': metric.measurement_timestamp.isoformat(),
                    'value': float(metric.numeric_value),
                    'window_seconds': metric.measurement_window_seconds,
                    'sample_size': metric.sample_size
                })
                
            return trends
        except SQLAlchemyError as e:
            raise DatabaseException(f"Database error retrieving metric trends: {e}")

class AuditCRUD(BaseCRUD):
    """CRUD operations for AuditLog."""
    
    def __init__(self):
        super().__init__(AuditLog)
        
    def log_event(
        self,
        db: Session,
        event_type: str,
        event_category: str,
        action: str,
        description: str,
        *,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ) -> AuditLog:
        """Log an audit event."""
        try:
            log_data = {
                'event_type': event_type,
                'event_category': event_category,
                'action': action,
                'description': description,
                'user_id': user_id,
                'resource_type': resource_type,
                'resource_id': resource_id,
                **kwargs
            }
            
            return self.create(db, obj_in=log_data)
        except Exception as e:
            raise DatabaseException(f"Error logging audit event: {e}")
            
    def get_security_events(
        self,
        db: Session,
        *,
        security_level: str = 'warning',
        hours: int = 24,
        skip: int = 0,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get security-related audit events."""
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            query = db.query(AuditLog).filter(
                and_(
                    AuditLog.security_level.in_(['warning', 'critical']),
                    AuditLog.created_at >= start_time
                )
            )
            
            if security_level == 'critical':
                query = query.filter(AuditLog.security_level == 'critical')
                
            return query.order_by(desc(AuditLog.created_at))\
                       .offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            raise DatabaseException(f"Database error retrieving security events: {e}")

class UserCRUD(BaseCRUD):
    """CRUD operations for UserActivity."""
    
    def __init__(self):
        super().__init__(UserActivity)
        
    def record_activity(
        self,
        db: Session,
        user_id: str,
        activity_type: str,
        category: str,
        *,
        activity_name: Optional[str] = None,
        **kwargs
    ) -> UserActivity:
        """Record user activity."""
        try:
            activity_data = {
                'user_id': user_id,
                'activity_type': activity_type,
                'category': category,
                'activity_name': activity_name,
                **kwargs
            }
            
            return self.create(db, obj_in=activity_data)
        except Exception as e:
            raise DatabaseException(f"Error recording user activity: {e}")
            
    def get_user_analytics(
        self,
        db: Session,
        user_id: str,
        *,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get user activity analytics."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            stats = db.query(
                func.count(UserActivity.id).label('total_activities'),
                func.count(func.distinct(UserActivity.activity_type)).label('unique_activities'),
                func.avg(UserActivity.duration_seconds).label('avg_duration'),
                func.sum(UserActivity.data_processed_mb).label('total_data_mb'),
                func.avg(UserActivity.success_rate).label('avg_success_rate')
            ).filter(
                and_(
                    UserActivity.user_id == user_id,
                    UserActivity.created_at >= start_date
                )
            ).first()
            
            return {
                'period_days': days,
                'total_activities': stats.total_activities or 0,
                'unique_activity_types': stats.unique_activities or 0,
                'avg_session_duration_seconds': float(stats.avg_duration) if stats.avg_duration else 0,
                'total_data_processed_mb': float(stats.total_data_mb) if stats.total_data_mb else 0,
                'avg_success_rate': float(stats.avg_success_rate) if stats.avg_success_rate else 0
            }
        except SQLAlchemyError as e:
            raise DatabaseException(f"Database error retrieving user analytics: {e}")

class ConfigCRUD(BaseCRUD):
    """CRUD operations for SystemConfiguration."""
    
    def __init__(self):
        super().__init__(SystemConfiguration)
        
    def get_config(
        self,
        db: Session,
        config_key: str,
        environment: str = 'production'
    ) -> Optional[SystemConfiguration]:
        """Get configuration by key and environment."""
        try:
            return db.query(SystemConfiguration).filter(
                and_(
                    SystemConfiguration.config_key == config_key,
                    SystemConfiguration.environment == environment,
                    SystemConfiguration.is_active == True
                )
            ).first()
        except SQLAlchemyError as e:
            raise DatabaseException(f"Database error retrieving configuration: {e}")
            
    def set_config(
        self,
        db: Session,
        config_key: str,
        config_value: Any,
        *,
        config_category: str,
        data_type: str,
        environment: str = 'production',
        changed_by: Optional[str] = None,
        **kwargs
    ) -> SystemConfiguration:
        """Set or update configuration value."""
        try:
            # Get existing config if it exists
            existing = self.get_config(db, config_key, environment)
            
            if existing:
                # Update existing configuration
                config_data = {
                    'config_value': config_value,
                    'version': existing.version + 1,
                    'changed_by': changed_by,
                    'previous_value': existing.config_value,
                    **kwargs
                }
                return self.update(db, db_obj=existing, obj_in=config_data)
            else:
                # Create new configuration
                config_data = {
                    'config_key': config_key,
                    'config_value': config_value,
                    'config_category': config_category,
                    'data_type': data_type,
                    'environment': environment,
                    'changed_by': changed_by,
                    **kwargs
                }
                return self.create(db, obj_in=config_data)
        except Exception as e:
            raise DatabaseException(f"Error setting configuration: {e}")
            
    def get_category_configs(
        self,
        db: Session,
        category: str,
        environment: str = 'production'
    ) -> List[SystemConfiguration]:
        """Get all configurations in a category."""
        try:
            return db.query(SystemConfiguration).filter(
                and_(
                    SystemConfiguration.config_category == category,
                    SystemConfiguration.environment == environment,
                    SystemConfiguration.is_active == True
                )
            ).order_by(SystemConfiguration.config_key).all()
        except SQLAlchemyError as e:
            raise DatabaseException(f"Database error retrieving category configs: {e}")

# Initialize CRUD instances
data_point_crud = DataPointCRUD()
cluster_crud = ClusterCRUD()
session_crud = SessionCRUD()
metrics_crud = MetricsCRUD()
audit_crud = AuditCRUD()
user_crud = UserCRUD()
config_crud = ConfigCRUD()