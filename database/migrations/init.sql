-- NeuroCluster Streamer API Database Schema
-- Initial migration script for PostgreSQL
-- Version: 1.0.0
-- Created: 2025-06-06

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text similarity searches

-- Create custom types
DO $$ BEGIN
    CREATE TYPE processing_status AS ENUM ('active', 'completed', 'failed', 'cancelled');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE cluster_health_status AS ENUM ('healthy', 'warning', 'critical', 'inactive');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE security_level AS ENUM ('info', 'warning', 'critical');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE completion_status AS ENUM ('completed', 'failed', 'cancelled', 'timeout');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE environment_type AS ENUM ('development', 'staging', 'production');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create processing_sessions table
CREATE TABLE IF NOT EXISTS processing_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_name VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    
    -- Session configuration
    algorithm_config JSONB NOT NULL,
    input_source VARCHAR(500),
    session_type VARCHAR(50) DEFAULT 'batch' NOT NULL,
    
    -- Status and lifecycle
    status processing_status DEFAULT 'active' NOT NULL,
    start_time TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    end_time TIMESTAMPTZ,
    total_duration_seconds DECIMAL(12, 3),
    
    -- Data statistics
    total_points_processed BIGINT DEFAULT 0 NOT NULL,
    total_points_clustered BIGINT DEFAULT 0 NOT NULL,
    total_outliers_detected BIGINT DEFAULT 0 NOT NULL,
    unique_clusters_created INTEGER DEFAULT 0 NOT NULL,
    
    -- Performance metrics
    avg_processing_time_ms DECIMAL(10, 3),
    max_processing_time_ms DECIMAL(10, 3),
    min_processing_time_ms DECIMAL(10, 3),
    throughput_points_per_sec DECIMAL(10, 3),
    
    -- Quality metrics
    overall_silhouette_score DECIMAL(5, 4),
    cluster_purity DECIMAL(5, 4),
    noise_ratio DECIMAL(5, 4),
    
    -- Resource utilization
    peak_memory_usage_mb DECIMAL(10, 2),
    avg_cpu_usage_percent DECIMAL(5, 2),
    disk_io_mb DECIMAL(10, 2),
    
    -- Error handling
    error_count INTEGER DEFAULT 0 NOT NULL,
    warning_count INTEGER DEFAULT 0 NOT NULL,
    last_error_message TEXT,
    last_error_time TIMESTAMPTZ,
    
    -- Configuration tracking
    algorithm_version VARCHAR(50) NOT NULL,
    api_version VARCHAR(20) NOT NULL,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CONSTRAINT ck_sessions_non_negative_processed CHECK (total_points_processed >= 0),
    CONSTRAINT ck_sessions_non_negative_throughput CHECK (throughput_points_per_sec >= 0),
    CONSTRAINT ck_sessions_valid_duration CHECK (total_duration_seconds >= 0)
);

-- Create clusters table
CREATE TABLE IF NOT EXISTS clusters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES processing_sessions(id) ON DELETE CASCADE,
    cluster_label VARCHAR(100) NOT NULL,
    
    -- Cluster properties
    centroid JSONB NOT NULL,
    dimensionality SMALLINT NOT NULL,
    radius DECIMAL(15, 6) NOT NULL,
    density DECIMAL(10, 6),
    
    -- Statistical properties
    point_count INTEGER DEFAULT 0 NOT NULL,
    min_points INTEGER NOT NULL,
    max_points INTEGER,
    
    -- Quality metrics
    cohesion DECIMAL(8, 6),
    separation DECIMAL(8, 6),
    silhouette_avg DECIMAL(5, 4),
    stability_score DECIMAL(5, 4),
    
    -- Health and lifecycle
    health_status cluster_health_status DEFAULT 'healthy' NOT NULL,
    is_active BOOLEAN DEFAULT TRUE NOT NULL,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    merge_candidate BOOLEAN DEFAULT FALSE,
    split_candidate BOOLEAN DEFAULT FALSE,
    
    -- Evolution tracking
    parent_cluster_id UUID REFERENCES clusters(id),
    generation SMALLINT DEFAULT 0 NOT NULL,
    split_count SMALLINT DEFAULT 0,
    merge_count SMALLINT DEFAULT 0,
    
    -- Performance metrics
    last_access_time TIMESTAMPTZ,
    access_frequency INTEGER DEFAULT 0,
    update_frequency INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CONSTRAINT ck_clusters_non_negative_points CHECK (point_count >= 0),
    CONSTRAINT ck_clusters_positive_radius CHECK (radius > 0),
    CONSTRAINT ck_clusters_non_negative_generation CHECK (generation >= 0),
    CONSTRAINT ck_clusters_positive_dimensionality CHECK (dimensionality > 0)
);

-- Create data_points table
CREATE TABLE IF NOT EXISTS data_points (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES processing_sessions(id) ON DELETE CASCADE,
    point_id VARCHAR(255) NOT NULL,
    
    -- Data point features and metadata
    features JSONB NOT NULL,
    normalized_features JSONB,
    dimensionality SMALLINT NOT NULL,
    
    -- Clustering results
    cluster_id UUID REFERENCES clusters(id) ON DELETE SET NULL,
    is_outlier BOOLEAN DEFAULT FALSE NOT NULL,
    outlier_score DECIMAL(5, 4),
    confidence_score DECIMAL(5, 4),
    
    -- Processing metadata
    processing_order BIGINT NOT NULL,
    processing_time_ms DECIMAL(8, 3),
    algorithm_version VARCHAR(50) NOT NULL,
    
    -- Similarity metrics
    nearest_cluster_distance DECIMAL(15, 6),
    second_nearest_distance DECIMAL(15, 6),
    silhouette_score DECIMAL(5, 4),
    
    -- Quality metrics
    stability_score DECIMAL(5, 4),
    novelty_score DECIMAL(5, 4),
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CONSTRAINT ck_data_points_positive_dimensionality CHECK (dimensionality > 0),
    CONSTRAINT ck_data_points_outlier_score_range CHECK (outlier_score >= 0 AND outlier_score <= 1),
    CONSTRAINT ck_data_points_confidence_range CHECK (confidence_score >= 0 AND confidence_score <= 1),
    CONSTRAINT uq_session_point UNIQUE (session_id, point_id)
);

-- Create performance_metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES processing_sessions(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_category VARCHAR(50) NOT NULL,
    
    -- Metric values
    numeric_value DECIMAL(20, 6),
    string_value VARCHAR(500),
    json_value JSONB,
    
    -- Measurement context
    measurement_timestamp TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    measurement_window_seconds DECIMAL(10, 3),
    sample_size INTEGER,
    
    -- Statistical properties
    min_value DECIMAL(20, 6),
    max_value DECIMAL(20, 6),
    avg_value DECIMAL(20, 6),
    std_dev DECIMAL(20, 6),
    percentile_95 DECIMAL(20, 6),
    percentile_99 DECIMAL(20, 6),
    
    -- Metadata
    tags JSONB,
    notes TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CONSTRAINT ck_metrics_positive_sample_size CHECK (sample_size > 0),
    CONSTRAINT ck_metrics_positive_window CHECK (measurement_window_seconds > 0)
);

-- Create audit_logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Event identification
    event_type VARCHAR(100) NOT NULL,
    event_category VARCHAR(50) NOT NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    
    -- Actor information
    user_id VARCHAR(255),
    user_email VARCHAR(320),
    user_role VARCHAR(100),
    api_key_id VARCHAR(255),
    client_ip INET,
    user_agent VARCHAR(1000),
    
    -- Request context
    request_id VARCHAR(255),
    session_id VARCHAR(255),
    endpoint VARCHAR(500),
    http_method VARCHAR(10),
    request_size_bytes BIGINT,
    response_size_bytes BIGINT,
    response_status SMALLINT,
    processing_time_ms DECIMAL(10, 3),
    
    -- Event details
    description TEXT NOT NULL,
    old_values JSONB,
    new_values JSONB,
    metadata JSONB,
    
    -- Security context
    security_level security_level DEFAULT 'info' NOT NULL,
    risk_score SMALLINT,
    threat_indicators TEXT[],
    
    -- Compliance and retention
    retention_policy VARCHAR(50) DEFAULT 'standard',
    compliance_tags TEXT[],
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CONSTRAINT ck_audit_valid_risk_score CHECK (risk_score >= 0 AND risk_score <= 100)
);

-- Create user_activities table
CREATE TABLE IF NOT EXISTS user_activities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    
    -- Activity details
    activity_type VARCHAR(100) NOT NULL,
    activity_name VARCHAR(200),
    category VARCHAR(50) NOT NULL,
    
    -- Context and metadata
    session_token VARCHAR(255),
    client_info JSONB,
    location_info JSONB,
    
    -- Performance and usage metrics
    duration_seconds DECIMAL(10, 3),
    data_processed_mb DECIMAL(10, 3),
    api_calls_count INTEGER DEFAULT 0,
    success_rate DECIMAL(5, 4),
    
    -- Resource utilization
    cpu_time_ms DECIMAL(10, 3),
    memory_peak_mb DECIMAL(10, 2),
    network_io_mb DECIMAL(10, 3),
    
    -- User preferences and behavior
    preferred_settings JSONB,
    feature_usage JSONB,
    interaction_path TEXT[],
    
    -- Outcome and feedback
    completion_status completion_status DEFAULT 'completed',
    error_message TEXT,
    user_satisfaction_score SMALLINT,
    feedback_text TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CONSTRAINT ck_user_activities_valid_satisfaction CHECK (user_satisfaction_score >= 1 AND user_satisfaction_score <= 5)
);

-- Create system_configurations table
CREATE TABLE IF NOT EXISTS system_configurations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(200) NOT NULL,
    config_category VARCHAR(100) NOT NULL,
    
    -- Configuration values
    config_value JSONB NOT NULL,
    default_value JSONB,
    data_type VARCHAR(50) NOT NULL,
    
    -- Validation and constraints
    validation_schema JSONB,
    min_value DECIMAL(20, 6),
    max_value DECIMAL(20, 6),
    allowed_values TEXT[],
    
    -- Metadata and documentation
    description TEXT,
    documentation_url VARCHAR(1000),
    example_value JSONB,
    
    -- Change management
    version INTEGER DEFAULT 1 NOT NULL,
    changed_by VARCHAR(255),
    change_reason TEXT,
    previous_value JSONB,
    
    -- Status and lifecycle
    is_active BOOLEAN DEFAULT TRUE NOT NULL,
    is_sensitive BOOLEAN DEFAULT FALSE NOT NULL,
    requires_restart BOOLEAN DEFAULT FALSE NOT NULL,
    environment environment_type DEFAULT 'production',
    
    -- Impact and dependencies
    affects_components TEXT[],
    dependencies TEXT[],
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    
    -- Constraints
    CONSTRAINT ck_config_positive_version CHECK (version > 0),
    CONSTRAINT uq_config_key_env UNIQUE (config_key, environment)
);

-- Create indexes for processing_sessions
CREATE INDEX IF NOT EXISTS idx_sessions_user ON processing_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON processing_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON processing_sessions(start_time);
CREATE INDEX IF NOT EXISTS idx_sessions_type ON processing_sessions(session_type);
CREATE INDEX IF NOT EXISTS idx_sessions_algorithm_version ON processing_sessions(algorithm_version);

-- Create indexes for clusters
CREATE INDEX IF NOT EXISTS idx_clusters_session ON clusters(session_id);
CREATE INDEX IF NOT EXISTS idx_clusters_active ON clusters(is_active);
CREATE INDEX IF NOT EXISTS idx_clusters_health ON clusters(health_status);
CREATE INDEX IF NOT EXISTS idx_clusters_label ON clusters(cluster_label);
CREATE INDEX IF NOT EXISTS idx_clusters_updated ON clusters(last_updated);
CREATE INDEX IF NOT EXISTS idx_clusters_parent ON clusters(parent_cluster_id);
CREATE INDEX IF NOT EXISTS idx_clusters_generation ON clusters(generation);

-- Create indexes for data_points
CREATE INDEX IF NOT EXISTS idx_data_points_session_order ON data_points(session_id, processing_order);
CREATE INDEX IF NOT EXISTS idx_data_points_cluster ON data_points(cluster_id);
CREATE INDEX IF NOT EXISTS idx_data_points_outlier ON data_points(is_outlier);
CREATE INDEX IF NOT EXISTS idx_data_points_created ON data_points(created_at);
CREATE INDEX IF NOT EXISTS idx_data_points_confidence ON data_points(confidence_score);
CREATE INDEX IF NOT EXISTS idx_data_points_outlier_score ON data_points(outlier_score);

-- Create indexes for performance_metrics
CREATE INDEX IF NOT EXISTS idx_metrics_session_name ON performance_metrics(session_id, metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_category ON performance_metrics(metric_category);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(measurement_timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_session_time ON performance_metrics(session_id, measurement_timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_numeric_value ON performance_metrics(numeric_value);

-- Create indexes for audit_logs
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_security ON audit_logs(security_level);
CREATE INDEX IF NOT EXISTS idx_audit_ip ON audit_logs(client_ip);
CREATE INDEX IF NOT EXISTS idx_audit_category ON audit_logs(event_category);

-- Create indexes for user_activities
CREATE INDEX IF NOT EXISTS idx_user_activity_user ON user_activities(user_id);
CREATE INDEX IF NOT EXISTS idx_user_activity_type ON user_activities(activity_type);
CREATE INDEX IF NOT EXISTS idx_user_activity_timestamp ON user_activities(created_at);
CREATE INDEX IF NOT EXISTS idx_user_activity_category ON user_activities(category);
CREATE INDEX IF NOT EXISTS idx_user_activity_completion ON user_activities(completion_status);

-- Create indexes for system_configurations
CREATE INDEX IF NOT EXISTS idx_config_key ON system_configurations(config_key);
CREATE INDEX IF NOT EXISTS idx_config_category ON system_configurations(config_category);
CREATE INDEX IF NOT EXISTS idx_config_active ON system_configurations(is_active);
CREATE INDEX IF NOT EXISTS idx_config_environment ON system_configurations(environment);
CREATE INDEX IF NOT EXISTS idx_config_version ON system_configurations(version);

-- Create GIN indexes for JSONB columns (for better JSON query performance)
CREATE INDEX IF NOT EXISTS idx_sessions_algorithm_config_gin ON processing_sessions USING GIN (algorithm_config);
CREATE INDEX IF NOT EXISTS idx_clusters_centroid_gin ON clusters USING GIN (centroid);
CREATE INDEX IF NOT EXISTS idx_data_points_features_gin ON data_points USING GIN (features);
CREATE INDEX IF NOT EXISTS idx_metrics_json_value_gin ON performance_metrics USING GIN (json_value);
CREATE INDEX IF NOT EXISTS idx_audit_metadata_gin ON audit_logs USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_config_value_gin ON system_configurations USING GIN (config_value);

-- Create functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_processing_sessions_updated_at BEFORE UPDATE ON processing_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_clusters_updated_at BEFORE UPDATE ON clusters
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_data_points_updated_at BEFORE UPDATE ON data_points
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_performance_metrics_updated_at BEFORE UPDATE ON performance_metrics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_audit_logs_updated_at BEFORE UPDATE ON audit_logs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_activities_updated_at BEFORE UPDATE ON user_activities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_system_configurations_updated_at BEFORE UPDATE ON system_configurations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function for calculating cluster stability
CREATE OR REPLACE FUNCTION calculate_cluster_stability(
    nearest_distance DECIMAL(15, 6),
    second_nearest_distance DECIMAL(15, 6)
) RETURNS DECIMAL(5, 4) AS $$
BEGIN
    IF nearest_distance IS NULL OR second_nearest_distance IS NULL THEN
        RETURN NULL;
    END IF;
    
    IF second_nearest_distance = 0 THEN
        RETURN 1.0;
    END IF;
    
    RETURN 1.0 - (nearest_distance / second_nearest_distance);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create function for cluster quality score
CREATE OR REPLACE FUNCTION calculate_cluster_quality(
    cohesion DECIMAL(8, 6),
    separation DECIMAL(8, 6),
    silhouette_avg DECIMAL(5, 4)
) RETURNS DECIMAL(5, 4) AS $$
BEGIN
    IF cohesion IS NULL OR separation IS NULL OR silhouette_avg IS NULL THEN
        RETURN NULL;
    END IF;
    
    RETURN (cohesion * 0.4 + separation * 0.3 + silhouette_avg * 0.3);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Insert default system configurations
INSERT INTO system_configurations (config_key, config_category, config_value, data_type, description, environment)
VALUES 
    ('ncs.similarity_threshold', 'algorithm', '0.85', 'float', 'Default similarity threshold for NCS clustering algorithm', 'production'),
    ('ncs.min_cluster_size', 'algorithm', '3', 'integer', 'Minimum number of points required to form a cluster', 'production'),
    ('ncs.max_clusters', 'algorithm', '1000', 'integer', 'Maximum number of clusters allowed in a session', 'production'),
    ('ncs.outlier_threshold', 'algorithm', '0.75', 'float', 'Threshold for outlier detection', 'production'),
    ('ncs.adaptive_threshold_enabled', 'algorithm', 'true', 'boolean', 'Enable adaptive threshold computation', 'production'),
    ('api.rate_limit_per_minute', 'performance', '1000', 'integer', 'API rate limit per minute per user', 'production'),
    ('api.max_batch_size', 'performance', '10000', 'integer', 'Maximum batch size for processing requests', 'production'),
    ('monitoring.metrics_retention_days', 'monitoring', '90', 'integer', 'Number of days to retain performance metrics', 'production'),
    ('security.jwt_expiry_hours', 'security', '24', 'integer', 'JWT token expiry time in hours', 'production'),
    ('security.max_failed_logins', 'security', '5', 'integer', 'Maximum failed login attempts before lockout', 'production')
ON CONFLICT (config_key, environment) DO NOTHING;

-- Create materialized view for session statistics (for dashboard performance)
CREATE MATERIALIZED VIEW IF NOT EXISTS session_statistics AS
SELECT 
    ps.id as session_id,
    ps.session_name,
    ps.user_id,
    ps.status,
    ps.start_time,
    ps.end_time,
    ps.total_duration_seconds,
    ps.total_points_processed,
    ps.total_points_clustered,
    ps.total_outliers_detected,
    ps.unique_clusters_created,
    ps.throughput_points_per_sec,
    ps.overall_silhouette_score,
    ps.noise_ratio,
    COUNT(DISTINCT c.id) as active_clusters,
    AVG(c.silhouette_avg) as avg_cluster_silhouette,
    AVG(c.stability_score) as avg_cluster_stability,
    COUNT(dp.id) FILTER (WHERE dp.is_outlier = true) as outlier_count,
    AVG(dp.confidence_score) as avg_confidence_score,
    CASE 
        WHEN ps.total_points_processed > 0 
        THEN (ps.total_points_clustered::DECIMAL / ps.total_points_processed) * 100
        ELSE 0 
    END as clustering_efficiency_percent
FROM processing_sessions ps
LEFT JOIN clusters c ON ps.id = c.session_id AND c.is_active = true
LEFT JOIN data_points dp ON ps.id = dp.session_id
GROUP BY ps.id, ps.session_name, ps.user_id, ps.status, ps.start_time, ps.end_time,
         ps.total_duration_seconds, ps.total_points_processed, ps.total_points_clustered,
         ps.total_outliers_detected, ps.unique_clusters_created, ps.throughput_points_per_sec,
         ps.overall_silhouette_score, ps.noise_ratio;

-- Create index on materialized view
CREATE INDEX IF NOT EXISTS idx_session_stats_user ON session_statistics(user_id);
CREATE INDEX IF NOT EXISTS idx_session_stats_status ON session_statistics(status);
CREATE INDEX IF NOT EXISTS idx_session_stats_start_time ON session_statistics(start_time);

-- Create function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_session_statistics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW session_statistics;
END;
$$ LANGUAGE plpgsql;

-- Set up row level security (RLS) policies for multi-tenancy
ALTER TABLE processing_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE clusters ENABLE ROW LEVEL SECURITY;
ALTER TABLE data_points ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_activities ENABLE ROW LEVEL SECURITY;

-- Create RLS policies (these would be customized based on your authentication system)
-- For now, creating basic policies that can be modified later

-- Policy for processing_sessions: users can only see their own sessions
CREATE POLICY user_sessions_policy ON processing_sessions
    FOR ALL TO authenticated_users
    USING (user_id = current_setting('app.current_user_id', true));

-- Policy for clusters: users can only see clusters from their sessions
CREATE POLICY user_clusters_policy ON clusters
    FOR ALL TO authenticated_users
    USING (session_id IN (
        SELECT id FROM processing_sessions 
        WHERE user_id = current_setting('app.current_user_id', true)
    ));

-- Policy for data_points: users can only see data points from their sessions
CREATE POLICY user_data_points_policy ON data_points
    FOR ALL TO authenticated_users
    USING (session_id IN (
        SELECT id FROM processing_sessions 
        WHERE user_id = current_setting('app.current_user_id', true)
    ));

-- Policy for performance_metrics: users can only see metrics from their sessions
CREATE POLICY user_metrics_policy ON performance_metrics
    FOR ALL TO authenticated_users
    USING (session_id IN (
        SELECT id FROM processing_sessions 
        WHERE user_id = current_setting('app.current_user_id', true)
    ));

-- Policy for user_activities: users can only see their own activities
CREATE POLICY user_activities_policy ON user_activities
    FOR ALL TO authenticated_users
    USING (user_id = current_setting('app.current_user_id', true));

-- Create role for the application
CREATE ROLE IF NOT EXISTS ncs_app_user;
CREATE ROLE IF NOT EXISTS authenticated_users;

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO ncs_app_user, authenticated_users;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO ncs_app_user, authenticated_users;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO ncs_app_user, authenticated_users;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO ncs_app_user, authenticated_users;

-- Create partition tables for large data (data_points and audit_logs)
-- Partition data_points by session_id for better performance
-- This is optional and can be implemented later if needed

-- Create database health check function
CREATE OR REPLACE FUNCTION check_database_health()
RETURNS TABLE(
    check_name TEXT,
    status TEXT,
    details JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'table_sizes'::TEXT,
        'ok'::TEXT,
        jsonb_build_object(
            'processing_sessions', (SELECT COUNT(*) FROM processing_sessions),
            'clusters', (SELECT COUNT(*) FROM clusters),
            'data_points', (SELECT COUNT(*) FROM data_points),
            'performance_metrics', (SELECT COUNT(*) FROM performance_metrics),
            'audit_logs', (SELECT COUNT(*) FROM audit_logs),
            'user_activities', (SELECT COUNT(*) FROM user_activities),
            'system_configurations', (SELECT COUNT(*) FROM system_configurations)
        );
        
    RETURN QUERY
    SELECT 
        'active_connections'::TEXT,
        'ok'::TEXT,
        jsonb_build_object(
            'active_connections', (
                SELECT COUNT(*) 
                FROM pg_stat_activity 
                WHERE state = 'active' AND datname = current_database()
            )
        );
        
    RETURN QUERY
    SELECT 
        'database_size'::TEXT,
        'ok'::TEXT,
        jsonb_build_object(
            'size_pretty', pg_size_pretty(pg_database_size(current_database())),
            'size_bytes', pg_database_size(current_database())
        );
END;
$$ LANGUAGE plpgsql;

-- Commit the transaction
COMMIT;

-- Success message
SELECT 'NeuroCluster Streamer API Database Schema initialized successfully!' as message;