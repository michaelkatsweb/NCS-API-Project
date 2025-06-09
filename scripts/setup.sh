#!/bin/bash
#
# NeuroCluster Streamer API - Environment Setup Script
# 
# This script sets up the development/production environment for the NCS API
# including dependencies, database, configuration, and initial data setup.
#
# Usage:
#   ./scripts/setup.sh [environment] [options]
#
# Environments:
#   development (default) - Local development setup
#   production           - Production environment setup
#   testing             - Testing environment setup
#
# Options:
#   --skip-deps         - Skip dependency installation
#   --skip-db          - Skip database setup
#   --skip-config      - Skip configuration generation
#   --force            - Force overwrite existing files
#   --help             - Show this help message
#

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-development}"
LOG_FILE="$PROJECT_ROOT/logs/setup.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
SKIP_DEPS=false
SKIP_DB=false
SKIP_CONFIG=false
FORCE=false
VERBOSE=false

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Create logs directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Log to file
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    # Log to console with colors
    case "$level" in
        "INFO")  echo -e "${GREEN}[INFO]${NC} $message" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $message" ;;
        "DEBUG") [[ "$VERBOSE" == "true" ]] && echo -e "${BLUE}[DEBUG]${NC} $message" ;;
    esac
}

# Error handler
error_exit() {
    log "ERROR" "$1"
    exit 1
}

# Success handler
success() {
    log "INFO" "$1"
}

# Show usage information
show_help() {
    cat << EOF
NeuroCluster Streamer API - Environment Setup Script

Usage: $0 [environment] [options]

Environments:
  development    Setup local development environment (default)
  production     Setup production environment  
  testing        Setup testing environment

Options:
  --skip-deps    Skip dependency installation
  --skip-db      Skip database setup
  --skip-config  Skip configuration generation
  --force        Force overwrite existing files
  --verbose      Enable verbose logging
  --help         Show this help message

Examples:
  $0                           # Setup development environment
  $0 production                # Setup production environment
  $0 development --skip-db     # Setup dev environment without database
  $0 production --force        # Force setup production environment

Environment Variables:
  DATABASE_URL     Override default database URL
  REDIS_URL        Override default Redis URL
  SECRET_KEY       Override JWT secret key generation

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            development|production|testing)
                ENVIRONMENT="$1"
                ;;
            --skip-deps)
                SKIP_DEPS=true
                ;;
            --skip-db)
                SKIP_DB=true
                ;;
            --skip-config)
                SKIP_CONFIG=true
                ;;
            --force)
                FORCE=true
                ;;
            --verbose)
                VERBOSE=true
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                error_exit "Unknown option: $1. Use --help for usage information."
                ;;
        esac
        shift
    done
}

# Check system requirements
check_requirements() {
    log "INFO" "Checking system requirements..."
    
    local requirements=(
        "python3:Python 3.8+ is required"
        "pip3:pip3 is required for Python package installation"
        "git:Git is required for version control"
    )
    
    # Check for required commands
    for req in "${requirements[@]}"; do
        local cmd="${req%%:*}"
        local desc="${req#*:}"
        
        if ! command -v "$cmd" &> /dev/null; then
            error_exit "$desc. Please install $cmd and try again."
        fi
        log "DEBUG" "Found $cmd: $(command -v "$cmd")"
    done
    
    # Check Python version
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    local min_version="3.8"
    
    if [[ "$(printf '%s\n' "$min_version" "$python_version" | sort -V | head -n1)" != "$min_version" ]]; then
        error_exit "Python $min_version+ is required. Found Python $python_version."
    fi
    
    log "INFO" "Python $python_version found - requirement satisfied"
    
    # Check available disk space
    local available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    local required_space=1048576  # 1GB in KB
    
    if [[ "$available_space" -lt "$required_space" ]]; then
        log "WARN" "Low disk space detected. At least 1GB free space recommended."
    fi
    
    success "System requirements check completed"
}

# Install system dependencies
install_system_deps() {
    if [[ "$SKIP_DEPS" == "true" ]]; then
        log "INFO" "Skipping system dependency installation"
        return 0
    fi
    
    log "INFO" "Installing system dependencies..."
    
    # Detect OS
    local os_type=""
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            os_type="debian"
        elif command -v yum &> /dev/null; then
            os_type="redhat"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        os_type="macos"
    fi
    
    case "$os_type" in
        "debian")
            log "INFO" "Detected Debian/Ubuntu system"
            sudo apt-get update
            sudo apt-get install -y \
                python3-dev \
                python3-venv \
                postgresql-client \
                redis-tools \
                build-essential \
                curl \
                wget \
                jq
            ;;
        "redhat")
            log "INFO" "Detected RedHat/CentOS system"
            sudo yum install -y \
                python3-devel \
                postgresql \
                redis \
                gcc \
                gcc-c++ \
                make \
                curl \
                wget \
                jq
            ;;
        "macos")
            log "INFO" "Detected macOS system"
            if command -v brew &> /dev/null; then
                brew install postgresql redis jq
            else
                log "WARN" "Homebrew not found. Please install dependencies manually."
            fi
            ;;
        *)
            log "WARN" "Unknown operating system. Please install dependencies manually."
            ;;
    esac
    
    success "System dependencies installation completed"
}

# Setup Python virtual environment
setup_python_env() {
    log "INFO" "Setting up Python virtual environment..."
    
    local venv_path="$PROJECT_ROOT/venv"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "$venv_path" ]] || [[ "$FORCE" == "true" ]]; then
        log "INFO" "Creating Python virtual environment at $venv_path"
        python3 -m venv "$venv_path"
    else
        log "INFO" "Virtual environment already exists at $venv_path"
    fi
    
    # Activate virtual environment
    source "$venv_path/bin/activate"
    
    # Upgrade pip
    log "INFO" "Upgrading pip..."
    python -m pip install --upgrade pip setuptools wheel
    
    # Install Python dependencies
    if [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
        log "INFO" "Installing Python dependencies from requirements.txt..."
        pip install -r "$PROJECT_ROOT/requirements.txt"
        
        # Install development dependencies if available
        if [[ -f "$PROJECT_ROOT/requirements-dev.txt" ]]; then
            log "INFO" "Installing development dependencies..."
            pip install -r "$PROJECT_ROOT/requirements-dev.txt"
        fi
    else
        error_exit "requirements.txt not found in project root"
    fi
    
    success "Python environment setup completed"
}

# Setup database
setup_database() {
    if [[ "$SKIP_DB" == "true" ]]; then
        log "INFO" "Skipping database setup"
        return 0
    fi
    
    log "INFO" "Setting up database..."
    
    # Database configuration based on environment
    local db_config
    case "$ENVIRONMENT" in
        "development")
            db_config="postgresql://ncs_dev:ncs_dev_password@localhost:5432/ncs_dev"
            ;;
        "testing")
            db_config="postgresql://ncs_test:ncs_test_password@localhost:5432/ncs_test"
            ;;
        "production")
            db_config="${DATABASE_URL:-postgresql://ncs_prod:ncs_prod_password@localhost:5432/ncs_prod}"
            ;;
    esac
    
    # Check if PostgreSQL is running
    if ! pg_isready -q; then
        log "WARN" "PostgreSQL is not running. Please start PostgreSQL service."
        
        # Try to start PostgreSQL (varies by system)
        if command -v systemctl &> /dev/null; then
            log "INFO" "Attempting to start PostgreSQL with systemctl..."
            sudo systemctl start postgresql || log "WARN" "Failed to start PostgreSQL"
        elif command -v brew &> /dev/null; then
            log "INFO" "Attempting to start PostgreSQL with brew..."
            brew services start postgresql || log "WARN" "Failed to start PostgreSQL"
        fi
    fi
    
    # Wait for PostgreSQL to be ready
    local max_attempts=30
    local attempt=1
    while ! pg_isready -q && [[ $attempt -le $max_attempts ]]; do
        log "INFO" "Waiting for PostgreSQL to be ready... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    if ! pg_isready -q; then
        error_exit "PostgreSQL is not available after $max_attempts attempts"
    fi
    
    # Run database migrations
    if [[ -f "$PROJECT_ROOT/database/migrations/init.sql" ]]; then
        log "INFO" "Running database migrations..."
        
        # Extract database connection details
        local db_name="${db_config##*/}"
        local db_user="${db_config#*//}"
        db_user="${db_user%%:*}"
        
        # Create database if it doesn't exist (for development/testing)
        if [[ "$ENVIRONMENT" != "production" ]]; then
            createdb "$db_name" 2>/dev/null || log "INFO" "Database $db_name already exists"
        fi
        
        # Run migrations using Python script if available
        if [[ -f "$PROJECT_ROOT/scripts/db_migrate.py" ]]; then
            python "$PROJECT_ROOT/scripts/db_migrate.py" --environment "$ENVIRONMENT"
        else
            log "WARN" "Database migration script not found. Please run migrations manually."
        fi
    fi
    
    success "Database setup completed"
}

# Setup Redis
setup_redis() {
    log "INFO" "Setting up Redis..."
    
    # Check if Redis is running
    if ! redis-cli ping &> /dev/null; then
        log "WARN" "Redis is not running. Please start Redis service."
        
        # Try to start Redis (varies by system)
        if command -v systemctl &> /dev/null; then
            log "INFO" "Attempting to start Redis with systemctl..."
            sudo systemctl start redis || log "WARN" "Failed to start Redis"
        elif command -v brew &> /dev/null; then
            log "INFO" "Attempting to start Redis with brew..."
            brew services start redis || log "WARN" "Failed to start Redis"
        fi
    fi
    
    # Test Redis connection
    if redis-cli ping &> /dev/null; then
        success "Redis is running and accessible"
    else
        log "WARN" "Redis is not accessible. Some features may not work properly."
    fi
}

# Generate configuration files
generate_config() {
    if [[ "$SKIP_CONFIG" == "true" ]]; then
        log "INFO" "Skipping configuration generation"
        return 0
    fi
    
    log "INFO" "Generating configuration files..."
    
    local env_file="$PROJECT_ROOT/.env"
    local env_example="$PROJECT_ROOT/.env.example"
    
    # Backup existing .env file
    if [[ -f "$env_file" ]] && [[ "$FORCE" != "true" ]]; then
        local backup_file="$env_file.backup.$(date +%Y%m%d_%H%M%S)"
        cp "$env_file" "$backup_file"
        log "INFO" "Backed up existing .env to $backup_file"
    fi
    
    # Generate secrets if the generation script exists
    local jwt_secret=""
    local api_keys=""
    if [[ -f "$PROJECT_ROOT/scripts/generate_secrets.py" ]]; then
        log "INFO" "Generating secrets..."
        local secrets_output=$(python "$PROJECT_ROOT/scripts/generate_secrets.py" --format json)
        jwt_secret=$(echo "$secrets_output" | jq -r '.jwt_secret')
        api_keys=$(echo "$secrets_output" | jq -r '.api_keys | join(",")')
    else
        log "WARN" "Secret generation script not found. Using default values."
        jwt_secret="your-secret-key-change-in-production"
        api_keys="api-key-1,api-key-2"
    fi
    
    # Create .env file based on environment
    cat > "$env_file" << EOF
# NeuroCluster Streamer API Configuration
# Environment: $ENVIRONMENT
# Generated: $(date)

# Environment
ENVIRONMENT=$ENVIRONMENT

# Database Configuration
DATABASE_URL=postgresql://ncs_${ENVIRONMENT}:ncs_${ENVIRONMENT}_password@localhost:5432/ncs_${ENVIRONMENT}

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Security Configuration
JWT_SECRET_KEY=$jwt_secret
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24

# API Configuration
API_KEYS=$api_keys
RATE_LIMIT_PER_MINUTE=1000

# Algorithm Configuration
DEFAULT_SIMILARITY_THRESHOLD=0.85
DEFAULT_MIN_CLUSTER_SIZE=3
DEFAULT_MAX_CLUSTERS=1000
DEFAULT_OUTLIER_THRESHOLD=0.75

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json

# Monitoring Configuration
ENABLE_METRICS=true
METRICS_PORT=8001

# Development specific settings
EOF

    if [[ "$ENVIRONMENT" == "development" ]]; then
        cat >> "$env_file" << EOF
DEBUG=true
RELOAD=true
WORKERS=1
EOF
    elif [[ "$ENVIRONMENT" == "production" ]]; then
        cat >> "$env_file" << EOF
DEBUG=false
RELOAD=false
WORKERS=4
EOF
    fi
    
    success "Configuration file generated at $env_file"
    
    # Set appropriate permissions
    chmod 600 "$env_file"
    log "INFO" "Set secure permissions (600) on .env file"
}

# Setup logging directory
setup_logging() {
    log "INFO" "Setting up logging directory..."
    
    local logs_dir="$PROJECT_ROOT/logs"
    mkdir -p "$logs_dir"
    
    # Create log rotation configuration if logrotate is available
    if command -v logrotate &> /dev/null; then
        local logrotate_config="/etc/logrotate.d/ncs-api"
        
        if [[ "$ENVIRONMENT" == "production" ]] && [[ ! -f "$logrotate_config" ]]; then
            log "INFO" "Creating logrotate configuration..."
            
            sudo tee "$logrotate_config" > /dev/null << EOF
$logs_dir/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF
            log "INFO" "Logrotate configuration created at $logrotate_config"
        fi
    fi
    
    success "Logging setup completed"
}

# Verify installation
verify_installation() {
    log "INFO" "Verifying installation..."
    
    local errors=0
    
    # Check Python environment
    if source "$PROJECT_ROOT/venv/bin/activate" && python -c "import fastapi, sqlalchemy, redis" 2>/dev/null; then
        success "Python dependencies verification passed"
    else
        log "ERROR" "Python dependencies verification failed"
        ((errors++))
    fi
    
    # Check configuration file
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        success "Configuration file exists"
    else
        log "ERROR" "Configuration file missing"
        ((errors++))
    fi
    
    # Check database connectivity (if not skipped)
    if [[ "$SKIP_DB" != "true" ]]; then
        if pg_isready -q; then
            success "Database connectivity verified"
        else
            log "ERROR" "Database not accessible"
            ((errors++))
        fi
    fi
    
    # Check Redis connectivity
    if redis-cli ping &> /dev/null; then
        success "Redis connectivity verified"
    else
        log "WARN" "Redis not accessible (optional)"
    fi
    
    if [[ $errors -eq 0 ]]; then
        success "Installation verification completed successfully"
        return 0
    else
        error_exit "Installation verification failed with $errors errors"
    fi
}

# Show next steps
show_next_steps() {
    cat << EOF

${GREEN}✅ Setup completed successfully!${NC}

${BLUE}Next steps:${NC}

1. Activate the Python virtual environment:
   ${YELLOW}source venv/bin/activate${NC}

2. Review and customize the configuration:
   ${YELLOW}nano .env${NC}

3. Start the development server:
   ${YELLOW}uvicorn main:app --reload${NC}

4. Or start the secure server:
   ${YELLOW}uvicorn main_secure:app --reload${NC}

5. Access the API documentation:
   ${YELLOW}http://localhost:8000/docs${NC}

6. Run tests to verify everything works:
   ${YELLOW}pytest tests/test_api.py -v${NC}

${BLUE}Useful commands:${NC}
- Run all tests: ${YELLOW}pytest${NC}
- Run performance tests: ${YELLOW}pytest tests/performance_test.py${NC}
- Check API health: ${YELLOW}curl http://localhost:8000/health${NC}
- View logs: ${YELLOW}tail -f logs/setup.log${NC}

${BLUE}Documentation:${NC}
- API Reference: docs/API_REFERENCE.md
- Deployment Guide: docs/DEPLOYMENT_GUIDE.md
- Troubleshooting: docs/TROUBLESHOOTING.md

For more information, visit: https://github.com/your-org/ncs-api

EOF
}

# Main setup function
main() {
    echo -e "${GREEN}NeuroCluster Streamer API - Environment Setup${NC}"
    echo -e "${BLUE}=============================================${NC}"
    echo ""
    
    # Parse command line arguments
    parse_args "$@"
    
    log "INFO" "Starting setup for $ENVIRONMENT environment"
    log "INFO" "Project root: $PROJECT_ROOT"
    log "INFO" "Log file: $LOG_FILE"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Run setup steps
    check_requirements
    install_system_deps
    setup_python_env
    setup_database
    setup_redis
    generate_config
    setup_logging
    verify_installation
    
    # Show completion message and next steps
    show_next_steps
    
    log "INFO" "Setup completed successfully for $ENVIRONMENT environment"
}

# Run main function with all arguments
main "$@"