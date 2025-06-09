#!/bin/bash
#
# NeuroCluster Streamer API - Deployment Automation Script
#
# This script automates deployment of the NCS API to various targets:
# - Docker containers (local/registry)
# - Kubernetes clusters 
# - Cloud platforms (AWS, GCP, Azure)
# - Traditional servers
#
# Usage:
#   ./scripts/deploy.sh [target] [environment] [options]
#
# Targets:
#   docker     - Deploy to Docker (local or registry)
#   k8s        - Deploy to Kubernetes cluster
#   aws        - Deploy to AWS (ECS/EKS)
#   gcp        - Deploy to Google Cloud Platform
#   azure      - Deploy to Azure Container Instances
#   server     - Deploy to traditional server
#
# Environments:
#   staging    - Staging environment
#   production - Production environment
#   testing    - Testing environment
#
# Options:
#   --build-only    - Only build, don't deploy
#   --no-build      - Skip build, deploy existing images
#   --rollback      - Rollback to previous version
#   --dry-run       - Show what would be deployed without deploying
#   --force         - Force deployment even with warnings
#   --help          - Show this help message
#

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOY_TARGET="${1:-docker}"
ENVIRONMENT="${2:-staging}"
LOG_FILE="$PROJECT_ROOT/logs/deploy.log"

# Default configuration
BUILD_ONLY=false
NO_BUILD=false
ROLLBACK=false
DRY_RUN=false
FORCE=false
VERBOSE=false

# Version and image configuration
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "latest")
IMAGE_NAME="ncs-api"
REGISTRY_URL="${REGISTRY_URL:-}"
FULL_IMAGE_NAME="${REGISTRY_URL:+$REGISTRY_URL/}$IMAGE_NAME:$VERSION"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Deployment configuration by environment
declare -A ENV_CONFIG
ENV_CONFIG[staging]="replicas=2;cpu_limit=500m;memory_limit=512Mi;cpu_request=250m;memory_request=256Mi"
ENV_CONFIG[production]="replicas=3;cpu_limit=1000m;memory_limit=1Gi;cpu_request=500m;memory_request=512Mi"
ENV_CONFIG[testing]="replicas=1;cpu_limit=250m;memory_limit=256Mi;cpu_request=100m;memory_request=128Mi"

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    case "$level" in
        "INFO")  echo -e "${GREEN}[INFO]${NC} $message" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $message" ;;
        "DEBUG") [[ "$VERBOSE" == "true" ]] && echo -e "${BLUE}[DEBUG]${NC} $message" ;;
        "DEPLOY") echo -e "${PURPLE}[DEPLOY]${NC} $message" ;;
    esac
}

# Error handler with cleanup
error_exit() {
    log "ERROR" "$1"
    cleanup_on_error
    exit 1
}

# Success handler
success() {
    log "INFO" "$1"
}

# Cleanup function for errors
cleanup_on_error() {
    log "INFO" "Performing cleanup due to error..."
    
    # Remove temporary files
    find /tmp -name "ncs-deploy-*" -type f -delete 2>/dev/null || true
    
    # Docker cleanup if needed
    if [[ "$DEPLOY_TARGET" == "docker" ]]; then
        docker system prune -f --filter "label=ncs-deploy=temp" 2>/dev/null || true
    fi
}

# Show usage information
show_help() {
    cat << EOF
NeuroCluster Streamer API - Deployment Automation Script

Usage: $0 [target] [environment] [options]

Targets:
  docker     Deploy to Docker (local or registry)
  k8s        Deploy to Kubernetes cluster  
  aws        Deploy to AWS (ECS/EKS)
  gcp        Deploy to Google Cloud Platform
  azure      Deploy to Azure Container Instances
  server     Deploy to traditional server

Environments:
  staging    Staging environment (default)
  production Production environment
  testing    Testing environment

Options:
  --build-only     Only build, don't deploy
  --no-build       Skip build, deploy existing images
  --rollback       Rollback to previous version
  --dry-run        Show what would be deployed without deploying
  --force          Force deployment even with warnings
  --verbose        Enable verbose logging
  --help           Show this help message

Environment Variables:
  REGISTRY_URL     Container registry URL (e.g., docker.io/myorg)
  KUBECONFIG       Path to Kubernetes config file
  AWS_PROFILE      AWS profile for deployment
  GCP_PROJECT      Google Cloud project ID
  AZURE_SUBSCRIPTION_ID  Azure subscription ID

Examples:
  $0 docker staging              # Deploy to Docker staging
  $0 k8s production              # Deploy to Kubernetes production  
  $0 docker staging --build-only # Only build Docker image
  $0 k8s production --dry-run    # Show K8s deployment plan
  $0 aws production --rollback   # Rollback AWS production deployment

EOF
}

# Parse command line arguments
parse_args() {
    local args=("$@")
    local i=0
    
    while [[ $i -lt ${#args[@]} ]]; do
        case "${args[$i]}" in
            docker|k8s|aws|gcp|azure|server)
                DEPLOY_TARGET="${args[$i]}"
                ;;
            staging|production|testing)
                ENVIRONMENT="${args[$i]}"
                ;;
            --build-only)
                BUILD_ONLY=true
                ;;
            --no-build)
                NO_BUILD=true
                ;;
            --rollback)
                ROLLBACK=true
                ;;
            --dry-run)
                DRY_RUN=true
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
                if [[ $i -lt 2 ]]; then
                    # Skip positional arguments (target and environment)
                    :
                else
                    error_exit "Unknown option: ${args[$i]}. Use --help for usage information."
                fi
                ;;
        esac
        ((i++))
    done
}

# Validate deployment configuration
validate_deployment() {
    log "INFO" "Validating deployment configuration..."
    
    # Check if git repository is clean (for production)
    if [[ "$ENVIRONMENT" == "production" ]] && [[ "$FORCE" != "true" ]]; then
        if ! git diff-index --quiet HEAD --; then
            error_exit "Git repository has uncommitted changes. Use --force to override or commit changes first."
        fi
        log "INFO" "Git repository is clean"
    fi
    
    # Validate environment configuration
    if [[ -z "${ENV_CONFIG[$ENVIRONMENT]:-}" ]]; then
        error_exit "Unknown environment: $ENVIRONMENT"
    fi
    
    # Parse environment configuration
    local env_config="${ENV_CONFIG[$ENVIRONMENT]}"
    IFS=';' read -ra CONFIG_PAIRS <<< "$env_config"
    declare -A CONFIG
    for pair in "${CONFIG_PAIRS[@]}"; do
        IFS='=' read -r key value <<< "$pair"
        CONFIG["$key"]="$value"
    done
    
    log "INFO" "Environment configuration loaded:"
    for key in "${!CONFIG[@]}"; do
        log "DEBUG" "  $key=${CONFIG[$key]}"
    done
    
    # Validate target-specific requirements
    case "$DEPLOY_TARGET" in
        docker)
            if ! command -v docker &> /dev/null; then
                error_exit "Docker is not installed or not in PATH"
            fi
            ;;
        k8s)
            if ! command -v kubectl &> /dev/null; then
                error_exit "kubectl is not installed or not in PATH"
            fi
            if ! kubectl cluster-info &> /dev/null; then
                error_exit "kubectl cannot connect to cluster. Check KUBECONFIG."
            fi
            ;;
        aws)
            if ! command -v aws &> /dev/null; then
                error_exit "AWS CLI is not installed or not in PATH"
            fi
            if ! aws sts get-caller-identity &> /dev/null; then
                error_exit "AWS CLI is not configured. Run 'aws configure' first."
            fi
            ;;
        gcp)
            if ! command -v gcloud &> /dev/null; then
                error_exit "Google Cloud SDK is not installed or not in PATH"
            fi
            if [[ -z "${GCP_PROJECT:-}" ]]; then
                error_exit "GCP_PROJECT environment variable is required for GCP deployment"
            fi
            ;;
        azure)
            if ! command -v az &> /dev/null; then
                error_exit "Azure CLI is not installed or not in PATH"
            fi
            ;;
    esac
    
    success "Deployment configuration validation completed"
}

# Build application
build_application() {
    if [[ "$NO_BUILD" == "true" ]]; then
        log "INFO" "Skipping build (--no-build specified)"
        return 0
    fi
    
    log "DEPLOY" "Building application..."
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Run tests before building (for production)
    if [[ "$ENVIRONMENT" == "production" ]] && [[ "$FORCE" != "true" ]]; then
        log "INFO" "Running tests before production build..."
        
        if ! python -m pytest tests/ -x --tb=short; then
            error_exit "Tests failed. Fix tests before deploying to production or use --force."
        fi
        success "All tests passed"
    fi
    
    # Build based on target
    case "$DEPLOY_TARGET" in
        docker|k8s|aws|gcp|azure)
            build_docker_image
            ;;
        server)
            build_for_server
            ;;
    esac
    
    success "Application build completed"
}

# Build Docker image
build_docker_image() {
    log "INFO" "Building Docker image: $FULL_IMAGE_NAME"
    
    local dockerfile="$PROJECT_ROOT/docker/Dockerfile"
    if [[ "$ENVIRONMENT" == "development" ]] && [[ -f "$PROJECT_ROOT/docker/Dockerfile.dev" ]]; then
        dockerfile="$PROJECT_ROOT/docker/Dockerfile.dev"
    fi
    
    if [[ ! -f "$dockerfile" ]]; then
        error_exit "Dockerfile not found: $dockerfile"
    fi
    
    # Build arguments
    local build_args=(
        --build-arg "VERSION=$VERSION"
        --build-arg "ENVIRONMENT=$ENVIRONMENT"
        --build-arg "BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        --label "ncs.version=$VERSION"
        --label "ncs.environment=$ENVIRONMENT"
        --label "ncs.build-date=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    )
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would build Docker image with:"
        log "INFO" "  Image: $FULL_IMAGE_NAME"
        log "INFO" "  Dockerfile: $dockerfile"
        log "INFO" "  Build args: ${build_args[*]}"
        return 0
    fi
    
    # Build the image
    if ! docker build \
        "${build_args[@]}" \
        -f "$dockerfile" \
        -t "$FULL_IMAGE_NAME" \
        -t "${IMAGE_NAME}:latest" \
        "$PROJECT_ROOT"; then
        error_exit "Docker build failed"
    fi
    
    # Push to registry if configured
    if [[ -n "$REGISTRY_URL" ]]; then
        log "INFO" "Pushing image to registry: $REGISTRY_URL"
        
        if ! docker push "$FULL_IMAGE_NAME"; then
            error_exit "Failed to push image to registry"
        fi
        
        success "Image pushed to registry"
    fi
    
    success "Docker image built successfully"
}

# Build for server deployment
build_for_server() {
    log "INFO" "Building application for server deployment..."
    
    # Create deployment package
    local package_name="ncs-api-$VERSION-$ENVIRONMENT.tar.gz"
    local temp_dir="/tmp/ncs-deploy-$$"
    
    mkdir -p "$temp_dir"
    
    # Copy application files
    rsync -av \
        --exclude='.git' \
        --exclude='venv' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.pytest_cache' \
        --exclude='logs' \
        "$PROJECT_ROOT/" "$temp_dir/"
    
    # Create version file
    echo "$VERSION" > "$temp_dir/VERSION"
    
    # Create deployment package
    cd "$(dirname "$temp_dir")"
    tar -czf "$PROJECT_ROOT/$package_name" "$(basename "$temp_dir")"
    
    # Cleanup
    rm -rf "$temp_dir"
    
    success "Server deployment package created: $package_name"
}

# Deploy to Docker
deploy_docker() {
    log "DEPLOY" "Deploying to Docker..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would deploy Docker container with image: $FULL_IMAGE_NAME"
        return 0
    fi
    
    # Stop existing container if running
    if docker ps -q -f name="$IMAGE_NAME" | grep -q .; then
        log "INFO" "Stopping existing container..."
        docker stop "$IMAGE_NAME" || true
        docker rm "$IMAGE_NAME" || true
    fi
    
    # Get environment configuration
    local env_config="${ENV_CONFIG[$ENVIRONMENT]}"
    IFS=';' read -ra CONFIG_PAIRS <<< "$env_config"
    declare -A CONFIG
    for pair in "${CONFIG_PAIRS[@]}"; do
        IFS='=' read -r key value <<< "$pair"
        CONFIG["$key"]="$value"
    done
    
    # Run new container
    local docker_args=(
        --name "$IMAGE_NAME"
        --detach
        --restart unless-stopped
        -p 8000:8000
        -p 8001:8001
        --env ENVIRONMENT="$ENVIRONMENT"
        --env-file "$PROJECT_ROOT/.env"
    )
    
    # Add resource limits if supported
    if [[ -n "${CONFIG[memory_limit]:-}" ]]; then
        docker_args+=(--memory "${CONFIG[memory_limit]}")
    fi
    
    if ! docker run "${docker_args[@]}" "$FULL_IMAGE_NAME"; then
        error_exit "Failed to start Docker container"
    fi
    
    # Wait for container to be healthy
    log "INFO" "Waiting for container to be healthy..."
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -sf "http://localhost:8000/health" > /dev/null 2>&1; then
            success "Container is healthy and responding"
            break
        fi
        
        log "INFO" "Waiting for container health check... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    if [[ $attempt -gt $max_attempts ]]; then
        error_exit "Container failed health check after $max_attempts attempts"
    fi
    
    success "Docker deployment completed"
}

# Deploy to Kubernetes
deploy_k8s() {
    log "DEPLOY" "Deploying to Kubernetes..."
    
    local k8s_dir="$PROJECT_ROOT/k8s"
    if [[ ! -d "$k8s_dir" ]]; then
        error_exit "Kubernetes manifests directory not found: $k8s_dir"
    fi
    
    # Get environment configuration
    local env_config="${ENV_CONFIG[$ENVIRONMENT]}"
    IFS=';' read -ra CONFIG_PAIRS <<< "$env_config"
    declare -A CONFIG
    for pair in "${CONFIG_PAIRS[@]}"; do
        IFS='=' read -r key value <<< "$pair"
        CONFIG["$key"]="$value"
    done
    
    # Create namespace if it doesn't exist
    local namespace="ncs-$ENVIRONMENT"
    if ! kubectl get namespace "$namespace" &> /dev/null; then
        log "INFO" "Creating namespace: $namespace"
        if [[ "$DRY_RUN" != "true" ]]; then
            kubectl create namespace "$namespace"
        fi
    fi
    
    # Process and apply manifests
    local manifests=(
        "namespace.yaml"
        "configmap.yaml"
        "secrets.yaml"
        "deployment.yaml"
        "service.yaml"
        "ingress.yaml"
        "hpa.yaml"
    )
    
    for manifest in "${manifests[@]}"; do
        local manifest_path="$k8s_dir/$manifest"
        
        if [[ ! -f "$manifest_path" ]]; then
            log "WARN" "Manifest not found, skipping: $manifest"
            continue
        fi
        
        log "INFO" "Processing manifest: $manifest"
        
        # Create temporary processed manifest
        local temp_manifest="/tmp/ncs-$manifest-$$"
        
        # Replace placeholders in manifest
        sed -e "s/{{ENVIRONMENT}}/$ENVIRONMENT/g" \
            -e "s/{{VERSION}}/$VERSION/g" \
            -e "s/{{IMAGE_NAME}}/$FULL_IMAGE_NAME/g" \
            -e "s/{{REPLICAS}}/${CONFIG[replicas]}/g" \
            -e "s/{{CPU_LIMIT}}/${CONFIG[cpu_limit]}/g" \
            -e "s/{{MEMORY_LIMIT}}/${CONFIG[memory_limit]}/g" \
            -e "s/{{CPU_REQUEST}}/${CONFIG[cpu_request]}/g" \
            -e "s/{{MEMORY_REQUEST}}/${CONFIG[memory_request]}/g" \
            "$manifest_path" > "$temp_manifest"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log "INFO" "[DRY RUN] Would apply manifest:"
            cat "$temp_manifest"
            echo "---"
        else
            if ! kubectl apply -f "$temp_manifest" -n "$namespace"; then
                rm -f "$temp_manifest"
                error_exit "Failed to apply manifest: $manifest"
            fi
        fi
        
        rm -f "$temp_manifest"
    done
    
    if [[ "$DRY_RUN" != "true" ]]; then
        # Wait for deployment to be ready
        log "INFO" "Waiting for deployment to be ready..."
        if ! kubectl wait --for=condition=available --timeout=300s deployment/ncs-api -n "$namespace"; then
            error_exit "Deployment failed to become available within 5 minutes"
        fi
        
        # Get service information
        log "INFO" "Deployment information:"
        kubectl get pods,services,ingress -n "$namespace" -l app=ncs-api
    fi
    
    success "Kubernetes deployment completed"
}

# Deploy to AWS
deploy_aws() {
    log "DEPLOY" "Deploying to AWS..."
    
    # Check if EKS or ECS
    local aws_target="${AWS_TARGET:-ecs}"
    
    case "$aws_target" in
        "eks")
            # Deploy to EKS (similar to k8s)
            log "INFO" "Deploying to AWS EKS..."
            deploy_k8s  # Reuse K8s deployment for EKS
            ;;
        "ecs")
            # Deploy to ECS
            deploy_aws_ecs
            ;;
        *)
            error_exit "Unknown AWS target: $aws_target. Use 'eks' or 'ecs'"
            ;;
    esac
}

# Deploy to AWS ECS
deploy_aws_ecs() {
    log "INFO" "Deploying to AWS ECS..."
    
    local cluster_name="ncs-$ENVIRONMENT"
    local service_name="ncs-api-$ENVIRONMENT"
    local task_family="ncs-api-$ENVIRONMENT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would deploy to ECS:"
        log "INFO" "  Cluster: $cluster_name"
        log "INFO" "  Service: $service_name"
        log "INFO" "  Task Family: $task_family"
        log "INFO" "  Image: $FULL_IMAGE_NAME"
        return 0
    fi
    
    # Create or update task definition
    local task_def_json=$(cat << EOF
{
    "family": "$task_family",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "512",
    "memory": "1024",
    "executionRoleArn": "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/ecsTaskExecutionRole",
    "containerDefinitions": [
        {
            "name": "ncs-api",
            "image": "$FULL_IMAGE_NAME",
            "portMappings": [
                {
                    "containerPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {
                    "name": "ENVIRONMENT",
                    "value": "$ENVIRONMENT"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/$task_family",
                    "awslogs-region": "$(aws configure get region)",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}
EOF
)
    
    # Register task definition
    local task_arn=$(aws ecs register-task-definition \
        --cli-input-json "$task_def_json" \
        --query 'taskDefinition.taskDefinitionArn' \
        --output text)
    
    if [[ -z "$task_arn" ]]; then
        error_exit "Failed to register ECS task definition"
    fi
    
    log "INFO" "Registered task definition: $task_arn"
    
    # Update service
    aws ecs update-service \
        --cluster "$cluster_name" \
        --service "$service_name" \
        --task-definition "$task_arn" > /dev/null
    
    # Wait for deployment to complete
    log "INFO" "Waiting for ECS service to stabilize..."
    aws ecs wait services-stable \
        --cluster "$cluster_name" \
        --services "$service_name"
    
    success "AWS ECS deployment completed"
}

# Rollback deployment
rollback_deployment() {
    log "DEPLOY" "Rolling back deployment..."
    
    case "$DEPLOY_TARGET" in
        docker)
            log "INFO" "Docker rollback not implemented yet"
            ;;
        k8s)
            kubectl rollout undo deployment/ncs-api -n "ncs-$ENVIRONMENT"
            kubectl rollout status deployment/ncs-api -n "ncs-$ENVIRONMENT"
            ;;
        aws)
            log "INFO" "AWS rollback not implemented yet"
            ;;
        *)
            error_exit "Rollback not supported for target: $DEPLOY_TARGET"
            ;;
    esac
    
    success "Rollback completed"
}

# Post-deployment verification
verify_deployment() {
    log "INFO" "Verifying deployment..."
    
    local health_url=""
    case "$DEPLOY_TARGET" in
        docker)
            health_url="http://localhost:8000/health"
            ;;
        k8s)
            # Get service URL
            local service_ip=$(kubectl get service ncs-api -n "ncs-$ENVIRONMENT" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
            if [[ -n "$service_ip" ]]; then
                health_url="http://$service_ip:8000/health"
            else
                log "WARN" "Could not determine service URL for health check"
                return 0
            fi
            ;;
        *)
            log "INFO" "Health check not implemented for target: $DEPLOY_TARGET"
            return 0
            ;;
    esac
    
    if [[ -n "$health_url" ]]; then
        log "INFO" "Checking health endpoint: $health_url"
        
        local max_attempts=10
        local attempt=1
        
        while [[ $attempt -le $max_attempts ]]; do
            if curl -sf "$health_url" > /dev/null 2>&1; then
                success "Health check passed"
                break
            fi
            
            log "INFO" "Health check attempt $attempt/$max_attempts..."
            sleep 5
            ((attempt++))
        done
        
        if [[ $attempt -gt $max_attempts ]]; then
            error_exit "Health check failed after $max_attempts attempts"
        fi
    fi
    
    success "Deployment verification completed"
}

# Show deployment summary
show_deployment_summary() {
    cat << EOF

${GREEN}🚀 Deployment completed successfully!${NC}

${BLUE}Deployment Summary:${NC}
- Target: ${YELLOW}$DEPLOY_TARGET${NC}
- Environment: ${YELLOW}$ENVIRONMENT${NC}
- Version: ${YELLOW}$VERSION${NC}
- Image: ${YELLOW}$FULL_IMAGE_NAME${NC}

${BLUE}Next steps:${NC}
EOF

    case "$DEPLOY_TARGET" in
        docker)
            cat << EOF
- Check container status: ${YELLOW}docker ps${NC}
- View logs: ${YELLOW}docker logs $IMAGE_NAME${NC}
- Access API: ${YELLOW}http://localhost:8000${NC}
- API docs: ${YELLOW}http://localhost:8000/docs${NC}
EOF
            ;;
        k8s)
            cat << EOF
- Check deployment: ${YELLOW}kubectl get deployments -n ncs-$ENVIRONMENT${NC}
- View pods: ${YELLOW}kubectl get pods -n ncs-$ENVIRONMENT${NC}
- Check logs: ${YELLOW}kubectl logs -f deployment/ncs-api -n ncs-$ENVIRONMENT${NC}
- Port forward: ${YELLOW}kubectl port-forward service/ncs-api 8000:8000 -n ncs-$ENVIRONMENT${NC}
EOF
            ;;
    esac
    
    cat << EOF

${BLUE}Monitoring:${NC}
- View deployment logs: ${YELLOW}tail -f logs/deploy.log${NC}
- Check health: ${YELLOW}curl http://your-endpoint/health${NC}

EOF
}

# Main deployment function
main() {
    echo -e "${GREEN}NeuroCluster Streamer API - Deployment Automation${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
    
    # Parse arguments
    parse_args "$@"
    
    log "INFO" "Starting deployment"
    log "INFO" "Target: $DEPLOY_TARGET"
    log "INFO" "Environment: $ENVIRONMENT"
    log "INFO" "Version: $VERSION"
    log "INFO" "Image: $FULL_IMAGE_NAME"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Handle rollback
    if [[ "$ROLLBACK" == "true" ]]; then
        rollback_deployment
        exit 0
    fi
    
    # Run deployment steps
    validate_deployment
    build_application
    
    # Stop here if build-only
    if [[ "$BUILD_ONLY" == "true" ]]; then
        log "INFO" "Build completed (--build-only specified)"
        exit 0
    fi
    
    # Deploy based on target
    case "$DEPLOY_TARGET" in
        docker)
            deploy_docker
            ;;
        k8s)
            deploy_k8s
            ;;
        aws)
            deploy_aws
            ;;
        gcp)
            error_exit "GCP deployment not implemented yet"
            ;;
        azure)
            error_exit "Azure deployment not implemented yet"
            ;;
        server)
            error_exit "Server deployment not implemented yet"
            ;;
        *)
            error_exit "Unknown deployment target: $DEPLOY_TARGET"
            ;;
    esac
    
    # Verify deployment
    if [[ "$DRY_RUN" != "true" ]]; then
        verify_deployment
        show_deployment_summary
    fi
    
    log "INFO" "Deployment completed successfully"
}

# Run main function with all arguments
main "$@"