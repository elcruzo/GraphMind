#!/bin/bash
# GraphMind Kubernetes Deployment Script
# Deploys GraphMind distributed GNN system to Kubernetes cluster

set -e

echo "🚀 GraphMind Kubernetes Deployment"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="graphmind"
IMAGE_TAG="${GRAPHMIND_IMAGE_TAG:-latest}"
REPLICAS="${GRAPHMIND_REPLICAS:-5}"
GPU_ENABLED="${GRAPHMIND_GPU_ENABLED:-true}"

print_section() {
    echo -e "\n${BLUE}🔧 $1${NC}"
    echo "----------------------------------------"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_section "Checking prerequisites"
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed"
        exit 1
    fi
    
    # Check if cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Kubernetes cluster is not accessible"
        exit 1
    fi
    
    # Check if Docker is available for building images
    if ! command -v docker &> /dev/null; then
        print_warning "Docker not available - using pre-built images"
    fi
    
    print_success "Prerequisites check passed"
}

# Build Docker image
build_image() {
    print_section "Building GraphMind Docker image"
    
    if command -v docker &> /dev/null; then
        echo "Building GraphMind image with tag: $IMAGE_TAG"
        
        docker build -t graphmind:$IMAGE_TAG .
        
        if [ $? -eq 0 ]; then
            print_success "Docker image built successfully"
            
            # Tag for registry if needed
            if [ ! -z "$DOCKER_REGISTRY" ]; then
                docker tag graphmind:$IMAGE_TAG $DOCKER_REGISTRY/graphmind:$IMAGE_TAG
                echo "Tagged for registry: $DOCKER_REGISTRY/graphmind:$IMAGE_TAG"
            fi
        else
            print_error "Docker image build failed"
            exit 1
        fi
    else
        print_warning "Docker not available, skipping image build"
    fi
}

# Push image to registry
push_image() {
    if [ ! -z "$DOCKER_REGISTRY" ] && command -v docker &> /dev/null; then
        print_section "Pushing image to registry"
        
        echo "Pushing to registry: $DOCKER_REGISTRY/graphmind:$IMAGE_TAG"
        docker push $DOCKER_REGISTRY/graphmind:$IMAGE_TAG
        
        if [ $? -eq 0 ]; then
            print_success "Image pushed to registry"
        else
            print_error "Failed to push image to registry"
            exit 1
        fi
    fi
}

# Create namespace and resources
create_namespace() {
    print_section "Creating namespace and resources"
    
    echo "Creating namespace: $NAMESPACE"
    kubectl apply -f k8s/namespace.yaml
    
    # Wait for namespace to be ready
    kubectl wait --for=condition=Active namespace/$NAMESPACE --timeout=30s
    
    print_success "Namespace created and active"
}

# Deploy Redis
deploy_redis() {
    print_section "Deploying Redis service discovery"
    
    echo "Deploying Redis master..."
    kubectl apply -f k8s/redis-deployment.yaml
    
    # Wait for Redis to be ready
    kubectl wait --for=condition=Available deployment/redis-master -n $NAMESPACE --timeout=300s
    
    print_success "Redis deployed and ready"
}

# Deploy monitoring stack
deploy_monitoring() {
    print_section "Deploying monitoring stack (Prometheus + Grafana)"
    
    echo "Deploying Prometheus and Grafana..."
    kubectl apply -f k8s/monitoring-stack.yaml
    
    # Wait for deployments to be ready
    kubectl wait --for=condition=Available deployment/prometheus -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=Available deployment/grafana -n $NAMESPACE --timeout=300s
    
    print_success "Monitoring stack deployed"
    
    # Get service URLs
    PROMETHEUS_PORT=$(kubectl get svc prometheus-service -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')
    GRAFANA_PORT=$(kubectl get svc grafana-service -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')
    
    echo "📊 Monitoring URLs:"
    echo "   Prometheus: http://localhost:$PROMETHEUS_PORT"
    echo "   Grafana: http://localhost:$GRAFANA_PORT (admin/graphmind_admin)"
}

# Deploy GraphMind nodes
deploy_graphmind() {
    print_section "Deploying GraphMind distributed nodes"
    
    # Update image in deployment if using custom registry
    if [ ! -z "$DOCKER_REGISTRY" ]; then
        sed -i.bak "s|graphmind:latest|$DOCKER_REGISTRY/graphmind:$IMAGE_TAG|g" k8s/graphmind-deployment.yaml
    elif [ "$IMAGE_TAG" != "latest" ]; then
        sed -i.bak "s|graphmind:latest|graphmind:$IMAGE_TAG|g" k8s/graphmind-deployment.yaml
    fi
    
    # Update replica count
    sed -i.bak "s|replicas: 5|replicas: $REPLICAS|g" k8s/graphmind-deployment.yaml
    
    echo "Deploying GraphMind StatefulSet with $REPLICAS nodes..."
    kubectl apply -f k8s/graphmind-deployment.yaml
    
    # Wait for StatefulSet to be ready
    echo "Waiting for GraphMind nodes to be ready (this may take several minutes)..."
    kubectl wait --for=condition=Ready pod -l app=graphmind -n $NAMESPACE --timeout=600s
    
    # Restore original deployment file
    if [ -f k8s/graphmind-deployment.yaml.bak ]; then
        mv k8s/graphmind-deployment.yaml.bak k8s/graphmind-deployment.yaml
    fi
    
    print_success "GraphMind nodes deployed and ready"
}

# Verify deployment
verify_deployment() {
    print_section "Verifying deployment"
    
    echo "Checking pod status..."
    kubectl get pods -n $NAMESPACE -o wide
    
    echo -e "\nChecking services..."
    kubectl get svc -n $NAMESPACE
    
    echo -e "\nChecking StatefulSet..."
    kubectl get statefulset -n $NAMESPACE
    
    echo -e "\nChecking PVCs..."
    kubectl get pvc -n $NAMESPACE
    
    # Check if all GraphMind nodes are running
    RUNNING_PODS=$(kubectl get pods -n $NAMESPACE -l app=graphmind --field-selector=status.phase=Running --no-headers | wc -l)
    
    if [ "$RUNNING_PODS" -eq "$REPLICAS" ]; then
        print_success "All $REPLICAS GraphMind nodes are running"
    else
        print_warning "Only $RUNNING_PODS out of $REPLICAS nodes are running"
    fi
    
    # Test consensus functionality
    echo -e "\nTesting consensus functionality..."
    POD_NAME=$(kubectl get pods -n $NAMESPACE -l app=graphmind -o jsonpath='{.items[0].metadata.name}')
    
    if [ ! -z "$POD_NAME" ]; then
        kubectl exec $POD_NAME -n $NAMESPACE -- curl -s http://localhost:8080/health > /dev/null
        if [ $? -eq 0 ]; then
            print_success "Health check passed on $POD_NAME"
        else
            print_warning "Health check failed on $POD_NAME"
        fi
    fi
}

# Display access information
display_access_info() {
    print_section "Access Information"
    
    # Get service information
    GRAFANA_PORT=$(kubectl get svc grafana-service -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "N/A")
    PROMETHEUS_PORT=$(kubectl get svc prometheus-service -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "N/A")
    GRAPHMIND_PORT=$(kubectl get svc graphmind-service -n $NAMESPACE -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "N/A")
    
    echo -e "${GREEN}🎉 GraphMind deployment completed successfully!${NC}"
    echo ""
    echo "📊 Monitoring Dashboards:"
    echo "  • Grafana: http://localhost:$GRAFANA_PORT (admin/graphmind_admin)"
    echo "  • Prometheus: http://localhost:$PROMETHEUS_PORT"
    echo ""
    echo "🌐 GraphMind API:"
    echo "  • HTTP: http://localhost:$GRAPHMIND_PORT"
    echo "  • gRPC: localhost:50051"
    echo ""
    echo "🔧 Useful Commands:"
    echo "  • View logs: kubectl logs -f -l app=graphmind -n $NAMESPACE"
    echo "  • Scale nodes: kubectl scale statefulset graphmind-nodes --replicas=7 -n $NAMESPACE"
    echo "  • Port forward: kubectl port-forward svc/grafana-service 3000:3000 -n $NAMESPACE"
    echo "  • Exec into pod: kubectl exec -it graphmind-nodes-0 -n $NAMESPACE -- /bin/bash"
    echo ""
    echo "📈 Next Steps:"
    echo "  1. Access Grafana to view system metrics"
    echo "  2. Run distributed GNN training workloads"
    echo "  3. Scale the cluster based on performance requirements"
    echo ""
    echo -e "${BLUE}Happy distributed computing! 🚀${NC}"
}

# Cleanup function
cleanup() {
    if [ "$1" = "clean" ]; then
        print_section "Cleaning up previous deployment"
        
        kubectl delete namespace $NAMESPACE --ignore-not-found=true
        
        # Wait for namespace deletion
        while kubectl get namespace $NAMESPACE &> /dev/null; do
            echo "Waiting for namespace deletion..."
            sleep 5
        done
        
        print_success "Previous deployment cleaned up"
    fi
}

# Main execution
main() {
    echo "Starting GraphMind Kubernetes deployment..."
    echo "Namespace: $NAMESPACE"
    echo "Image tag: $IMAGE_TAG"
    echo "Replicas: $REPLICAS"
    echo "GPU enabled: $GPU_ENABLED"
    echo ""
    
    # Handle cleanup if requested
    if [ "$1" = "clean" ]; then
        cleanup clean
        echo "Cleanup completed. Re-run without 'clean' to deploy."
        exit 0
    fi
    
    # Main deployment flow
    check_prerequisites
    build_image
    push_image
    create_namespace
    deploy_redis
    deploy_monitoring
    deploy_graphmind
    verify_deployment
    display_access_info
}

# Run main function with arguments
main "$@"