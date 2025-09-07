#!/bin/bash
# GraphMind Distributed Infrastructure Setup Script
# Sets up the distributed system infrastructure and runs basic tests

set -e  # Exit on any error

echo "🚀 GraphMind Distributed Infrastructure Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GRAPHMIND_HOME=$(pwd)
VENV_NAME="graphmind_env"
REDIS_PORT=6379
NUM_TEST_NODES=3

echo -e "${BLUE}📍 GraphMind Home: ${GRAPHMIND_HOME}${NC}"

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}🔧 $1${NC}"
    echo "----------------------------------------"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Setup Python virtual environment
setup_python_env() {
    print_section "Setting up Python environment"
    
    if command_exists python3; then
        PYTHON_CMD=python3
    elif command_exists python; then
        PYTHON_CMD=python
    else
        print_error "Python not found. Please install Python 3.8 or higher."
        exit 1
    fi
    
    echo "Using Python: $(which $PYTHON_CMD)"
    echo "Python version: $($PYTHON_CMD --version)"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_NAME" ]; then
        echo "Creating virtual environment: $VENV_NAME"
        $PYTHON_CMD -m venv $VENV_NAME
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source $VENV_NAME/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    print_success "Python environment ready"
}

# Install dependencies
install_dependencies() {
    print_section "Installing Python dependencies"
    
    # Install core dependencies
    pip install -r requirements.txt
    
    # Install additional development dependencies
    pip install \
        pytest==7.4.3 \
        pytest-asyncio==0.21.1 \
        pytest-timeout==2.2.0 \
        pytest-xdist==3.5.0 \
        coverage==7.3.2 \
        black==23.11.0 \
        isort==5.12.0 \
        mypy==1.7.1 \
        flake8==6.1.0
    
    print_success "Dependencies installed"
}

# Generate gRPC protocol buffer files
generate_grpc_files() {
    print_section "Generating gRPC protocol buffer files"
    
    cd src/distributed
    
    # Generate Python gRPC files
    python -m grpc_tools.protoc \
        --python_out=. \
        --grpc_python_out=. \
        --proto_path=. \
        consensus.proto
    
    if [ $? -eq 0 ]; then
        print_success "gRPC files generated successfully"
    else
        print_error "Failed to generate gRPC files"
        exit 1
    fi
    
    cd $GRAPHMIND_HOME
}

# Setup Redis for service discovery
setup_redis() {
    print_section "Setting up Redis for service discovery"
    
    if command_exists redis-server; then
        echo "Redis found: $(which redis-server)"
        
        # Check if Redis is running
        if pgrep redis-server >/dev/null; then
            print_warning "Redis is already running"
        else
            echo "Starting Redis server on port $REDIS_PORT..."
            redis-server --port $REDIS_PORT --daemonize yes --logfile redis.log
            sleep 2
            
            if pgrep redis-server >/dev/null; then
                print_success "Redis server started"
            else
                print_error "Failed to start Redis server"
                exit 1
            fi
        fi
        
        # Test Redis connection
        if redis-cli -p $REDIS_PORT ping | grep -q PONG; then
            print_success "Redis connection test passed"
        else
            print_error "Redis connection test failed"
            exit 1
        fi
    else
        print_warning "Redis not found. Please install Redis:"
        echo "  Ubuntu/Debian: sudo apt-get install redis-server"
        echo "  macOS: brew install redis"
        echo "  Or run: docker run -d -p 6379:6379 redis:alpine"
    fi
}

# Create necessary directories
create_directories() {
    print_section "Creating necessary directories"
    
    mkdir -p keys
    mkdir -p logs
    mkdir -p data
    mkdir -p config
    mkdir -p tests/integration
    
    # Set appropriate permissions
    chmod 700 keys  # Secure key directory
    
    print_success "Directories created"
}

# Run basic tests
run_tests() {
    print_section "Running basic tests"
    
    # Run unit tests
    echo "Running unit tests..."
    python -m pytest tests/ -v --timeout=30 -x
    
    if [ $? -eq 0 ]; then
        print_success "Unit tests passed"
    else
        print_error "Unit tests failed"
        exit 1
    fi
}

# Test distributed node startup
test_node_startup() {
    print_section "Testing distributed node startup"
    
    echo "Testing single node startup..."
    
    # Start a test node in background
    python distributed_node.py --node-id test-node-1 --log-level DEBUG &
    NODE_PID=$!
    
    # Wait for startup
    sleep 5
    
    # Check if node is still running
    if kill -0 $NODE_PID 2>/dev/null; then
        print_success "Node startup test passed"
        
        # Stop the test node
        kill $NODE_PID
        wait $NODE_PID 2>/dev/null
        print_success "Node stopped cleanly"
    else
        print_error "Node startup test failed"
        exit 1
    fi
}

# Create sample distributed network
create_sample_network() {
    print_section "Creating sample distributed network configuration"
    
    # Create sample multi-node configuration
    for i in $(seq 1 $NUM_TEST_NODES); do
        port=$((8080 + i - 1))
        grpc_port=$((50051 + i - 1))
        
        cat > config/node${i}_config.yaml << EOF
node:
  hostname: "localhost"
  port: $port
  grpc_port: $grpc_port

discovery:
  backend: "redis"
  backend_config:
    host: "localhost"
    port: 6379

consensus:
  byzantine_threshold: 0.33
  view_timeout: 10.0

byzantine:
  detection_threshold: 0.7
  evidence_window: 100

grpc:
  connection_timeout: 30.0
  max_workers: 10

monitoring:
  logging:
    level: "INFO"
EOF
    done
    
    print_success "Sample network configuration created for $NUM_TEST_NODES nodes"
}

# Create startup script for multi-node testing
create_startup_script() {
    print_section "Creating multi-node startup script"
    
    cat > scripts/start_test_network.sh << 'EOF'
#!/bin/bash
# Start test network with multiple nodes

NUM_NODES=3
PIDS=()

echo "🚀 Starting GraphMind test network with $NUM_NODES nodes"

# Start nodes
for i in $(seq 1 $NUM_NODES); do
    echo "Starting node$i..."
    python distributed_node.py \
        --node-id "node$i" \
        --config "config/node${i}_config.yaml" \
        --log-level INFO &
    
    PIDS+=($!)
    sleep 2
done

echo "✅ All nodes started. PIDs: ${PIDS[*]}"
echo "Press Ctrl+C to stop all nodes"

# Wait for interrupt
trap 'echo "🛑 Stopping all nodes..."; kill ${PIDS[*]}; wait; echo "All nodes stopped."; exit 0' INT
wait
EOF
    
    chmod +x scripts/start_test_network.sh
    print_success "Multi-node startup script created"
}

# Print final instructions
print_instructions() {
    print_section "Setup Complete! 🎉"
    
    echo -e "${GREEN}GraphMind distributed infrastructure is ready!${NC}"
    echo ""
    echo "🔧 What was set up:"
    echo "  • Python virtual environment: $VENV_NAME"
    echo "  • All dependencies installed"
    echo "  • gRPC protocol buffer files generated"
    echo "  • Redis service discovery backend"
    echo "  • Sample configurations for $NUM_TEST_NODES nodes"
    echo "  • Test scripts and logging directories"
    echo ""
    echo "🚀 Next steps:"
    echo ""
    echo "1. Start a single node:"
    echo "   python distributed_node.py --node-id node1"
    echo ""
    echo "2. Start a test network:"
    echo "   ./scripts/start_test_network.sh"
    echo ""
    echo "3. Run integration tests:"
    echo "   python -m pytest tests/integration/ -v"
    echo ""
    echo "4. Monitor logs:"
    echo "   tail -f logs/graphmind.log"
    echo ""
    echo "📚 Configuration files:"
    echo "  • Main config: config/node_config.yaml"
    echo "  • Node configs: config/node{1,2,3}_config.yaml"
    echo ""
    echo "🔧 Useful commands:"
    echo "  • Check Redis: redis-cli ping"
    echo "  • View Redis keys: redis-cli keys 'graphmind*'"
    echo "  • Monitor gRPC: grpcurl -plaintext localhost:50051 list"
    echo ""
    echo -e "${BLUE}Happy distributed computing! 🌐${NC}"
}

# Main execution
main() {
    echo "Starting setup process..."
    
    # Check if we're in the right directory
    if [ ! -f "distributed_node.py" ]; then
        print_error "Please run this script from the GraphMind root directory"
        exit 1
    fi
    
    setup_python_env
    install_dependencies
    generate_grpc_files
    setup_redis
    create_directories
    create_sample_network
    create_startup_script
    run_tests
    test_node_startup
    
    print_instructions
}

# Run main function
main "$@"