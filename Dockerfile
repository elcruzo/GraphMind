# Multi-stage Docker build for GraphMind distributed GNN training
# Optimized for production deployment with minimal image size

# Stage 1: Base environment with system dependencies
FROM nvidia/cuda:11.8-devel-ubuntu22.04 AS base

LABEL maintainer="Ayomide Caleb Adekoya <ayomideadekoya11@gmail.com>"
LABEL version="1.0"
LABEL description="GraphMind: Distributed Graph Neural Networks with Byzantine Fault Tolerant Consensus"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libopenmpi-dev \
    openmpi-bin \
    redis-server \
    supervisor \
    nano \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash graphmind
USER graphmind
WORKDIR /home/graphmind

# Stage 2: Python environment setup
FROM base AS python-env

# Create virtual environment
RUN python3 -m venv /home/graphmind/venv
ENV PATH="/home/graphmind/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
RUN pip install --no-cache-dir \
    torch-geometric==2.4.0 \
    torch-cluster==1.6.3+pt21cu118 \
    torch-scatter==2.1.2+pt21cu118 \
    torch-sparse==0.6.18+pt21cu118 \
    torch-spline-conv==1.2.2+pt21cu118 \
    --find-links https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Stage 3: Application dependencies
FROM python-env AS app-deps

# Copy requirements first for better caching
COPY --chown=graphmind:graphmind requirements.txt /home/graphmind/
RUN pip install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip install --no-cache-dir \
    gunicorn==21.2.0 \
    prometheus-client==0.19.0 \
    structlog==23.2.0 \
    kubernetes==28.1.0

# Stage 4: Application code
FROM app-deps AS app

# Create directory structure
RUN mkdir -p /home/graphmind/app/{src,config,logs,data,keys,scripts}

# Copy application code
COPY --chown=graphmind:graphmind src/ /home/graphmind/app/src/
COPY --chown=graphmind:graphmind config/ /home/graphmind/app/config/
COPY --chown=graphmind:graphmind scripts/ /home/graphmind/app/scripts/
COPY --chown=graphmind:graphmind distributed_node.py /home/graphmind/app/
COPY --chown=graphmind:graphmind setup.py /home/graphmind/app/

# Set working directory
WORKDIR /home/graphmind/app

# Make scripts executable
RUN chmod +x scripts/*.sh

# Install GraphMind in development mode
RUN pip install -e .

# Generate gRPC files
WORKDIR /home/graphmind/app/src/distributed
RUN python -m grpc_tools.protoc \
    --python_out=. \
    --grpc_python_out=. \
    --proto_path=. \
    consensus.proto

WORKDIR /home/graphmind/app

# Stage 5: Production image
FROM app AS production

# Copy supervisor configuration
COPY --chown=graphmind:graphmind docker/supervisord.conf /home/graphmind/

# Copy entrypoint script
COPY --chown=graphmind:graphmind docker/entrypoint.sh /home/graphmind/
RUN chmod +x /home/graphmind/entrypoint.sh

# Health check
COPY --chown=graphmind:graphmind docker/healthcheck.py /home/graphmind/
RUN chmod +x /home/graphmind/healthcheck.py

# Set up health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 /home/graphmind/healthcheck.py

# Expose ports
EXPOSE 8080 50051 9090

# Set environment variables
ENV GRAPHMIND_CONFIG_PATH=/home/graphmind/app/config/node_config.yaml
ENV GRAPHMIND_LOG_LEVEL=INFO
ENV GRAPHMIND_NODE_ID=""

# Volume mounts
VOLUME ["/home/graphmind/app/logs"]
VOLUME ["/home/graphmind/app/data"]
VOLUME ["/home/graphmind/app/keys"]

# Default command
ENTRYPOINT ["/home/graphmind/entrypoint.sh"]
CMD ["distributed_node"]

# Metadata
LABEL org.opencontainers.image.title="GraphMind"
LABEL org.opencontainers.image.description="Distributed Graph Neural Networks with Byzantine Fault Tolerant Consensus"
LABEL org.opencontainers.image.vendor="Ayomide Caleb Adekoya"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.version="1.0"
LABEL org.opencontainers.image.source="https://github.com/elcruzo/GraphMind"