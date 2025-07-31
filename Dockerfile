# Multi-stage build for GraphMind
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as builder

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libblas-dev \
    liblapack-dev \
    libopenmpi-dev \
    openmpi-bin \
    openmpi-common \
    libhdf5-dev \
    libgraphviz-dev \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools
RUN pip3 install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir -r requirements.txt

# Production stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as production

# Install runtime dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    curl \
    git \
    libopenmpi3 \
    openmpi-bin \
    libhdf5-103 \
    libgraphviz4 \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Create non-root user
RUN groupadd -r graphmind && useradd -r -g graphmind graphmind

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/dist-packages /usr/local/lib/python3.9/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY research/ ./research/
COPY benchmarks/ ./benchmarks/
COPY docker/entrypoint.sh ./entrypoint.sh

# Create directories for data, results, and logs
RUN mkdir -p /app/data /app/results /app/logs /app/checkpoints && \
    chown -R graphmind:graphmind /app

# Make entrypoint executable
RUN chmod +x ./entrypoint.sh

# Switch to non-root user
USER graphmind

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# MPI environment
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Expose ports for distributed computing
EXPOSE 23456 23457 23458

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD python src/utils/health_check.py || exit 1

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# Default command (consensus benchmark)
CMD ["consensus-benchmark"]