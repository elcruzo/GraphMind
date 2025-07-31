#!/bin/bash
set -e

# GraphMind Entrypoint Script

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Default environment variables
export PYTHONPATH=/app
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# Create necessary directories
mkdir -p /app/logs /app/results /app/data

log "Starting GraphMind with command: $@"

case "$1" in
    "consensus-benchmark")
        log "Running consensus algorithm benchmark..."
        mpirun -np ${NUM_NODES:-4} python src/benchmarks/consensus_benchmark.py
        ;;
    
    "partitioning-benchmark")
        log "Running graph partitioning benchmark..."
        python src/benchmarks/partitioning_benchmark.py
        ;;
        
    "distributed-train")
        log "Starting distributed GNN training..."
        mpirun -np ${NUM_NODES:-4} python src/distributed_train.py --config config/training_config.yaml
        ;;
        
    "research-experiment")
        log "Running research experiment..."
        python src/research/run_experiment.py --config ${EXPERIMENT_CONFIG:-config/research_config.yaml}
        ;;
        
    *)
        log "Unknown command: $1"
        log "Available commands: consensus-benchmark, partitioning-benchmark, distributed-train, research-experiment"
        exit 1
        ;;
esac