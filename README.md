# GraphMind

Distributed GNN training with Byzantine fault tolerant consensus.

## Overview

GraphMind implements distributed graph neural network training with three core components:

1. **TA-BFT Consensus** - Topology-aware Byzantine fault tolerant consensus using eigenvector centrality for validator weighting
2. **Adaptive Graph Partitioning** - Multi-level partitioning with spectral clustering and simulated annealing
3. **FedTopo Aggregation** - Federated learning with graph Laplacian regularization

## Architecture

```
Edge Nodes (Local GNN Training)
            │
            ▼
    TA-BFT Consensus Network
            │
            ▼
    Global Model Aggregation
    ├── AMP Partitioner
    ├── FedTopo Aggregator
    └── PORA Optimizer
```

## Installation

```bash
git clone https://github.com/elcruzo/GraphMind
cd GraphMind

conda create -n graphmind python=3.9
conda activate graphmind

pip install -r requirements.txt

# MPI for distributed training
sudo apt-get install libopenmpi-dev
pip install mpi4py
```

## Usage

### Distributed Training

```bash
# Single machine
mpirun -np 8 python src/distributed_train.py --config config/training_config.yaml

# Multi-machine cluster
mpirun -f hostfile -np 32 python src/distributed_train.py --config config/training_config.yaml
```

### Byzantine Simulation

```bash
python src/byzantine_simulation.py --failures 2 --nodes 10 --graph cora
```

### Production Mode

```bash
python distributed_node.py --config config/node_config.yaml --node-id node1
```

## Configuration

```yaml
aggregator:
  method: "topology_aware"  # topology_aware, coordinate_median, trimmed_mean
  regularization_strength: 0.1
  personalization_rate: 0.3
  byzantine_robust:
    trim_ratio: 0.2
```

## Algorithms

### TA-BFT Consensus

- Incorporates graph topology into Byzantine agreement
- Eigenvector centrality weighting for validator selection
- Maintains safety under f < n/3 Byzantine failures
- O(n^2) message complexity, O(log n) round complexity

### Adaptive Multi-Level Partitioning

- Dynamic repartitioning based on community evolution
- Multi-objective optimization: cut-size, load balance, communication cost
- Spectral clustering + simulated annealing

### FedTopo

- Structure-aware parameter aggregation using graph Laplacian
- Personalized GNN models with global consensus
- Convergence guarantees under non-IID data

## Performance

| Dataset  | Nodes | Byzantine | TA-BFT Rounds | PBFT Rounds | Speedup |
|----------|-------|-----------|---------------|-------------|---------|
| Cora     | 100   | 10%       | 3.2           | 8.1         | 2.5x    |
| CiteSeer | 500   | 20%       | 4.7           | 12.3        | 2.6x    |
| PubMed   | 1000  | 30%       | 6.1           | 18.7        | 3.1x    |

## Requirements

- Python 3.9+
- PyTorch 2.0+
- torch-geometric 2.4+
- NetworkX 3.0+
- OpenMPI (for distributed training)
- CUDA 11.8+ (optional, for GPU)

## License

MIT
