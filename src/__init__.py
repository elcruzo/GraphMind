"""
GraphMind: Distributed Graph Neural Networks with Byzantine Fault Tolerant Consensus

A research-level implementation of distributed GNN training featuring novel consensus 
algorithms, adaptive graph partitioning, and topologically-aware federated learning 
for edge computing environments.
"""

__version__ = "0.1.0"
__author__ = "Ayomide Caleb Adekoya"

# Import main components
from src.consensus.ta_bft import TopologyAwareBFT
from src.partitioning.adaptive_multilevel import AdaptiveMultiLevelPartitioner
from src.federated.fedtopo_aggregator import FederatedTopoAggregator, FederatedGNNModel

# Export main classes
__all__ = [
    "TopologyAwareBFT",
    "AdaptiveMultiLevelPartitioner", 
    "FederatedTopoAggregator",
    "FederatedGNNModel",
]