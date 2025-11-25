"""
GraphMind: Distributed Graph Neural Networks with Byzantine Fault Tolerant Consensus

A research-level implementation of distributed GNN training featuring novel consensus 
algorithms, adaptive graph partitioning, and topologically-aware federated learning 
for edge computing environments.
"""

__version__ = "0.1.0"
__author__ = "Ayomide Caleb Adekoya"

__all__ = [
    "TopologyAwareBFT",
    "AdaptiveMultiLevelPartitioner", 
    "FederatedTopoAggregator",
    "FederatedGNNModel",
]


def __getattr__(name):
    """Lazy import to avoid loading torch_geometric unless needed"""
    if name == "TopologyAwareBFT":
        from src.consensus.ta_bft import TopologyAwareBFT
        return TopologyAwareBFT
    elif name == "AdaptiveMultiLevelPartitioner":
        from src.partitioning.adaptive_multilevel import AdaptiveMultiLevelPartitioner
        return AdaptiveMultiLevelPartitioner
    elif name == "FederatedTopoAggregator":
        from src.federated.fedtopo_aggregator import FederatedTopoAggregator
        return FederatedTopoAggregator
    elif name == "FederatedGNNModel":
        from src.federated.fedtopo_aggregator import FederatedGNNModel
        return FederatedGNNModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
