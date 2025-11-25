"""
Distributed system components for GraphMind

This package provides distributed infrastructure for Byzantine fault-tolerant
graph neural network training with consensus algorithms and node management.
"""

__all__ = [
    'NodeDiscoveryService',
    'DistributedNodeManager', 
    'NodeInfo',
    'NodeStatus',
    'ServiceBackend',
    'HealthCheck'
]


def __getattr__(name):
    """Lazy import to avoid loading optional dependencies"""
    if name in __all__:
        from . import node_discovery
        return getattr(node_discovery, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
