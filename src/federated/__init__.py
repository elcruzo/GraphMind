"""Federated learning components for GraphMind"""

__all__ = ['FederatedTopoAggregator', 'FederatedGNNModel']


def __getattr__(name):
    if name in __all__:
        from . import fedtopo_aggregator
        return getattr(fedtopo_aggregator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
