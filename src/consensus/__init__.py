"""Consensus algorithms for GraphMind"""

__all__ = ['TopologyAwareBFT']


def __getattr__(name):
    if name == 'TopologyAwareBFT':
        from .ta_bft import TopologyAwareBFT
        return TopologyAwareBFT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
