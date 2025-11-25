"""Graph partitioning algorithms for GraphMind"""

__all__ = ['AdaptiveMultiLevelPartitioner', 'PartitionConstraints', 'PartitionResult']


def __getattr__(name):
    if name in __all__:
        from . import adaptive_multilevel
        return getattr(adaptive_multilevel, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
