"""
Byzantine Fault Tolerance Components for GraphMind

This package provides comprehensive Byzantine fault detection, evidence collection,
and tolerance mechanisms for distributed graph neural network training.
"""

__all__ = [
    'ByzantineFaultDetector',
    'ByzantineToleranceManager',
    'ByzantineEvidence', 
    'NodeBehaviorProfile',
    'FaultType',
    'EvidenceType'
]


def __getattr__(name):
    """Lazy import to avoid loading optional dependencies"""
    if name in __all__:
        from . import fault_detector
        return getattr(fault_detector, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
