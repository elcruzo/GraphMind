"""
Metrics collection and tracking utilities for GraphMind
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class Metric:
    """Single metric measurement"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and manages metrics for distributed training
    """
    
    def __init__(self, node_id: int = 0):
        self.node_id = node_id
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.start_time = time.time()
        
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time() - self.start_time,
            tags=tags or {}
        )
        self.metrics[name].append(metric)
    
    def get_metric(self, name: str) -> List[float]:
        """Get all values for a metric"""
        return [m.value for m in self.metrics.get(name, [])]
    
    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value for a metric"""
        values = self.metrics.get(name, [])
        return values[-1].value if values else None
    
    def get_average(self, name: str) -> Optional[float]:
        """Get average value for a metric"""
        values = self.get_metric(name)
        return np.mean(values) if values else None
    
    def get_statistics(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        values = self.get_metric(name)
        if not values:
            return {}
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
    
    def export_json(self, filepath: str):
        """Export metrics to JSON file"""
        export_data = {
            'node_id': self.node_id,
            'duration': time.time() - self.start_time,
            'metrics': {}
        }
        
        for name, metrics in self.metrics.items():
            export_data['metrics'][name] = {
                'values': [m.value for m in metrics],
                'timestamps': [m.timestamp for m in metrics],
                'statistics': self.get_statistics(name)
            }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.start_time = time.time()


class PerformanceTracker:
    """
    Track performance metrics for algorithms
    """
    
    def __init__(self):
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        
    def start_timer(self, name: str) -> 'TimerContext':
        """Start a timer"""
        return TimerContext(self, name)
    
    def record_time(self, name: str, duration: float):
        """Record a time measurement"""
        self.timers[name].append(duration)
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter"""
        self.counters[name] += value
    
    def set_gauge(self, name: str, value: float):
        """Set a gauge value"""
        self.gauges[name] = value
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a timer"""
        times = self.timers.get(name, [])
        if not times:
            return {}
        
        return {
            'count': len(times),
            'total': sum(times),
            'mean': np.mean(times),
            'std': np.std(times),
            'min': min(times),
            'max': max(times),
            'p50': np.percentile(times, 50),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {
            'timers': {},
            'counters': dict(self.counters),
            'gauges': dict(self.gauges)
        }
        
        for name in self.timers:
            summary['timers'][name] = self.get_timer_stats(name)
        
        return summary
    
    def print_summary(self):
        """Print performance summary"""
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        # Timers
        if self.timers:
            print("\nTimers:")
            for name, stats in sorted(self.get_summary()['timers'].items()):
                if stats:
                    print(f"  {name}:")
                    print(f"    Count: {stats['count']}")
                    print(f"    Total: {stats['total']:.3f}s")
                    print(f"    Mean: {stats['mean']:.3f}s")
                    print(f"    P95: {stats['p95']:.3f}s")
        
        # Counters
        if self.counters:
            print("\nCounters:")
            for name, value in sorted(self.counters.items()):
                print(f"  {name}: {value}")
        
        # Gauges
        if self.gauges:
            print("\nGauges:")
            for name, value in sorted(self.gauges.items()):
                print(f"  {name}: {value:.3f}")
        
        print("="*60)


class TimerContext:
    """Context manager for timing code blocks"""
    
    def __init__(self, tracker: PerformanceTracker, name: str):
        self.tracker = tracker
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.tracker.record_time(self.name, duration)