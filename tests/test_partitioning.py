"""
Unit tests for Adaptive Multi-Level Partitioning (AMP) algorithm

Tests the correctness and performance of the graph partitioning implementation.
"""

import pytest
import numpy as np
import networkx as nx
import time
from typing import Dict, List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.partitioning.adaptive_multilevel import (
    AdaptiveMultiLevelPartitioner,
    PartitionConstraints,
    PartitionResult,
    PartitionMetrics
)


class TestAdaptiveMultiLevelPartitioner:
    """Test suite for AMP algorithm"""
    
    @pytest.fixture
    def create_test_graph(self) -> nx.Graph:
        """Create a simple test graph"""
        # Create a graph with known structure
        G = nx.Graph()
        # Add two loosely connected clusters
        # Cluster 1: nodes 0-4
        for i in range(5):
            for j in range(i+1, 5):
                G.add_edge(i, j)
        
        # Cluster 2: nodes 5-9
        for i in range(5, 10):
            for j in range(i+1, 10):
                G.add_edge(i, j)
        
        # Connect clusters with single edge
        G.add_edge(4, 5)
        
        return G
    
    @pytest.fixture
    def create_large_test_graph(self) -> nx.Graph:
        """Create a larger test graph for performance testing"""
        # Create a graph with community structure
        return nx.generators.community.planted_partition_graph(
            [20, 20, 20, 20],  # 4 communities of 20 nodes each
            0.8,  # High intra-community edge probability
            0.05  # Low inter-community edge probability
        )
    
    def test_initialization(self):
        """Test partitioner initialization"""
        partitioner = AdaptiveMultiLevelPartitioner(
            objective_weights={'cut': 0.5, 'balance': 0.3, 'communication': 0.2},
            coarsening_threshold=500,
            refinement_iterations=20
        )
        
        assert partitioner.objective_weights['cut'] == 0.5
        assert partitioner.objective_weights['balance'] == 0.3
        assert partitioner.objective_weights['communication'] == 0.2
        assert partitioner.coarsening_threshold == 500
        assert partitioner.refinement_iterations == 20
    
    def test_basic_partitioning(self, create_test_graph):
        """Test basic graph partitioning"""
        graph = create_test_graph
        partitioner = AdaptiveMultiLevelPartitioner()
        
        # Partition into 2 parts
        result = partitioner.partition(graph, num_partitions=2)
        
        assert isinstance(result, PartitionResult)
        assert result.num_partitions == 2
        assert len(result.partition_assignment) == graph.number_of_nodes()
        
        # Check all nodes are assigned
        for node in graph.nodes():
            assert node in result.partition_assignment
            assert 0 <= result.partition_assignment[node] < 2
        
        # For this graph structure, cut size should be small (ideally 1)
        assert result.cut_size <= 3  # Allow some flexibility
    
    def test_partition_constraints(self, create_test_graph):
        """Test partitioning with constraints"""
        graph = create_test_graph
        partitioner = AdaptiveMultiLevelPartitioner()
        
        # Add node weights
        node_weights = {i: 1.0 + 0.1 * i for i in graph.nodes()}
        
        constraints = PartitionConstraints(
            max_load_imbalance=0.1,
            min_partition_size=3,
            node_weights=node_weights
        )
        
        result = partitioner.partition(graph, num_partitions=2, constraints=constraints)
        
        # Check load balance
        assert result.load_balance >= 0.9  # Max 10% imbalance
        
        # Check partition sizes
        partition_sizes = {}
        for node, part in result.partition_assignment.items():
            if part not in partition_sizes:
                partition_sizes[part] = 0
            partition_sizes[part] += 1
        
        for size in partition_sizes.values():
            assert size >= constraints.min_partition_size
    
    def test_multi_level_coarsening(self, create_large_test_graph):
        """Test multi-level coarsening process"""
        graph = create_large_test_graph
        partitioner = AdaptiveMultiLevelPartitioner(coarsening_threshold=20)
        
        # Test coarsening
        hierarchy = partitioner._multilevel_coarsening(graph, PartitionConstraints())
        
        # Check hierarchy properties
        assert len(hierarchy) > 1  # Should have multiple levels
        assert hierarchy[0].number_of_nodes() == graph.number_of_nodes()
        
        # Each level should be smaller than previous
        for i in range(1, len(hierarchy)):
            assert hierarchy[i].number_of_nodes() < hierarchy[i-1].number_of_nodes()
        
        # Coarsest graph should be small enough
        assert hierarchy[-1].number_of_nodes() <= partitioner.coarsening_threshold
    
    def test_heavy_edge_matching(self, create_test_graph):
        """Test heavy-edge matching algorithm"""
        graph = create_test_graph
        partitioner = AdaptiveMultiLevelPartitioner()
        
        # Add edge weights
        for u, v in graph.edges():
            graph[u][v]['weight'] = 1.0 + abs(u - v) * 0.1
        
        constraints = PartitionConstraints(
            edge_weights={(u, v): graph[u][v]['weight'] for u, v in graph.edges()}
        )
        
        matching = partitioner._compute_heavy_edge_matching(graph, constraints)
        
        # Check matching properties
        matched_nodes = set()
        for u, v in matching:
            # No node should be matched twice
            assert u not in matched_nodes
            assert v not in matched_nodes
            matched_nodes.add(u)
            matched_nodes.add(v)
            
            # Matched edge should exist
            assert graph.has_edge(u, v) or graph.has_edge(v, u)
    
    def test_spectral_clustering(self, create_test_graph):
        """Test spectral clustering component"""
        graph = create_test_graph
        partitioner = AdaptiveMultiLevelPartitioner()
        
        # Test spectral clustering
        partition = partitioner._spectral_clustering(graph, num_partitions=2)
        
        # Check partition validity
        assert len(partition) == graph.number_of_nodes()
        for node in graph.nodes():
            assert node in partition
            assert 0 <= partition[node] < 2
    
    def test_simulated_annealing(self, create_test_graph):
        """Test simulated annealing optimization"""
        graph = create_test_graph
        partitioner = AdaptiveMultiLevelPartitioner(
            annealing_schedule={
                'initial_temp': 10.0,
                'cooling_rate': 0.95,
                'min_temp': 0.1,
                'max_iterations': 100
            }
        )
        
        # Create initial partition
        initial_partition = {i: i % 2 for i in graph.nodes()}
        
        # Run annealing
        optimized = partitioner._simulated_annealing_optimization(
            graph, initial_partition, PartitionConstraints()
        )
        
        # Score should improve
        initial_score = partitioner._compute_objective_score(
            graph, initial_partition, PartitionConstraints()
        )
        optimized_score = partitioner._compute_objective_score(
            graph, optimized, PartitionConstraints()
        )
        
        assert optimized_score >= initial_score
    
    def test_objective_score_computation(self, create_test_graph):
        """Test multi-objective score computation"""
        graph = create_test_graph
        partitioner = AdaptiveMultiLevelPartitioner()
        
        # Perfect partition (separating two clusters)
        perfect_partition = {i: 0 if i < 5 else 1 for i in graph.nodes()}
        
        # Bad partition (splitting clusters)
        bad_partition = {i: i % 2 for i in graph.nodes()}
        
        perfect_score = partitioner._compute_objective_score(
            graph, perfect_partition, PartitionConstraints()
        )
        bad_score = partitioner._compute_objective_score(
            graph, bad_partition, PartitionConstraints()
        )
        
        # Perfect partition should have better score
        assert perfect_score > bad_score
    
    def test_partition_metrics(self, create_test_graph):
        """Test partition quality metrics computation"""
        graph = create_test_graph
        partitioner = AdaptiveMultiLevelPartitioner()
        
        partition = {i: 0 if i < 5 else 1 for i in graph.nodes()}
        metrics = partitioner._compute_partition_metrics(
            graph, partition, PartitionConstraints()
        )
        
        assert isinstance(metrics, PartitionMetrics)
        assert 0 <= metrics.cut_ratio <= 1
        assert metrics.edge_cut == 1  # Only one edge between clusters
        assert metrics.max_partition_size == 5
        assert metrics.min_partition_size == 5
        assert metrics.size_variance == 0  # Equal sizes
    
    def test_different_partition_counts(self, create_large_test_graph):
        """Test partitioning with different numbers of partitions"""
        graph = create_large_test_graph
        partitioner = AdaptiveMultiLevelPartitioner()
        
        for num_partitions in [2, 4, 8]:
            result = partitioner.partition(graph, num_partitions)
            
            # Check correct number of partitions
            unique_partitions = set(result.partition_assignment.values())
            assert len(unique_partitions) == num_partitions
            
            # All partitions should be used
            for i in range(num_partitions):
                assert i in unique_partitions
    
    def test_performance_metrics(self, create_large_test_graph):
        """Test performance metric collection"""
        graph = create_large_test_graph
        partitioner = AdaptiveMultiLevelPartitioner()
        
        # Run multiple partitions
        for _ in range(3):
            partitioner.partition(graph, num_partitions=4)
        
        metrics = partitioner.get_performance_metrics()
        
        assert metrics['total_partitions'] == 3
        assert 'avg_partitioning_time' in metrics
        assert 'avg_coarsening_time' in metrics
        assert 'avg_refinement_time' in metrics
        assert len(metrics['best_objective_scores']) == 3
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        partitioner = AdaptiveMultiLevelPartitioner()
        
        # Empty graph
        empty_graph = nx.Graph()
        result = partitioner.partition(empty_graph, 2)
        assert len(result.partition_assignment) == 0
        
        # Single node graph
        single_node = nx.Graph()
        single_node.add_node(0)
        result = partitioner.partition(single_node, 2)
        assert result.partition_assignment[0] in [0, 1]
        
        # Complete graph (worst case for partitioning)
        complete = nx.complete_graph(10)
        result = partitioner.partition(complete, 2)
        assert result.cut_size > 0  # Must cut some edges
    
    @pytest.mark.parametrize("graph_type,expected_quality", [
        ("grid", 0.8),      # Grid graphs partition well
        ("random", 0.5),    # Random graphs partition moderately
        ("complete", 0.3),  # Complete graphs partition poorly
    ])
    def test_different_graph_types(self, graph_type, expected_quality):
        """Test partitioning on different graph types"""
        if graph_type == "grid":
            graph = nx.grid_2d_graph(10, 10)
            # Relabel to integers
            mapping = {node: i for i, node in enumerate(graph.nodes())}
            graph = nx.relabel_nodes(graph, mapping)
        elif graph_type == "random":
            graph = nx.erdos_renyi_graph(100, 0.1)
        elif graph_type == "complete":
            graph = nx.complete_graph(50)
        
        partitioner = AdaptiveMultiLevelPartitioner()
        result = partitioner.partition(graph, num_partitions=4)
        
        # Check partition quality
        quality_score = result.load_balance * (1 - result.cut_size / graph.number_of_edges())
        assert quality_score >= expected_quality * 0.5  # Allow some variance


class TestPartitioningIntegration:
    """Integration tests for partitioning with other components"""
    
    def test_scalability(self):
        """Test partitioning scalability"""
        partitioner = AdaptiveMultiLevelPartitioner()
        
        sizes = [100, 500, 1000]
        times = []
        
        for size in sizes:
            graph = nx.barabasi_albert_graph(size, 3)
            
            start_time = time.time()
            result = partitioner.partition(graph, num_partitions=8)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Basic validity checks
            assert len(result.partition_assignment) == size
            assert result.num_partitions == 8
        
        # Check time scaling (should be roughly O(n log n))
        # Allow for some variance
        for i in range(1, len(times)):
            ratio = times[i] / times[i-1]
            size_ratio = sizes[i] / sizes[i-1]
            expected_ratio = size_ratio * np.log(sizes[i]) / np.log(sizes[i-1])
            assert ratio < expected_ratio * 3  # Allow 3x variance
    
    def test_repeatability(self, create_test_graph):
        """Test that partitioning is deterministic with same inputs"""
        graph = create_test_graph
        partitioner1 = AdaptiveMultiLevelPartitioner()
        partitioner2 = AdaptiveMultiLevelPartitioner()
        
        # Note: Due to randomness in spectral clustering and annealing,
        # results may vary. In production, would set random seeds.
        result1 = partitioner1.partition(graph, 2)
        result2 = partitioner2.partition(graph, 2)
        
        # At least check same quality metrics
        assert abs(result1.objective_score - result2.objective_score) < 0.2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])