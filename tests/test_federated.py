"""
Unit tests for Federated GNN with Topological Aggregation

Tests the FedTopo algorithm implementation including privacy preservation
and convergence properties.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Dict, List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.federated.fedtopo_aggregator import (
    FederatedTopoAggregator,
    FederatedGNNModel,
    GraphLaplacianRegularizer,
    FederatedNode,
    AggregationResult
)
from torch_geometric.data import Data


class TestGraphLaplacianRegularizer:
    """Test suite for Graph Laplacian regularization"""
    
    @pytest.fixture
    def create_test_graph(self) -> nx.Graph:
        """Create a simple test graph"""
        return nx.cycle_graph(4)
    
    def test_initialization(self, create_test_graph):
        """Test regularizer initialization"""
        graph = create_test_graph
        regularizer = GraphLaplacianRegularizer(graph, regularization_strength=0.1)
        
        assert regularizer.regularization_strength == 0.1
        assert regularizer.laplacian is not None
        assert regularizer.laplacian.shape == (4, 4)
    
    def test_laplacian_computation(self, create_test_graph):
        """Test normalized Laplacian computation"""
        graph = create_test_graph
        regularizer = GraphLaplacianRegularizer(graph)
        
        laplacian = regularizer.laplacian
        
        # Check Laplacian properties
        # 1. Symmetric
        assert torch.allclose(laplacian, laplacian.t())
        
        # 2. Row sums should be close to 0 (for normalized Laplacian, close to 1)
        # 3. Diagonal elements should be positive
        for i in range(laplacian.shape[0]):
            assert laplacian[i, i] > 0
    
    def test_structural_loss(self, create_test_graph):
        """Test structural loss computation"""
        graph = create_test_graph
        regularizer = GraphLaplacianRegularizer(graph, regularization_strength=0.1)
        
        # Create test parameters (4 nodes, 10-dim parameters)
        params = [torch.randn(10) for _ in range(4)]
        
        loss = regularizer.compute_structural_loss(params)
        
        assert isinstance(loss, float)
        assert loss >= 0  # Loss should be non-negative
        
        # Test with identical parameters (should have zero loss)
        identical_params = [torch.ones(10) for _ in range(4)]
        identical_loss = regularizer.compute_structural_loss(identical_params)
        assert identical_loss < loss  # Should be smaller


class TestFederatedTopoAggregator:
    """Test suite for FedTopo aggregator"""
    
    @pytest.fixture
    def create_network_topology(self) -> nx.Graph:
        """Create test network topology"""
        return nx.complete_graph(5)
    
    @pytest.fixture
    def create_local_parameters(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Create mock local parameters from nodes"""
        params = {}
        for node_id in range(5):
            params[node_id] = {
                'layer1.weight': torch.randn(64, 32),
                'layer1.bias': torch.randn(64),
                'layer2.weight': torch.randn(10, 64),
                'layer2.bias': torch.randn(10)
            }
        return params
    
    def test_initialization(self, create_network_topology):
        """Test aggregator initialization"""
        graph = create_network_topology
        aggregator = FederatedTopoAggregator(
            graph_structure=graph,
            privacy_budget=1.0,
            regularization_strength=0.1,
            personalization_rate=0.3
        )
        
        assert aggregator.privacy_budget == 1.0
        assert aggregator.personalization_rate == 0.3
        assert len(aggregator.topology_weights) == 5
        
        # Check topology weights sum to 1
        total_weight = sum(aggregator.topology_weights.values())
        assert abs(total_weight - 1.0) < 1e-6
    
    def test_topology_weight_computation(self):
        """Test topology-aware weight computation"""
        # Star topology - center should have higher weight
        star = nx.star_graph(4)
        aggregator = FederatedTopoAggregator(star)
        
        weights = aggregator.topology_weights
        
        # Center node (0) should have highest weight
        center_weight = weights[0]
        for node in range(1, 5):
            assert center_weight > weights[node]
    
    def test_aggregation_weights(self, create_network_topology):
        """Test aggregation weight computation"""
        graph = create_network_topology
        aggregator = FederatedTopoAggregator(graph)
        
        node_capabilities = {i: 0.8 + 0.05 * i for i in range(5)}
        
        weights = aggregator._compute_aggregation_weights(node_capabilities, round_number=10)
        
        # Check all nodes have weights
        assert len(weights) == 5
        
        # Weights should incorporate capabilities
        # Higher capability nodes should have higher weights (generally)
        assert weights[4] >= weights[0] * 0.8  # Allow some variance
    
    def test_privacy_noise_addition(self, create_network_topology, create_local_parameters):
        """Test differential privacy noise addition"""
        graph = create_network_topology
        aggregator = FederatedTopoAggregator(graph, privacy_budget=1.0)
        
        local_params = create_local_parameters
        
        # Add privacy noise
        private_params = aggregator._add_privacy_noise(local_params)
        
        # Check noise was added
        for node_id in local_params:
            for layer_name in local_params[node_id]:
                original = local_params[node_id][layer_name]
                private = private_params[node_id][layer_name]
                
                # Should not be identical (noise added)
                assert not torch.allclose(original, private)
                
                # But should be similar (controlled noise)
                diff = torch.norm(original - private)
                assert diff < torch.norm(original) * 2  # Noise shouldn't dominate
    
    def test_parameter_aggregation(self, create_network_topology, create_local_parameters):
        """Test full parameter aggregation"""
        graph = create_network_topology
        aggregator = FederatedTopoAggregator(graph)
        
        local_params = create_local_parameters
        node_capabilities = {i: 1.0 for i in range(5)}
        
        result = aggregator.aggregate_parameters(
            local_params,
            node_capabilities,
            round_number=1
        )
        
        assert isinstance(result, AggregationResult)
        assert result.global_parameters is not None
        assert len(result.global_parameters) == 4  # 4 layers
        assert result.privacy_spent > 0
        assert result.structural_loss >= 0
        assert result.convergence_metric >= 0
    
    def test_personalization(self, create_network_topology):
        """Test model personalization"""
        graph = create_network_topology
        aggregator = FederatedTopoAggregator(graph, personalization_rate=0.3)
        
        # Create global and local parameters
        global_params = {
            'layer1.weight': torch.ones(10, 10),
            'layer1.bias': torch.ones(10)
        }
        
        local_params = {
            'layer1.weight': torch.zeros(10, 10),
            'layer1.bias': torch.zeros(10)
        }
        
        # Apply personalization for high-degree node (node 0 in complete graph)
        personalized = aggregator.apply_personalization(global_params, local_params, node_id=0)
        
        # Should be weighted combination
        for layer_name in global_params:
            pers_param = personalized[layer_name]
            # Should be between global and local values
            assert torch.all(pers_param >= 0)
            assert torch.all(pers_param <= 1)
            # Should not be exactly global or local
            assert not torch.allclose(pers_param, global_params[layer_name])
            assert not torch.allclose(pers_param, local_params[layer_name])
    
    def test_convergence_analysis(self, create_network_topology):
        """Test convergence analysis functionality"""
        graph = create_network_topology
        aggregator = FederatedTopoAggregator(graph)
        
        # Simulate decreasing convergence metrics
        aggregator.metrics['convergence_history'] = [1.0, 0.8, 0.6, 0.4, 0.2]
        
        analysis = aggregator.get_convergence_analysis()
        
        assert 'converged' in analysis
        assert 'convergence_rate' in analysis
        assert 'estimated_rounds_to_convergence' in analysis
        
        # With decreasing values, should show positive convergence
        assert analysis['convergence_rate'] > 0
    
    def test_metrics_reset(self, create_network_topology):
        """Test metrics reset functionality"""
        graph = create_network_topology
        aggregator = FederatedTopoAggregator(graph)
        
        # Add some metrics
        aggregator.metrics['aggregation_rounds'] = 10
        aggregator.metrics['convergence_history'] = [1.0, 0.5]
        
        # Reset
        aggregator.reset_metrics()
        
        assert aggregator.metrics['aggregation_rounds'] == 0
        assert len(aggregator.metrics['convergence_history']) == 0


class TestFederatedGNNModel:
    """Test suite for Federated GNN model"""
    
    def test_model_initialization(self):
        """Test GNN model initialization"""
        model = FederatedGNNModel(
            input_dim=32,
            hidden_dim=64,
            output_dim=10,
            num_layers=2,
            model_type='gcn',
            dropout=0.5
        )
        
        assert len(model.convs) == 2
        assert len(model.batch_norms) == 2
    
    def test_forward_pass(self):
        """Test model forward pass"""
        model = FederatedGNNModel(
            input_dim=32,
            hidden_dim=64,
            output_dim=10,
            num_layers=2,
            model_type='gcn'
        )
        
        # Create test data
        x = torch.randn(20, 32)  # 20 nodes, 32 features
        edge_index = torch.randint(0, 20, (2, 50))  # 50 edges
        
        # Forward pass
        out = model(x, edge_index)
        
        assert out.shape == (20, 10)  # Output dimension should match
    
    def test_parameter_dict_operations(self):
        """Test getting and setting parameters as dictionary"""
        model = FederatedGNNModel(
            input_dim=32,
            hidden_dim=64,
            output_dim=10,
            num_layers=2
        )
        
        # Get parameters
        param_dict = model.get_parameters_dict()
        
        assert isinstance(param_dict, dict)
        assert len(param_dict) > 0
        
        # Modify parameters
        for key in param_dict:
            param_dict[key] = torch.zeros_like(param_dict[key])
        
        # Set parameters
        model.set_parameters_dict(param_dict)
        
        # Verify parameters were set
        new_param_dict = model.get_parameters_dict()
        for key in param_dict:
            assert torch.allclose(new_param_dict[key], torch.zeros_like(new_param_dict[key]))
    
    @pytest.mark.parametrize("model_type", ["gcn", "gat", "sage"])
    def test_different_model_types(self, model_type):
        """Test different GNN architectures"""
        model = FederatedGNNModel(
            input_dim=32,
            hidden_dim=64,
            output_dim=10,
            num_layers=2,
            model_type=model_type
        )
        
        x = torch.randn(20, 32)
        edge_index = torch.randint(0, 20, (2, 50))
        
        out = model(x, edge_index)
        assert out.shape == (20, 10)


class TestFederatedIntegration:
    """Integration tests for federated learning system"""
    
    def test_full_federated_round(self):
        """Test complete federated learning round"""
        # Create network
        network = nx.cycle_graph(4)
        
        # Create models for each node
        models = {}
        for i in range(4):
            models[i] = FederatedGNNModel(
                input_dim=16,
                hidden_dim=32,
                output_dim=4,
                num_layers=2
            )
        
        # Create aggregator
        aggregator = FederatedTopoAggregator(
            graph_structure=network,
            privacy_budget=1.0
        )
        
        # Get local parameters
        local_params = {
            i: models[i].get_parameters_dict()
            for i in range(4)
        }
        
        # Run aggregation
        node_capabilities = {i: 1.0 for i in range(4)}
        result = aggregator.aggregate_parameters(
            local_params,
            node_capabilities,
            round_number=1
        )
        
        # Apply personalized updates
        for i in range(4):
            personalized = aggregator.apply_personalization(
                result.global_parameters,
                local_params[i],
                node_id=i
            )
            models[i].set_parameters_dict(personalized)
        
        # Verify models were updated
        new_params = models[0].get_parameters_dict()
        for key in new_params:
            assert not torch.allclose(new_params[key], local_params[0][key])
    
    def test_convergence_over_rounds(self):
        """Test convergence behavior over multiple rounds"""
        network = nx.complete_graph(3)
        aggregator = FederatedTopoAggregator(network)
        
        # Simulate multiple rounds with decreasing variance
        for round_num in range(5):
            # Create parameters with decreasing variance
            variance = 1.0 / (round_num + 1)
            local_params = {}
            
            for node_id in range(3):
                base_weight = torch.ones(10, 10)
                noise = torch.randn(10, 10) * variance
                local_params[node_id] = {
                    'weight': base_weight + noise
                }
            
            # Aggregate
            result = aggregator.aggregate_parameters(
                local_params,
                {i: 1.0 for i in range(3)},
                round_num
            )
        
        # Check convergence metrics
        analysis = aggregator.get_convergence_analysis()
        
        # Should show improvement over rounds
        history = aggregator.metrics['convergence_history']
        assert len(history) == 5
        
        # Later rounds should have lower convergence metric (closer consensus)
        assert history[-1] < history[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])