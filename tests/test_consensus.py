"""
Unit tests for TA-BFT consensus algorithm

Tests the correctness and performance of the Topology-Aware Byzantine
Fault Tolerant consensus implementation.
"""

import pytest
import asyncio
import numpy as np
import networkx as nx
from typing import Dict, List
from cryptography.hazmat.primitives.asymmetric import rsa

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.consensus.ta_bft import TopologyAwareBFT, ConsensusMessage, ConsensusResult, MessageType


class TestTopologyAwareBFT:
    """Test suite for TA-BFT consensus algorithm"""
    
    @pytest.fixture
    def create_network_keys(self, num_nodes: int = 4) -> Dict[int, tuple]:
        """Create RSA key pairs for test network"""
        keys = {}
        public_keys = {}
        
        for i in range(num_nodes):
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            public_key = private_key.public_key()
            keys[i] = private_key
            public_keys[i] = public_key
            
        return keys, public_keys
    
    @pytest.fixture
    def create_test_topology(self, num_nodes: int = 4) -> nx.Graph:
        """Create test network topology"""
        # Create a simple ring topology
        return nx.cycle_graph(num_nodes)
    
    def test_initialization(self, create_network_keys, create_test_topology):
        """Test TA-BFT initialization"""
        num_nodes = 4
        keys, public_keys = create_network_keys
        topology = create_test_topology
        
        # Initialize consensus for node 0
        consensus = TopologyAwareBFT(
            node_id=0,
            graph_topology=topology,
            private_key=keys[0],
            public_keys=public_keys,
            byzantine_threshold=1/3
        )
        
        assert consensus.node_id == 0
        assert consensus.byzantine_threshold == 1/3
        assert len(consensus.centrality_weights) == num_nodes
        assert consensus.quorum_size >= 3  # For 4 nodes with f=1, quorum >= 2f+1 = 3
    
    def test_centrality_computation(self, create_test_topology):
        """Test eigenvector centrality computation"""
        topology = create_test_topology
        
        # For a ring topology, all nodes should have equal centrality
        keys, public_keys = create_network_keys
        consensus = TopologyAwareBFT(
            node_id=0,
            graph_topology=topology,
            private_key=keys[0],
            public_keys=public_keys
        )
        
        weights = consensus.centrality_weights
        
        # Check weights sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        
        # For ring topology, weights should be approximately equal
        expected_weight = 1.0 / len(topology)
        for weight in weights.values():
            assert abs(weight - expected_weight) < 0.1
    
    def test_topology_fingerprint(self, create_network_keys):
        """Test topology fingerprint computation"""
        # Create two identical topologies
        topology1 = nx.cycle_graph(4)
        topology2 = nx.cycle_graph(4)
        
        keys, public_keys = create_network_keys
        
        consensus1 = TopologyAwareBFT(0, topology1, keys[0], public_keys)
        consensus2 = TopologyAwareBFT(1, topology2, keys[1], public_keys)
        
        # Fingerprints should match for identical topologies
        assert consensus1.topology_fingerprint == consensus2.topology_fingerprint
        
        # Different topology should have different fingerprint
        topology3 = nx.complete_graph(4)
        consensus3 = TopologyAwareBFT(2, topology3, keys[2], public_keys)
        assert consensus1.topology_fingerprint != consensus3.topology_fingerprint
    
    def test_message_signing_and_verification(self, create_network_keys, create_test_topology):
        """Test message signing and verification"""
        keys, public_keys = create_network_keys
        topology = create_test_topology
        
        consensus = TopologyAwareBFT(0, topology, keys[0], public_keys)
        
        # Create a test message
        message = ConsensusMessage(
            msg_type=MessageType.PREPARE,
            view=0,
            sequence=1,
            proposal="test_value",
            sender_id=0,
            timestamp=123456.789,
            topology_proof=consensus._create_topology_proof()
        )
        
        # Sign the message
        signature = consensus._sign_message(message)
        assert signature is not None
        assert len(signature) > 0
        
        # Verify the signature
        message.signature = signature
        assert consensus._verify_signature(message, signature) is True
        
        # Verify with wrong signature should fail
        bad_signature = b"invalid_signature"
        assert consensus._verify_signature(message, bad_signature) is False
    
    def test_topology_proof_validation(self, create_network_keys, create_test_topology):
        """Test topology proof validation"""
        keys, public_keys = create_network_keys
        topology = create_test_topology
        
        consensus0 = TopologyAwareBFT(0, topology, keys[0], public_keys)
        consensus1 = TopologyAwareBFT(1, topology, keys[1], public_keys)
        
        # Create message with valid topology proof
        message = ConsensusMessage(
            msg_type=MessageType.PREPARE,
            view=0,
            sequence=1,
            proposal="test",
            sender_id=0,
            timestamp=123456.789,
            topology_proof=consensus0._create_topology_proof()
        )
        
        # Should validate successfully on node with same topology
        assert consensus1._validate_topology_proof(message) is True
        
        # Invalid topology proof should fail
        message.topology_proof['fingerprint'] = "invalid_fingerprint"
        assert consensus1._validate_topology_proof(message) is False
    
    def test_quorum_computation(self):
        """Test topology-aware quorum computation"""
        # Test different network sizes and Byzantine thresholds
        test_cases = [
            (4, 1/3, 3),   # 4 nodes, f=1, quorum=3
            (7, 1/3, 5),   # 7 nodes, f=2, quorum=5
            (10, 1/3, 7),  # 10 nodes, f=3, quorum=7
        ]
        
        for num_nodes, byzantine_threshold, expected_min_quorum in test_cases:
            topology = nx.cycle_graph(num_nodes)
            keys, public_keys = create_network_keys(num_nodes)
            
            consensus = TopologyAwareBFT(
                0, topology, keys[0], public_keys,
                byzantine_threshold=byzantine_threshold
            )
            
            assert consensus.quorum_size >= expected_min_quorum
    
    @pytest.mark.asyncio
    async def test_consensus_proposal(self, create_network_keys, create_test_topology):
        """Test basic consensus proposal (mock version)"""
        keys, public_keys = create_network_keys
        topology = create_test_topology
        
        consensus = TopologyAwareBFT(0, topology, keys[0], public_keys)
        
        # Mock the consensus execution
        # In real tests, would need to simulate network communication
        proposed_value = "test_value_123"
        
        # Test proposal creation
        prepare_msg = ConsensusMessage(
            msg_type=MessageType.PREPARE,
            view=0,
            sequence=1,
            proposal=proposed_value,
            sender_id=0,
            timestamp=123456.789,
            topology_proof=consensus._create_topology_proof()
        )
        
        assert prepare_msg.proposal == proposed_value
        assert prepare_msg.msg_type == MessageType.PREPARE
    
    def test_performance_metrics(self, create_network_keys, create_test_topology):
        """Test performance metrics collection"""
        keys, public_keys = create_network_keys
        topology = create_test_topology
        
        consensus = TopologyAwareBFT(0, topology, keys[0], public_keys)
        
        # Get initial metrics
        metrics = consensus.get_performance_metrics()
        
        assert 'rounds_completed' in metrics
        assert 'messages_sent' in metrics
        assert 'topology_validations' in metrics
        assert 'quorum_size' in metrics
        assert 'topology_nodes' in metrics
        
        # Reset metrics
        consensus.reset_metrics()
        new_metrics = consensus.get_performance_metrics()
        assert new_metrics['rounds_completed'] == 0
        assert new_metrics['messages_sent'] == 0


class TestConsensusIntegration:
    """Integration tests for consensus algorithm"""
    
    @pytest.mark.asyncio
    async def test_multi_node_consensus(self):
        """Test consensus with multiple nodes (simplified)"""
        num_nodes = 4
        topology = nx.complete_graph(num_nodes)
        
        # Create keys
        keys = {}
        public_keys = {}
        for i in range(num_nodes):
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            keys[i] = private_key
            public_keys[i] = private_key.public_key()
        
        # Create consensus instances
        consensus_nodes = []
        for i in range(num_nodes):
            consensus = TopologyAwareBFT(
                i, topology, keys[i], public_keys,
                byzantine_threshold=1/3
            )
            consensus_nodes.append(consensus)
        
        # Verify all nodes have consistent view
        fingerprint0 = consensus_nodes[0].topology_fingerprint
        for consensus in consensus_nodes[1:]:
            assert consensus.topology_fingerprint == fingerprint0
    
    def test_byzantine_threshold_validation(self):
        """Test Byzantine threshold constraints"""
        topology = nx.complete_graph(10)
        
        # Valid thresholds
        valid_thresholds = [0.1, 0.2, 0.33, 1/3]
        
        for threshold in valid_thresholds:
            keys, public_keys = create_network_keys(10)
            consensus = TopologyAwareBFT(
                0, topology, keys[0], public_keys,
                byzantine_threshold=threshold
            )
            assert consensus.byzantine_threshold == threshold
    
    def test_topology_types(self):
        """Test consensus with different topology types"""
        topologies = [
            ('ring', nx.cycle_graph(6)),
            ('complete', nx.complete_graph(6)),
            ('star', nx.star_graph(5)),
            ('grid', nx.grid_2d_graph(3, 2)),
        ]
        
        for name, topology in topologies:
            # Relabel grid nodes to integers
            if name == 'grid':
                mapping = {node: i for i, node in enumerate(topology.nodes())}
                topology = nx.relabel_nodes(topology, mapping)
            
            num_nodes = len(topology)
            keys, public_keys = create_network_keys(num_nodes)
            
            consensus = TopologyAwareBFT(
                0, topology, keys[0], public_keys
            )
            
            # Verify initialization succeeded
            assert consensus.quorum_size > 0
            assert len(consensus.centrality_weights) == num_nodes
            assert consensus.topology_fingerprint is not None


def create_network_keys(num_nodes: int) -> tuple:
    """Helper function to create network keys"""
    keys = {}
    public_keys = {}
    
    for i in range(num_nodes):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        keys[i] = private_key
        public_keys[i] = private_key.public_key()
    
    return keys, public_keys


if __name__ == '__main__':
    pytest.main([__file__, '-v'])