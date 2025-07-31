"""
Topology-Aware Byzantine Fault Tolerant Consensus Algorithm

This module implements the novel TA-BFT consensus protocol that incorporates
graph topology into Byzantine agreement decisions through eigenvector centrality
weighting and structural validation.

Author: Ayomide Caleb Adekoya
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of consensus messages in TA-BFT protocol"""
    PREPARE = "prepare"
    PROMISE = "promise" 
    COMMIT = "commit"
    VIEW_CHANGE = "view_change"
    NEW_VIEW = "new_view"

@dataclass
class ConsensusMessage:
    """TA-BFT consensus message structure"""
    msg_type: MessageType
    view: int
    sequence: int
    proposal: Any
    sender_id: int
    timestamp: float
    topology_proof: Optional[Dict] = None
    signature: Optional[bytes] = None
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary for serialization"""
        return {
            'msg_type': self.msg_type.value,
            'view': self.view,
            'sequence': self.sequence, 
            'proposal': self.proposal,
            'sender_id': self.sender_id,
            'timestamp': self.timestamp,
            'topology_proof': self.topology_proof
        }

@dataclass 
class ConsensusResult:
    """Result of consensus algorithm execution"""
    decided: bool
    value: Any
    view: int
    sequence: int
    rounds: int
    proof: Dict
    execution_time: float

class TopologyAwareBFT:
    """
    Topology-Aware Byzantine Fault Tolerant Consensus Protocol
    
    Key innovations:
    1. Eigenvector centrality weighting for validator selection
    2. Graph structure validation in consensus messages
    3. Topology-aware quorum computation
    4. Structural integrity verification
    """
    
    def __init__(
        self,
        node_id: int,
        graph_topology: nx.Graph,
        private_key: rsa.RSAPrivateKey,
        public_keys: Dict[int, rsa.RSAPublicKey],
        byzantine_threshold: float = 1/3,
        view_timeout: float = 10.0
    ):
        self.node_id = node_id
        self.topology = graph_topology
        self.private_key = private_key
        self.public_keys = public_keys
        self.byzantine_threshold = byzantine_threshold
        self.view_timeout = view_timeout
        
        # Consensus state
        self.current_view = 0
        self.current_sequence = 0
        self.decided_values = {}
        self.message_log = {}
        
        # Topology analysis
        self.centrality_weights = self._compute_centrality_weights()
        self.topology_fingerprint = self._compute_topology_fingerprint()
        self.quorum_size = self._compute_topology_quorum()
        
        # Performance metrics
        self.metrics = {
            'rounds_completed': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'consensus_latency': [],
            'topology_validations': 0
        }
        
        logger.info(f"TA-BFT initialized for node {node_id} with {len(graph_topology)} nodes")
        
    def _compute_centrality_weights(self) -> Dict[int, float]:
        """
        Compute eigenvector centrality weights for topology-aware validation
        
        Returns:
            Dictionary mapping node IDs to centrality weights
        """
        try:
            # Compute eigenvector centrality
            centrality = nx.eigenvector_centrality(self.topology, max_iter=1000)
            
            # Normalize weights to sum to 1
            total_weight = sum(centrality.values())
            normalized_weights = {
                node: weight / total_weight 
                for node, weight in centrality.items()
            }
            
            logger.debug(f"Computed centrality weights: {normalized_weights}")
            return normalized_weights
            
        except nx.NetworkXError:
            # Fallback to uniform weights if centrality computation fails
            n_nodes = len(self.topology)
            uniform_weight = 1.0 / n_nodes
            return {node: uniform_weight for node in self.topology.nodes()}
    
    def _compute_topology_fingerprint(self) -> str:
        """
        Compute cryptographic fingerprint of graph topology
        
        Returns:
            SHA-256 hash of canonical graph representation
        """
        # Create canonical adjacency matrix representation
        nodes = sorted(self.topology.nodes())
        adj_matrix = nx.adjacency_matrix(self.topology, nodelist=nodes)
        
        # Convert to canonical string representation
        canonical_repr = adj_matrix.toarray().tobytes()
        
        # Compute SHA-256 hash
        fingerprint = hashlib.sha256(canonical_repr).hexdigest()
        
        logger.debug(f"Topology fingerprint: {fingerprint[:16]}...")
        return fingerprint
    
    def _compute_topology_quorum(self) -> int:
        """
        Compute topology-aware quorum size based on graph connectivity
        
        Returns:
            Minimum number of nodes needed for valid quorum
        """
        n = len(self.topology)
        f = int(n * self.byzantine_threshold)  # Max Byzantine nodes
        
        # Standard BFT quorum: n >= 3f + 1, quorum >= 2f + 1
        standard_quorum = 2 * f + 1
        
        # Topology adjustment based on connectivity
        avg_degree = sum(dict(self.topology.degree()).values()) / n
        connectivity_factor = min(1.0, avg_degree / (n - 1))
        
        # Adjust quorum based on graph connectivity
        adjusted_quorum = max(
            standard_quorum,
            int(standard_quorum * (2 - connectivity_factor))
        )
        
        logger.info(f"Computed quorum size: {adjusted_quorum} (standard: {standard_quorum})")
        return adjusted_quorum
    
    def _sign_message(self, message: ConsensusMessage) -> bytes:
        """Sign consensus message with private key"""
        message_dict = message.to_dict()
        message_bytes = json.dumps(message_dict, sort_keys=True).encode()
        
        signature = self.private_key.sign(
            message_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature
    
    def _verify_signature(self, message: ConsensusMessage, signature: bytes) -> bool:
        """Verify message signature"""
        try:
            public_key = self.public_keys.get(message.sender_id)
            if not public_key:
                return False
                
            message_dict = message.to_dict()
            message_bytes = json.dumps(message_dict, sort_keys=True).encode()
            
            public_key.verify(
                signature,
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
            
        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            return False
    
    def _validate_topology_proof(self, message: ConsensusMessage) -> bool:
        """
        Validate topology proof in consensus message
        
        Ensures message sender has consistent view of graph topology
        """
        if not message.topology_proof:
            return False
            
        try:
            # Verify topology fingerprint matches
            sender_fingerprint = message.topology_proof.get('fingerprint')
            if sender_fingerprint != self.topology_fingerprint:
                logger.warning(f"Topology fingerprint mismatch from node {message.sender_id}")
                return False
            
            # Verify sender's centrality weight
            claimed_weight = message.topology_proof.get('centrality_weight')
            expected_weight = self.centrality_weights.get(message.sender_id, 0)
            
            if abs(claimed_weight - expected_weight) > 1e-6:
                logger.warning(f"Centrality weight mismatch from node {message.sender_id}")
                return False
                
            self.metrics['topology_validations'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Topology proof validation error: {e}")
            return False
    
    def _create_topology_proof(self) -> Dict:
        """Create topology proof for outgoing messages"""
        return {
            'fingerprint': self.topology_fingerprint,
            'centrality_weight': self.centrality_weights.get(self.node_id, 0),
            'timestamp': time.time()
        }
    
    async def propose_value(self, value: Any) -> ConsensusResult:
        """
        Initiate consensus on a proposed value
        
        Args:
            value: Value to achieve consensus on
            
        Returns:
            ConsensusResult with consensus outcome
        """
        start_time = time.time()
        self.current_sequence += 1
        
        logger.info(f"Node {self.node_id} proposing value for sequence {self.current_sequence}")
        
        # Create and broadcast PREPARE message
        prepare_msg = ConsensusMessage(
            msg_type=MessageType.PREPARE,
            view=self.current_view,
            sequence=self.current_sequence,
            proposal=value,
            sender_id=self.node_id,
            timestamp=time.time(),
            topology_proof=self._create_topology_proof()
        )
        
        prepare_msg.signature = self._sign_message(prepare_msg)
        
        # Broadcast prepare message
        await self._broadcast_message(prepare_msg)
        self.metrics['messages_sent'] += 1
        
        # Execute consensus protocol
        result = await self._execute_consensus_protocol(value, start_time)
        
        execution_time = time.time() - start_time
        self.metrics['consensus_latency'].append(execution_time)
        
        return result
    
    async def _execute_consensus_protocol(self, proposed_value: Any, start_time: float) -> ConsensusResult:
        """
        Execute the full TA-BFT consensus protocol
        
        Three-phase protocol:
        1. PREPARE phase: Collect topology-weighted prepare votes
        2. PROMISE phase: Verify Byzantine quorum and promise to accept
        3. COMMIT phase: Final commitment with Byzantine validation
        """
        rounds = 0
        
        # Phase 1: PREPARE - Collect prepare votes with topology validation
        logger.debug(f"Phase 1: PREPARE (sequence {self.current_sequence})")
        prepare_votes = await self._collect_prepare_votes(proposed_value)
        rounds += 1
        
        if len(prepare_votes) < self.quorum_size:
            return ConsensusResult(
                decided=False,
                value=None,
                view=self.current_view,
                sequence=self.current_sequence,
                rounds=rounds,
                proof={'phase': 'prepare', 'votes': len(prepare_votes)},
                execution_time=time.time() - start_time
            )
        
        # Phase 2: PROMISE - Verify quorum and send promise
        logger.debug(f"Phase 2: PROMISE (sequence {self.current_sequence})")
        promise_votes = await self._collect_promise_votes(prepare_votes)
        rounds += 1
        
        if len(promise_votes) < self.quorum_size:
            return ConsensusResult(
                decided=False,
                value=None,
                view=self.current_view,
                sequence=self.current_sequence,
                rounds=rounds,
                proof={'phase': 'promise', 'votes': len(promise_votes)},
                execution_time=time.time() - start_time
            )
        
        # Phase 3: COMMIT - Final Byzantine verification and commitment
        logger.debug(f"Phase 3: COMMIT (sequence {self.current_sequence})")
        commit_success = await self._execute_commit_phase(promise_votes, proposed_value)
        rounds += 1
        
        self.metrics['rounds_completed'] = rounds
        
        if commit_success:
            self.decided_values[self.current_sequence] = proposed_value
            logger.info(f"Consensus DECIDED on value for sequence {self.current_sequence}")
            
            return ConsensusResult(
                decided=True,
                value=proposed_value,
                view=self.current_view,
                sequence=self.current_sequence,
                rounds=rounds,
                proof={
                    'prepare_votes': len(prepare_votes),
                    'promise_votes': len(promise_votes),
                    'topology_validated': True
                },
                execution_time=time.time() - start_time
            )
        else:
            return ConsensusResult(
                decided=False,
                value=None,
                view=self.current_view,
                sequence=self.current_sequence,
                rounds=rounds,
                proof={'phase': 'commit', 'byzantine_detected': True},
                execution_time=time.time() - start_time
            )
    
    async def _collect_prepare_votes(self, proposed_value: Any) -> List[ConsensusMessage]:
        """Collect PREPARE votes with topology validation"""
        prepare_votes = []
        
        # Wait for prepare messages from other nodes
        await asyncio.sleep(0.1)  # Simulated network delay
        
        # In actual implementation, this would collect real network messages
        # For now, simulate Byzantine-tolerant prepare collection
        
        return prepare_votes
    
    async def _collect_promise_votes(self, prepare_votes: List[ConsensusMessage]) -> List[ConsensusMessage]:
        """Collect PROMISE votes after prepare phase validation"""
        promise_votes = []
        
        # Validate prepare votes using topology-aware Byzantine detection
        valid_prepares = []
        for vote in prepare_votes:
            if (self._verify_signature(vote, vote.signature) and 
                self._validate_topology_proof(vote)):
                valid_prepares.append(vote)
        
        # If sufficient valid prepares, send promise
        if len(valid_prepares) >= self.quorum_size:
            promise_msg = ConsensusMessage(
                msg_type=MessageType.PROMISE,
                view=self.current_view,
                sequence=self.current_sequence,
                proposal=prepare_votes[0].proposal,
                sender_id=self.node_id,
                timestamp=time.time(),
                topology_proof=self._create_topology_proof()
            )
            
            promise_msg.signature = self._sign_message(promise_msg)
            await self._broadcast_message(promise_msg)
            
            promise_votes.append(promise_msg)
        
        return promise_votes
    
    async def _execute_commit_phase(self, promise_votes: List[ConsensusMessage], value: Any) -> bool:
        """Execute final commit phase with Byzantine validation"""
        
        # Verify weighted quorum using topology-aware validation
        total_weight = sum(
            self.centrality_weights.get(vote.sender_id, 0)
            for vote in promise_votes
        )
        
        # Require majority of topology-weighted voting power
        required_weight = 0.5 + self.byzantine_threshold
        
        if total_weight >= required_weight:
            # Broadcast commit message
            commit_msg = ConsensusMessage(
                msg_type=MessageType.COMMIT,
                view=self.current_view,
                sequence=self.current_sequence,
                proposal=value,
                sender_id=self.node_id,
                timestamp=time.time(),
                topology_proof=self._create_topology_proof()
            )
            
            commit_msg.signature = self._sign_message(commit_msg)
            await self._broadcast_message(commit_msg)
            
            return True
        
        return False
    
    async def _broadcast_message(self, message: ConsensusMessage):
        """Broadcast message to all nodes in topology"""
        # In actual implementation, this would use network communication
        # For now, just log the broadcast
        logger.debug(f"Broadcasting {message.msg_type.value} message from node {self.node_id}")
        self.metrics['messages_sent'] += 1
    
    def get_performance_metrics(self) -> Dict:
        """Get consensus algorithm performance metrics"""
        metrics = self.metrics.copy()
        
        if metrics['consensus_latency']:
            metrics['avg_latency'] = np.mean(metrics['consensus_latency'])
            metrics['p95_latency'] = np.percentile(metrics['consensus_latency'], 95)
            metrics['p99_latency'] = np.percentile(metrics['consensus_latency'], 99)
        
        metrics['quorum_size'] = self.quorum_size
        metrics['topology_nodes'] = len(self.topology)
        metrics['centrality_distribution'] = self.centrality_weights
        
        return metrics
    
    def reset_metrics(self):
        """Reset performance metrics counters"""
        self.metrics = {
            'rounds_completed': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'consensus_latency': [],
            'topology_validations': 0
        }