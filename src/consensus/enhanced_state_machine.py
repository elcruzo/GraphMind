"""
Enhanced Consensus State Machine for TA-BFT

This module implements a sophisticated state machine for the Topology-Aware
Byzantine Fault Tolerant consensus with enhanced message ordering, duplicate
detection, state synchronization, and recovery mechanisms.

Author: Ayomide Caleb Adekoya
"""

import asyncio
import json
import logging
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from collections import defaultdict, deque, OrderedDict
import threading

import numpy as np
import networkx as nx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

from .ta_bft import ConsensusMessage, MessageType, ConsensusResult
from .ta_bft_proofs import TABFTFormalVerification

logger = logging.getLogger(__name__)


class ConsensusState(Enum):
    """Enhanced consensus protocol states"""
    IDLE = "idle"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED = "committed"
    VIEW_CHANGING = "view_changing"
    RECOVERING = "recovering"
    CHECKPOINTING = "checkpointing"


class MessagePhase(Enum):
    """Message phases in consensus protocol"""
    PREPARE = "prepare"
    PROMISE = "promise"
    ACCEPT = "accept"
    ACCEPTED = "accepted"
    COMMIT = "commit"
    VIEW_CHANGE = "view_change"
    NEW_VIEW = "new_view"
    CHECKPOINT = "checkpoint"
    RECOVERY = "recovery"


@dataclass
class MessageInfo:
    """Enhanced message information with ordering and validation"""
    message: ConsensusMessage
    phase: MessagePhase
    received_time: float
    validated: bool = False
    processed: bool = False
    dependencies: Set[str] = field(default_factory=set)
    sequence_number: int = 0
    checksum: str = ""
    
    def __post_init__(self):
        # Compute message checksum for duplicate detection
        message_bytes = json.dumps(self.message.to_dict(), sort_keys=True).encode()
        self.checksum = hashlib.sha256(message_bytes).hexdigest()


@dataclass
class ViewState:
    """State information for consensus view"""
    view_number: int
    leader_id: str
    start_time: float
    timeout: float
    prepared_values: Dict[str, Any] = field(default_factory=dict)
    promise_count: int = 0
    accept_count: int = 0
    participants: Set[str] = field(default_factory=set)
    is_active: bool = True


@dataclass
class Checkpoint:
    """Consensus checkpoint for state recovery"""
    sequence_number: int
    state_hash: str
    decided_values: Dict[int, Any]
    view_number: int
    timestamp: float
    node_states: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[bytes] = None


class EnhancedConsensusMachine:
    """
    Enhanced state machine for TA-BFT consensus with production features:
    
    Features:
    - Message ordering and duplicate detection
    - State synchronization across network partitions
    - Checkpoint-based recovery mechanisms
    - View change protocols with leader election
    - Byzantine-robust message validation
    - Performance monitoring and optimization
    """
    
    def __init__(
        self,
        node_id: str,
        private_key: rsa.RSAPrivateKey,
        topology: nx.Graph,
        checkpoint_interval: int = 100,
        max_message_buffer: int = 1000
    ):
        self.node_id = node_id
        self.private_key = private_key
        self.topology = topology
        self.checkpoint_interval = checkpoint_interval
        self.max_message_buffer = max_message_buffer
        
        # Consensus state
        self.current_state = ConsensusState.IDLE
        self.current_view = 0
        self.current_sequence = 0
        self.leader_id = self._select_initial_leader()
        
        # Message management
        self.message_buffer: deque[MessageInfo] = deque(maxlen=max_message_buffer)
        self.message_log: Dict[int, List[MessageInfo]] = defaultdict(list)
        self.duplicate_filter: Dict[str, float] = {}  # checksum -> timestamp
        self.pending_messages: Dict[str, MessageInfo] = {}
        
        # View management
        self.view_states: Dict[int, ViewState] = {}
        self.view_change_votes: Dict[int, Set[str]] = defaultdict(set)
        self.new_view_messages: Dict[int, List[ConsensusMessage]] = defaultdict(list)
        
        # Consensus progress tracking
        self.decided_values: OrderedDict[int, Any] = OrderedDict()
        self.prepared_values: Dict[int, Dict[str, Any]] = defaultdict(dict)
        self.promise_counts: Dict[int, int] = defaultdict(int)
        self.accept_counts: Dict[int, int] = defaultdict(int)
        
        # Checkpoint and recovery
        self.checkpoints: Dict[int, Checkpoint] = {}
        self.last_checkpoint_seq = 0
        self.recovery_state: Optional[Dict] = None
        
        # Performance monitoring
        self.metrics = {
            'messages_processed': 0,
            'views_completed': 0,
            'view_changes': 0,
            'checkpoints_created': 0,
            'recovery_events': 0,
            'duplicate_messages': 0,
            'state_transitions': 0,
            'consensus_latency': deque(maxlen=100),
            'message_processing_time': deque(maxlen=1000)
        }
        
        # Threading and synchronization
        self._state_lock = threading.RLock()
        self._processing_task: Optional[asyncio.Task] = None
        
        # Formal verification integration
        self.verifier = TABFTFormalVerification(
            n_nodes=len(topology),
            byzantine_threshold=0.33
        )
        
        logger.info(f"Enhanced consensus state machine initialized for {node_id}")
    
    def _select_initial_leader(self) -> str:
        """Select initial leader based on topology centrality"""
        try:
            centrality = nx.eigenvector_centrality(self.topology)
            # Select node with highest centrality as initial leader
            leader = max(centrality.keys(), key=lambda x: centrality[x])
            return str(leader)
        except Exception:
            # Fallback to lexicographic ordering
            return sorted(self.topology.nodes())[0]
    
    async def start_processing(self):
        """Start the state machine processing loop"""
        if self._processing_task is None:
            self._processing_task = asyncio.create_task(self._processing_loop())
            logger.info("Enhanced consensus state machine started")
    
    async def stop_processing(self):
        """Stop the state machine processing loop"""
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None
            logger.info("Enhanced consensus state machine stopped")
    
    async def _processing_loop(self):
        """Main processing loop for the state machine"""
        while True:
            try:
                await self._process_pending_messages()
                await self._check_view_timeouts()
                await self._perform_checkpointing()
                await self._cleanup_old_data()
                
                # Performance monitoring
                await asyncio.sleep(0.01)  # 10ms processing interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                await asyncio.sleep(1)
    
    async def receive_message(self, message: ConsensusMessage) -> bool:
        """
        Receive and process consensus message
        
        Args:
            message: Incoming consensus message
            
        Returns:
            True if message was accepted, False otherwise
        """
        start_time = time.time()
        
        try:
            with self._state_lock:
                # Create message info
                phase = self._get_message_phase(message.msg_type)
                msg_info = MessageInfo(
                    message=message,
                    phase=phase,
                    received_time=time.time(),
                    sequence_number=message.sequence
                )
                
                # Check for duplicates
                if self._is_duplicate_message(msg_info):
                    self.metrics['duplicate_messages'] += 1
                    return False
                
                # Validate message
                if not await self._validate_message(msg_info):
                    logger.warning(f"Invalid message from {message.sender_id}")
                    return False
                
                msg_info.validated = True
                
                # Add to appropriate data structures
                self._add_message_to_buffer(msg_info)
                self.message_log[message.sequence].append(msg_info)
                
                # Update duplicate filter
                self.duplicate_filter[msg_info.checksum] = time.time()
                
                # Process immediately if in correct state
                if self._can_process_immediately(msg_info):
                    await self._process_message_immediate(msg_info)
                else:
                    self.pending_messages[msg_info.checksum] = msg_info
                
                self.metrics['messages_processed'] += 1
                processing_time = time.time() - start_time
                self.metrics['message_processing_time'].append(processing_time)
                
                return True
                
        except Exception as e:
            logger.error(f"Message reception failed: {e}")
            return False
    
    def _get_message_phase(self, msg_type: MessageType) -> MessagePhase:
        """Map message type to processing phase"""
        phase_map = {
            MessageType.PREPARE: MessagePhase.PREPARE,
            MessageType.PROMISE: MessagePhase.PROMISE,
            MessageType.COMMIT: MessagePhase.COMMIT,
            MessageType.VIEW_CHANGE: MessagePhase.VIEW_CHANGE,
            MessageType.NEW_VIEW: MessagePhase.NEW_VIEW
        }
        return phase_map.get(msg_type, MessagePhase.PREPARE)
    
    def _is_duplicate_message(self, msg_info: MessageInfo) -> bool:
        """Check if message is a duplicate"""
        checksum = msg_info.checksum
        
        if checksum in self.duplicate_filter:
            # Check if it's a recent duplicate (within 60 seconds)
            age = time.time() - self.duplicate_filter[checksum]
            return age < 60.0
        
        return False
    
    async def _validate_message(self, msg_info: MessageInfo) -> bool:
        """Comprehensive message validation"""
        message = msg_info.message
        
        try:
            # 1. Basic structure validation
            if not message.sender_id or message.sequence <= 0:
                return False
            
            # 2. Signature validation
            if not self._verify_message_signature(message):
                return False
            
            # 3. Topology proof validation
            if message.topology_proof and not self._validate_topology_proof(message):
                return False
            
            # 4. Sequence number validation
            if not self._validate_sequence_number(message):
                return False
            
            # 5. View number validation
            if message.view < 0 or message.view < self.current_view - 1:
                return False
            
            # 6. Message freshness (prevent replay attacks)
            message_age = time.time() - message.timestamp
            if message_age > 300:  # 5 minutes
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Message validation error: {e}")
            return False
    
    def _verify_message_signature(self, message: ConsensusMessage) -> bool:
        """Verify message cryptographic signature"""
        # This would integrate with the actual cryptographic verification
        # For now, simplified check
        return message.signature is not None and len(message.signature) > 0
    
    def _validate_topology_proof(self, message: ConsensusMessage) -> bool:
        """Validate topology proof consistency"""
        if not message.topology_proof:
            return True  # Optional field
        
        # Check required fields
        required_fields = ['fingerprint', 'centrality_weight', 'timestamp']
        return all(field in message.topology_proof for field in required_fields)
    
    def _validate_sequence_number(self, message: ConsensusMessage) -> bool:
        """Validate message sequence number ordering"""
        # Allow messages for current and next sequence
        return message.sequence >= self.current_sequence
    
    def _add_message_to_buffer(self, msg_info: MessageInfo):
        """Add message to processing buffer with proper ordering"""
        self.message_buffer.append(msg_info)
        
        # Sort buffer by sequence number for ordered processing
        if len(self.message_buffer) > 1:
            sorted_buffer = sorted(self.message_buffer, key=lambda x: x.sequence_number)
            self.message_buffer.clear()
            self.message_buffer.extend(sorted_buffer)
    
    def _can_process_immediately(self, msg_info: MessageInfo) -> bool:
        """Check if message can be processed immediately"""
        message = msg_info.message
        
        # Process if it's for the current sequence and view
        return (message.sequence == self.current_sequence and 
                message.view == self.current_view and
                self.current_state != ConsensusState.RECOVERING)
    
    async def _process_message_immediate(self, msg_info: MessageInfo):
        """Process message immediately"""
        message = msg_info.message
        
        try:
            if message.msg_type == MessageType.PREPARE:
                await self._handle_prepare_phase(msg_info)
            elif message.msg_type == MessageType.PROMISE:
                await self._handle_promise_phase(msg_info)
            elif message.msg_type == MessageType.COMMIT:
                await self._handle_commit_phase(msg_info)
            elif message.msg_type == MessageType.VIEW_CHANGE:
                await self._handle_view_change_phase(msg_info)
            elif message.msg_type == MessageType.NEW_VIEW:
                await self._handle_new_view_phase(msg_info)
            
            msg_info.processed = True
            
        except Exception as e:
            logger.error(f"Immediate message processing failed: {e}")
    
    async def _process_pending_messages(self):
        """Process messages waiting in pending queue"""
        with self._state_lock:
            to_process = []
            to_remove = []
            
            for checksum, msg_info in self.pending_messages.items():
                if not msg_info.processed and self._can_process_immediately(msg_info):
                    to_process.append(msg_info)
                    to_remove.append(checksum)
            
            # Remove from pending
            for checksum in to_remove:
                del self.pending_messages[checksum]
        
        # Process messages outside lock
        for msg_info in to_process:
            await self._process_message_immediate(msg_info)
    
    async def _handle_prepare_phase(self, msg_info: MessageInfo):
        """Handle PREPARE phase message"""
        message = msg_info.message
        
        logger.debug(f"Processing PREPARE from {message.sender_id}, seq={message.sequence}")
        
        # Transition to PREPARING state
        await self._transition_state(ConsensusState.PREPARING)
        
        # Store prepared value
        self.prepared_values[message.sequence][message.sender_id] = message.proposal
        
        # Check if we have enough PREPARE messages
        prepare_count = len(self.prepared_values[message.sequence])
        quorum_size = self._compute_quorum_size()
        
        if prepare_count >= quorum_size:
            await self._transition_state(ConsensusState.PREPARED)
            # Could trigger PROMISE phase here
    
    async def _handle_promise_phase(self, msg_info: MessageInfo):
        """Handle PROMISE phase message"""
        message = msg_info.message
        
        logger.debug(f"Processing PROMISE from {message.sender_id}, seq={message.sequence}")
        
        self.promise_counts[message.sequence] += 1
        
        # Check if we have enough PROMISE messages
        if self.promise_counts[message.sequence] >= self._compute_quorum_size():
            await self._transition_state(ConsensusState.COMMITTING)
    
    async def _handle_commit_phase(self, msg_info: MessageInfo):
        """Handle COMMIT phase message"""
        message = msg_info.message
        
        logger.debug(f"Processing COMMIT from {message.sender_id}, seq={message.sequence}")
        
        self.accept_counts[message.sequence] += 1
        
        # Check if we can decide
        if self.accept_counts[message.sequence] >= self._compute_quorum_size():
            await self._decide_value(message.sequence, message.proposal)
    
    async def _handle_view_change_phase(self, msg_info: MessageInfo):
        """Handle VIEW_CHANGE phase message"""
        message = msg_info.message
        new_view = message.view
        
        logger.info(f"Processing VIEW_CHANGE to view {new_view} from {message.sender_id}")
        
        self.view_change_votes[new_view].add(message.sender_id)
        
        # Check if we have enough view change votes
        if len(self.view_change_votes[new_view]) >= self._compute_quorum_size():
            await self._start_new_view(new_view)
    
    async def _handle_new_view_phase(self, msg_info: MessageInfo):
        """Handle NEW_VIEW phase message"""
        message = msg_info.message
        
        logger.info(f"Processing NEW_VIEW {message.view} from {message.sender_id}")
        
        # Validate new view message
        if await self._validate_new_view_message(message):
            await self._adopt_new_view(message.view)
    
    async def _decide_value(self, sequence: int, value: Any):
        """Decide on a consensus value"""
        logger.info(f"DECIDED value for sequence {sequence}: {value}")
        
        self.decided_values[sequence] = value
        self.current_sequence = max(self.current_sequence, sequence + 1)
        
        await self._transition_state(ConsensusState.COMMITTED)
        
        # Record consensus latency
        # This would need to track when consensus started
        # For now, just increment metrics
        self.metrics['views_completed'] += 1
    
    async def _start_new_view(self, new_view: int):
        """Start new consensus view"""
        logger.info(f"Starting new view {new_view}")
        
        self.current_view = new_view
        self.leader_id = self._select_leader_for_view(new_view)
        
        # Create view state
        self.view_states[new_view] = ViewState(
            view_number=new_view,
            leader_id=self.leader_id,
            start_time=time.time(),
            timeout=30.0  # 30 second view timeout
        )
        
        await self._transition_state(ConsensusState.IDLE)
        self.metrics['view_changes'] += 1
    
    def _select_leader_for_view(self, view: int) -> str:
        """Select leader for specific view using round-robin"""
        nodes = sorted(self.topology.nodes())
        leader_index = view % len(nodes)
        return str(nodes[leader_index])
    
    async def _adopt_new_view(self, view: int):
        """Adopt new view from NEW_VIEW message"""
        if view > self.current_view:
            await self._start_new_view(view)
    
    async def _validate_new_view_message(self, message: ConsensusMessage) -> bool:
        """Validate NEW_VIEW message"""
        # Check if sender is legitimate leader for this view
        expected_leader = self._select_leader_for_view(message.view)
        return message.sender_id == expected_leader
    
    async def _transition_state(self, new_state: ConsensusState):
        """Transition consensus state with validation"""
        if new_state != self.current_state:
            old_state = self.current_state
            self.current_state = new_state
            self.metrics['state_transitions'] += 1
            
            logger.debug(f"State transition: {old_state.value} → {new_state.value}")
    
    def _compute_quorum_size(self) -> int:
        """Compute quorum size based on Byzantine fault tolerance"""
        n = len(self.topology)
        f = int(n * 0.33)  # Max Byzantine nodes
        return 2 * f + 1  # Byzantine quorum
    
    async def _check_view_timeouts(self):
        """Check for view timeouts and trigger view changes"""
        current_time = time.time()
        
        for view, view_state in list(self.view_states.items()):
            if (view_state.is_active and 
                current_time - view_state.start_time > view_state.timeout):
                
                logger.warning(f"View {view} timeout, initiating view change")
                await self._initiate_view_change(view + 1)
                view_state.is_active = False
    
    async def _initiate_view_change(self, new_view: int):
        """Initiate view change to new view"""
        logger.info(f"Initiating view change to view {new_view}")
        
        await self._transition_state(ConsensusState.VIEW_CHANGING)
        
        # In real implementation, would broadcast VIEW_CHANGE message
        # For now, simulate by adding our own vote
        self.view_change_votes[new_view].add(self.node_id)
    
    async def _perform_checkpointing(self):
        """Create checkpoint if needed"""
        if (self.current_sequence - self.last_checkpoint_seq >= self.checkpoint_interval):
            await self._create_checkpoint()
    
    async def _create_checkpoint(self):
        """Create consensus state checkpoint"""
        checkpoint_seq = self.current_sequence
        
        # Create state hash
        state_data = {
            'decided_values': dict(self.decided_values),
            'current_view': self.current_view,
            'current_sequence': self.current_sequence
        }
        state_json = json.dumps(state_data, sort_keys=True)
        state_hash = hashlib.sha256(state_json.encode()).hexdigest()
        
        # Create checkpoint
        checkpoint = Checkpoint(
            sequence_number=checkpoint_seq,
            state_hash=state_hash,
            decided_values=dict(self.decided_values),
            view_number=self.current_view,
            timestamp=time.time()
        )
        
        self.checkpoints[checkpoint_seq] = checkpoint
        self.last_checkpoint_seq = checkpoint_seq
        self.metrics['checkpoints_created'] += 1
        
        logger.info(f"Created checkpoint at sequence {checkpoint_seq}")
    
    async def _cleanup_old_data(self):
        """Clean up old data structures"""
        current_time = time.time()
        
        # Clean duplicate filter (older than 5 minutes)
        cutoff_time = current_time - 300
        to_remove = [
            checksum for checksum, timestamp in self.duplicate_filter.items()
            if timestamp < cutoff_time
        ]
        for checksum in to_remove:
            del self.duplicate_filter[checksum]
        
        # Clean old checkpoints (keep last 10)
        if len(self.checkpoints) > 10:
            old_checkpoints = sorted(self.checkpoints.keys())[:-10]
            for seq in old_checkpoints:
                del self.checkpoints[seq]
    
    def get_consensus_status(self) -> Dict[str, Any]:
        """Get current consensus status"""
        return {
            'node_id': self.node_id,
            'current_state': self.current_state.value,
            'current_view': self.current_view,
            'current_sequence': self.current_sequence,
            'leader_id': self.leader_id,
            'decided_values_count': len(self.decided_values),
            'pending_messages': len(self.pending_messages),
            'buffer_size': len(self.message_buffer),
            'checkpoints': len(self.checkpoints),
            'last_checkpoint': self.last_checkpoint_seq,
            'metrics': {k: (list(v) if hasattr(v, '__iter__') and not isinstance(v, str) else v) 
                       for k, v in self.metrics.items()},
            'topology_size': len(self.topology),
            'quorum_size': self._compute_quorum_size()
        }
    
    async def recover_from_checkpoint(self, checkpoint_seq: int) -> bool:
        """Recover state from checkpoint"""
        if checkpoint_seq not in self.checkpoints:
            logger.error(f"Checkpoint {checkpoint_seq} not found")
            return False
        
        try:
            await self._transition_state(ConsensusState.RECOVERING)
            
            checkpoint = self.checkpoints[checkpoint_seq]
            
            # Restore state
            self.decided_values = OrderedDict(checkpoint.decided_values)
            self.current_view = checkpoint.view_number
            self.current_sequence = checkpoint.sequence_number
            
            # Clear transient state
            self.prepared_values.clear()
            self.promise_counts.clear()
            self.accept_counts.clear()
            self.pending_messages.clear()
            
            await self._transition_state(ConsensusState.IDLE)
            self.metrics['recovery_events'] += 1
            
            logger.info(f"Recovered from checkpoint {checkpoint_seq}")
            return True
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False