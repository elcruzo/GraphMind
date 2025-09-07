"""
Byzantine Fault Detection and Tolerance for GraphMind

This module implements sophisticated Byzantine fault detection, evidence collection,
and tolerance mechanisms for distributed graph neural network training.

Author: Ayomide Caleb Adekoya
"""

import asyncio
import json
import logging
import time
import hashlib
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from collections import defaultdict, deque

import numpy as np
import torch
import networkx as nx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

from ..consensus.ta_bft import ConsensusMessage, MessageType
from ..distributed.node_discovery import NodeInfo, NodeStatus

logger = logging.getLogger(__name__)


class FaultType(Enum):
    """Types of Byzantine faults"""
    UNKNOWN = "unknown"
    SILENT = "silent"              # Node stops responding
    CRASH = "crash"                # Node crashes and stops
    ARBITRARY = "arbitrary"        # Node sends arbitrary/incorrect data
    SELFISH = "selfish"            # Node acts selfishly but not maliciously
    EQUIVOCATION = "equivocation"  # Node sends different messages to different nodes
    TIMING = "timing"              # Node violates timing assumptions
    CORRUPTION = "corruption"      # Node sends corrupted data


class EvidenceType(Enum):
    """Types of Byzantine evidence"""
    MESSAGE_INCONSISTENCY = "message_inconsistency"
    SIGNATURE_FAILURE = "signature_failure"
    TIMING_VIOLATION = "timing_violation"
    PARAMETER_ANOMALY = "parameter_anomaly"
    CONSENSUS_DEVIATION = "consensus_deviation"
    HEALTH_DEGRADATION = "health_degradation"
    STATISTICAL_OUTLIER = "statistical_outlier"


@dataclass
class ByzantineEvidence:
    """Evidence of Byzantine behavior"""
    evidence_id: str
    suspect_id: str
    evidence_type: EvidenceType
    fault_type: FaultType
    timestamp: float
    confidence: float  # 0.0 to 1.0
    details: Dict[str, Any]
    witness_ids: Set[str] = field(default_factory=set)
    verified: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'evidence_id': self.evidence_id,
            'suspect_id': self.suspect_id,
            'evidence_type': self.evidence_type.value,
            'fault_type': self.fault_type.value,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'details': self.details,
            'witness_ids': list(self.witness_ids),
            'verified': self.verified
        }


@dataclass
class NodeBehaviorProfile:
    """Behavioral profile of a node"""
    node_id: str
    message_patterns: Dict[str, Any] = field(default_factory=dict)
    timing_profile: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    parameter_history: deque = field(default_factory=lambda: deque(maxlen=100))
    health_history: deque = field(default_factory=lambda: deque(maxlen=100))
    consensus_participation: Dict[str, int] = field(default_factory=dict)
    anomaly_score: float = 0.0
    last_updated: float = field(default_factory=time.time)


class ByzantineFaultDetector:
    """
    Sophisticated Byzantine fault detection system
    
    Features:
    - Multi-modal fault detection (timing, consensus, parameters, health)
    - Statistical outlier detection with machine learning
    - Evidence collection and verification
    - Adaptive thresholds based on network conditions
    - Byzantine-robust aggregation for model parameters
    """
    
    def __init__(
        self,
        node_id: str,
        detection_threshold: float = 0.7,
        evidence_window: int = 100,
        verification_quorum: float = 0.5
    ):
        self.node_id = node_id
        self.detection_threshold = detection_threshold
        self.evidence_window = evidence_window
        self.verification_quorum = verification_quorum
        
        # Node behavior tracking
        self.node_profiles: Dict[str, NodeBehaviorProfile] = {}
        self.evidence_log: List[ByzantineEvidence] = []
        self.suspected_nodes: Set[str] = set()
        self.verified_byzantine: Set[str] = set()
        
        # Detection algorithms
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=3)
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            'timing_deviation': 2.0,
            'parameter_anomaly': 0.8,
            'health_degradation': 0.3,
            'consensus_deviation': 0.5
        }
        
        # Callbacks
        self.on_byzantine_detected: Optional[Callable[[ByzantineEvidence], None]] = None
        self.on_node_verified_byzantine: Optional[Callable[[str], None]] = None
        
        # Performance metrics
        self.metrics = {
            'detections': 0,
            'false_positives': 0,
            'true_positives': 0,
            'evidence_collected': 0,
            'verifications_performed': 0
        }
        
        logger.info(f"Byzantine fault detector initialized for node {node_id}")
    
    def update_node_profile(
        self,
        node_id: str,
        message: Optional[ConsensusMessage] = None,
        health_score: Optional[float] = None,
        model_params: Optional[torch.Tensor] = None
    ):
        """Update behavioral profile for a node"""
        if node_id not in self.node_profiles:
            self.node_profiles[node_id] = NodeBehaviorProfile(node_id=node_id)
        
        profile = self.node_profiles[node_id]
        current_time = time.time()
        
        # Update message patterns
        if message:
            msg_type = message.msg_type.value
            if msg_type not in profile.message_patterns:
                profile.message_patterns[msg_type] = {
                    'count': 0,
                    'avg_size': 0,
                    'last_seen': 0,
                    'intervals': deque(maxlen=50)
                }
            
            pattern = profile.message_patterns[msg_type]
            pattern['count'] += 1
            
            # Update timing intervals
            if pattern['last_seen'] > 0:
                interval = current_time - pattern['last_seen']
                pattern['intervals'].append(interval)
                profile.timing_profile[msg_type].append(interval)
            
            pattern['last_seen'] = current_time
            
            # Detect timing anomalies
            self._detect_timing_anomalies(node_id, message)
            
            # Detect message inconsistencies
            self._detect_message_inconsistencies(node_id, message)
        
        # Update health score
        if health_score is not None:
            profile.health_history.append(health_score)
            self._detect_health_anomalies(node_id, health_score)
        
        # Update model parameters
        if model_params is not None:
            param_hash = hashlib.sha256(model_params.numpy().tobytes()).hexdigest()
            profile.parameter_history.append({
                'timestamp': current_time,
                'hash': param_hash,
                'norm': torch.norm(model_params).item()
            })
            
            self._detect_parameter_anomalies(node_id, model_params)
        
        # Update anomaly score using ensemble methods
        profile.anomaly_score = self._calculate_anomaly_score(profile)
        profile.last_updated = current_time
        
        # Check if node should be flagged as suspect
        if profile.anomaly_score > self.detection_threshold:
            self._flag_suspicious_node(node_id, profile.anomaly_score)
    
    def _detect_timing_anomalies(self, node_id: str, message: ConsensusMessage):
        """Detect timing-based Byzantine behavior"""
        profile = self.node_profiles[node_id]
        msg_type = message.msg_type.value
        
        if msg_type in profile.timing_profile and len(profile.timing_profile[msg_type]) > 5:
            intervals = profile.timing_profile[msg_type][-20:]  # Recent intervals
            
            if len(intervals) >= 3:
                mean_interval = statistics.mean(intervals)
                std_interval = statistics.stdev(intervals)
                
                # Detect outliers using z-score
                latest_interval = intervals[-1]
                if std_interval > 0:
                    z_score = abs((latest_interval - mean_interval) / std_interval)
                    
                    if z_score > self.adaptive_thresholds['timing_deviation']:
                        evidence = ByzantineEvidence(
                            evidence_id=f"timing_{node_id}_{int(time.time())}",
                            suspect_id=node_id,
                            evidence_type=EvidenceType.TIMING_VIOLATION,
                            fault_type=FaultType.TIMING,
                            timestamp=time.time(),
                            confidence=min(z_score / 10.0, 0.9),
                            details={
                                'message_type': msg_type,
                                'z_score': z_score,
                                'mean_interval': mean_interval,
                                'actual_interval': latest_interval,
                                'std_deviation': std_interval
                            }
                        )
                        
                        self._collect_evidence(evidence)
    
    def _detect_message_inconsistencies(self, node_id: str, message: ConsensusMessage):
        """Detect message inconsistency Byzantine behavior"""
        try:
            # Check signature validity
            if not self._verify_message_signature(message):
                evidence = ByzantineEvidence(
                    evidence_id=f"signature_{node_id}_{int(time.time())}",
                    suspect_id=node_id,
                    evidence_type=EvidenceType.SIGNATURE_FAILURE,
                    fault_type=FaultType.ARBITRARY,
                    timestamp=time.time(),
                    confidence=0.95,
                    details={
                        'message_type': message.msg_type.value,
                        'sequence': message.sequence,
                        'view': message.view
                    }
                )
                
                self._collect_evidence(evidence)
            
            # Check topology proof consistency
            if message.topology_proof and not self._verify_topology_proof(message):
                evidence = ByzantineEvidence(
                    evidence_id=f"topology_{node_id}_{int(time.time())}",
                    suspect_id=node_id,
                    evidence_type=EvidenceType.CONSENSUS_DEVIATION,
                    fault_type=FaultType.ARBITRARY,
                    timestamp=time.time(),
                    confidence=0.8,
                    details={
                        'message_type': message.msg_type.value,
                        'topology_fingerprint': message.topology_proof.get('fingerprint', 'missing')
                    }
                )
                
                self._collect_evidence(evidence)
                
        except Exception as e:
            logger.error(f"Error detecting message inconsistencies: {e}")
    
    def _detect_health_anomalies(self, node_id: str, health_score: float):
        """Detect health-based anomalies"""
        profile = self.node_profiles[node_id]
        
        if len(profile.health_history) > 5:
            recent_scores = list(profile.health_history)[-10:]
            trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            
            # Detect rapid health degradation
            if trend < -0.1 and health_score < self.adaptive_thresholds['health_degradation']:
                evidence = ByzantineEvidence(
                    evidence_id=f"health_{node_id}_{int(time.time())}",
                    suspect_id=node_id,
                    evidence_type=EvidenceType.HEALTH_DEGRADATION,
                    fault_type=FaultType.SILENT,
                    timestamp=time.time(),
                    confidence=abs(trend),
                    details={
                        'health_score': health_score,
                        'trend': trend,
                        'recent_scores': recent_scores
                    }
                )
                
                self._collect_evidence(evidence)
    
    def _detect_parameter_anomalies(self, node_id: str, model_params: torch.Tensor):
        """Detect model parameter anomalies using statistical methods"""
        profile = self.node_profiles[node_id]
        
        if len(profile.parameter_history) > 3:
            # Calculate parameter statistics
            recent_norms = [p['norm'] for p in list(profile.parameter_history)[-10:]]
            current_norm = torch.norm(model_params).item()
            
            if len(recent_norms) >= 3:
                mean_norm = statistics.mean(recent_norms)
                std_norm = statistics.stdev(recent_norms) if len(recent_norms) > 1 else 0
                
                if std_norm > 0:
                    z_score = abs((current_norm - mean_norm) / std_norm)
                    
                    if z_score > 3.0:  # 3-sigma rule
                        evidence = ByzantineEvidence(
                            evidence_id=f"param_{node_id}_{int(time.time())}",
                            suspect_id=node_id,
                            evidence_type=EvidenceType.PARAMETER_ANOMALY,
                            fault_type=FaultType.ARBITRARY,
                            timestamp=time.time(),
                            confidence=min(z_score / 10.0, 0.9),
                            details={
                                'parameter_norm': current_norm,
                                'mean_norm': mean_norm,
                                'z_score': z_score,
                                'std_deviation': std_norm
                            }
                        )
                        
                        self._collect_evidence(evidence)
    
    def _calculate_anomaly_score(self, profile: NodeBehaviorProfile) -> float:
        """Calculate overall anomaly score for a node"""
        scores = []
        
        # Timing anomaly score
        timing_scores = []
        for msg_type, intervals in profile.timing_profile.items():
            if len(intervals) > 5:
                recent_intervals = intervals[-10:]
                if len(recent_intervals) > 1:
                    cv = statistics.stdev(recent_intervals) / statistics.mean(recent_intervals)
                    timing_scores.append(min(cv, 1.0))
        
        if timing_scores:
            scores.append(statistics.mean(timing_scores))
        
        # Health anomaly score
        if len(profile.health_history) > 0:
            recent_health = list(profile.health_history)[-5:]
            health_score = 1.0 - statistics.mean(recent_health)
            scores.append(health_score)
        
        # Parameter anomaly score (using isolation forest if enough data)
        if len(profile.parameter_history) > 10:
            norms = [p['norm'] for p in list(profile.parameter_history)[-20:]]
            try:
                # Reshape for sklearn
                X = np.array(norms).reshape(-1, 1)
                
                # Fit isolation forest
                isolation_score = self.isolation_forest.fit(X).decision_function(X)
                # Convert to 0-1 scale (lower scores = more anomalous)
                param_anomaly = max(0, 1.0 + isolation_score[-1] / 2.0)
                scores.append(param_anomaly)
            except Exception as e:
                logger.debug(f"Failed to calculate isolation forest score: {e}")
        
        # Return weighted average of scores
        return statistics.mean(scores) if scores else 0.0
    
    def _flag_suspicious_node(self, node_id: str, anomaly_score: float):
        """Flag a node as suspicious based on anomaly score"""
        if node_id not in self.suspected_nodes:
            self.suspected_nodes.add(node_id)
            
            evidence = ByzantineEvidence(
                evidence_id=f"statistical_{node_id}_{int(time.time())}",
                suspect_id=node_id,
                evidence_type=EvidenceType.STATISTICAL_OUTLIER,
                fault_type=FaultType.UNKNOWN,
                timestamp=time.time(),
                confidence=anomaly_score,
                details={
                    'anomaly_score': anomaly_score,
                    'threshold': self.detection_threshold
                }
            )
            
            self._collect_evidence(evidence)
            
            logger.warning(f"Node {node_id} flagged as suspicious (score: {anomaly_score:.3f})")
    
    def _collect_evidence(self, evidence: ByzantineEvidence):
        """Collect and store Byzantine evidence"""
        self.evidence_log.append(evidence)
        self.metrics['evidence_collected'] += 1
        
        # Keep evidence window manageable
        if len(self.evidence_log) > self.evidence_window:
            self.evidence_log = self.evidence_log[-self.evidence_window:]
        
        logger.info(f"Collected evidence: {evidence.evidence_type.value} for {evidence.suspect_id}")
        
        # Trigger callback if registered
        if self.on_byzantine_detected:
            self.on_byzantine_detected(evidence)
        
        # Auto-verify if confidence is very high
        if evidence.confidence > 0.9:
            asyncio.create_task(self._verify_evidence(evidence))
    
    async def _verify_evidence(self, evidence: ByzantineEvidence) -> bool:
        """Verify Byzantine evidence through consensus"""
        try:
            # Collect evidence from other nodes for verification
            # In a real implementation, this would query other nodes
            
            # For now, mark as verified if confidence is high enough
            if evidence.confidence > self.detection_threshold:
                evidence.verified = True
                
                # Add to verified Byzantine set if enough evidence
                suspect_evidence = [
                    e for e in self.evidence_log 
                    if e.suspect_id == evidence.suspect_id and e.verified
                ]
                
                if len(suspect_evidence) >= 3:  # Multiple verified pieces of evidence
                    self.verified_byzantine.add(evidence.suspect_id)
                    
                    if self.on_node_verified_byzantine:
                        self.on_node_verified_byzantine(evidence.suspect_id)
                    
                    logger.error(f"Node {evidence.suspect_id} verified as Byzantine!")
                
                self.metrics['verifications_performed'] += 1
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Evidence verification failed: {e}")
            return False
    
    def _verify_message_signature(self, message: ConsensusMessage) -> bool:
        """Verify message signature (simplified for demo)"""
        # In real implementation, would verify using public key
        return message.signature is not None and len(message.signature) > 0
    
    def _verify_topology_proof(self, message: ConsensusMessage) -> bool:
        """Verify topology proof consistency (simplified for demo)"""
        # In real implementation, would check against current topology
        if not message.topology_proof:
            return False
        
        return 'fingerprint' in message.topology_proof
    
    def get_suspicious_nodes(self) -> List[str]:
        """Get list of nodes currently under suspicion"""
        return list(self.suspected_nodes)
    
    def get_verified_byzantine_nodes(self) -> List[str]:
        """Get list of verified Byzantine nodes"""
        return list(self.verified_byzantine)
    
    def get_node_trust_score(self, node_id: str) -> float:
        """Get trust score for a node (1.0 = fully trusted, 0.0 = Byzantine)"""
        if node_id in self.verified_byzantine:
            return 0.0
        
        if node_id in self.node_profiles:
            return 1.0 - self.node_profiles[node_id].anomaly_score
        
        return 0.5  # Unknown node gets neutral score
    
    def get_evidence_for_node(self, node_id: str) -> List[ByzantineEvidence]:
        """Get all evidence collected for a specific node"""
        return [e for e in self.evidence_log if e.suspect_id == node_id]
    
    def byzantine_robust_aggregation(
        self,
        values: List[Tuple[str, torch.Tensor]],
        aggregation_method: str = "trimmed_mean"
    ) -> torch.Tensor:
        """
        Perform Byzantine-robust aggregation of values
        
        Args:
            values: List of (node_id, tensor) pairs
            aggregation_method: Aggregation method to use
            
        Returns:
            Aggregated tensor
        """
        if not values:
            raise ValueError("No values to aggregate")
        
        # Filter out verified Byzantine nodes
        filtered_values = [
            (node_id, tensor) for node_id, tensor in values
            if node_id not in self.verified_byzantine
        ]
        
        if not filtered_values:
            logger.warning("All nodes are Byzantine, using original values")
            filtered_values = values
        
        tensors = [tensor for _, tensor in filtered_values]
        
        if aggregation_method == "trimmed_mean":
            return self._trimmed_mean_aggregation(tensors)
        elif aggregation_method == "median":
            return self._median_aggregation(tensors)
        elif aggregation_method == "trust_weighted":
            return self._trust_weighted_aggregation(filtered_values)
        else:
            # Default: simple average
            return torch.mean(torch.stack(tensors), dim=0)
    
    def _trimmed_mean_aggregation(
        self,
        tensors: List[torch.Tensor],
        trim_ratio: float = 0.2
    ) -> torch.Tensor:
        """Trimmed mean aggregation (removes outliers)"""
        if len(tensors) == 1:
            return tensors[0]
        
        stacked = torch.stack(tensors)
        
        # Sort along the batch dimension
        sorted_tensors, _ = torch.sort(stacked, dim=0)
        
        # Calculate trim indices
        n = len(tensors)
        trim_count = int(n * trim_ratio // 2)
        
        if trim_count > 0:
            # Remove top and bottom trim_count values
            trimmed = sorted_tensors[trim_count:-trim_count]
        else:
            trimmed = sorted_tensors
        
        return torch.mean(trimmed, dim=0)
    
    def _median_aggregation(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Median aggregation (robust to outliers)"""
        stacked = torch.stack(tensors)
        return torch.median(stacked, dim=0)[0]
    
    def _trust_weighted_aggregation(
        self,
        values: List[Tuple[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Trust-weighted aggregation based on node trust scores"""
        weights = []
        tensors = []
        
        for node_id, tensor in values:
            trust_score = self.get_node_trust_score(node_id)
            weights.append(trust_score)
            tensors.append(tensor)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            # All nodes have zero trust, use equal weights
            weights = [1.0] * len(weights)
            total_weight = len(weights)
        
        normalized_weights = [w / total_weight for w in weights]
        
        # Compute weighted average
        result = torch.zeros_like(tensors[0])
        for weight, tensor in zip(normalized_weights, tensors):
            result += weight * tensor
        
        return result
    
    def update_adaptive_thresholds(self, network_conditions: Dict[str, float]):
        """Update detection thresholds based on network conditions"""
        # Adjust thresholds based on network latency, load, etc.
        base_multiplier = network_conditions.get('latency_multiplier', 1.0)
        
        self.adaptive_thresholds['timing_deviation'] = 2.0 * base_multiplier
        self.adaptive_thresholds['parameter_anomaly'] = 0.8 / base_multiplier
        
        logger.debug(f"Updated adaptive thresholds: {self.adaptive_thresholds}")
    
    def get_detection_metrics(self) -> Dict[str, Any]:
        """Get detection performance metrics"""
        return {
            **self.metrics,
            'suspected_nodes': len(self.suspected_nodes),
            'verified_byzantine': len(self.verified_byzantine),
            'evidence_count': len(self.evidence_log),
            'active_profiles': len(self.node_profiles),
            'detection_threshold': self.detection_threshold,
            'adaptive_thresholds': self.adaptive_thresholds.copy()
        }
    
    def reset_node_suspicion(self, node_id: str):
        """Reset suspicion for a node (for testing/recovery)"""
        self.suspected_nodes.discard(node_id)
        if node_id in self.node_profiles:
            self.node_profiles[node_id].anomaly_score = 0.0
        
        # Remove evidence
        self.evidence_log = [
            e for e in self.evidence_log 
            if e.suspect_id != node_id
        ]
        
        logger.info(f"Reset suspicion for node {node_id}")


class ByzantineToleranceManager:
    """
    High-level manager for Byzantine fault tolerance
    
    Coordinates detection, reporting, and recovery mechanisms
    """
    
    def __init__(
        self,
        node_id: str,
        fault_detector: ByzantineFaultDetector
    ):
        self.node_id = node_id
        self.detector = fault_detector
        
        # Set up callbacks
        self.detector.on_byzantine_detected = self._handle_byzantine_detection
        self.detector.on_node_verified_byzantine = self._handle_verified_byzantine
        
        # Recovery strategies
        self.recovery_strategies = {
            FaultType.SILENT: self._handle_silent_fault,
            FaultType.CRASH: self._handle_crash_fault,
            FaultType.ARBITRARY: self._handle_arbitrary_fault,
            FaultType.TIMING: self._handle_timing_fault
        }
        
        logger.info(f"Byzantine tolerance manager initialized for {node_id}")
    
    def _handle_byzantine_detection(self, evidence: ByzantineEvidence):
        """Handle new Byzantine evidence"""
        logger.warning(f"Byzantine behavior detected: {evidence.evidence_type.value} "
                      f"from {evidence.suspect_id} (confidence: {evidence.confidence:.3f})")
        
        # Trigger appropriate recovery strategy
        recovery_func = self.recovery_strategies.get(evidence.fault_type)
        if recovery_func:
            asyncio.create_task(recovery_func(evidence))
    
    def _handle_verified_byzantine(self, node_id: str):
        """Handle verified Byzantine node"""
        logger.error(f"Node {node_id} verified as Byzantine - initiating isolation")
        
        # In production, would:
        # 1. Exclude from consensus
        # 2. Ignore future messages
        # 3. Report to network administrators
        # 4. Update routing tables
    
    async def _handle_silent_fault(self, evidence: ByzantineEvidence):
        """Handle silent/non-responsive nodes"""
        logger.info(f"Handling silent fault from {evidence.suspect_id}")
        # Implement timeout-based recovery
    
    async def _handle_crash_fault(self, evidence: ByzantineEvidence):
        """Handle crashed nodes"""
        logger.info(f"Handling crash fault from {evidence.suspect_id}")
        # Implement crash recovery mechanisms
    
    async def _handle_arbitrary_fault(self, evidence: ByzantineEvidence):
        """Handle arbitrary/malicious behavior"""
        logger.warning(f"Handling arbitrary fault from {evidence.suspect_id}")
        # Implement strong isolation mechanisms
    
    async def _handle_timing_fault(self, evidence: ByzantineEvidence):
        """Handle timing violations"""
        logger.info(f"Handling timing fault from {evidence.suspect_id}")
        # Implement timeout adjustments
    
    def get_network_health(self) -> Dict[str, Any]:
        """Get overall network health metrics"""
        metrics = self.detector.get_detection_metrics()
        
        total_nodes = len(self.detector.node_profiles)
        byzantine_ratio = len(self.detector.verified_byzantine) / max(total_nodes, 1)
        
        return {
            **metrics,
            'total_nodes': total_nodes,
            'byzantine_ratio': byzantine_ratio,
            'network_trust_score': 1.0 - byzantine_ratio,
            'consensus_safety': byzantine_ratio < 0.33  # BFT threshold
        }