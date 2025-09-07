"""
gRPC Client Implementation for GraphMind Distributed Consensus

This module provides client-side gRPC communication for distributed consensus,
health monitoring, and model synchronization.

Author: Ayomide Caleb Adekoya
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, AsyncIterator, Any

import grpc
from grpc import aio
import torch
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

# Import generated protobuf classes
try:
    import consensus_pb2 as pb2
    import consensus_pb2_grpc as pb2_grpc
except ImportError:
    # Mock for development
    class MockPb2:
        def __getattr__(self, name):
            return lambda **kwargs: type('MockMessage', (), kwargs)()
    
    class MockPb2Grpc:
        def __getattr__(self, name):
            return type('MockStub', (), {
                '__getattr__': lambda self, method: lambda *args, **kwargs: None
            })()
    
    pb2 = MockPb2()
    pb2_grpc = MockPb2Grpc()

from .node_discovery import NodeInfo, NodeStatus
from ..consensus.ta_bft import ConsensusMessage, MessageType, ConsensusResult

logger = logging.getLogger(__name__)


class GraphMindGrpcClient:
    """
    gRPC client for GraphMind distributed operations
    
    Provides methods for consensus communication, health monitoring,
    and model synchronization with Byzantine fault tolerance.
    """
    
    def __init__(
        self,
        node_id: str,
        private_key: rsa.RSAPrivateKey,
        connection_timeout: float = 30.0,
        retry_attempts: int = 3
    ):
        self.node_id = node_id
        self.private_key = private_key
        self.connection_timeout = connection_timeout
        self.retry_attempts = retry_attempts
        
        # Connection pool
        self.connections: Dict[str, aio.Channel] = {}
        self.stubs: Dict[str, Dict[str, Any]] = {}
        
        # Streaming connections
        self.consensus_streams: Dict[str, AsyncIterator] = {}
        
        # Metrics
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'connection_failures': 0,
            'consensus_calls': 0,
            'health_checks': 0
        }
        
        logger.info(f"gRPC client initialized for node {node_id}")
    
    async def connect_to_node(self, node_info: NodeInfo) -> bool:
        """
        Establish gRPC connection to a node
        
        Args:
            node_info: Target node information
            
        Returns:
            True if connection successful, False otherwise
        """
        address = f"{node_info.hostname}:{node_info.port}"
        
        try:
            if node_info.node_id in self.connections:
                # Check if existing connection is healthy
                if await self._test_connection(node_info.node_id):
                    return True
                else:
                    await self.disconnect_from_node(node_info.node_id)
            
            # Create new connection
            channel = aio.insecure_channel(
                address,
                options=[
                    ('grpc.keepalive_time_ms', 10000),
                    ('grpc.keepalive_timeout_ms', 5000),
                    ('grpc.keepalive_permit_without_calls', True),
                    ('grpc.http2.max_pings_without_data', 0),
                    ('grpc.http2.min_ping_interval_without_data_ms', 300000)
                ]
            )
            
            # Create stubs
            consensus_stub = pb2_grpc.ConsensusServiceStub(channel)
            model_sync_stub = pb2_grpc.ModelSyncServiceStub(channel)
            byzantine_stub = pb2_grpc.ByzantineDetectionServiceStub(channel)
            
            # Test connection
            try:
                health_request = pb2.HealthCheckRequest(
                    node_id=self.node_id,
                    timestamp=int(time.time() * 1000)
                )
                
                await asyncio.wait_for(
                    consensus_stub.HealthCheck(health_request),
                    timeout=self.connection_timeout
                )
                
            except Exception as e:
                await channel.close()
                raise Exception(f"Connection test failed: {e}")
            
            # Store connection
            self.connections[node_info.node_id] = channel
            self.stubs[node_info.node_id] = {
                'consensus': consensus_stub,
                'model_sync': model_sync_stub,
                'byzantine': byzantine_stub
            }
            
            logger.info(f"Connected to node {node_info.node_id} at {address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {node_info.node_id} at {address}: {e}")
            self.metrics['connection_failures'] += 1
            return False
    
    async def disconnect_from_node(self, node_id: str):
        """Disconnect from a node"""
        try:
            if node_id in self.connections:
                await self.connections[node_id].close()
                del self.connections[node_id]
                
            if node_id in self.stubs:
                del self.stubs[node_id]
                
            if node_id in self.consensus_streams:
                del self.consensus_streams[node_id]
                
            logger.info(f"Disconnected from node {node_id}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from {node_id}: {e}")
    
    async def _test_connection(self, node_id: str) -> bool:
        """Test if connection to node is healthy"""
        try:
            if node_id not in self.stubs:
                return False
            
            stub = self.stubs[node_id]['consensus']
            health_request = pb2.HealthCheckRequest(
                node_id=self.node_id,
                timestamp=int(time.time() * 1000)
            )
            
            await asyncio.wait_for(
                stub.HealthCheck(health_request),
                timeout=5.0
            )
            
            return True
            
        except Exception:
            return False
    
    async def send_consensus_message(
        self,
        target_node_id: str,
        message: ConsensusMessage
    ) -> Optional[Any]:
        """
        Send consensus message to target node
        
        Args:
            target_node_id: Target node identifier
            message: Consensus message to send
            
        Returns:
            Message acknowledgment or None on failure
        """
        try:
            if target_node_id not in self.stubs:
                logger.warning(f"No connection to {target_node_id}")
                return None
            
            stub = self.stubs[target_node_id]['consensus']
            
            # Convert to protobuf message
            pb_message = self._consensus_message_to_pb(message)
            
            # Send message with retry logic
            for attempt in range(self.retry_attempts):
                try:
                    response = await asyncio.wait_for(
                        stub.SendMessage(pb_message),
                        timeout=self.connection_timeout
                    )
                    
                    self.metrics['messages_sent'] += 1
                    self.metrics['consensus_calls'] += 1
                    
                    return response
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout sending to {target_node_id}, attempt {attempt + 1}")
                    if attempt == self.retry_attempts - 1:
                        raise
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                    
        except Exception as e:
            logger.error(f"Failed to send consensus message to {target_node_id}: {e}")
            return None
    
    async def propose_consensus_value(
        self,
        target_node_id: str,
        value: Any
    ) -> Optional[ConsensusResult]:
        """
        Propose value for consensus to target node
        
        Args:
            target_node_id: Target node identifier
            value: Value to propose for consensus
            
        Returns:
            Consensus result or None on failure
        """
        try:
            if target_node_id not in self.stubs:
                return None
            
            stub = self.stubs[target_node_id]['consensus']
            
            # Create consensus message
            message = ConsensusMessage(
                msg_type=MessageType.PREPARE,
                view=0,
                sequence=int(time.time() * 1000),
                proposal=value,
                sender_id=self.node_id,
                timestamp=time.time(),
                topology_proof=self._create_topology_proof()
            )
            
            # Sign message
            message.signature = self._sign_message(message)
            
            # Convert to protobuf
            pb_message = self._consensus_message_to_pb(message)
            
            # Send proposal
            response = await asyncio.wait_for(
                stub.ProposeValue(pb_message),
                timeout=self.connection_timeout
            )
            
            self.metrics['consensus_calls'] += 1
            
            # Convert response to internal format
            return self._pb_to_consensus_result(response)
            
        except Exception as e:
            logger.error(f"Failed to propose consensus value to {target_node_id}: {e}")
            return None
    
    async def start_consensus_stream(self, target_node_id: str) -> Optional[AsyncIterator]:
        """
        Start bidirectional consensus stream with target node
        
        Args:
            target_node_id: Target node identifier
            
        Returns:
            Stream iterator or None on failure
        """
        try:
            if target_node_id not in self.stubs:
                return None
            
            stub = self.stubs[target_node_id]['consensus']
            
            # Create message queue for outgoing messages
            outgoing_queue = asyncio.Queue()
            
            async def message_generator():
                while True:
                    message = await outgoing_queue.get()
                    if message is None:  # Sentinel for stream termination
                        break
                    yield message
            
            # Start streaming call
            stream = stub.ConsensusStream(message_generator())
            
            self.consensus_streams[target_node_id] = {
                'stream': stream,
                'queue': outgoing_queue
            }
            
            logger.info(f"Started consensus stream with {target_node_id}")
            return stream
            
        except Exception as e:
            logger.error(f"Failed to start consensus stream with {target_node_id}: {e}")
            return None
    
    async def health_check(self, target_node_id: str) -> Optional[Dict[str, Any]]:
        """
        Perform health check on target node
        
        Args:
            target_node_id: Target node identifier
            
        Returns:
            Health status information or None on failure
        """
        try:
            if target_node_id not in self.stubs:
                return None
            
            stub = self.stubs[target_node_id]['consensus']
            
            request = pb2.HealthCheckRequest(
                node_id=self.node_id,
                timestamp=int(time.time() * 1000)
            )
            
            response = await asyncio.wait_for(
                stub.HealthCheck(request),
                timeout=10.0
            )
            
            self.metrics['health_checks'] += 1
            
            return {
                'node_id': response.node_id,
                'status': self._pb_to_node_status(response.status),
                'health_score': response.health_score,
                'timestamp': response.timestamp,
                'metrics': dict(response.metrics)
            }
            
        except Exception as e:
            logger.warning(f"Health check failed for {target_node_id}: {e}")
            return None
    
    async def sync_model(
        self,
        target_node_id: str,
        model_id: str,
        current_version: int
    ) -> Optional[List[Any]]:
        """
        Synchronize model with target node
        
        Args:
            target_node_id: Target node identifier
            model_id: Model identifier
            current_version: Current model version
            
        Returns:
            List of model updates or None on failure
        """
        try:
            if target_node_id not in self.stubs:
                return None
            
            stub = self.stubs[target_node_id]['model_sync']
            
            request = pb2.ModelSyncRequest(
                model_id=model_id,
                current_version=current_version,
                requester_id=self.node_id
            )
            
            response = await asyncio.wait_for(
                stub.SyncModel(request),
                timeout=30.0
            )
            
            if response.success:
                return list(response.updates)
            else:
                logger.error(f"Model sync failed: {response.error_message}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to sync model with {target_node_id}: {e}")
            return None
    
    async def broadcast_to_all(
        self,
        message: ConsensusMessage,
        exclude_nodes: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Broadcast consensus message to all connected nodes
        
        Args:
            message: Message to broadcast
            exclude_nodes: Nodes to exclude from broadcast
            
        Returns:
            Dictionary mapping node IDs to success status
        """
        exclude_nodes = exclude_nodes or []
        results = {}
        
        tasks = []
        node_ids = []
        
        for node_id in self.stubs:
            if node_id not in exclude_nodes:
                task = self.send_consensus_message(node_id, message)
                tasks.append(task)
                node_ids.append(node_id)
        
        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for node_id, response in zip(node_ids, responses):
                results[node_id] = not isinstance(response, Exception)
        
        return results
    
    def _consensus_message_to_pb(self, message: ConsensusMessage):
        """Convert ConsensusMessage to protobuf"""
        msg_type_map = {
            MessageType.PREPARE: pb2.PREPARE,
            MessageType.PROMISE: pb2.PROMISE,
            MessageType.COMMIT: pb2.COMMIT,
            MessageType.VIEW_CHANGE: pb2.VIEW_CHANGE,
            MessageType.NEW_VIEW: pb2.NEW_VIEW
        }
        
        topology_proof = None
        if message.topology_proof:
            topology_proof = pb2.TopologyProof(
                fingerprint=message.topology_proof.get('fingerprint', ''),
                centrality_weight=message.topology_proof.get('centrality_weight', 0.0),
                timestamp=message.topology_proof.get('timestamp', 0),
                neighbor_weights=message.topology_proof.get('neighbor_weights', {})
            )
        
        return pb2.ConsensusMessage(
            msg_type=msg_type_map.get(message.msg_type, pb2.PREPARE),
            view=message.view,
            sequence=message.sequence,
            proposal=message.proposal if isinstance(message.proposal, bytes) else json.dumps(message.proposal).encode(),
            sender_id=message.sender_id,
            timestamp=int(message.timestamp * 1000),
            topology_proof=topology_proof,
            signature=message.signature or b''
        )
    
    def _pb_to_consensus_result(self, pb_result) -> ConsensusResult:
        """Convert protobuf ConsensusResult to internal format"""
        return ConsensusResult(
            decided=pb_result.decided,
            value=pb_result.value,
            view=pb_result.view,
            sequence=pb_result.sequence,
            rounds=pb_result.rounds,
            proof=dict(pb_result.proof),
            execution_time=pb_result.execution_time
        )
    
    def _pb_to_node_status(self, pb_status) -> NodeStatus:
        """Convert protobuf NodeStatus to internal format"""
        status_map = {
            pb2.NODE_JOINING: NodeStatus.JOINING,
            pb2.NODE_ACTIVE: NodeStatus.ACTIVE,
            pb2.NODE_UNHEALTHY: NodeStatus.UNHEALTHY,
            pb2.NODE_LEAVING: NodeStatus.LEAVING,
            pb2.NODE_OFFLINE: NodeStatus.OFFLINE
        }
        return status_map.get(pb_status, NodeStatus.OFFLINE)
    
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
    
    def _create_topology_proof(self) -> Dict[str, Any]:
        """Create topology proof for messages"""
        return {
            'fingerprint': 'mock_fingerprint',  # Would use actual topology
            'centrality_weight': 1.0,
            'timestamp': time.time()
        }
    
    async def cleanup(self):
        """Clean up all connections"""
        for node_id in list(self.connections.keys()):
            await self.disconnect_from_node(node_id)
        
        logger.info("gRPC client cleanup completed")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics"""
        return {
            **self.metrics,
            'connected_nodes': len(self.connections),
            'active_streams': len(self.consensus_streams)
        }