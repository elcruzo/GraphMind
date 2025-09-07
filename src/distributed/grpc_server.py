"""
gRPC Server Implementation for GraphMind Distributed Consensus

This module implements the gRPC server for handling distributed consensus,
health monitoring, and model synchronization in Byzantine fault-tolerant
graph neural network training.

Author: Ayomide Caleb Adekoya
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, AsyncIterator

import grpc
from grpc import aio
import torch
import numpy as np
import networkx as nx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

# Import generated protobuf classes
# Note: These would be generated from the .proto file using:
# python -m grpc_tools.protoc --python_out=. --grpc_python_out=. consensus.proto

try:
    import consensus_pb2 as pb2
    import consensus_pb2_grpc as pb2_grpc
except ImportError:
    # For development, create mock classes
    class MockPb2:
        def __getattr__(self, name):
            return lambda **kwargs: type('MockMessage', (), kwargs)()
    
    class MockPb2Grpc:
        def __getattr__(self, name):
            return type('MockServicer', (), {})
    
    pb2 = MockPb2()
    pb2_grpc = MockPb2Grpc()

from .node_discovery import NodeDiscoveryService, NodeInfo, NodeStatus
from ..consensus.ta_bft import TopologyAwareBFT, ConsensusMessage, MessageType

logger = logging.getLogger(__name__)


class ConsensusServicer(pb2_grpc.ConsensusServiceServicer):
    """
    gRPC servicer for consensus operations
    """
    
    def __init__(
        self,
        node_id: str,
        consensus_algorithm: TopologyAwareBFT,
        node_discovery: NodeDiscoveryService
    ):
        self.node_id = node_id
        self.consensus = consensus_algorithm
        self.discovery = node_discovery
        
        # Message queues for streaming
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.active_streams: Dict[str, bool] = {}
        
        logger.info(f"Consensus servicer initialized for node {node_id}")
    
    async def ProposeValue(self, request, context):
        """Handle consensus value proposal"""
        try:
            # Convert protobuf message to internal format
            message = self._pb_to_consensus_message(request)
            
            # Validate message signature
            if not self._validate_message_signature(message):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Invalid message signature")
                return pb2.ConsensusResult()
            
            # Execute consensus
            result = await self.consensus.propose_value(message.proposal)
            
            # Convert result to protobuf
            return self._consensus_result_to_pb(result)
            
        except Exception as e:
            logger.error(f"ProposeValue error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.ConsensusResult()
    
    async def SendMessage(self, request, context):
        """Handle consensus message sending"""
        try:
            message = self._pb_to_consensus_message(request)
            
            if not self._validate_message_signature(message):
                return pb2.MessageAck(
                    success=False,
                    message="Invalid signature",
                    timestamp=int(time.time() * 1000)
                )
            
            # Process message based on type
            await self._process_consensus_message(message)
            
            # Broadcast to active streams
            await self._broadcast_to_streams(request)
            
            return pb2.MessageAck(
                success=True,
                message="Message processed",
                timestamp=int(time.time() * 1000)
            )
            
        except Exception as e:
            logger.error(f"SendMessage error: {e}")
            return pb2.MessageAck(
                success=False,
                message=str(e),
                timestamp=int(time.time() * 1000)
            )
    
    async def ConsensusStream(self, request_iterator, context):
        """Bidirectional streaming for real-time consensus"""
        client_id = context.peer()
        logger.info(f"Starting consensus stream for {client_id}")
        
        # Create message queue for this client
        message_queue = asyncio.Queue()
        self.message_queues[client_id] = message_queue
        self.active_streams[client_id] = True
        
        try:
            # Start tasks for handling incoming and outgoing messages
            incoming_task = asyncio.create_task(
                self._handle_incoming_stream(request_iterator, client_id)
            )
            outgoing_task = asyncio.create_task(
                self._handle_outgoing_stream(message_queue)
            )
            
            # Wait for either task to complete
            done, pending = await asyncio.wait(
                [incoming_task, outgoing_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
            
        finally:
            # Cleanup
            self.active_streams[client_id] = False
            if client_id in self.message_queues:
                del self.message_queues[client_id]
            
            logger.info(f"Consensus stream ended for {client_id}")
    
    async def _handle_incoming_stream(self, request_iterator, client_id: str):
        """Handle incoming messages from consensus stream"""
        try:
            async for request in request_iterator:
                message = self._pb_to_consensus_message(request)
                
                if self._validate_message_signature(message):
                    await self._process_consensus_message(message)
                    
                    # Echo to other streams
                    await self._broadcast_to_streams(request, exclude=client_id)
                else:
                    logger.warning(f"Invalid signature from {client_id}")
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Incoming stream error for {client_id}: {e}")
    
    async def _handle_outgoing_stream(self, message_queue: asyncio.Queue) -> AsyncIterator:
        """Handle outgoing messages to consensus stream"""
        try:
            while True:
                message = await message_queue.get()
                yield message
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Outgoing stream error: {e}")
    
    async def _broadcast_to_streams(self, message, exclude: str = None):
        """Broadcast message to all active streams"""
        for client_id, queue in self.message_queues.items():
            if client_id != exclude and self.active_streams.get(client_id, False):
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    logger.warning(f"Message queue full for {client_id}")
                except Exception as e:
                    logger.error(f"Broadcast error to {client_id}: {e}")
    
    async def HealthCheck(self, request, context):
        """Handle health check request"""
        try:
            node_info = self.discovery.nodes.get(request.node_id)
            
            if node_info:
                return pb2.HealthCheckResponse(
                    node_id=self.node_id,
                    status=self._node_status_to_pb(node_info.status),
                    health_score=node_info.health_score,
                    timestamp=int(time.time() * 1000),
                    metrics=self._get_health_metrics()
                )
            else:
                return pb2.HealthCheckResponse(
                    node_id=self.node_id,
                    status=pb2.NODE_UNKNOWN,
                    health_score=0.0,
                    timestamp=int(time.time() * 1000)
                )
                
        except Exception as e:
            logger.error(f"HealthCheck error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.HealthCheckResponse()
    
    def _pb_to_consensus_message(self, pb_message) -> ConsensusMessage:
        """Convert protobuf message to internal ConsensusMessage"""
        try:
            # Map protobuf MessageType to internal MessageType
            msg_type_map = {
                pb2.PREPARE: MessageType.PREPARE,
                pb2.PROMISE: MessageType.PROMISE,
                pb2.COMMIT: MessageType.COMMIT,
                pb2.VIEW_CHANGE: MessageType.VIEW_CHANGE,
                pb2.NEW_VIEW: MessageType.NEW_VIEW
            }
            
            topology_proof = None
            if pb_message.topology_proof:
                topology_proof = {
                    'fingerprint': pb_message.topology_proof.fingerprint,
                    'centrality_weight': pb_message.topology_proof.centrality_weight,
                    'timestamp': pb_message.topology_proof.timestamp,
                    'neighbor_weights': dict(pb_message.topology_proof.neighbor_weights)
                }
            
            return ConsensusMessage(
                msg_type=msg_type_map.get(pb_message.msg_type, MessageType.PREPARE),
                view=pb_message.view,
                sequence=pb_message.sequence,
                proposal=pb_message.proposal,
                sender_id=pb_message.sender_id,
                timestamp=pb_message.timestamp,
                topology_proof=topology_proof,
                signature=pb_message.signature
            )
            
        except Exception as e:
            logger.error(f"Failed to convert protobuf message: {e}")
            raise
    
    def _consensus_result_to_pb(self, result):
        """Convert internal ConsensusResult to protobuf"""
        return pb2.ConsensusResult(
            decided=result.decided,
            value=result.value if isinstance(result.value, bytes) else json.dumps(result.value).encode(),
            view=result.view,
            sequence=result.sequence,
            rounds=result.rounds,
            proof={k: v for k, v in result.proof.items()},
            execution_time=result.execution_time
        )
    
    def _node_status_to_pb(self, status: NodeStatus):
        """Convert internal NodeStatus to protobuf"""
        status_map = {
            NodeStatus.JOINING: pb2.NODE_JOINING,
            NodeStatus.ACTIVE: pb2.NODE_ACTIVE,
            NodeStatus.UNHEALTHY: pb2.NODE_UNHEALTHY,
            NodeStatus.LEAVING: pb2.NODE_LEAVING,
            NodeStatus.OFFLINE: pb2.NODE_OFFLINE
        }
        return status_map.get(status, pb2.NODE_UNKNOWN)
    
    def _validate_message_signature(self, message: ConsensusMessage) -> bool:
        """Validate message signature using sender's public key"""
        try:
            # Get public key for sender
            node_info = self.discovery.nodes.get(message.sender_id)
            if not node_info:
                return False
            
            public_key = serialization.load_pem_public_key(node_info.public_key)
            
            # Create message bytes for verification
            message_dict = message.to_dict()
            message_bytes = json.dumps(message_dict, sort_keys=True).encode()
            
            # Verify signature
            public_key.verify(
                message.signature,
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception as e:
            logger.warning(f"Signature validation failed: {e}")
            return False
    
    async def _process_consensus_message(self, message: ConsensusMessage):
        """Process incoming consensus message"""
        try:
            # Update metrics
            self.consensus.metrics['messages_received'] += 1
            
            # Handle different message types
            if message.msg_type == MessageType.PREPARE:
                await self._handle_prepare_message(message)
            elif message.msg_type == MessageType.PROMISE:
                await self._handle_promise_message(message)
            elif message.msg_type == MessageType.COMMIT:
                await self._handle_commit_message(message)
            elif message.msg_type == MessageType.VIEW_CHANGE:
                await self._handle_view_change_message(message)
            
            # Store in message log
            if message.sequence not in self.consensus.message_log:
                self.consensus.message_log[message.sequence] = []
            self.consensus.message_log[message.sequence].append(message)
            
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    async def _handle_prepare_message(self, message: ConsensusMessage):
        """Handle PREPARE message in consensus protocol"""
        logger.debug(f"Handling PREPARE from {message.sender_id}, seq={message.sequence}")
        # Implementation would integrate with consensus algorithm
    
    async def _handle_promise_message(self, message: ConsensusMessage):
        """Handle PROMISE message in consensus protocol"""
        logger.debug(f"Handling PROMISE from {message.sender_id}, seq={message.sequence}")
    
    async def _handle_commit_message(self, message: ConsensusMessage):
        """Handle COMMIT message in consensus protocol"""
        logger.debug(f"Handling COMMIT from {message.sender_id}, seq={message.sequence}")
    
    async def _handle_view_change_message(self, message: ConsensusMessage):
        """Handle VIEW_CHANGE message in consensus protocol"""
        logger.debug(f"Handling VIEW_CHANGE from {message.sender_id}, view={message.view}")
    
    def _get_health_metrics(self) -> Dict[str, float]:
        """Get current health metrics"""
        return {
            'cpu_usage': 0.5,  # Would get actual metrics
            'memory_usage': 0.3,
            'disk_usage': 0.2,
            'consensus_latency': np.mean(self.consensus.metrics.get('consensus_latency', [0])),
            'message_throughput': len(self.consensus.metrics.get('consensus_latency', [])) / 60.0
        }


class ModelSyncServicer(pb2_grpc.ModelSyncServiceServicer):
    """
    gRPC servicer for model synchronization
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.models: Dict[str, torch.nn.Module] = {}
        self.model_versions: Dict[str, int] = {}
        
    async def SyncModel(self, request, context):
        """Synchronize model parameters"""
        try:
            model_id = request.model_id
            current_version = request.current_version
            
            if model_id in self.models and self.model_versions.get(model_id, 0) > current_version:
                # Return model updates
                model = self.models[model_id]
                parameters = self._serialize_model_parameters(model)
                
                update = pb2.ModelUpdate(
                    model_id=model_id,
                    version=self.model_versions[model_id],
                    parameters=parameters,
                    sender_id=self.node_id,
                    timestamp=int(time.time() * 1000)
                )
                
                return pb2.ModelSyncResponse(
                    model_id=model_id,
                    updates=[update],
                    success=True
                )
            else:
                return pb2.ModelSyncResponse(
                    model_id=model_id,
                    updates=[],
                    success=True,
                    error_message="No updates available"
                )
                
        except Exception as e:
            logger.error(f"Model sync error: {e}")
            return pb2.ModelSyncResponse(
                model_id=request.model_id,
                updates=[],
                success=False,
                error_message=str(e)
            )
    
    def _serialize_model_parameters(self, model: torch.nn.Module) -> bytes:
        """Serialize model parameters to bytes"""
        state_dict = model.state_dict()
        return torch.save(state_dict, None)  # Save to bytes


class GraphMindGrpcServer:
    """
    Main gRPC server for GraphMind distributed operations
    """
    
    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        consensus_algorithm: TopologyAwareBFT,
        node_discovery: NodeDiscoveryService,
        max_workers: int = 10
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.consensus = consensus_algorithm
        self.discovery = node_discovery
        self.max_workers = max_workers
        
        # gRPC server
        self.server = None
        
        # Servicers
        self.consensus_servicer = ConsensusServicer(
            node_id, consensus_algorithm, node_discovery
        )
        self.model_sync_servicer = ModelSyncServicer(node_id)
        
        logger.info(f"gRPC server initialized for {node_id} on {host}:{port}")
    
    async def start(self):
        """Start the gRPC server"""
        self.server = aio.server(
            ThreadPoolExecutor(max_workers=self.max_workers)
        )
        
        # Add servicers
        pb2_grpc.add_ConsensusServiceServicer_to_server(
            self.consensus_servicer, self.server
        )
        pb2_grpc.add_ModelSyncServiceServicer_to_server(
            self.model_sync_servicer, self.server
        )
        
        # Configure server
        listen_addr = f"{self.host}:{self.port}"
        self.server.add_insecure_port(listen_addr)
        
        # Start server
        await self.server.start()
        logger.info(f"gRPC server started on {listen_addr}")
        
        # Keep server running
        await self.server.wait_for_termination()
    
    async def stop(self):
        """Stop the gRPC server"""
        if self.server:
            await self.server.stop(grace=5)
            logger.info("gRPC server stopped")
    
    def get_consensus_servicer(self) -> ConsensusServicer:
        """Get consensus servicer instance"""
        return self.consensus_servicer
    
    def get_model_sync_servicer(self) -> ModelSyncServicer:
        """Get model sync servicer instance"""
        return self.model_sync_servicer