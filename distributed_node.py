#!/usr/bin/env python3
"""
GraphMind Distributed Node Integration Script

This script integrates all Phase 1 distributed infrastructure components:
- Node discovery and health monitoring
- gRPC communication for consensus
- Byzantine fault detection and tolerance
- TA-BFT consensus with topology awareness

Usage:
    python distributed_node.py --config config/node_config.yaml --node-id node1
    
Author: Ayomide Caleb Adekoya
"""

import asyncio
import argparse
import logging
import signal
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.distributed.node_discovery import (
    DistributedNodeManager, 
    NodeDiscoveryService, 
    ServiceBackend,
    HealthCheck
)
from src.distributed.grpc_server import GraphMindGrpcServer
from src.distributed.grpc_client import GraphMindGrpcClient
from src.consensus.ta_bft import TopologyAwareBFT
from src.byzantine.fault_detector import (
    ByzantineFaultDetector,
    ByzantineToleranceManager
)
import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('graphmind_node.log')
    ]
)
logger = logging.getLogger(__name__)


class GraphMindDistributedNode:
    """
    Main distributed node class integrating all Phase 1 components
    
    Features:
    - Service discovery and health monitoring
    - gRPC-based communication
    - Byzantine fault tolerance
    - TA-BFT consensus algorithm
    - Production-ready error handling and monitoring
    """
    
    def __init__(self, config: Dict[str, Any], node_id: str):
        self.config = config
        self.node_id = node_id
        
        # Node configuration
        self.hostname = config['node']['hostname']
        self.port = config['node']['port']
        self.grpc_port = config['node']['grpc_port']
        
        # Generate or load cryptographic keys
        self.private_key, self.public_key = self._setup_keys()
        
        # Core components
        self.node_manager: Optional[DistributedNodeManager] = None
        self.grpc_server: Optional[GraphMindGrpcServer] = None
        self.grpc_client: Optional[GraphMindGrpcClient] = None
        self.consensus: Optional[TopologyAwareBFT] = None
        self.fault_detector: Optional[ByzantineFaultDetector] = None
        self.byzantine_manager: Optional[ByzantineToleranceManager] = None
        
        # State management
        self.running = False
        self.startup_complete = False
        
        # Performance metrics
        self.metrics = {
            'start_time': 0,
            'messages_processed': 0,
            'consensus_rounds': 0,
            'byzantine_detections': 0,
            'uptime': 0
        }
        
        logger.info(f"GraphMind distributed node {node_id} initialized")
    
    def _setup_keys(self) -> tuple:
        """Setup cryptographic keys for the node"""
        key_file = Path(f"keys/{self.node_id}_private.pem")
        
        if key_file.exists():
            # Load existing key
            with open(key_file, 'rb') as f:
                private_key = serialization.load_pem_private_key(f.read(), password=None)
        else:
            # Generate new key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Save private key
            key_file.parent.mkdir(exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
        
        public_key = private_key.public_key()
        logger.info(f"Cryptographic keys setup for node {self.node_id}")
        
        return private_key, public_key
    
    async def start(self):
        """Start the distributed node"""
        logger.info(f"Starting GraphMind distributed node {self.node_id}")
        
        try:
            # Initialize components in dependency order
            await self._initialize_components()
            
            # Start all services
            await self._start_services()
            
            # Mark as running
            self.running = True
            self.startup_complete = True
            self.metrics['start_time'] = asyncio.get_event_loop().time()
            
            logger.info(f"Node {self.node_id} started successfully on {self.hostname}:{self.port}")
            
            # Wait for shutdown signal
            await self._wait_for_shutdown()
            
        except Exception as e:
            logger.error(f"Failed to start node {self.node_id}: {e}")
            await self.stop()
            raise
    
    async def _initialize_components(self):
        """Initialize all distributed system components"""
        logger.info("Initializing distributed system components...")
        
        # 1. Initialize Byzantine fault detector
        self.fault_detector = ByzantineFaultDetector(
            node_id=self.node_id,
            detection_threshold=self.config['byzantine']['detection_threshold'],
            evidence_window=self.config['byzantine']['evidence_window']
        )
        
        self.byzantine_manager = ByzantineToleranceManager(
            node_id=self.node_id,
            fault_detector=self.fault_detector
        )
        
        # 2. Initialize node discovery and management
        discovery_config = self.config['discovery']
        health_checks = [
            HealthCheck(
                endpoint="/health",
                interval=30.0,
                timeout=5.0
            )
        ]
        
        self.node_manager = DistributedNodeManager(
            node_id=self.node_id,
            hostname=self.hostname,
            port=self.port,
            private_key=self.private_key,
            discovery_config={
                'backend': ServiceBackend(discovery_config['backend']),
                'backend_config': discovery_config['backend_config'],
                'health_checks': health_checks
            }
        )
        
        # 3. Initialize gRPC client for inter-node communication
        self.grpc_client = GraphMindGrpcClient(
            node_id=self.node_id,
            private_key=self.private_key,
            connection_timeout=self.config['grpc']['connection_timeout']
        )
        
        # 4. Initialize consensus algorithm
        # Start with empty topology, will be updated by discovery
        initial_topology = nx.Graph()
        initial_topology.add_node(self.node_id)
        
        self.consensus = TopologyAwareBFT(
            node_id=int(self.node_id.replace('node', '')),  # Convert to int ID
            graph_topology=initial_topology,
            private_key=self.private_key,
            public_keys={},  # Will be populated by discovery
            byzantine_threshold=self.config['consensus']['byzantine_threshold'],
            view_timeout=self.config['consensus']['view_timeout']
        )
        
        # 5. Initialize gRPC server
        self.grpc_server = GraphMindGrpcServer(
            node_id=self.node_id,
            host=self.hostname,
            port=self.grpc_port,
            consensus_algorithm=self.consensus,
            node_discovery=self.node_manager.discovery_service,
            max_workers=self.config['grpc']['max_workers']
        )
        
        # Set up inter-component callbacks
        self._setup_callbacks()
        
        logger.info("All components initialized successfully")
    
    def _setup_callbacks(self):
        """Setup callbacks between components"""
        # Node discovery callbacks
        def on_topology_changed(topology):
            """Update consensus topology when network changes"""
            if self.consensus:
                # Update consensus algorithm with new topology
                self.consensus.topology = topology
                self.consensus.centrality_weights = self.consensus._compute_centrality_weights()
                self.consensus.quorum_size = self.consensus._compute_topology_quorum()
                logger.info(f"Updated consensus topology: {len(topology.nodes())} nodes")
        
        def on_node_joined(node_info):
            """Handle new node joining"""
            logger.info(f"New node joined: {node_info.node_id}")
            
            # Update fault detector
            if self.fault_detector:
                self.fault_detector.update_node_profile(
                    node_info.node_id,
                    health_score=node_info.health_score
                )
            
            # Establish gRPC connection
            if self.grpc_client:
                asyncio.create_task(self.grpc_client.connect_to_node(node_info))
        
        def on_node_left(node_id):
            """Handle node leaving"""
            logger.info(f"Node left: {node_id}")
            
            # Disconnect gRPC
            if self.grpc_client:
                asyncio.create_task(self.grpc_client.disconnect_from_node(node_id))
        
        # Set callbacks
        self.node_manager.discovery_service.on_topology_changed = on_topology_changed
        self.node_manager.discovery_service.on_node_joined = on_node_joined
        self.node_manager.discovery_service.on_node_left = on_node_left
        
        # Byzantine detection callbacks
        def on_byzantine_detected(evidence):
            """Handle Byzantine behavior detection"""
            self.metrics['byzantine_detections'] += 1
            logger.warning(f"Byzantine behavior detected: {evidence.evidence_type.value}")
        
        self.fault_detector.on_byzantine_detected = on_byzantine_detected
    
    async def _start_services(self):
        """Start all background services"""
        logger.info("Starting background services...")
        
        # Start node discovery
        await self.node_manager.start()
        
        # Start gRPC server
        asyncio.create_task(self.grpc_server.start())
        
        # Wait for initial discovery
        await asyncio.sleep(2)
        
        logger.info("All services started")
    
    async def _wait_for_shutdown(self):
        """Wait for shutdown signal"""
        shutdown_event = asyncio.Event()
        
        def signal_handler():
            logger.info("Received shutdown signal")
            shutdown_event.set()
        
        # Setup signal handlers
        if sys.platform != 'win32':
            for sig in (signal.SIGTERM, signal.SIGINT):
                asyncio.get_event_loop().add_signal_handler(
                    sig, signal_handler
                )
        
        # Wait for shutdown
        await shutdown_event.wait()
    
    async def stop(self):
        """Stop the distributed node"""
        logger.info(f"Stopping GraphMind node {self.node_id}")
        
        self.running = False
        
        try:
            # Stop services in reverse order
            if self.grpc_server:
                await self.grpc_server.stop()
            
            if self.grpc_client:
                await self.grpc_client.cleanup()
            
            if self.node_manager:
                await self.node_manager.stop()
            
            logger.info(f"Node {self.node_id} stopped successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def propose_consensus_value(self, value: Any) -> bool:
        """Propose a value for consensus"""
        if not self.consensus:
            return False
        
        try:
            result = await self.consensus.propose_value(value)
            self.metrics['consensus_rounds'] += 1
            
            return result.decided
            
        except Exception as e:
            logger.error(f"Consensus proposal failed: {e}")
            return False
    
    def get_node_status(self) -> Dict[str, Any]:
        """Get comprehensive node status"""
        uptime = 0
        if self.metrics['start_time'] > 0:
            uptime = asyncio.get_event_loop().time() - self.metrics['start_time']
        
        status = {
            'node_id': self.node_id,
            'running': self.running,
            'startup_complete': self.startup_complete,
            'uptime': uptime,
            'hostname': self.hostname,
            'port': self.port,
            'grpc_port': self.grpc_port,
            'metrics': self.metrics
        }
        
        # Add component status
        if self.node_manager:
            status['connected_nodes'] = len(self.node_manager.discovery_service.nodes)
            status['topology_size'] = len(self.node_manager.get_consensus_topology().nodes())
        
        if self.fault_detector:
            status['byzantine_status'] = self.fault_detector.get_detection_metrics()
        
        if self.consensus:
            status['consensus_metrics'] = self.consensus.get_performance_metrics()
        
        return status
    
    def get_network_overview(self) -> Dict[str, Any]:
        """Get overview of the entire network"""
        if not self.node_manager:
            return {}
        
        nodes = self.node_manager.discovery_service.get_active_nodes()
        topology = self.node_manager.get_consensus_topology()
        
        return {
            'total_nodes': len(nodes),
            'active_nodes': len([n for n in nodes if n.status.value == 'active']),
            'topology': {
                'nodes': len(topology.nodes()),
                'edges': len(topology.edges()),
                'density': nx.density(topology) if len(topology.nodes()) > 1 else 0.0
            },
            'byzantine_status': self.byzantine_manager.get_network_health() if self.byzantine_manager else {},
            'node_health': {
                node.node_id: node.health_score 
                for node in nodes
            }
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Return default configuration
        return {
            'node': {
                'hostname': 'localhost',
                'port': 8080,
                'grpc_port': 50051
            },
            'discovery': {
                'backend': 'redis',
                'backend_config': {
                    'host': 'localhost',
                    'port': 6379
                }
            },
            'consensus': {
                'byzantine_threshold': 0.33,
                'view_timeout': 10.0
            },
            'byzantine': {
                'detection_threshold': 0.7,
                'evidence_window': 100
            },
            'grpc': {
                'connection_timeout': 30.0,
                'max_workers': 10
            }
        }


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='GraphMind Distributed Node')
    parser.add_argument('--config', default='config/node_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--node-id', required=True,
                       help='Unique node identifier')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load configuration
    config = load_config(args.config)
    
    # Create and start node
    node = GraphMindDistributedNode(config, args.node_id)
    
    try:
        await node.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Node failed: {e}")
        return 1
    finally:
        await node.stop()
    
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))