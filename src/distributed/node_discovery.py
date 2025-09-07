"""
Distributed Node Discovery System for GraphMind

This module implements service discovery, health monitoring, and dynamic topology
management for Byzantine fault-tolerant distributed GNN training.

Author: Ayomide Caleb Adekoya
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
from concurrent.futures import ThreadPoolExecutor
import threading

import aiohttp
import etcd3
import consul
import redis
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import networkx as nx

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node status enumeration"""
    JOINING = "joining"
    ACTIVE = "active"
    UNHEALTHY = "unhealthy"
    LEAVING = "leaving"
    OFFLINE = "offline"


class ServiceBackend(Enum):
    """Service discovery backend types"""
    ETCD = "etcd"
    CONSUL = "consul"
    REDIS = "redis"


@dataclass
class NodeInfo:
    """Node information structure"""
    node_id: str
    hostname: str
    port: int
    public_key: bytes
    capabilities: Dict[str, Any]
    status: NodeStatus = NodeStatus.JOINING
    last_seen: float = field(default_factory=time.time)
    health_score: float = 1.0
    consensus_weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        data['public_key'] = self.public_key.hex()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeInfo':
        """Create from dictionary"""
        node_data = data.copy()
        node_data['status'] = NodeStatus(node_data['status'])
        node_data['public_key'] = bytes.fromhex(node_data['public_key'])
        return cls(**node_data)


@dataclass
class HealthCheck:
    """Health check configuration"""
    endpoint: str
    interval: float = 30.0
    timeout: float = 5.0
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    
    
class NodeDiscoveryService:
    """
    Distributed node discovery and health monitoring service
    
    Features:
    - Service registration and discovery
    - Health monitoring with configurable checks
    - Dynamic topology updates
    - Byzantine fault detection through health scoring
    - Multi-backend support (etcd, Consul, Redis)
    """
    
    def __init__(
        self,
        node_id: str,
        hostname: str,
        port: int,
        private_key: rsa.RSAPrivateKey,
        backend: ServiceBackend = ServiceBackend.ETCD,
        backend_config: Optional[Dict[str, Any]] = None,
        health_checks: Optional[List[HealthCheck]] = None
    ):
        self.node_id = node_id
        self.hostname = hostname
        self.port = port
        self.private_key = private_key
        self.backend = backend
        self.backend_config = backend_config or {}
        
        # Generate public key
        public_key = private_key.public_key()
        self.public_key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Node state
        self.node_info = NodeInfo(
            node_id=node_id,
            hostname=hostname,
            port=port,
            public_key=self.public_key_bytes,
            capabilities={
                'consensus': True,
                'gnn_training': True,
                'graph_partitioning': True
            }
        )
        
        # Service discovery backend
        self.client = None
        self.nodes: Dict[str, NodeInfo] = {}
        self.topology = nx.Graph()
        
        # Health monitoring
        self.health_checks = health_checks or []
        self.health_monitor_task = None
        self.health_history: Dict[str, List[float]] = {}
        
        # Event callbacks
        self.on_node_joined: Optional[Callable[[NodeInfo], None]] = None
        self.on_node_left: Optional[Callable[[str], None]] = None
        self.on_topology_changed: Optional[Callable[[nx.Graph], None]] = None
        
        # Async components
        self.session = None
        self.running = False
        self._lock = asyncio.Lock()
        
        logger.info(f"Node discovery service initialized for {node_id}")
    
    async def start(self):
        """Start the node discovery service"""
        logger.info(f"Starting node discovery service for {self.node_id}")
        
        try:
            # Initialize backend client
            await self._init_backend()
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Register this node
            await self._register_node()
            
            # Start background tasks
            self.running = True
            
            # Start health monitoring
            if self.health_checks:
                self.health_monitor_task = asyncio.create_task(
                    self._health_monitor_loop()
                )
            
            # Start discovery loop
            asyncio.create_task(self._discovery_loop())
            
            logger.info(f"Node discovery service started for {self.node_id}")
            
        except Exception as e:
            logger.error(f"Failed to start node discovery: {e}")
            raise
    
    async def stop(self):
        """Stop the node discovery service"""
        logger.info(f"Stopping node discovery service for {self.node_id}")
        
        self.running = False
        
        # Cancel health monitoring
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            
        # Deregister node
        try:
            await self._deregister_node()
        except Exception as e:
            logger.warning(f"Failed to deregister node: {e}")
        
        # Close HTTP session
        if self.session:
            await self.session.close()
        
        # Close backend client
        if hasattr(self.client, 'close'):
            try:
                await self.client.close()
            except:
                pass
        
        logger.info("Node discovery service stopped")
    
    async def _init_backend(self):
        """Initialize service discovery backend"""
        if self.backend == ServiceBackend.ETCD:
            self.client = etcd3.Etcd3Client(
                host=self.backend_config.get('host', 'localhost'),
                port=self.backend_config.get('port', 2379)
            )
        elif self.backend == ServiceBackend.CONSUL:
            self.client = consul.Consul(
                host=self.backend_config.get('host', 'localhost'),
                port=self.backend_config.get('port', 8500)
            )
        elif self.backend == ServiceBackend.REDIS:
            self.client = redis.asyncio.Redis(
                host=self.backend_config.get('host', 'localhost'),
                port=self.backend_config.get('port', 6379),
                decode_responses=True
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    async def _register_node(self):
        """Register this node with service discovery"""
        key = f"graphmind/nodes/{self.node_id}"
        value = json.dumps(self.node_info.to_dict())
        
        try:
            if self.backend == ServiceBackend.ETCD:
                # Register with TTL lease
                lease = self.client.lease(30)  # 30 second TTL
                self.client.put(key, value, lease=lease)
                
                # Keep lease alive
                asyncio.create_task(self._keep_alive_etcd(lease))
                
            elif self.backend == ServiceBackend.CONSUL:
                # Register service with health check
                self.client.agent.service.register(
                    name="graphmind",
                    service_id=self.node_id,
                    address=self.hostname,
                    port=self.port,
                    meta=self.node_info.to_dict(),
                    check=consul.Check.http(
                        f"http://{self.hostname}:{self.port}/health",
                        interval="30s"
                    )
                )
                
            elif self.backend == ServiceBackend.REDIS:
                # Set with expiration
                await self.client.setex(key, 30, value)
                
                # Keep alive
                asyncio.create_task(self._keep_alive_redis(key, value))
            
            logger.info(f"Node {self.node_id} registered successfully")
            
        except Exception as e:
            logger.error(f"Failed to register node: {e}")
            raise
    
    async def _deregister_node(self):
        """Deregister this node from service discovery"""
        self.node_info.status = NodeStatus.LEAVING
        
        if self.backend == ServiceBackend.CONSUL:
            self.client.agent.service.deregister(self.node_id)
        elif self.backend == ServiceBackend.REDIS:
            key = f"graphmind/nodes/{self.node_id}"
            await self.client.delete(key)
    
    async def _keep_alive_etcd(self, lease):
        """Keep etcd lease alive"""
        while self.running:
            try:
                lease.refresh()
                await asyncio.sleep(10)  # Refresh every 10 seconds
            except Exception as e:
                logger.warning(f"Failed to refresh etcd lease: {e}")
                break
    
    async def _keep_alive_redis(self, key: str, value: str):
        """Keep Redis key alive"""
        while self.running:
            try:
                await asyncio.sleep(15)  # Refresh every 15 seconds
                await self.client.setex(key, 30, value)
            except Exception as e:
                logger.warning(f"Failed to refresh Redis key: {e}")
                break
    
    async def _discovery_loop(self):
        """Main discovery loop for monitoring nodes"""
        while self.running:
            try:
                await self._discover_nodes()
                await self._update_topology()
                await asyncio.sleep(5)  # Discovery interval
                
            except Exception as e:
                logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(10)
    
    async def _discover_nodes(self):
        """Discover all active nodes"""
        try:
            discovered_nodes = {}
            
            if self.backend == ServiceBackend.ETCD:
                # List all nodes under prefix
                response = self.client.get_prefix("graphmind/nodes/")
                for value, metadata in response:
                    if value:
                        node_data = json.loads(value.decode())
                        node_info = NodeInfo.from_dict(node_data)
                        discovered_nodes[node_info.node_id] = node_info
            
            elif self.backend == ServiceBackend.CONSUL:
                # Get healthy services
                _, services = self.client.health.service("graphmind", passing=True)
                for service in services:
                    node_data = service['Service']['Meta']
                    node_info = NodeInfo.from_dict(node_data)
                    discovered_nodes[node_info.node_id] = node_info
            
            elif self.backend == ServiceBackend.REDIS:
                # Scan for node keys
                keys = []
                async for key in self.client.scan_iter("graphmind/nodes/*"):
                    keys.append(key)
                
                if keys:
                    values = await self.client.mget(keys)
                    for value in values:
                        if value:
                            node_data = json.loads(value)
                            node_info = NodeInfo.from_dict(node_data)
                            discovered_nodes[node_info.node_id] = node_info
            
            # Update local node registry
            async with self._lock:
                # Detect new nodes
                new_nodes = set(discovered_nodes.keys()) - set(self.nodes.keys())
                for node_id in new_nodes:
                    node_info = discovered_nodes[node_id]
                    self.nodes[node_id] = node_info
                    logger.info(f"Discovered new node: {node_id}")
                    
                    if self.on_node_joined:
                        self.on_node_joined(node_info)
                
                # Detect departed nodes
                departed_nodes = set(self.nodes.keys()) - set(discovered_nodes.keys())
                for node_id in departed_nodes:
                    if node_id != self.node_id:  # Don't remove self
                        logger.info(f"Node departed: {node_id}")
                        del self.nodes[node_id]
                        
                        if self.on_node_left:
                            self.on_node_left(node_id)
                
                # Update existing nodes
                for node_id, node_info in discovered_nodes.items():
                    if node_id in self.nodes and node_id != self.node_id:
                        self.nodes[node_id] = node_info
            
        except Exception as e:
            logger.error(f"Node discovery error: {e}")
    
    async def _update_topology(self):
        """Update network topology based on discovered nodes"""
        try:
            async with self._lock:
                # Create new topology
                new_topology = nx.Graph()
                
                # Add all active nodes
                active_nodes = [
                    node_id for node_id, node_info in self.nodes.items()
                    if node_info.status in [NodeStatus.ACTIVE, NodeStatus.JOINING]
                ]
                
                new_topology.add_nodes_from(active_nodes)
                
                # Add edges based on connectivity (full mesh for now)
                # In production, this would be based on network topology
                for i, node1 in enumerate(active_nodes):
                    for node2 in active_nodes[i+1:]:
                        new_topology.add_edge(node1, node2)
                
                # Update topology if changed
                if not nx.is_isomorphic(self.topology, new_topology):
                    self.topology = new_topology
                    logger.info(f"Topology updated: {len(active_nodes)} nodes")
                    
                    if self.on_topology_changed:
                        self.on_topology_changed(self.topology)
        
        except Exception as e:
            logger.error(f"Topology update error: {e}")
    
    async def _health_monitor_loop(self):
        """Health monitoring loop"""
        while self.running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(30)  # Health check interval
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_checks(self):
        """Perform health checks on all nodes"""
        if not self.session:
            return
        
        tasks = []
        async with self._lock:
            for node_id, node_info in self.nodes.items():
                if node_id != self.node_id:  # Don't check self
                    task = self._check_node_health(node_info)
                    tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_node_health(self, node_info: NodeInfo):
        """Check health of a specific node"""
        try:
            health_scores = []
            
            for health_check in self.health_checks:
                url = f"http://{node_info.hostname}:{node_info.port}{health_check.endpoint}"
                
                try:
                    async with self.session.get(
                        url, 
                        timeout=aiohttp.ClientTimeout(total=health_check.timeout)
                    ) as response:
                        if response.status == 200:
                            health_scores.append(1.0)
                        else:
                            health_scores.append(0.5)
                except:
                    health_scores.append(0.0)
            
            # Calculate average health score
            avg_health = sum(health_scores) / len(health_scores) if health_scores else 0.0
            
            # Update node health
            async with self._lock:
                if node_info.node_id in self.nodes:
                    self.nodes[node_info.node_id].health_score = avg_health
                    self.nodes[node_info.node_id].last_seen = time.time()
                    
                    # Update status based on health
                    if avg_health < 0.3:
                        self.nodes[node_info.node_id].status = NodeStatus.UNHEALTHY
                    elif avg_health > 0.7:
                        self.nodes[node_info.node_id].status = NodeStatus.ACTIVE
            
            # Track health history
            if node_info.node_id not in self.health_history:
                self.health_history[node_info.node_id] = []
            
            self.health_history[node_info.node_id].append(avg_health)
            
            # Keep only recent history (last 100 checks)
            if len(self.health_history[node_info.node_id]) > 100:
                self.health_history[node_info.node_id] = self.health_history[node_info.node_id][-100:]
            
        except Exception as e:
            logger.warning(f"Health check failed for {node_info.node_id}: {e}")
    
    def get_active_nodes(self) -> List[NodeInfo]:
        """Get list of active nodes"""
        return [
            node_info for node_info in self.nodes.values()
            if node_info.status == NodeStatus.ACTIVE
        ]
    
    def get_topology(self) -> nx.Graph:
        """Get current network topology"""
        return self.topology.copy()
    
    def get_node_health(self, node_id: str) -> Optional[float]:
        """Get health score for a node"""
        node_info = self.nodes.get(node_id)
        return node_info.health_score if node_info else None
    
    def get_byzantine_suspects(self, threshold: float = 0.3) -> List[str]:
        """Get nodes suspected of being Byzantine based on health scores"""
        suspects = []
        for node_id, node_info in self.nodes.items():
            if node_info.health_score < threshold:
                suspects.append(node_id)
        return suspects
    
    def update_node_status(self, node_id: str, status: NodeStatus):
        """Update status of a node"""
        if node_id in self.nodes:
            self.nodes[node_id].status = status
            logger.info(f"Updated {node_id} status to {status.value}")


class DistributedNodeManager:
    """
    High-level manager for distributed node operations
    
    Integrates node discovery with Byzantine consensus and GNN training
    """
    
    def __init__(
        self,
        node_id: str,
        hostname: str,
        port: int,
        private_key: rsa.RSAPrivateKey,
        discovery_config: Dict[str, Any]
    ):
        self.node_id = node_id
        self.discovery_service = NodeDiscoveryService(
            node_id=node_id,
            hostname=hostname,
            port=port,
            private_key=private_key,
            **discovery_config
        )
        
        # Set up callbacks
        self.discovery_service.on_node_joined = self._on_node_joined
        self.discovery_service.on_node_left = self._on_node_left
        self.discovery_service.on_topology_changed = self._on_topology_changed
        
        # Consensus integration
        self.consensus_nodes: Dict[str, Any] = {}
        self.public_keys: Dict[str, rsa.RSAPublicKey] = {}
        
    async def start(self):
        """Start the distributed node manager"""
        await self.discovery_service.start()
    
    async def stop(self):
        """Stop the distributed node manager"""
        await self.discovery_service.stop()
    
    def _on_node_joined(self, node_info: NodeInfo):
        """Handle new node joining"""
        logger.info(f"Node joined: {node_info.node_id}")
        
        # Add to consensus participants
        try:
            public_key = serialization.load_pem_public_key(node_info.public_key)
            self.public_keys[node_info.node_id] = public_key
        except Exception as e:
            logger.error(f"Failed to load public key for {node_info.node_id}: {e}")
    
    def _on_node_left(self, node_id: str):
        """Handle node leaving"""
        logger.info(f"Node left: {node_id}")
        
        # Remove from consensus
        if node_id in self.public_keys:
            del self.public_keys[node_id]
    
    def _on_topology_changed(self, topology: nx.Graph):
        """Handle topology changes"""
        logger.info(f"Topology changed: {len(topology.nodes())} nodes, {len(topology.edges())} edges")
    
    def get_consensus_topology(self) -> nx.Graph:
        """Get topology for consensus algorithm"""
        return self.discovery_service.get_topology()
    
    def get_public_keys(self) -> Dict[str, rsa.RSAPublicKey]:
        """Get public keys for all active nodes"""
        return self.public_keys.copy()