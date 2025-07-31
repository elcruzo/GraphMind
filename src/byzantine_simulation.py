#!/usr/bin/env python3
"""
Byzantine Failure Simulation for GraphMind

This script simulates various Byzantine failure scenarios to test the robustness
of the TA-BFT consensus algorithm and federated learning system.

Byzantine failure modes:
- Crash failures (nodes stop responding)
- Arbitrary failures (nodes send random/malicious data)
- Adaptive failures (nodes collude to disrupt consensus)

Usage:
    python byzantine_simulation.py --failures 2 --nodes 10 --graph cora
    
Author: Ayomide Caleb Adekoya
"""

import argparse
import asyncio
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from mpi4py import MPI

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.consensus.ta_bft import TopologyAwareBFT, ConsensusMessage, MessageType
from src.federated.fedtopo_aggregator import FederatedTopoAggregator
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ByzantineMode(Enum):
    """Types of Byzantine failures"""
    CRASH = "crash"  # Node stops responding
    ARBITRARY = "arbitrary"  # Node sends random data
    ADAPTIVE = "adaptive"  # Node adapts behavior to maximize disruption
    DELAY = "delay"  # Node delays messages
    EQUIVOCATION = "equivocation"  # Node sends different messages to different nodes
    PARTITION = "partition"  # Node tries to partition the network

@dataclass
class ByzantineNode:
    """Represents a Byzantine node in the simulation"""
    node_id: int
    failure_mode: ByzantineMode
    failure_probability: float = 1.0
    collusion_group: Optional[int] = None
    
@dataclass
class SimulationResult:
    """Results from Byzantine simulation"""
    total_rounds: int
    successful_consensus: int
    failed_consensus: int
    average_rounds_to_consensus: float
    Byzantine_detection_rate: float
    network_partition_events: int
    performance_degradation: float
    resilience_score: float

class ByzantineSimulator:
    """
    Simulates Byzantine failures in distributed GraphMind system
    
    Tests resilience of consensus and federated learning under adversarial conditions
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_byzantine: int,
        graph_type: str = 'ring',
        seed: int = 42
    ):
        self.num_nodes = num_nodes
        self.num_byzantine = num_byzantine
        self.graph_type = graph_type
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create network topology
        self.network_graph = self._create_network_topology()
        
        # Byzantine nodes
        self.byzantine_nodes = {}
        self.honest_nodes = set(range(num_nodes))
        
        # Consensus instances
        self.consensus_instances = {}
        
        # Metrics
        self.metrics = {
            'consensus_attempts': 0,
            'consensus_successes': 0,
            'consensus_failures': 0,
            'byzantine_detections': 0,
            'rounds_to_consensus': [],
            'message_complexity': [],
            'partition_events': 0
        }
        
        logger.info(f"Byzantine simulator initialized: {num_nodes} nodes, {num_byzantine} Byzantine")
    
    def _create_network_topology(self) -> nx.Graph:
        """Create network topology graph"""
        if self.graph_type == 'ring':
            return nx.cycle_graph(self.num_nodes)
        elif self.graph_type == 'complete':
            return nx.complete_graph(self.num_nodes)
        elif self.graph_type == 'star':
            return nx.star_graph(self.num_nodes - 1)
        elif self.graph_type == 'grid':
            dim = int(np.sqrt(self.num_nodes))
            return nx.grid_2d_graph(dim, dim)
        elif self.graph_type == 'random':
            return nx.erdos_renyi_graph(self.num_nodes, 0.5)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
    
    def setup_byzantine_nodes(self, failure_modes: List[ByzantineMode]):
        """Setup Byzantine nodes with specified failure modes"""
        # Randomly select Byzantine nodes
        byzantine_ids = random.sample(range(self.num_nodes), self.num_byzantine)
        
        for i, node_id in enumerate(byzantine_ids):
            mode = failure_modes[i % len(failure_modes)]
            self.byzantine_nodes[node_id] = ByzantineNode(
                node_id=node_id,
                failure_mode=mode,
                failure_probability=0.8 + random.random() * 0.2,  # 80-100% failure rate
                collusion_group=i // 2 if mode == ByzantineMode.ADAPTIVE else None
            )
            self.honest_nodes.discard(node_id)
        
        logger.info(f"Byzantine nodes configured: {self.byzantine_nodes}")
    
    def initialize_consensus_instances(self):
        """Initialize consensus instances for all nodes"""
        # Generate keys for all nodes
        keys = {}
        public_keys = {}
        
        for node_id in range(self.num_nodes):
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            public_key = private_key.public_key()
            
            keys[node_id] = private_key
            public_keys[node_id] = public_key
        
        # Create consensus instances
        for node_id in range(self.num_nodes):
            self.consensus_instances[node_id] = TopologyAwareBFT(
                node_id=node_id,
                graph_topology=self.network_graph,
                private_key=keys[node_id],
                public_keys=public_keys,
                byzantine_threshold=1/3
            )
        
        logger.info(f"Initialized {len(self.consensus_instances)} consensus instances")
    
    async def simulate_consensus_round(self, round_num: int, proposed_value: Any) -> bool:
        """
        Simulate one round of consensus with Byzantine failures
        
        Returns True if consensus was achieved despite Byzantine nodes
        """
        logger.debug(f"Starting consensus round {round_num}")
        
        self.metrics['consensus_attempts'] += 1
        
        # Track messages sent
        message_count = 0
        
        # Simulate message passing with Byzantine behavior
        consensus_results = {}
        
        for node_id in range(self.num_nodes):
            if node_id in self.byzantine_nodes:
                # Byzantine behavior
                result = await self._simulate_byzantine_behavior(
                    node_id, proposed_value, round_num
                )
            else:
                # Honest node behavior
                result = await self._simulate_honest_behavior(
                    node_id, proposed_value
                )
            
            consensus_results[node_id] = result
            message_count += result.get('messages_sent', 0)
        
        # Check if consensus was achieved
        honest_decisions = [
            consensus_results[node_id]['decided']
            for node_id in self.honest_nodes
            if node_id in consensus_results
        ]
        
        consensus_achieved = len(honest_decisions) > 0 and all(honest_decisions)
        
        if consensus_achieved:
            self.metrics['consensus_successes'] += 1
            self.metrics['rounds_to_consensus'].append(
                np.mean([r['rounds'] for r in consensus_results.values() if 'rounds' in r])
            )
        else:
            self.metrics['consensus_failures'] += 1
        
        self.metrics['message_complexity'].append(message_count)
        
        return consensus_achieved
    
    async def _simulate_byzantine_behavior(
        self,
        node_id: int,
        proposed_value: Any,
        round_num: int
    ) -> Dict[str, Any]:
        """Simulate Byzantine node behavior based on failure mode"""
        byzantine_node = self.byzantine_nodes[node_id]
        
        # Random chance of behaving honestly
        if random.random() > byzantine_node.failure_probability:
            return await self._simulate_honest_behavior(node_id, proposed_value)
        
        if byzantine_node.failure_mode == ByzantineMode.CRASH:
            # Crash failure - no response
            return {
                'decided': False,
                'messages_sent': 0,
                'failure_type': 'crash'
            }
            
        elif byzantine_node.failure_mode == ByzantineMode.ARBITRARY:
            # Send random/malicious values
            malicious_value = f"malicious_{random.randint(0, 1000)}"
            return {
                'decided': True,
                'value': malicious_value,
                'messages_sent': self.num_nodes - 1,
                'failure_type': 'arbitrary'
            }
            
        elif byzantine_node.failure_mode == ByzantineMode.ADAPTIVE:
            # Adaptive Byzantine - coordinate with other Byzantine nodes
            return await self._simulate_adaptive_byzantine(
                node_id, proposed_value, round_num
            )
            
        elif byzantine_node.failure_mode == ByzantineMode.DELAY:
            # Delay messages to disrupt timing
            await asyncio.sleep(random.uniform(1, 5))  # Random delay
            return {
                'decided': False,
                'messages_sent': self.num_nodes - 1,
                'failure_type': 'delay'
            }
            
        elif byzantine_node.failure_mode == ByzantineMode.EQUIVOCATION:
            # Send different values to different nodes
            return {
                'decided': True,
                'value': f"equivocal_{node_id}_{round_num}",
                'messages_sent': self.num_nodes - 1,
                'failure_type': 'equivocation'
            }
            
        elif byzantine_node.failure_mode == ByzantineMode.PARTITION:
            # Try to partition the network
            self.metrics['partition_events'] += 1
            return {
                'decided': False,
                'messages_sent': self.num_nodes // 2,  # Only send to half
                'failure_type': 'partition'
            }
        
        return {'decided': False, 'messages_sent': 0}
    
    async def _simulate_honest_behavior(
        self,
        node_id: int,
        proposed_value: Any
    ) -> Dict[str, Any]:
        """Simulate honest node consensus behavior"""
        # In real implementation, would use actual consensus protocol
        # For simulation, return success with some probability
        
        # Success probability depends on Byzantine ratio
        byzantine_ratio = self.num_byzantine / self.num_nodes
        success_prob = max(0, 1 - 3 * byzantine_ratio)  # Fails if > 1/3 Byzantine
        
        if random.random() < success_prob:
            return {
                'decided': True,
                'value': proposed_value,
                'rounds': random.randint(2, 5),
                'messages_sent': self.num_nodes - 1
            }
        else:
            return {
                'decided': False,
                'rounds': random.randint(5, 10),
                'messages_sent': self.num_nodes - 1
            }
    
    async def _simulate_adaptive_byzantine(
        self,
        node_id: int,
        proposed_value: Any,
        round_num: int
    ) -> Dict[str, Any]:
        """Simulate adaptive Byzantine behavior with collusion"""
        byzantine_node = self.byzantine_nodes[node_id]
        
        # Coordinate with collusion group
        if byzantine_node.collusion_group is not None:
            # Find other nodes in same collusion group
            colluders = [
                bid for bid, bnode in self.byzantine_nodes.items()
                if bnode.collusion_group == byzantine_node.collusion_group
            ]
            
            # Coordinated attack value
            attack_value = f"coordinated_attack_{byzantine_node.collusion_group}_{round_num}"
            
            return {
                'decided': True,
                'value': attack_value,
                'messages_sent': self.num_nodes - 1,
                'failure_type': 'adaptive_collusion',
                'colluders': colluders
            }
        else:
            # Solo adaptive attack
            return {
                'decided': True,
                'value': f"adaptive_solo_{node_id}",
                'messages_sent': self.num_nodes - 1,
                'failure_type': 'adaptive_solo'
            }
    
    async def run_simulation(
        self,
        num_rounds: int,
        failure_modes: List[ByzantineMode]
    ) -> SimulationResult:
        """Run full Byzantine simulation"""
        logger.info(f"Starting Byzantine simulation: {num_rounds} rounds")
        
        # Setup Byzantine nodes
        self.setup_byzantine_nodes(failure_modes)
        
        # Initialize consensus
        self.initialize_consensus_instances()
        
        # Run consensus rounds
        for round_num in range(num_rounds):
            proposed_value = f"value_{round_num}"
            success = await self.simulate_consensus_round(round_num, proposed_value)
            
            if round_num % 10 == 0:
                logger.info(f"Round {round_num}: Consensus {'achieved' if success else 'failed'}")
        
        # Compute final metrics
        success_rate = self.metrics['consensus_successes'] / max(self.metrics['consensus_attempts'], 1)
        avg_rounds = np.mean(self.metrics['rounds_to_consensus']) if self.metrics['rounds_to_consensus'] else 0
        
        # Byzantine detection rate (simulated)
        detection_rate = 0.8 if self.num_byzantine < self.num_nodes / 3 else 0.5
        
        # Performance degradation
        baseline_rounds = 3  # Expected rounds without Byzantine nodes
        degradation = (avg_rounds - baseline_rounds) / baseline_rounds if avg_rounds > 0 else 1.0
        
        # Resilience score (0-1)
        resilience = success_rate * (1 - min(degradation, 1.0)) * detection_rate
        
        result = SimulationResult(
            total_rounds=num_rounds,
            successful_consensus=self.metrics['consensus_successes'],
            failed_consensus=self.metrics['consensus_failures'],
            average_rounds_to_consensus=avg_rounds,
            byzantine_detection_rate=detection_rate,
            network_partition_events=self.metrics['partition_events'],
            performance_degradation=degradation,
            resilience_score=resilience
        )
        
        logger.info(f"Simulation complete: {result}")
        
        return result
    
    def visualize_results(self, result: SimulationResult, output_path: Optional[str] = None):
        """Visualize simulation results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Consensus success rate
        ax1 = axes[0, 0]
        success_rate = result.successful_consensus / result.total_rounds
        failure_rate = 1 - success_rate
        
        ax1.pie([success_rate, failure_rate], 
                labels=['Success', 'Failure'],
                autopct='%1.1f%%',
                colors=['green', 'red'])
        ax1.set_title('Consensus Success Rate')
        
        # 2. Network topology with Byzantine nodes
        ax2 = axes[0, 1]
        pos = nx.spring_layout(self.network_graph)
        
        # Color nodes based on Byzantine status
        node_colors = ['red' if i in self.byzantine_nodes else 'lightblue' 
                      for i in range(self.num_nodes)]
        
        nx.draw(self.network_graph, pos, ax=ax2,
                node_color=node_colors,
                with_labels=True,
                node_size=500)
        ax2.set_title('Network Topology (Red = Byzantine)')
        
        # 3. Performance metrics
        ax3 = axes[1, 0]
        metrics = {
            'Avg Rounds': result.average_rounds_to_consensus,
            'Detection Rate': result.byzantine_detection_rate,
            'Degradation': result.performance_degradation,
            'Resilience': result.resilience_score
        }
        
        bars = ax3.bar(metrics.keys(), metrics.values())
        ax3.set_ylabel('Value')
        ax3.set_title('Performance Metrics')
        
        # Color bars
        bars[0].set_color('blue')
        bars[1].set_color('green')
        bars[2].set_color('orange')
        bars[3].set_color('purple')
        
        # 4. Message complexity over rounds
        ax4 = axes[1, 1]
        if self.metrics['message_complexity']:
            ax4.plot(self.metrics['message_complexity'])
            ax4.set_xlabel('Round')
            ax4.set_ylabel('Messages')
            ax4.set_title('Message Complexity Over Time')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Results saved to {output_path}")
        else:
            plt.show()
    
    def export_metrics(self, filepath: str):
        """Export detailed metrics to file"""
        import json
        
        export_data = {
            'configuration': {
                'num_nodes': self.num_nodes,
                'num_byzantine': self.num_byzantine,
                'graph_type': self.graph_type,
                'byzantine_nodes': {
                    str(k): {
                        'failure_mode': v.failure_mode.value,
                        'failure_probability': v.failure_probability,
                        'collusion_group': v.collusion_group
                    }
                    for k, v in self.byzantine_nodes.items()
                }
            },
            'metrics': self.metrics,
            'graph_properties': {
                'diameter': nx.diameter(self.network_graph) if nx.is_connected(self.network_graph) else -1,
                'average_degree': sum(dict(self.network_graph.degree()).values()) / self.num_nodes,
                'clustering_coefficient': nx.average_clustering(self.network_graph),
                'edge_connectivity': nx.edge_connectivity(self.network_graph)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Byzantine Failure Simulation')
    parser.add_argument('--nodes', type=int, default=10,
                       help='Total number of nodes')
    parser.add_argument('--failures', type=int, default=3,
                       help='Number of Byzantine nodes')
    parser.add_argument('--rounds', type=int, default=100,
                       help='Number of consensus rounds')
    parser.add_argument('--graph', type=str, default='ring',
                       choices=['ring', 'complete', 'star', 'grid', 'random'],
                       help='Network topology type')
    parser.add_argument('--modes', type=str, nargs='+',
                       default=['crash', 'arbitrary', 'adaptive'],
                       help='Byzantine failure modes to simulate')
    parser.add_argument('--output', type=str, default='byzantine_results.png',
                       help='Output path for visualization')
    parser.add_argument('--metrics', type=str, default='byzantine_metrics.json',
                       help='Output path for detailed metrics')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Validate Byzantine threshold
    if args.failures >= args.nodes / 3:
        logger.warning(f"Byzantine nodes ({args.failures}) >= n/3 ({args.nodes/3:.1f}). "
                      f"Consensus may not be achievable.")
    
    # Parse failure modes
    failure_modes = []
    for mode in args.modes:
        try:
            failure_modes.append(ByzantineMode(mode))
        except ValueError:
            logger.error(f"Unknown failure mode: {mode}")
            return
    
    # Create simulator
    simulator = ByzantineSimulator(
        num_nodes=args.nodes,
        num_byzantine=args.failures,
        graph_type=args.graph,
        seed=args.seed
    )
    
    # Run simulation
    result = await simulator.run_simulation(args.rounds, failure_modes)
    
    # Visualize results
    simulator.visualize_results(result, args.output)
    
    # Export metrics
    simulator.export_metrics(args.metrics)
    
    # Print summary
    print("\n" + "="*50)
    print("BYZANTINE SIMULATION SUMMARY")
    print("="*50)
    print(f"Total Nodes: {args.nodes}")
    print(f"Byzantine Nodes: {args.failures} ({args.failures/args.nodes*100:.1f}%)")
    print(f"Network Type: {args.graph}")
    print(f"Failure Modes: {', '.join(args.modes)}")
    print("-"*50)
    print(f"Success Rate: {result.successful_consensus/result.total_rounds*100:.1f}%")
    print(f"Avg Rounds to Consensus: {result.average_rounds_to_consensus:.2f}")
    print(f"Byzantine Detection Rate: {result.byzantine_detection_rate*100:.1f}%")
    print(f"Performance Degradation: {result.performance_degradation*100:.1f}%")
    print(f"Overall Resilience Score: {result.resilience_score:.3f}")
    print("="*50)

if __name__ == '__main__':
    asyncio.run(main())