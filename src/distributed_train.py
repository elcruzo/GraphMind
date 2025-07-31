#!/usr/bin/env python3
"""
Distributed Training Script for GraphMind

This script orchestrates distributed GNN training using the TA-BFT consensus,
adaptive partitioning, and federated topological aggregation components.

Usage:
    mpirun -np 8 python distributed_train.py --config config/consensus_config.yaml
    
Author: Ayomide Caleb Adekoya
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid, Reddit, PPI
import networkx as nx
from mpi4py import MPI

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.consensus.ta_bft import TopologyAwareBFT, ConsensusResult
from src.partitioning.adaptive_multilevel import AdaptiveMultiLevelPartitioner, PartitionConstraints
from src.federated.fedtopo_aggregator import FederatedTopoAggregator, FederatedGNNModel
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DistributedGraphMindTrainer:
    """
    Main class for distributed GraphMind training
    
    Coordinates:
    - Graph partitioning across nodes
    - Local GNN training
    - Byzantine consensus for parameter aggregation
    - Federated learning with topological awareness
    """
    
    def __init__(self, config: Dict[str, Any], comm: MPI.Comm):
        self.config = config
        self.comm = comm
        self.rank = comm.Get_rank()
        self.world_size = comm.Get_size()
        
        # Set device
        self.device = torch.device(
            f"cuda:{self.rank % torch.cuda.device_count()}" 
            if torch.cuda.is_available() else "cpu"
        )
        
        logger.info(f"Node {self.rank}/{self.world_size} initialized on {self.device}")
        
        # Initialize components
        self.dataset = None
        self.local_data = None
        self.model = None
        self.optimizer = None
        self.consensus = None
        self.partitioner = None
        self.aggregator = None
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'consensus_time': [],
            'aggregation_time': [],
            'communication_cost': []
        }
    
    def setup(self):
        """Setup all components for distributed training"""
        logger.info(f"Node {self.rank}: Setting up distributed training")
        
        # Load dataset and create topology
        self._load_dataset()
        self._create_network_topology()
        
        # Partition graph across nodes
        self._partition_graph()
        
        # Initialize model
        self._initialize_model()
        
        # Setup consensus mechanism
        self._setup_consensus()
        
        # Initialize federated aggregator
        self._setup_federated_aggregator()
        
        logger.info(f"Node {self.rank}: Setup complete")
    
    def _load_dataset(self):
        """Load graph dataset based on configuration"""
        dataset_name = self.config['dataset']['name']
        dataset_path = self.config['dataset']['path']
        
        logger.info(f"Loading dataset: {dataset_name}")
        
        if dataset_name == 'cora':
            self.dataset = Planetoid(root=dataset_path, name='Cora')
        elif dataset_name == 'citeseer':
            self.dataset = Planetoid(root=dataset_path, name='CiteSeer')
        elif dataset_name == 'pubmed':
            self.dataset = Planetoid(root=dataset_path, name='PubMed')
        elif dataset_name == 'reddit':
            self.dataset = Reddit(root=dataset_path)
        elif dataset_name == 'ppi':
            self.dataset = PPI(root=dataset_path)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Get global data
        self.global_data = self.dataset[0]
        
        # Broadcast dataset info to all nodes
        dataset_info = {
            'num_features': self.dataset.num_features,
            'num_classes': self.dataset.num_classes,
            'num_nodes': self.global_data.num_nodes,
            'num_edges': self.global_data.num_edges
        }
        
        dataset_info = self.comm.bcast(dataset_info, root=0)
        
        logger.info(f"Dataset loaded: {dataset_info}")
    
    def _create_network_topology(self):
        """Create network topology for distributed nodes"""
        # Create a connected graph topology for MPI nodes
        if self.config['topology']['type'] == 'ring':
            self.network_graph = nx.cycle_graph(self.world_size)
        elif self.config['topology']['type'] == 'complete':
            self.network_graph = nx.complete_graph(self.world_size)
        elif self.config['topology']['type'] == 'star':
            self.network_graph = nx.star_graph(self.world_size - 1)
        elif self.config['topology']['type'] == 'random':
            p = self.config['topology'].get('connection_probability', 0.5)
            self.network_graph = nx.erdos_renyi_graph(self.world_size, p)
            # Ensure connected
            if not nx.is_connected(self.network_graph):
                # Add edges to make connected
                components = list(nx.connected_components(self.network_graph))
                for i in range(1, len(components)):
                    u = list(components[i-1])[0]
                    v = list(components[i])[0]
                    self.network_graph.add_edge(u, v)
        else:
            # Default to ring topology
            self.network_graph = nx.cycle_graph(self.world_size)
        
        logger.info(f"Network topology created: {self.config['topology']['type']}")
    
    def _partition_graph(self):
        """Partition the graph dataset across distributed nodes"""
        if self.rank == 0:
            logger.info("Partitioning graph across nodes")
            
            # Initialize partitioner
            self.partitioner = AdaptiveMultiLevelPartitioner(
                objective_weights=self.config['partitioner']['objective_weights'],
                coarsening_threshold=self.config['partitioner']['coarsening_threshold'],
                refinement_iterations=self.config['partitioner']['refinement_iterations']
            )
            
            # Create graph from data
            edge_index = self.global_data.edge_index.numpy()
            data_graph = nx.Graph()
            data_graph.add_edges_from(zip(edge_index[0], edge_index[1]))
            
            # Partition
            constraints = PartitionConstraints(
                max_load_imbalance=self.config['partitioner']['max_load_imbalance']
            )
            
            partition_result = self.partitioner.partition(
                data_graph, 
                self.world_size,
                constraints
            )
            
            partition_assignment = partition_result.partition_assignment
            
            logger.info(f"Partitioning complete: cut_size={partition_result.cut_size}, "
                       f"balance={partition_result.load_balance:.3f}")
        else:
            partition_assignment = None
        
        # Broadcast partition assignment
        partition_assignment = self.comm.bcast(partition_assignment, root=0)
        
        # Extract local partition
        self._extract_local_partition(partition_assignment)
    
    def _extract_local_partition(self, partition_assignment: Dict[int, int]):
        """Extract local data partition for this node"""
        # Get nodes assigned to this rank
        local_nodes = [node for node, part in partition_assignment.items() if part == self.rank]
        local_nodes = torch.tensor(local_nodes, dtype=torch.long)
        
        # Extract local subgraph
        if len(local_nodes) > 0:
            # Get node features
            local_x = self.global_data.x[local_nodes]
            local_y = self.global_data.y[local_nodes]
            
            # Extract edges within local partition
            edge_index = self.global_data.edge_index
            mask = torch.isin(edge_index[0], local_nodes) & torch.isin(edge_index[1], local_nodes)
            local_edge_index = edge_index[:, mask]
            
            # Remap node indices to local
            node_mapping = {int(node): i for i, node in enumerate(local_nodes)}
            remapped_edges = []
            for i in range(local_edge_index.size(1)):
                src = int(local_edge_index[0, i])
                dst = int(local_edge_index[1, i])
                if src in node_mapping and dst in node_mapping:
                    remapped_edges.append([node_mapping[src], node_mapping[dst]])
            
            if remapped_edges:
                local_edge_index = torch.tensor(remapped_edges, dtype=torch.long).t()
            else:
                local_edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            # Create local data object
            self.local_data = Data(
                x=local_x,
                edge_index=local_edge_index,
                y=local_y,
                num_nodes=len(local_nodes)
            )
            
            # Create train/val/test masks
            total_nodes = len(local_nodes)
            indices = torch.randperm(total_nodes)
            train_size = int(0.6 * total_nodes)
            val_size = int(0.2 * total_nodes)
            
            self.local_data.train_mask = torch.zeros(total_nodes, dtype=torch.bool)
            self.local_data.val_mask = torch.zeros(total_nodes, dtype=torch.bool)
            self.local_data.test_mask = torch.zeros(total_nodes, dtype=torch.bool)
            
            self.local_data.train_mask[indices[:train_size]] = True
            self.local_data.val_mask[indices[train_size:train_size+val_size]] = True
            self.local_data.test_mask[indices[train_size+val_size:]] = True
            
        else:
            # Empty partition
            self.local_data = Data(
                x=torch.zeros((0, self.dataset.num_features)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                y=torch.zeros(0, dtype=torch.long),
                num_nodes=0
            )
        
        self.local_data = self.local_data.to(self.device)
        
        logger.info(f"Node {self.rank}: Local partition has {self.local_data.num_nodes} nodes "
                   f"and {self.local_data.edge_index.size(1)} edges")
    
    def _initialize_model(self):
        """Initialize local GNN model"""
        model_config = self.config['model']
        
        self.model = FederatedGNNModel(
            input_dim=self.dataset.num_features,
            hidden_dim=model_config['hidden_dim'],
            output_dim=self.dataset.num_classes,
            num_layers=model_config['num_layers'],
            model_type=model_config['type'],
            dropout=model_config['dropout']
        ).to(self.device)
        
        # Initialize optimizer
        optimizer_config = self.config['optimizer']
        if optimizer_config['type'] == 'adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config['weight_decay']
            )
        else:
            self.optimizer = SGD(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config['weight_decay']
            )
        
        logger.info(f"Node {self.rank}: Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _setup_consensus(self):
        """Setup Byzantine consensus mechanism"""
        # Generate RSA keys for this node
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        
        # Exchange public keys
        public_key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        all_public_keys_bytes = self.comm.allgather(public_key_bytes)
        
        # Deserialize public keys
        public_keys = {}
        for i, key_bytes in enumerate(all_public_keys_bytes):
            public_keys[i] = serialization.load_pem_public_key(key_bytes)
        
        # Initialize consensus
        self.consensus = TopologyAwareBFT(
            node_id=self.rank,
            graph_topology=self.network_graph,
            private_key=private_key,
            public_keys=public_keys,
            byzantine_threshold=self.config['consensus']['byzantine_threshold'],
            view_timeout=self.config['consensus']['view_timeout']
        )
        
        logger.info(f"Node {self.rank}: Consensus mechanism initialized")
    
    def _setup_federated_aggregator(self):
        """Setup federated aggregator with topology awareness"""
        self.aggregator = FederatedTopoAggregator(
            graph_structure=self.network_graph,
            privacy_budget=self.config['privacy']['budget'],
            regularization_strength=self.config['aggregator']['regularization_strength'],
            personalization_rate=self.config['aggregator']['personalization_rate']
        )
        
        logger.info(f"Node {self.rank}: Federated aggregator initialized")
    
    async def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch locally"""
        self.model.train()
        
        if self.local_data.num_nodes == 0:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        # Forward pass
        out = self.model(self.local_data.x, self.local_data.edge_index)
        
        # Compute loss only on training nodes
        if self.local_data.train_mask.sum() > 0:
            loss = F.cross_entropy(
                out[self.local_data.train_mask],
                self.local_data.y[self.local_data.train_mask]
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            pred = out.argmax(dim=1)
            correct = (pred[self.local_data.train_mask] == self.local_data.y[self.local_data.train_mask]).sum()
            accuracy = float(correct) / float(self.local_data.train_mask.sum())
            
            return {'loss': float(loss), 'accuracy': accuracy}
        else:
            return {'loss': 0.0, 'accuracy': 0.0}
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        
        if self.local_data.num_nodes == 0 or self.local_data.val_mask.sum() == 0:
            return {'val_accuracy': 0.0}
        
        with torch.no_grad():
            out = self.model(self.local_data.x, self.local_data.edge_index)
            pred = out.argmax(dim=1)
            correct = (pred[self.local_data.val_mask] == self.local_data.y[self.local_data.val_mask]).sum()
            accuracy = float(correct) / float(self.local_data.val_mask.sum())
        
        return {'val_accuracy': accuracy}
    
    async def federated_aggregation(self, round_num: int):
        """Perform federated aggregation with consensus"""
        start_time = time.time()
        
        # Get local model parameters
        local_params = self.model.get_parameters_dict()
        
        # Gather all parameters (in practice, would use consensus)
        all_params = self.comm.allgather(local_params)
        
        # Create parameter dictionary
        param_dict = {i: params for i, params in enumerate(all_params)}
        
        # Node capabilities (could be based on compute power, data size, etc.)
        node_capabilities = {i: 1.0 for i in range(self.world_size)}
        
        # Perform federated aggregation
        aggregation_result = self.aggregator.aggregate_parameters(
            param_dict,
            node_capabilities,
            round_num
        )
        
        # Apply personalization
        personalized_params = self.aggregator.apply_personalization(
            aggregation_result.global_parameters,
            local_params,
            self.rank
        )
        
        # Update local model
        self.model.set_parameters_dict(personalized_params)
        
        aggregation_time = time.time() - start_time
        self.metrics['aggregation_time'].append(aggregation_time)
        
        logger.info(f"Node {self.rank}: Aggregation completed in {aggregation_time:.2f}s")
    
    async def run_training(self):
        """Main training loop"""
        num_rounds = self.config['training']['num_rounds']
        local_epochs = self.config['training']['local_epochs']
        
        logger.info(f"Starting distributed training for {num_rounds} rounds")
        
        for round_num in range(num_rounds):
            round_start = time.time()
            
            # Local training
            for epoch in range(local_epochs):
                train_metrics = await self.train_epoch(epoch)
                
            self.metrics['train_loss'].append(train_metrics['loss'])
            self.metrics['train_accuracy'].append(train_metrics['accuracy'])
            
            # Evaluation
            eval_metrics = self.evaluate()
            self.metrics['val_accuracy'].append(eval_metrics['val_accuracy'])
            
            # Federated aggregation with consensus
            await self.federated_aggregation(round_num)
            
            # Log progress
            if self.rank == 0:
                logger.info(
                    f"Round {round_num + 1}/{num_rounds}: "
                    f"Loss={train_metrics['loss']:.4f}, "
                    f"Train Acc={train_metrics['accuracy']:.4f}, "
                    f"Val Acc={eval_metrics['val_accuracy']:.4f}"
                )
            
            # Synchronize nodes
            self.comm.Barrier()
            
            round_time = time.time() - round_start
            logger.debug(f"Node {self.rank}: Round {round_num + 1} completed in {round_time:.2f}s")
    
    def save_results(self):
        """Save training results and model"""
        if self.rank == 0:
            results_dir = Path(self.config['output']['results_dir'])
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            metrics_path = results_dir / 'training_metrics.npy'
            np.save(metrics_path, self.metrics)
            
            # Save model
            model_path = results_dir / 'final_model.pt'
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'final_metrics': {
                    'train_accuracy': self.metrics['train_accuracy'][-1],
                    'val_accuracy': self.metrics['val_accuracy'][-1]
                }
            }, model_path)
            
            # Save consensus metrics
            consensus_metrics = self.consensus.get_performance_metrics()
            consensus_path = results_dir / 'consensus_metrics.yaml'
            with open(consensus_path, 'w') as f:
                yaml.dump(consensus_metrics, f)
            
            # Save aggregator convergence analysis
            convergence_analysis = self.aggregator.get_convergence_analysis()
            convergence_path = results_dir / 'convergence_analysis.yaml'
            with open(convergence_path, 'w') as f:
                yaml.dump(convergence_analysis, f)
            
            logger.info(f"Results saved to {results_dir}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set defaults
    defaults = {
        'dataset': {
            'name': 'cora',
            'path': './data'
        },
        'topology': {
            'type': 'ring'
        },
        'model': {
            'type': 'gcn',
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.5
        },
        'optimizer': {
            'type': 'adam',
            'lr': 0.01,
            'weight_decay': 5e-4
        },
        'training': {
            'num_rounds': 100,
            'local_epochs': 1
        },
        'consensus': {
            'byzantine_threshold': 0.33,
            'view_timeout': 10.0
        },
        'partitioner': {
            'objective_weights': {'cut': 0.4, 'balance': 0.4, 'communication': 0.2},
            'coarsening_threshold': 100,
            'refinement_iterations': 10,
            'max_load_imbalance': 0.1
        },
        'aggregator': {
            'regularization_strength': 0.1,
            'personalization_rate': 0.3
        },
        'privacy': {
            'budget': 1.0
        },
        'output': {
            'results_dir': './results'
        }
    }
    
    # Merge with loaded config
    def merge_dicts(d1, d2):
        for k, v in d1.items():
            if k in d2 and isinstance(v, dict) and isinstance(d2[k], dict):
                merge_dicts(v, d2[k])
            elif k not in d2:
                d2[k] = v
    
    merge_dicts(defaults, config)
    
    return config

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='GraphMind Distributed Training')
    parser.add_argument('--config', type=str, default='config/consensus_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Set random seeds
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer
    trainer = DistributedGraphMindTrainer(config, comm)
    
    # Setup
    trainer.setup()
    
    # Run training
    await trainer.run_training()
    
    # Save results
    trainer.save_results()
    
    logger.info(f"Node {rank}: Training completed")

if __name__ == '__main__':
    asyncio.run(main())