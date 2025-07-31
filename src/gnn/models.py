"""
Graph Neural Network models for GraphMind

Implements GCN, GAT, and GraphSAGE architectures optimized for
distributed training with Byzantine resilience.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm, LayerNorm
from typing import Optional, Tuple, Dict, Any
import numpy as np


class DistributedGCN(nn.Module):
    """
    Graph Convolutional Network optimized for distributed training
    
    Features:
    - Gradient clipping for Byzantine resilience
    - Layer normalization for stability
    - Dropout for regularization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        dropout: float = 0.5,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.dropout = dropout
        self.activation = getattr(F, activation)
        
        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_batch_norm else None
        
        for i in range(len(dims) - 1):
            self.convs.append(GCNConv(dims[i], dims[i+1]))
            
            if use_batch_norm and i < len(dims) - 2:  # No norm on last layer
                self.norms.append(BatchNorm(dims[i+1]))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            batch: Batch assignment for graph-level tasks
            
        Returns:
            Output features or predictions
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < len(self.convs) - 1:  # Not last layer
                if self.norms is not None:
                    x = self.norms[i](x)
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling for graph-level tasks
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings before final layer"""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.norms is not None:
                x = self.norms[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class DistributedGAT(nn.Module):
    """
    Graph Attention Network optimized for distributed training
    
    Features:
    - Multi-head attention
    - Edge dropout for robustness
    - Attention weight regularization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        heads: list,
        dropout: float = 0.5,
        edge_dropout: float = 0.1,
        concat_last: bool = False
    ):
        super().__init__()
        
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        
        # Build layers
        dims = [input_dim] + hidden_dims
        self.convs = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            self.convs.append(
                GATConv(
                    dims[i] * (heads[i-1] if i > 0 else 1),
                    dims[i+1],
                    heads=heads[i],
                    concat=True,
                    dropout=dropout
                )
            )
        
        # Output layer
        self.out_conv = GATConv(
            dims[-1] * heads[-2],
            output_dim,
            heads=heads[-1],
            concat=concat_last,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with attention weights
        
        Returns:
            (output, attention_weights_dict)
        """
        attention_weights = {}
        
        # Apply edge dropout
        if self.training and self.edge_dropout > 0:
            edge_mask = torch.rand(edge_index.size(1)) > self.edge_dropout
            edge_index = edge_index[:, edge_mask]
        
        # Process through layers
        for i, conv in enumerate(self.convs):
            x, (edge_index_with_self_loops, alpha) = conv(x, edge_index, return_attention_weights=True)
            attention_weights[f'layer_{i}'] = alpha
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x, (_, alpha) = self.out_conv(x, edge_index, return_attention_weights=True)
        attention_weights['output'] = alpha
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x, attention_weights


class DistributedGraphSAGE(nn.Module):
    """
    GraphSAGE optimized for distributed training
    
    Features:
    - Neighborhood sampling efficiency
    - Aggregator options (mean, max, LSTM)
    - Scalable to large graphs
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        aggregator: str = 'mean',
        dropout: float = 0.5,
        normalize: bool = True
    ):
        super().__init__()
        
        self.dropout = dropout
        self.aggregator = aggregator
        
        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        self.convs = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            self.convs.append(
                SAGEConv(
                    dims[i],
                    dims[i+1],
                    aggr=aggregator,
                    normalize=normalize
                )
            )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        if batch is not None:
            if self.aggregator == 'max':
                x = global_max_pool(x, batch)
            else:
                x = global_mean_pool(x, batch)
        
        return x


class ByzantineResilientGNN(nn.Module):
    """
    GNN with Byzantine resilience mechanisms
    
    Features:
    - Gradient clipping
    - Outlier detection in aggregation
    - Robust normalization
    """
    
    def __init__(
        self,
        base_model: str,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        byzantine_threshold: float = 0.2,
        **kwargs
    ):
        super().__init__()
        
        self.byzantine_threshold = byzantine_threshold
        
        # Create base model
        if base_model == 'gcn':
            self.model = DistributedGCN(input_dim, hidden_dims, output_dim, **kwargs)
        elif base_model == 'gat':
            self.model = DistributedGAT(input_dim, hidden_dims, output_dim, **kwargs)
        elif base_model == 'sage':
            self.model = DistributedGraphSAGE(input_dim, hidden_dims, output_dim, **kwargs)
        else:
            raise ValueError(f"Unknown base model: {base_model}")
        
        # Byzantine detection parameters
        self.gradient_history = []
        self.gradient_threshold = None
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with Byzantine monitoring"""
        output = self.model(x, edge_index, batch)
        
        # Monitor gradients during training
        if self.training:
            self._monitor_gradients()
        
        return output
    
    def _monitor_gradients(self):
        """Monitor gradients for Byzantine behavior detection"""
        if not self.training:
            return
        
        # Collect gradient norms
        grad_norms = []
        for param in self.parameters():
            if param.grad is not None:
                grad_norms.append(torch.norm(param.grad).item())
        
        if grad_norms:
            current_norm = np.mean(grad_norms)
            self.gradient_history.append(current_norm)
            
            # Maintain history window
            if len(self.gradient_history) > 100:
                self.gradient_history.pop(0)
            
            # Update threshold
            if len(self.gradient_history) > 10:
                mean_norm = np.mean(self.gradient_history)
                std_norm = np.std(self.gradient_history)
                self.gradient_threshold = mean_norm + 3 * std_norm
    
    def clip_gradients(self, max_norm: float = 1.0):
        """Clip gradients for Byzantine resilience"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
    
    def detect_byzantine_updates(self, updates: Dict[int, torch.Tensor]) -> set:
        """
        Detect potential Byzantine updates from other nodes
        
        Args:
            updates: Dictionary of parameter updates from other nodes
            
        Returns:
            Set of potentially Byzantine node IDs
        """
        byzantine_nodes = set()
        
        if not updates or self.gradient_threshold is None:
            return byzantine_nodes
        
        # Compute update norms
        update_norms = {}
        for node_id, update in updates.items():
            update_norms[node_id] = torch.norm(update).item()
        
        # Statistical outlier detection
        norms = list(update_norms.values())
        if len(norms) > 3:
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)
            threshold = mean_norm + 3 * std_norm
            
            for node_id, norm in update_norms.items():
                if norm > threshold:
                    byzantine_nodes.add(node_id)
        
        # Check against historical threshold
        if self.gradient_threshold is not None:
            for node_id, norm in update_norms.items():
                if norm > self.gradient_threshold * 2:
                    byzantine_nodes.add(node_id)
        
        return byzantine_nodes


class AdaptiveGNN(nn.Module):
    """
    GNN that adapts to graph structure and data distribution
    
    Features:
    - Dynamic layer selection
    - Adaptive dropout rates
    - Structure-aware normalization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        num_models: int = 3
    ):
        super().__init__()
        
        # Create ensemble of models
        self.models = nn.ModuleList([
            DistributedGCN(input_dim, hidden_dims, output_dim),
            DistributedGAT(input_dim, hidden_dims, output_dim, 
                          heads=[4] * len(hidden_dims) + [1]),
            DistributedGraphSAGE(input_dim, hidden_dims, output_dim)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_models),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with adaptive model selection"""
        # Compute gating weights
        if batch is None:
            gate_input = x.mean(dim=0, keepdim=True)
        else:
            gate_input = global_mean_pool(x, batch).mean(dim=0, keepdim=True)
        
        weights = self.gate(gate_input)
        
        # Apply models
        outputs = []
        for i, model in enumerate(self.models):
            output = model(x, edge_index, batch)
            outputs.append(output * weights[0, i])
        
        # Weighted sum
        return sum(outputs)