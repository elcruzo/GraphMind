"""
Graph utility functions for GraphMind
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx


def graph_to_tensor(graph: nx.Graph) -> Data:
    """
    Convert NetworkX graph to PyTorch Geometric Data object
    
    Args:
        graph: NetworkX graph
        
    Returns:
        PyTorch Geometric Data object
    """
    # Get node features if available
    if 'x' in next(iter(graph.nodes(data=True)))[1]:
        x = torch.tensor([graph.nodes[n]['x'] for n in graph.nodes()], dtype=torch.float)
    else:
        # Use degree as default feature
        degrees = dict(graph.degree())
        x = torch.tensor([[degrees[n]] for n in graph.nodes()], dtype=torch.float)
    
    # Get edge indices
    edge_index = torch.tensor(list(graph.edges())).t().contiguous()
    
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
    
    # Get edge attributes if available
    edge_attr = None
    if graph.number_of_edges() > 0 and 'weight' in next(iter(graph.edges(data=True)))[2]:
        weights = []
        for u, v in graph.edges():
            weights.append(graph[u][v].get('weight', 1.0))
            weights.append(graph[u][v].get('weight', 1.0))  # For reverse edge
        edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
    
    # Get node labels if available
    if 'y' in next(iter(graph.nodes(data=True)))[1]:
        y = torch.tensor([graph.nodes[n]['y'] for n in graph.nodes()], dtype=torch.long)
    else:
        y = None
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def tensor_to_graph(data: Data) -> nx.Graph:
    """
    Convert PyTorch Geometric Data object to NetworkX graph
    
    Args:
        data: PyTorch Geometric Data object
        
    Returns:
        NetworkX graph
    """
    graph = to_networkx(data, to_undirected=True)
    
    # Add node features
    if data.x is not None:
        for i, features in enumerate(data.x):
            graph.nodes[i]['x'] = features.numpy()
    
    # Add node labels
    if data.y is not None:
        for i, label in enumerate(data.y):
            graph.nodes[i]['y'] = int(label)
    
    return graph


def compute_graph_statistics(graph: nx.Graph) -> Dict[str, Any]:
    """
    Compute various statistics for a graph
    
    Args:
        graph: NetworkX graph
        
    Returns:
        Dictionary of graph statistics
    """
    stats = {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'is_connected': nx.is_connected(graph),
        'num_components': nx.number_connected_components(graph),
    }
    
    if stats['num_nodes'] > 0:
        degrees = [d for n, d in graph.degree()]
        stats['avg_degree'] = np.mean(degrees)
        stats['max_degree'] = np.max(degrees)
        stats['min_degree'] = np.min(degrees)
        stats['degree_std'] = np.std(degrees)
        
        # Clustering coefficient
        stats['avg_clustering'] = nx.average_clustering(graph)
        
        # If connected, compute diameter and radius
        if stats['is_connected']:
            stats['diameter'] = nx.diameter(graph)
            stats['radius'] = nx.radius(graph)
            stats['avg_shortest_path'] = nx.average_shortest_path_length(graph)
        
        # Centrality measures
        try:
            degree_cent = nx.degree_centrality(graph)
            stats['avg_degree_centrality'] = np.mean(list(degree_cent.values()))
            
            if stats['is_connected']:
                betweenness = nx.betweenness_centrality(graph)
                stats['avg_betweenness_centrality'] = np.mean(list(betweenness.values()))
        except Exception:
            pass
    
    return stats


def visualize_graph_partition(
    graph: nx.Graph,
    partition: Dict[int, int],
    title: str = "Graph Partition",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize graph with partition coloring
    
    Args:
        graph: NetworkX graph
        partition: Dictionary mapping nodes to partition IDs
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get partition colors
    num_partitions = len(set(partition.values()))
    colors = plt.cm.Set3(np.linspace(0, 1, num_partitions))
    node_colors = [colors[partition[node]] for node in graph.nodes()]
    
    # Compute layout
    if graph.number_of_nodes() < 100:
        pos = nx.spring_layout(graph, k=2/np.sqrt(graph.number_of_nodes()), iterations=50)
    else:
        pos = nx.kamada_kawai_layout(graph)
    
    # Draw nodes and edges
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_colors, node_size=300,
        alpha=0.9, ax=ax
    )
    
    # Draw edges with different styles for inter-partition edges
    edge_colors = []
    edge_widths = []
    for u, v in graph.edges():
        if partition[u] == partition[v]:
            edge_colors.append('gray')
            edge_widths.append(1.0)
        else:
            edge_colors.append('red')
            edge_widths.append(2.0)
    
    nx.draw_networkx_edges(
        graph, pos, edge_color=edge_colors, width=edge_widths,
        alpha=0.5, ax=ax
    )
    
    # Add labels for small graphs
    if graph.number_of_nodes() < 50:
        nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
    
    # Add partition legend
    for i in range(num_partitions):
        ax.scatter([], [], c=[colors[i]], s=100, label=f'Partition {i}')
    ax.legend(loc='upper right')
    
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_synthetic_graph(
    graph_type: str,
    num_nodes: int,
    **kwargs
) -> nx.Graph:
    """
    Create synthetic graph for testing
    
    Args:
        graph_type: Type of graph ('erdos_renyi', 'barabasi_albert', 'watts_strogatz', 'community')
        num_nodes: Number of nodes
        **kwargs: Additional parameters for graph generation
        
    Returns:
        NetworkX graph
    """
    if graph_type == 'erdos_renyi':
        p = kwargs.get('p', 0.1)
        graph = nx.erdos_renyi_graph(num_nodes, p)
        
    elif graph_type == 'barabasi_albert':
        m = kwargs.get('m', 3)
        graph = nx.barabasi_albert_graph(num_nodes, m)
        
    elif graph_type == 'watts_strogatz':
        k = kwargs.get('k', 4)
        p = kwargs.get('p', 0.1)
        graph = nx.watts_strogatz_graph(num_nodes, k, p)
        
    elif graph_type == 'community':
        num_communities = kwargs.get('num_communities', 4)
        p_in = kwargs.get('p_in', 0.8)
        p_out = kwargs.get('p_out', 0.05)
        
        sizes = [num_nodes // num_communities] * num_communities
        sizes[-1] += num_nodes % num_communities
        
        graph = nx.generators.community.planted_partition_graph(
            sizes, p_in, p_out
        )
        
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    return graph


def add_node_features(
    graph: nx.Graph,
    feature_dim: int,
    feature_type: str = 'random'
) -> nx.Graph:
    """
    Add features to graph nodes
    
    Args:
        graph: NetworkX graph
        feature_dim: Feature dimension
        feature_type: Type of features ('random', 'degree', 'one_hot')
        
    Returns:
        Graph with node features
    """
    if feature_type == 'random':
        for node in graph.nodes():
            graph.nodes[node]['x'] = np.random.randn(feature_dim)
            
    elif feature_type == 'degree':
        max_degree = max(dict(graph.degree()).values())
        for node in graph.nodes():
            degree = graph.degree(node)
            features = np.zeros(feature_dim)
            features[0] = degree / max_degree
            if feature_dim > 1:
                features[1:] = np.random.randn(feature_dim - 1) * 0.1
            graph.nodes[node]['x'] = features
            
    elif feature_type == 'one_hot':
        for i, node in enumerate(graph.nodes()):
            features = np.zeros(feature_dim)
            if i < feature_dim:
                features[i] = 1.0
            graph.nodes[node]['x'] = features
            
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    return graph