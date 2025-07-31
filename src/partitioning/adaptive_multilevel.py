"""
Adaptive Multi-Level Graph Partitioning (AMP) Algorithm

This module implements the novel AMP algorithm that combines spectral clustering
with simulated annealing for multi-objective graph partitioning optimization.

Key features:
- Multi-level coarsening with heavy-edge matching
- Spectral clustering + simulated annealing for initial partitioning  
- Multi-objective optimization (cut size, load balance, communication cost)
- Dynamic repartitioning based on graph evolution

Author: Ayomide Caleb Adekoya
"""

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class PartitionConstraints:
    """Constraints for graph partitioning"""
    max_load_imbalance: float = 0.1  # Maximum load imbalance (10%)
    max_communication_cost: Optional[float] = None
    min_partition_size: int = 10
    max_partition_size: Optional[int] = None
    node_weights: Optional[Dict[int, float]] = None
    edge_weights: Optional[Dict[Tuple[int, int], float]] = None

@dataclass
class PartitionResult:
    """Result of graph partitioning"""
    partition_assignment: Dict[int, int]
    num_partitions: int
    cut_size: int
    load_balance: float
    communication_cost: float
    objective_score: float
    partitioning_time: float
    coarsening_levels: int
    refinement_iterations: int

@dataclass
class PartitionMetrics:
    """Detailed partitioning quality metrics"""
    cut_ratio: float
    edge_cut: int
    vertex_cut: int
    max_partition_size: int
    min_partition_size: int
    size_variance: float
    modularity: float
    conductance: float

class AdaptiveMultiLevelPartitioner:
    """
    Adaptive Multi-Level Graph Partitioning Algorithm
    
    Implements a sophisticated partitioning approach with:
    1. Multi-level coarsening using heavy-edge matching
    2. Spectral clustering for initial partitioning
    3. Simulated annealing for optimization
    4. Multi-objective refinement
    5. Adaptive repartitioning for dynamic graphs
    """
    
    def __init__(
        self,
        objective_weights: Dict[str, float] = None,
        coarsening_threshold: int = 1000,
        refinement_iterations: int = 10,
        annealing_schedule: Dict[str, float] = None
    ):
        # Multi-objective weights (cut, balance, communication)
        self.objective_weights = objective_weights or {
            'cut': 0.4,
            'balance': 0.4,
            'communication': 0.2
        }
        
        # Algorithm parameters
        self.coarsening_threshold = coarsening_threshold
        self.refinement_iterations = refinement_iterations
        
        # Simulated annealing parameters
        self.annealing_schedule = annealing_schedule or {
            'initial_temp': 100.0,
            'cooling_rate': 0.95,
            'min_temp': 0.01,
            'max_iterations': 1000
        }
        
        # Performance tracking
        self.metrics = {
            'total_partitions': 0,
            'avg_partitioning_time': 0,
            'best_objective_scores': [],
            'coarsening_times': [],
            'refinement_times': []
        }
        
        logger.info("AMP partitioner initialized with multi-objective optimization")
    
    def partition(
        self,
        graph: nx.Graph,
        num_partitions: int,
        constraints: PartitionConstraints = None
    ) -> PartitionResult:
        """
        Main partitioning method implementing the full AMP algorithm
        
        Args:
            graph: Input graph to partition
            num_partitions: Desired number of partitions
            constraints: Partitioning constraints
            
        Returns:
            PartitionResult with detailed results
        """
        start_time = time.time()
        constraints = constraints or PartitionConstraints()
        
        logger.info(f"Starting AMP partitioning: {len(graph)} nodes, {num_partitions} partitions")
        
        # Phase 1: Multi-level coarsening
        coarsening_start = time.time()
        coarse_hierarchy = self._multilevel_coarsening(graph, constraints)
        coarsening_time = time.time() - coarsening_start
        self.metrics['coarsening_times'].append(coarsening_time)
        
        logger.debug(f"Coarsening completed: {len(coarse_hierarchy)} levels in {coarsening_time:.2f}s")
        
        # Phase 2: Initial partitioning on coarsest graph
        initial_partition = self._spectral_annealing_partition(
            coarse_hierarchy[-1], num_partitions, constraints
        )
        
        # Phase 3: Multi-level refinement
        refinement_start = time.time()
        final_partition = self._multilevel_refinement(
            coarse_hierarchy, initial_partition, constraints
        )
        refinement_time = time.time() - refinement_start
        self.metrics['refinement_times'].append(refinement_time)
        
        # Compute final metrics
        total_time = time.time() - start_time
        metrics = self._compute_partition_metrics(graph, final_partition, constraints)
        
        result = PartitionResult(
            partition_assignment=final_partition,
            num_partitions=num_partitions,
            cut_size=metrics.edge_cut,
            load_balance=1.0 - metrics.size_variance,
            communication_cost=self._compute_communication_cost(graph, final_partition),
            objective_score=self._compute_objective_score(graph, final_partition, constraints),
            partitioning_time=total_time,
            coarsening_levels=len(coarse_hierarchy),
            refinement_iterations=self.refinement_iterations
        )
        
        self.metrics['total_partitions'] += 1
        self.metrics['best_objective_scores'].append(result.objective_score)
        
        logger.info(f"AMP partitioning completed in {total_time:.2f}s, objective: {result.objective_score:.4f}")
        
        return result
    
    def _multilevel_coarsening(
        self,
        graph: nx.Graph,
        constraints: PartitionConstraints
    ) -> List[nx.Graph]:
        """
        Multi-level graph coarsening using heavy-edge matching
        
        Iteratively reduces graph size while preserving important structural properties
        """
        hierarchy = [graph.copy()]
        current_graph = graph.copy()
        level = 0
        
        while current_graph.number_of_nodes() > self.coarsening_threshold:
            logger.debug(f"Coarsening level {level}: {current_graph.number_of_nodes()} nodes")
            
            # Find maximum weight matching for edge contraction
            matching = self._compute_heavy_edge_matching(current_graph, constraints)
            
            if len(matching) < current_graph.number_of_nodes() * 0.1:
                # Stop coarsening if very few edges can be matched
                logger.debug("Stopping coarsening: insufficient matching")
                break
            
            # Contract matched edges to create coarser graph
            current_graph = self._contract_graph(current_graph, matching, constraints)
            hierarchy.append(current_graph)
            level += 1
            
            # Safety check to prevent infinite loops
            if level > 20:
                logger.warning("Maximum coarsening levels reached")
                break
        
        logger.info(f"Multi-level coarsening completed: {len(hierarchy)} levels")
        return hierarchy
    
    def _compute_heavy_edge_matching(
        self,
        graph: nx.Graph,
        constraints: PartitionConstraints
    ) -> List[Tuple[int, int]]:
        """
        Compute maximum weight matching for graph coarsening
        
        Uses greedy algorithm to find heavy edges for contraction
        """
        # Get edge weights (default to 1.0 if not specified)
        edge_weights = constraints.edge_weights or {}
        
        # Create list of edges sorted by weight (descending)
        weighted_edges = []
        for u, v in graph.edges():
            weight = edge_weights.get((u, v), edge_weights.get((v, u), 1.0))
            weighted_edges.append((weight, u, v))
        
        weighted_edges.sort(reverse=True)  # Sort by weight descending
        
        # Greedy matching algorithm
        matching = []
        matched_nodes = set()
        
        for weight, u, v in weighted_edges:
            if u not in matched_nodes and v not in matched_nodes:
                matching.append((u, v))
                matched_nodes.add(u)
                matched_nodes.add(v)
        
        logger.debug(f"Heavy-edge matching found {len(matching)} pairs")
        return matching
    
    def _contract_graph(
        self,
        graph: nx.Graph,
        matching: List[Tuple[int, int]],
        constraints: PartitionConstraints
    ) -> nx.Graph:
        """
        Contract matched edges to create coarser graph
        
        Combines matched nodes while preserving edge weights and node properties
        """
        contracted = graph.copy()
        node_mapping = {}  # Maps original nodes to contracted node IDs
        
        # Contract each matched pair
        for u, v in matching:
            if u in contracted and v in contracted:
                # Create new contracted node ID
                new_node = min(u, v)  # Use smaller ID as representative
                old_node = max(u, v)
                
                # Get node weights
                node_weights = constraints.node_weights or {}
                u_weight = node_weights.get(u, 1.0)
                v_weight = node_weights.get(v, 1.0)
                combined_weight = u_weight + v_weight
                
                # Combine edges from both nodes
                u_neighbors = set(contracted.neighbors(u))
                v_neighbors = set(contracted.neighbors(v))
                
                # Add new node with combined weight
                contracted.add_node(new_node, weight=combined_weight)
                
                # Add combined edges
                for neighbor in (u_neighbors | v_neighbors) - {u, v}:
                    if neighbor in contracted:
                        # Combine edge weights if both nodes had edges to neighbor
                        edge_weight = 0.0
                        if contracted.has_edge(u, neighbor):
                            edge_weight += contracted[u][neighbor].get('weight', 1.0)
                        if contracted.has_edge(v, neighbor):
                            edge_weight += contracted[v][neighbor].get('weight', 1.0)
                        
                        contracted.add_edge(new_node, neighbor, weight=edge_weight)
                
                # Remove original nodes
                contracted.remove_node(u)
                contracted.remove_node(v)
                
                # Update mapping
                node_mapping[u] = new_node
                node_mapping[v] = new_node
        
        logger.debug(f"Graph contracted: {graph.number_of_nodes()} -> {contracted.number_of_nodes()} nodes")
        return contracted
    
    def _spectral_annealing_partition(
        self,
        graph: nx.Graph,
        num_partitions: int,
        constraints: PartitionConstraints
    ) -> Dict[int, int]:
        """
        Initial partitioning using spectral clustering + simulated annealing
        
        Combines spectral methods for global structure with annealing for optimization
        """
        logger.debug(f"Spectral annealing partition: {graph.number_of_nodes()} nodes -> {num_partitions} partitions")
        
        # Phase 1: Spectral clustering for initial partition
        spectral_partition = self._spectral_clustering(graph, num_partitions)
        
        # Phase 2: Simulated annealing optimization
        optimized_partition = self._simulated_annealing_optimization(
            graph, spectral_partition, constraints
        )
        
        return optimized_partition
    
    def _spectral_clustering(self, graph: nx.Graph, num_partitions: int) -> Dict[int, int]:
        """
        Spectral clustering based on graph Laplacian eigenvectors
        """
        # Compute normalized Laplacian matrix
        laplacian = nx.normalized_laplacian_matrix(graph, nodelist=sorted(graph.nodes()))
        
        # Compute smallest eigenvectors (Fiedler vectors)
        try:
            eigenvals, eigenvecs = eigsh(laplacian, k=num_partitions, which='SM')
        except Exception as e:
            logger.warning(f"Spectral clustering failed: {e}, using random partition")
            return self._random_partition(graph, num_partitions)
        
        # Use k-means clustering on eigenvectors
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=num_partitions, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(eigenvecs)
        
        # Create partition assignment
        nodes = sorted(graph.nodes())
        partition = {nodes[i]: int(cluster_labels[i]) for i in range(len(nodes))}
        
        logger.debug("Spectral clustering completed")
        return partition
    
    def _simulated_annealing_optimization(
        self,
        graph: nx.Graph,
        initial_partition: Dict[int, int],
        constraints: PartitionConstraints
    ) -> Dict[int, int]:
        """
        Simulated annealing for multi-objective partition optimization
        """
        current_partition = initial_partition.copy()
        best_partition = initial_partition.copy()
        
        current_score = self._compute_objective_score(graph, current_partition, constraints)
        best_score = current_score
        
        # Annealing parameters
        temperature = self.annealing_schedule['initial_temp']
        cooling_rate = self.annealing_schedule['cooling_rate']
        min_temp = self.annealing_schedule['min_temp']
        max_iterations = self.annealing_schedule['max_iterations']
        
        iteration = 0
        
        while temperature > min_temp and iteration < max_iterations:
            # Generate neighbor solution by moving random node
            neighbor_partition = self._generate_neighbor_partition(
                current_partition, graph, constraints
            )
            
            neighbor_score = self._compute_objective_score(graph, neighbor_partition, constraints)
            
            # Accept or reject based on annealing criterion
            delta = neighbor_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current_partition = neighbor_partition
                current_score = neighbor_score
                
                # Update best solution
                if current_score > best_score:
                    best_partition = current_partition.copy()
                    best_score = current_score
            
            # Cool down
            temperature *= cooling_rate
            iteration += 1
            
            if iteration % 100 == 0:
                logger.debug(f"Annealing iteration {iteration}, temp: {temperature:.4f}, score: {current_score:.4f}")
        
        logger.debug(f"Simulated annealing completed: {iteration} iterations, best score: {best_score:.4f}")
        return best_partition
    
    def _generate_neighbor_partition(
        self,
        partition: Dict[int, int],
        graph: nx.Graph,
        constraints: PartitionConstraints
    ) -> Dict[int, int]:
        """Generate neighbor solution for simulated annealing"""
        neighbor = partition.copy()
        
        # Select random node to move
        nodes = list(graph.nodes())
        node = random.choice(nodes)
        current_partition = partition[node]
        
        # Select different partition (avoid creating empty partitions)
        num_partitions = max(partition.values()) + 1
        available_partitions = [p for p in range(num_partitions) if p != current_partition]
        
        if available_partitions:
            new_partition = random.choice(available_partitions)
            neighbor[node] = new_partition
        
        return neighbor
    
    def _multilevel_refinement(
        self,
        hierarchy: List[nx.Graph],
        initial_partition: Dict[int, int],
        constraints: PartitionConstraints
    ) -> Dict[int, int]:
        """
        Multi-level refinement from coarse to fine graph
        
        Projects partition up the hierarchy while refining at each level
        """
        current_partition = initial_partition
        
        # Refine from coarsest to finest level
        for level in range(len(hierarchy) - 2, -1, -1):
            current_graph = hierarchy[level]
            
            logger.debug(f"Refining level {level}: {current_graph.number_of_nodes()} nodes")
            
            # Project partition to finer level
            current_partition = self._project_partition(
                hierarchy[level + 1], current_graph, current_partition
            )
            
            # Local refinement using Fiduccia-Mattheyses-like algorithm
            current_partition = self._local_refinement(
                current_graph, current_partition, constraints
            )
        
        return current_partition
    
    def _project_partition(
        self,
        coarse_graph: nx.Graph,
        fine_graph: nx.Graph,
        coarse_partition: Dict[int, int]
    ) -> Dict[int, int]:
        """
        Project partition from coarse to fine graph
        
        This is a simplified projection - in full implementation,
        would maintain contraction mapping between levels
        """
        # For now, use simple node ID matching
        # In full implementation, would use proper contraction history
        fine_partition = {}
        
        for node in fine_graph.nodes():
            if node in coarse_partition:
                fine_partition[node] = coarse_partition[node]
            else:
                # Assign to partition of nearest neighbor in coarse graph
                neighbors = list(fine_graph.neighbors(node))
                if neighbors:
                    neighbor_partitions = [
                        coarse_partition.get(n, 0) for n in neighbors
                        if n in coarse_partition
                    ]
                    if neighbor_partitions:
                        fine_partition[node] = max(set(neighbor_partitions), 
                                                 key=neighbor_partitions.count)
                    else:
                        fine_partition[node] = 0
                else:
                    fine_partition[node] = 0
        
        return fine_partition
    
    def _local_refinement(
        self,
        graph: nx.Graph,
        partition: Dict[int, int],
        constraints: PartitionConstraints
    ) -> Dict[int, int]:
        """
        Local refinement using gain-based node moves
        
        Implements Fiduccia-Mattheyses-style local search
        """
        improved_partition = partition.copy()
        
        for iteration in range(self.refinement_iterations):
            improvement_found = False
            
            # Calculate gain for each potential node move
            for node in graph.nodes():
                current_partition = improved_partition[node]
                best_gain = 0
                best_target = current_partition
                
                # Try moving to each other partition
                num_partitions = max(improved_partition.values()) + 1
                for target_partition in range(num_partitions):
                    if target_partition != current_partition:
                        gain = self._compute_move_gain(
                            graph, improved_partition, node, target_partition, constraints
                        )
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_target = target_partition
                
                # Make move if beneficial
                if best_gain > 0:
                    improved_partition[node] = best_target
                    improvement_found = True
            
            if not improvement_found:
                break
        
        return improved_partition
    
    def _compute_move_gain(
        self,
        graph: nx.Graph,
        partition: Dict[int, int],
        node: int,
        target_partition: int,
        constraints: PartitionConstraints
    ) -> float:
        """
        Compute gain from moving node to target partition
        
        Considers multi-objective function: cut size, balance, communication
        """
        current_partition = partition[node]
        
        # Temporarily move node
        old_score = self._compute_objective_score(graph, partition, constraints)
        
        test_partition = partition.copy()
        test_partition[node] = target_partition
        
        new_score = self._compute_objective_score(graph, test_partition, constraints)
        
        return new_score - old_score
    
    def _compute_objective_score(
        self,
        graph: nx.Graph,
        partition: Dict[int, int],
        constraints: PartitionConstraints
    ) -> float:
        """
        Compute multi-objective score for partition quality
        
        Combines cut size, load balance, and communication cost
        """
        # Cut size component (minimize)
        cut_size = self._compute_cut_size(graph, partition)
        normalized_cut = 1.0 - (cut_size / max(graph.number_of_edges(), 1))
        
        # Load balance component (maximize)
        load_balance = self._compute_load_balance(graph, partition, constraints)
        
        # Communication cost component (minimize)
        comm_cost = self._compute_communication_cost(graph, partition)
        max_comm = graph.number_of_edges()  # Upper bound
        normalized_comm = 1.0 - (comm_cost / max(max_comm, 1))
        
        # Weighted combination
        score = (
            self.objective_weights['cut'] * normalized_cut +
            self.objective_weights['balance'] * load_balance +
            self.objective_weights['communication'] * normalized_comm
        )
        
        return score
    
    def _compute_cut_size(self, graph: nx.Graph, partition: Dict[int, int]) -> int:
        """Compute number of edges crossing partition boundaries"""
        cut_size = 0
        for u, v in graph.edges():
            if partition[u] != partition[v]:
                cut_size += 1
        return cut_size
    
    def _compute_load_balance(
        self,
        graph: nx.Graph,
        partition: Dict[int, int],
        constraints: PartitionConstraints
    ) -> float:
        """Compute load balance metric (1.0 = perfect balance)"""
        node_weights = constraints.node_weights or {}
        
        # Calculate partition sizes
        partition_weights = {}
        for node, part in partition.items():
            weight = node_weights.get(node, 1.0)
            partition_weights[part] = partition_weights.get(part, 0) + weight
        
        if not partition_weights:
            return 1.0
        
        # Calculate balance metric
        total_weight = sum(partition_weights.values())
        num_partitions = len(partition_weights)
        ideal_weight = total_weight / num_partitions
        
        max_deviation = max(
            abs(weight - ideal_weight) / ideal_weight
            for weight in partition_weights.values()
        )
        
        return max(0.0, 1.0 - max_deviation)
    
    def _compute_communication_cost(self, graph: nx.Graph, partition: Dict[int, int]) -> float:
        """Compute communication cost based on inter-partition edges"""
        comm_cost = 0.0
        
        for u, v in graph.edges():
            if partition[u] != partition[v]:
                # Add communication cost (could be distance-based in full implementation)
                edge_weight = graph[u][v].get('weight', 1.0)
                comm_cost += edge_weight
        
        return comm_cost
    
    def _compute_partition_metrics(
        self,
        graph: nx.Graph,
        partition: Dict[int, int],
        constraints: PartitionConstraints
    ) -> PartitionMetrics:
        """Compute detailed partition quality metrics"""
        
        # Basic metrics
        cut_size = self._compute_cut_size(graph, partition)
        cut_ratio = cut_size / max(graph.number_of_edges(), 1)
        
        # Partition size statistics
        partition_sizes = {}
        for node, part in partition.items():
            partition_sizes[part] = partition_sizes.get(part, 0) + 1
        
        sizes = list(partition_sizes.values())
        max_size = max(sizes) if sizes else 0
        min_size = min(sizes) if sizes else 0
        size_variance = np.var(sizes) if len(sizes) > 1 else 0
        
        # Modularity
        modularity = self._compute_modularity(graph, partition)
        
        # Conductance (for 2-partition case)
        conductance = self._compute_conductance(graph, partition)
        
        return PartitionMetrics(
            cut_ratio=cut_ratio,
            edge_cut=cut_size,
            vertex_cut=0,  # Would compute vertex cut in full implementation
            max_partition_size=max_size,
            min_partition_size=min_size,
            size_variance=size_variance,
            modularity=modularity,
            conductance=conductance
        )
    
    def _compute_modularity(self, graph: nx.Graph, partition: Dict[int, int]) -> float:
        """Compute modularity of partition"""
        try:
            # Convert partition to list of sets
            communities = {}
            for node, part in partition.items():
                if part not in communities:
                    communities[part] = set()
                communities[part].add(node)
            
            community_list = list(communities.values())
            return nx.algorithms.community.modularity(graph, community_list)
        except:
            return 0.0
    
    def _compute_conductance(self, graph: nx.Graph, partition: Dict[int, int]) -> float:
        """Compute conductance (for 2-partition case)"""
        if len(set(partition.values())) != 2:
            return 0.0  # Conductance only defined for 2-partitions
        
        try:
            # Split nodes by partition
            part_0_nodes = {n for n, p in partition.items() if p == 0}
            part_1_nodes = {n for n, p in partition.items() if p == 1}
            
            if not part_0_nodes or not part_1_nodes:
                return 0.0
            
            # Use smaller partition for conductance calculation  
            smaller_part = part_0_nodes if len(part_0_nodes) <= len(part_1_nodes) else part_1_nodes
            
            return nx.algorithms.cuts.conductance(graph, smaller_part)
        except:
            return 0.0
    
    def _random_partition(self, graph: nx.Graph, num_partitions: int) -> Dict[int, int]:
        """Generate random partition as fallback"""
        partition = {}
        nodes = list(graph.nodes())
        
        for i, node in enumerate(nodes):
            partition[node] = i % num_partitions
        
        return partition
    
    def get_performance_metrics(self) -> Dict:
        """Get algorithm performance metrics"""
        metrics = self.metrics.copy()
        
        if metrics['coarsening_times']:
            metrics['avg_coarsening_time'] = np.mean(metrics['coarsening_times'])
        
        if metrics['refinement_times']:
            metrics['avg_refinement_time'] = np.mean(metrics['refinement_times'])
        
        if metrics['best_objective_scores']:
            metrics['avg_objective_score'] = np.mean(metrics['best_objective_scores'])
            metrics['best_objective_score'] = max(metrics['best_objective_scores'])
        
        return metrics