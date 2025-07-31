"""
Pareto-Optimal Resource Allocation (PORA) Algorithm

This module implements multi-objective optimization for resource allocation
in distributed graph learning, finding Pareto frontiers and Nash equilibria.

Author: Ayomide Caleb Adekoya
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResourceConstraints:
    """Resource constraints for optimization"""
    max_computation: float  # Maximum computation units
    max_communication: float  # Maximum communication bandwidth
    max_memory: float  # Maximum memory usage
    min_accuracy: float  # Minimum required accuracy
    node_capabilities: Dict[int, Dict[str, float]]  # Per-node capabilities


@dataclass
class AllocationResult:
    """Result of resource allocation optimization"""
    allocation: Dict[int, Dict[str, float]]  # Node -> Resource allocation
    objectives: Dict[str, float]  # Objective values
    pareto_front: List[Tuple[float, ...]]  # Pareto frontier points
    nash_equilibrium: Optional[Dict[int, float]]  # Nash equilibrium solution
    optimization_time: float


class ParetoOptimalResourceAllocator:
    """
    Implements Pareto-Optimal Resource Allocation (PORA) for distributed GNN training
    
    Key features:
    1. Multi-objective optimization (accuracy, communication, computation)
    2. Evolutionary algorithm for Pareto frontier discovery
    3. Game-theoretic Nash equilibrium analysis
    4. Dynamic pricing mechanism for incentivization
    """
    
    def __init__(
        self,
        num_nodes: int,
        objectives: List[str] = ['accuracy', 'communication', 'computation'],
        population_size: int = 100,
        generations: int = 50
    ):
        self.num_nodes = num_nodes
        self.objectives = objectives
        self.population_size = population_size
        self.generations = generations
        
        # Setup DEAP for multi-objective optimization
        self._setup_deap()
        
        # Metrics tracking
        self.metrics = {
            'pareto_fronts': [],
            'hypervolume': [],
            'optimization_iterations': 0
        }
        
        logger.info(f"PORA initialized for {num_nodes} nodes with objectives: {objectives}")
    
    def _setup_deap(self):
        """Setup DEAP framework for evolutionary optimization"""
        # Create fitness class (minimize communication and computation, maximize accuracy)
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        weights = []
        for obj in self.objectives:
            if obj == 'accuracy':
                weights.append(1.0)  # Maximize
            else:
                weights.append(-1.0)  # Minimize
                
        creator.create("FitnessMulti", base.Fitness, weights=tuple(weights))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        # Create toolbox
        self.toolbox = base.Toolbox()
        
        # Gene: allocation percentage for each node (0.1 to 1.0)
        self.toolbox.register("gene", np.random.uniform, 0.1, 1.0)
        
        # Individual: allocation for all nodes
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.gene,
            n=self.num_nodes * 3  # 3 resources per node
        )
        
        # Population
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )
        
        # Genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        self.toolbox.register("select", tools.selNSGA2)
    
    def optimize(
        self,
        constraints: ResourceConstraints,
        node_data_sizes: Dict[int, int],
        graph_properties: Dict[str, float]
    ) -> AllocationResult:
        """
        Main optimization method to find Pareto-optimal resource allocation
        
        Args:
            constraints: Resource constraints
            node_data_sizes: Data size for each node
            graph_properties: Graph topology properties
            
        Returns:
            AllocationResult with Pareto-optimal allocations
        """
        import time
        start_time = time.time()
        
        logger.info("Starting PORA optimization")
        
        # Define evaluation function
        def evaluate(individual):
            allocation = self._decode_individual(individual)
            
            # Check constraints
            if not self._check_constraints(allocation, constraints):
                return float('-inf'), float('inf'), float('inf')
            
            # Compute objectives
            accuracy = self._compute_accuracy_objective(
                allocation, node_data_sizes, graph_properties
            )
            communication = self._compute_communication_objective(
                allocation, graph_properties
            )
            computation = self._compute_computation_objective(
                allocation, constraints
            )
            
            return accuracy, communication, computation
        
        self.toolbox.register("evaluate", evaluate)
        
        # Run evolutionary algorithm
        population = self.toolbox.population(n=self.population_size)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        
        # Run NSGA-II
        pareto_front = tools.ParetoFront()
        
        population, logbook = algorithms.eaMuPlusLambda(
            population,
            self.toolbox,
            mu=self.population_size,
            lambda_=self.population_size,
            cxpb=0.7,
            mutpb=0.3,
            ngen=self.generations,
            stats=stats,
            halloffame=pareto_front,
            verbose=False
        )
        
        # Extract Pareto front
        pareto_points = [ind.fitness.values for ind in pareto_front]
        
        # Find best compromise solution (closest to ideal point)
        best_individual = self._find_best_compromise(pareto_front)
        best_allocation = self._decode_individual(best_individual)
        
        # Compute Nash equilibrium
        nash_equilibrium = self._compute_nash_equilibrium(
            best_allocation, constraints, node_data_sizes
        )
        
        # Prepare result
        objectives = {
            'accuracy': best_individual.fitness.values[0],
            'communication': best_individual.fitness.values[1],
            'computation': best_individual.fitness.values[2]
        }
        
        optimization_time = time.time() - start_time
        
        result = AllocationResult(
            allocation=best_allocation,
            objectives=objectives,
            pareto_front=pareto_points,
            nash_equilibrium=nash_equilibrium,
            optimization_time=optimization_time
        )
        
        # Update metrics
        self.metrics['pareto_fronts'].append(pareto_points)
        self.metrics['optimization_iterations'] += 1
        
        logger.info(f"PORA optimization completed in {optimization_time:.2f}s")
        
        return result
    
    def _decode_individual(self, individual: List[float]) -> Dict[int, Dict[str, float]]:
        """Decode individual to resource allocation"""
        allocation = {}
        
        for i in range(self.num_nodes):
            base_idx = i * 3
            allocation[i] = {
                'computation': individual[base_idx],
                'communication': individual[base_idx + 1],
                'memory': individual[base_idx + 2]
            }
        
        return allocation
    
    def _check_constraints(
        self,
        allocation: Dict[int, Dict[str, float]],
        constraints: ResourceConstraints
    ) -> bool:
        """Check if allocation satisfies constraints"""
        total_computation = sum(a['computation'] for a in allocation.values())
        total_communication = sum(a['communication'] for a in allocation.values())
        total_memory = sum(a['memory'] for a in allocation.values())
        
        if total_computation > constraints.max_computation:
            return False
        if total_communication > constraints.max_communication:
            return False
        if total_memory > constraints.max_memory:
            return False
        
        # Check per-node constraints
        for node_id, alloc in allocation.items():
            if node_id in constraints.node_capabilities:
                caps = constraints.node_capabilities[node_id]
                if alloc['computation'] > caps.get('max_computation', float('inf')):
                    return False
                if alloc['memory'] > caps.get('max_memory', float('inf')):
                    return False
        
        return True
    
    def _compute_accuracy_objective(
        self,
        allocation: Dict[int, Dict[str, float]],
        node_data_sizes: Dict[int, int],
        graph_properties: Dict[str, float]
    ) -> float:
        """
        Compute expected accuracy based on resource allocation
        
        Model: accuracy = f(computation, data_size, graph_connectivity)
        """
        total_accuracy = 0.0
        
        for node_id, alloc in allocation.items():
            data_size = node_data_sizes.get(node_id, 1000)
            
            # Accuracy increases with computation and data size
            node_accuracy = (
                0.5 * np.log1p(alloc['computation'] * 10) +
                0.3 * np.log1p(data_size / 1000) +
                0.2 * np.log1p(alloc['memory'] * 5)
            )
            
            # Weight by node importance (could use centrality)
            node_weight = 1.0 / self.num_nodes
            total_accuracy += node_accuracy * node_weight
        
        # Scale to [0, 1]
        return min(1.0, total_accuracy)
    
    def _compute_communication_objective(
        self,
        allocation: Dict[int, Dict[str, float]],
        graph_properties: Dict[str, float]
    ) -> float:
        """Compute communication cost based on allocation"""
        total_comm = 0.0
        
        # Communication increases with distributed computation
        variance = np.var([a['computation'] for a in allocation.values()])
        
        for node_id, alloc in allocation.items():
            # Higher computation requires more communication
            node_comm = alloc['communication'] * alloc['computation']
            total_comm += node_comm
        
        # Add penalty for imbalanced allocation (requires more coordination)
        total_comm *= (1 + variance)
        
        return total_comm
    
    def _compute_computation_objective(
        self,
        allocation: Dict[int, Dict[str, float]],
        constraints: ResourceConstraints
    ) -> float:
        """Compute total computation cost"""
        total_comp = 0.0
        
        for node_id, alloc in allocation.items():
            # Consider node efficiency
            efficiency = constraints.node_capabilities.get(
                node_id, {}
            ).get('efficiency', 1.0)
            
            node_comp = alloc['computation'] / efficiency
            total_comp += node_comp
        
        return total_comp
    
    def _find_best_compromise(self, pareto_front: List) -> List[float]:
        """
        Find best compromise solution using distance to ideal point
        """
        if not pareto_front:
            return []
        
        # Get objective values
        objectives = np.array([ind.fitness.values for ind in pareto_front])
        
        # Normalize objectives
        obj_min = objectives.min(axis=0)
        obj_max = objectives.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1  # Avoid division by zero
        
        normalized = (objectives - obj_min) / obj_range
        
        # Ideal point (best value for each objective)
        ideal_point = np.zeros(len(self.objectives))
        for i, obj in enumerate(self.objectives):
            if obj == 'accuracy':
                ideal_point[i] = 1.0  # Maximum
            else:
                ideal_point[i] = 0.0  # Minimum
        
        # Find closest to ideal point
        distances = np.linalg.norm(normalized - ideal_point, axis=1)
        best_idx = np.argmin(distances)
        
        return pareto_front[best_idx]
    
    def _compute_nash_equilibrium(
        self,
        allocation: Dict[int, Dict[str, float]],
        constraints: ResourceConstraints,
        node_data_sizes: Dict[int, int]
    ) -> Dict[int, float]:
        """
        Compute Nash equilibrium using game theory
        
        Each node is a player choosing resource contribution level
        """
        # Simplified Nash equilibrium computation
        # In practice, would use more sophisticated game-theoretic analysis
        
        equilibrium = {}
        
        for node_id in range(self.num_nodes):
            # Utility function: benefit - cost
            data_size = node_data_sizes.get(node_id, 1000)
            
            # Benefit increases with allocation and data size
            benefit = np.log1p(allocation[node_id]['computation'] * data_size / 1000)
            
            # Cost is proportional to resource contribution
            cost = (
                allocation[node_id]['computation'] * 0.5 +
                allocation[node_id]['communication'] * 0.3 +
                allocation[node_id]['memory'] * 0.2
            )
            
            # Nash equilibrium strategy (simplified)
            equilibrium[node_id] = benefit / (cost + 1e-6)
        
        # Normalize
        total = sum(equilibrium.values())
        if total > 0:
            equilibrium = {k: v/total for k, v in equilibrium.items()}
        
        return equilibrium
    
    def visualize_pareto_front(
        self,
        result: AllocationResult,
        objectives_to_plot: Tuple[str, str] = ('accuracy', 'communication'),
        save_path: Optional[str] = None
    ):
        """Visualize Pareto frontier"""
        if len(result.pareto_front) == 0:
            logger.warning("No Pareto front to visualize")
            return
        
        # Get indices of objectives to plot
        obj_indices = [self.objectives.index(obj) for obj in objectives_to_plot]
        
        # Extract objective values
        pareto_points = np.array(result.pareto_front)
        x_values = pareto_points[:, obj_indices[0]]
        y_values = pareto_points[:, obj_indices[1]]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot Pareto front
        plt.scatter(x_values, y_values, c='blue', s=50, alpha=0.6, label='Pareto Front')
        
        # Highlight best compromise
        best_point = [result.objectives[objectives_to_plot[0]], 
                     result.objectives[objectives_to_plot[1]]]
        plt.scatter(best_point[0], best_point[1], c='red', s=200, 
                   marker='*', label='Best Compromise')
        
        plt.xlabel(objectives_to_plot[0].capitalize())
        plt.ylabel(objectives_to_plot[1].capitalize())
        plt.title('Pareto Frontier for Resource Allocation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def compute_hypervolume(self, pareto_front: List[Tuple[float, ...]]) -> float:
        """
        Compute hypervolume indicator for Pareto front quality
        """
        if not pareto_front:
            return 0.0
        
        # Reference point (worst case for all objectives)
        ref_point = []
        for i, obj in enumerate(self.objectives):
            if obj == 'accuracy':
                ref_point.append(0.0)  # Minimum accuracy
            else:
                ref_point.append(10.0)  # Maximum cost
        
        # Simple hypervolume computation for 2D
        if len(self.objectives) == 2:
            points = sorted(pareto_front, key=lambda x: x[0])
            volume = 0.0
            prev_x = ref_point[0]
            
            for point in points:
                if point[1] < ref_point[1]:
                    volume += (point[0] - prev_x) * (ref_point[1] - point[1])
                    prev_x = point[0]
            
            return volume
        else:
            # For higher dimensions, use approximation
            # This is a simplified version
            return np.prod([max(0, ref_point[i] - min(p[i] for p in pareto_front)) 
                           for i in range(len(self.objectives))])
    
    def dynamic_pricing_mechanism(
        self,
        allocation: Dict[int, Dict[str, float]],
        market_conditions: Dict[str, float]
    ) -> Dict[int, float]:
        """
        Compute dynamic prices for resources to incentivize participation
        
        Args:
            allocation: Current resource allocation
            market_conditions: Market parameters (supply, demand, etc.)
            
        Returns:
            Prices for each node
        """
        prices = {}
        
        # Base price from supply and demand
        base_price = market_conditions.get('base_price', 1.0)
        demand_factor = market_conditions.get('demand_factor', 1.0)
        
        for node_id, alloc in allocation.items():
            # Price increases with scarcity
            total_resources = sum(alloc.values())
            scarcity_factor = 1.0 / (total_resources + 0.1)
            
            # Price decreases with higher contribution
            contribution_discount = 0.9 ** total_resources
            
            # Node-specific price
            node_price = base_price * demand_factor * scarcity_factor * contribution_discount
            prices[node_id] = node_price
        
        return prices