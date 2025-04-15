from typing import Dict, Any, List
import numpy as np # type: ignore
from .base import BaseExperiment
from ..algorithms.gaussian import cluster_gaussian

class Experiment2(BaseExperiment):
    """Experiment 2: Varying dimensions and delta"""
    
    def run_single_iteration(self, seed: int) -> Dict[str, Any]:
        """
        Run a single Monte Carlo iteration for Experiment 2
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing iteration results
        """
        np.random.seed(seed)
        
        # Initialize output arrays
        mean_budgets = np.zeros([len(self.config.num_items_array), len(self.config.delta_array)])
        lower_budgets = np.zeros([len(self.config.num_items_array), len(self.config.delta_array)])
        upper_budgets = np.zeros([len(self.config.num_items_array), len(self.config.delta_array)])
        
        # Iterate over different num_items values
        for n_idx, num_items in enumerate(self.config.num_items_array):
            # Generate true clusters
            true_clusters = np.random.permutation(
                1 * (np.array(range(num_items)) < self.config.theta * num_items)
            )
            
            # Generate data matrix
            M = self._generate_data_matrix(num_items, true_clusters)
            
            # Iterate over different delta values
            for delta_idx, delta in enumerate(self.config.delta_array):
                # Run cluster algorithm
                cluster_clusters, cluster_budget = cluster_gaussian(M, delta)
                
                # Store results
                mean_budgets[n_idx, delta_idx] = cluster_budget
                lower_budgets[n_idx, delta_idx] = cluster_budget * 0.9  # 10% lower bound
                upper_budgets[n_idx, delta_idx] = cluster_budget * 1.1  # 10% upper bound
        
        return {
            'mean_budgets': mean_budgets,
            'lower_budgets': lower_budgets,
            'upper_budgets': upper_budgets
        }
        
    def process_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Process results from all iterations
        
        Args:
            results: List of results from each iteration
        """
        # Initialize result arrays
        self.results = {
            'mean_budgets': np.zeros([len(self.config.num_items_array), len(self.config.delta_array)]),
            'lower_budgets': np.zeros([len(self.config.num_items_array), len(self.config.delta_array)]),
            'upper_budgets': np.zeros([len(self.config.num_items_array), len(self.config.delta_array)]),
            'num_items_array': self.config.num_items_array,
            'delta_array': self.config.delta_array
        }
        
        # Store results from each iteration
        for k, result in enumerate(results):
            for key in ['mean_budgets', 'lower_budgets', 'upper_budgets']:
                self.results[key] += result[key] / len(results)
                
    def _generate_data_matrix(self, num_items: int, true_clusters: np.ndarray) -> np.ndarray:
        """Generate data matrix for given number of items"""
        num_features = num_items * self.config.multip  # Calculate num_features from num_items
        m_a = np.zeros(num_features)
        m_b = np.zeros(num_features)
        m_b[range(self.config.sparsity)] = np.zeros(self.config.sparsity) + self.config.average_signal_strength/np.sqrt(self.config.sparsity)
        
        M = np.zeros([num_items, num_features])
        for i in range(num_items):
            if true_clusters[i] == 1:
                M[i,:] = m_a
            else:
                M[i,:] = m_b
        return M 