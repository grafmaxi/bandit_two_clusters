from typing import Dict, Any, List
import numpy as np # type: ignore
from .base import BaseExperiment
from ..algorithms.gaussian import cr_gaussian, cbc_gaussian, cluster_gaussian
from ..utils.clustering import get_cluster_representatives

class Experiment3(BaseExperiment):
    """Experiment 3: Varying proportion with constant gap"""
    
    def run_single_iteration(self, seed: int) -> Dict[str, Any]:
        """
        Run a single Monte Carlo iteration for Experiment 3
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing iteration results
        """
        np.random.seed(seed)
        
        # Initialize output arrays
        cr_budgets = np.zeros(len(self.config.theta_vec))
        cbc_budgets = np.zeros(len(self.config.theta_vec))
        cluster_budgets = np.zeros(len(self.config.theta_vec))
        
        # Iterate over different theta values
        for theta_idx, theta in enumerate(self.config.theta_vec):
            # Generate true clusters
            true_clusters = np.random.permutation(
                1 * (np.array(range(self.config.num_items)) < theta * self.config.num_items)
            )
            i_a, i_b = get_cluster_representatives(true_clusters)
            
            # Generate data matrix
            M = self._generate_data_matrix(true_clusters)
            
            # Run CR algorithm
            _, cr_budget = cr_gaussian(M, self.config.delta/2)
            cr_budgets[theta_idx] = cr_budget
            
            # Run CBC algorithm
            if true_clusters[0] == 0:
                _, cbc_budget = cbc_gaussian(M, i_a, self.config.delta/2)
            else:
                _, cbc_budget = cbc_gaussian(M, i_b, self.config.delta/2)
            cbc_budgets[theta_idx] = cbc_budget
            
            # Run cluster algorithm
            _, cluster_budget = cluster_gaussian(M, self.config.delta)
            cluster_budgets[theta_idx] = cluster_budget
        
        return {
            'cr_budgets': cr_budgets,
            'cbc_budgets': cbc_budgets,
            'cluster_budgets': cluster_budgets
        }
        
    def process_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Process results from all iterations
        
        Args:
            results: List of results from each iteration
        """
        # Initialize result arrays
        self.results = {
            'cr_budgets_means': np.zeros(len(self.config.theta_vec)),
            'cbc_budgets_means': np.zeros(len(self.config.theta_vec)),
            'cluster_budgets_means': np.zeros(len(self.config.theta_vec)),
            'theta_vec': self.config.theta_vec
        }
        
        # Store results from each iteration
        for k, result in enumerate(results):
            for key in ['cr_budgets', 'cbc_budgets', 'cluster_budgets']:
                self.results[f"{key}_means"] += result[key] / len(results)
                
    def _generate_data_matrix(self, true_clusters: np.ndarray) -> np.ndarray:
        """Generate data matrix"""
        m_a = np.zeros(self.config.num_features)
        m_b = np.zeros(self.config.num_features)
        m_b[range(self.config.sparsity)] = np.zeros(self.config.sparsity) + self.config.average_signal_strength/np.sqrt(self.config.sparsity)
        
        M = np.zeros([self.config.num_items, self.config.num_features])
        for i in range(self.config.num_items):
            if true_clusters[i] == 1:
                M[i,:] = m_a
            else:
                M[i,:] = m_b
        return M 