from typing import Dict, Any, List
import numpy as np # type: ignore
from .base import BaseExperiment
from ..algorithms.gaussian import cr_gaussian, cbc_gaussian, cluster_gaussian
from ..algorithms.kmeans import kmeans_budget
from ..utils.clustering import clusters_equal, get_cluster_representatives

class Experiment1(BaseExperiment):
    """
    Experiment 1: Varying sparsity and time grid sizes.
    """
    def __init__(self, config: Any):
        super().__init__(config)
        # Initialize all required attributes from config
        self.spars_grid_size = config.spars_grid_size
        self.time_grid_size = config.time_grid_size
        self.num_items = config.num_items  
        self.num_features = config.num_features 
        self.theta = config.theta
        self.delta = config.delta
        self.t_min = config.t_min
        self.t_max = config.t_max
        
    def run_single_iteration(self, seed: int) -> Dict[str, Any]:
        print(seed)
        """
        Run a single iteration of the experiment.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing results for this iteration
        """
        np.random.seed(seed)
        
        # Initialize output arrays
        cr_errors = np.zeros(self.spars_grid_size)
        cr_budgets = np.zeros(self.spars_grid_size)
        cbc_errors = np.zeros(self.spars_grid_size)
        cbc_budgets = np.zeros(self.spars_grid_size)
        cluster_errors = np.zeros((self.spars_grid_size, self.time_grid_size))
        cluster_budgets = np.zeros((self.spars_grid_size, self.time_grid_size))
        km_errors = np.zeros((self.spars_grid_size, self.time_grid_size))
        km_budgets = np.zeros((self.spars_grid_size, self.time_grid_size))
        # Generate true clusters
        true_clusters = np.random.permutation(
            1 * (np.array(range(self.num_items)) < self.theta * self.num_items)
        )
        i_a, i_b = get_cluster_representatives(true_clusters)
        # Iterate over sparsity levels
        for s_ind in range(self.spars_grid_size):
            # Generate data matrix
            sparsity = int((self.num_features-1)*s_ind/(self.spars_grid_size-1)+1)
            M = self._generate_data_matrix(sparsity, true_clusters)
            # Run CR algorithm
            cr_arms, cr_budget = cr_gaussian(M, self.delta/2)
            cr_budgets[s_ind] = cr_budget
            cr_errors[s_ind] = 1 if true_clusters[0] == true_clusters[cr_arms[0]] else 0
            
            # Run CBC algorithm
            if true_clusters[0] == 0:
                cbc_clusters, cbc_budget = cbc_gaussian(M, i_a, self.delta/2)
            else:
                cbc_clusters, cbc_budget = cbc_gaussian(M, i_b, self.delta/2)
            cbc_budgets[s_ind] = cbc_budget
            cbc_errors[s_ind] = 1 - clusters_equal(true_clusters, cbc_clusters)
            
            # Run cluster algorithm for each time grid size
            for t_ind in range(self.time_grid_size):
                cluster_clusters, cluster_budget = cluster_gaussian(M, self.delta)
                
                # Handle case where clustering failed
                if cluster_clusters is None:
                    cluster_errors[s_ind, t_ind] = 1.0  # Maximum error
                    cluster_budgets[s_ind, t_ind] = cluster_budget
                    continue
                    
                # Calculate error and budget
                cluster_errors[s_ind, t_ind] = 1 - clusters_equal(true_clusters, cluster_clusters)
                cluster_budgets[s_ind, t_ind] = cluster_budget
                
                # Run K-means for each time grid size
                t = self.t_min + (self.t_max-self.t_min)*(t_ind/(self.time_grid_size-1))
                km_budgets[s_ind, t_ind] = int(t/(self.num_items*self.num_features))*(self.num_items*self.num_features)
                km_clusters = kmeans_budget(M, t)
                km_errors[s_ind, t_ind] = 1 - clusters_equal(true_clusters, km_clusters)
        return {
            'cr_errors': cr_errors,
            'cr_budgets': cr_budgets,
            'cbc_errors': cbc_errors,
            'cbc_budgets': cbc_budgets,
            'cluster_errors': cluster_errors,
            'cluster_budgets': cluster_budgets,
            'km_errors': km_errors,
            'km_budgets': km_budgets
        }
        
    def process_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Process results from all iterations
        
        Args:
            results: List of results from each iteration
        """
        # Initialize result arrays
        self.results = {
            'cr_errors': np.zeros([self.spars_grid_size, self.config.monte_carlo_runs]),
            'cr_budgets': np.zeros([self.spars_grid_size, self.config.monte_carlo_runs]),
            'cbc_errors': np.zeros([self.spars_grid_size, self.config.monte_carlo_runs]),
            'cbc_budgets': np.zeros([self.spars_grid_size, self.config.monte_carlo_runs]),
            'cluster_errors': np.zeros([self.spars_grid_size, self.time_grid_size, self.config.monte_carlo_runs]),
            'cluster_budgets': np.zeros([self.spars_grid_size, self.time_grid_size, self.config.monte_carlo_runs]),
            'km_errors': np.zeros([self.spars_grid_size, self.time_grid_size, self.config.monte_carlo_runs]),
            'km_budgets': np.zeros([self.spars_grid_size, self.time_grid_size, self.config.monte_carlo_runs])
        }
        
        # Store results from each iteration
        for k, result in enumerate(results):
            for key in self.results:
                self.results[key][..., k] = result[key]
                
    def _generate_data_matrix(self, sparsity: int, true_clusters: np.ndarray) -> np.ndarray:
        """Generate data matrix for given sparsity level"""
        m_a = np.zeros(self.config.num_features)
        m_b = np.zeros(self.config.num_features)
        m_b[range(sparsity)] = np.zeros(sparsity) + self.config.average_signal_strength/np.sqrt(sparsity)
        
        M = np.zeros([self.config.num_items, self.config.num_features])
        for i in range(self.config.num_items):
            if true_clusters[i] == 1:
                M[i,:] = m_a
            else:
                M[i,:] = m_b
        return M 