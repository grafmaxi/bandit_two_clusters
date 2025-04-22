"""Experiment 2: varying dimensions and delta parameters."""

import numpy as np
from typing import Tuple, Dict, Any
from src.algorithms.gaussian import cluster_gaussian
from src.utils.clustering import clusters_equal
from src.utils.data_generation import generate_clusters, generate_data_matrix
from src.configs.config_2 import config_2

def simulation_iteration_2(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run one iteration of Experiment 2.
    
    This experiment tests the cluster algorithm with:
    - Varying matrix dimensions (n x d)
    - Different delta parameters
    - Fixed sparsity level
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - cluster_errors: Error rates for cluster algorithm
        - cluster_budgets: Budgets used by cluster algorithm
    """
    # Set random seed
    np.random.seed(seed)
    print(seed)
    
    # Initialize output arrays
    num_items_count = config_2["num_items_count"]
    delta_count = config_2["delta_count"]
    
    cluster_budgets = np.zeros([num_items_count, delta_count])
    cluster_errors = np.zeros([num_items_count, delta_count])
    
    # Iterate over different matrix dimensions
    for items_idx in range(num_items_count):
        num_items = config_2["num_items_array"][items_idx]
        num_features = config_2["num_features_array"][items_idx]
        
        # Generate true clusters
        true_clusters = generate_clusters(num_items, config_2["theta"])
        
        # Generate data matrix
        M = generate_data_matrix(
            num_items,
            num_features,
            config_2["sparsity"],
            config_2["signal_strength"],
            true_clusters
        )
        
        # Test different delta parameters
        for delta_idx in range(delta_count):
            trial_clusters, trial_budgets = cluster_gaussian(M, config_2["delta_array"][delta_idx])
            cluster_budgets[items_idx, delta_idx] = trial_budgets
            cluster_errors[items_idx, delta_idx] = 1 - int(clusters_equal(true_clusters, trial_clusters))
    
    return cluster_errors, cluster_budgets