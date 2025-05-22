"""Experiment 4: Comparison of Adaptive Sampling and Bernoulli Sequential Algorithms."""

import numpy as np
import time
from typing import Dict, Any

# Import the algorithms to compare
from src.algorithms.bernoulli_sequential import cluster_bernoulli

# Import the configuration and utilities
from src.configs.config_4 import config_4
from src.utils.clustering import clusters_equal
from src.utils.data_generation import generate_data_matrix_for_bernoulli, generate_clusters

def simulation_iteration_4_cluster(seed: int) -> Dict[str, Any]:
    """
    Run one iteration of Experiment 4.

    Compares adaptive_clustering against cluster_bernoulli algorithm.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing results (errors, budgets, runtimes) for each algorithm.
    """
    # Set random seed for this iteration
    np.random.seed(seed)

    # --- Data Generation ---
    # Use the pre-defined true sigma from config
    true_sigma = generate_clusters(config_4["num_items"], config_4["theta"] )
    num_items = config_4["num_items"]
    signal_strength = config_4["signal_strength"]
    feature_grid = config_4["feature_grid"]
    feature_grid_size = len(feature_grid)
    sparsity_grid = config_4["sparsity_grid"]
    cluster_errors = np.zeros(feature_grid_size)
    cluster_budgets = np.zeros(feature_grid_size)
    for i in range(feature_grid_size):
        num_features = feature_grid[i]
        sparsity = sparsity_grid[i]
        spars_indices = np.random.choice(num_features, sparsity, replace=False)
        p = np.full((2, num_features), 0.5)
        p[0, spars_indices] = 0.5 + signal_strength/2
        p[1, spars_indices] = 0.5 - signal_strength/2
        true_sigma = generate_clusters(num_items, 0.5) 
        data_matrix = np.zeros((num_items, num_features))
        for j in range(num_items):
            if true_sigma[j] == 1:
                data_matrix[j,:] = p[0,:]
            else:
                data_matrix[j,:] = p[1,:]
        result = cluster_bernoulli(data_matrix, config_4["delta"])
        cluster_errors[i] = 1 - clusters_equal(result[0], true_sigma)
        cluster_budgets[i] = result[1]
    return cluster_errors, cluster_budgets


