"""
Experiment 4: Comparison of Adaptive Sampling and Bernoulli Sequential Algorithms.
"""

import numpy as np
from typing import Dict, Any

# Import the algorithms to compare
from src.algorithms.bernoulli_sequential import cluster_bernoulli

# Import the configuration and utilities
from src.configs.config_4 import config_4
from src.utils.clustering import clusters_equal
from src.utils.data_generation import generate_clusters


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

    # Initialize experiment parameters from config
    num_items = config_4["num_items"]
    signal_strength = config_4["signal_strength"]
    feature_grid = config_4["feature_grid"]
    feature_grid_size = len(feature_grid)
    sparsity_grid = config_4["sparsity_grid"]

    # Initialize result arrays
    cluster_errors = np.zeros(feature_grid_size)
    sample_costs = np.zeros(feature_grid_size)

    # Generate true cluster assignments
    true_clusters = generate_clusters(num_items, config_4["theta"])

    # Run experiment for each feature dimension
    for i in range(feature_grid_size):
        num_features = feature_grid[i]
        sparsity = sparsity_grid[i]

        # Generate sparse feature indices
        spars_indices = np.random.choice(num_features, sparsity, replace=False)

        # Set up probability matrix for Bernoulli trials
        means_matrix = np.full((2, num_features), 0.5)
        means_matrix[0, spars_indices] = 0.5 + signal_strength / 2
        means_matrix[1, spars_indices] = 0.5 - signal_strength / 2

        # Generate data matrix based on cluster assignments
        data_matrix = np.zeros((num_items, num_features))
        for j in range(num_items):
            data_matrix[j, :] = means_matrix[1 if true_clusters[j]
                                             == 1 else 0, :]

        # Run clustering algorithm
        predicted_clusters, sample_cost = cluster_bernoulli(
            data_matrix, config_4["delta"])

        # Record results
        cluster_errors[i] = 0 if predicted_clusters is None else 1 - \
            clusters_equal(predicted_clusters, true_clusters)
        sample_costs[i] = sample_cost

    return cluster_errors, sample_costs
