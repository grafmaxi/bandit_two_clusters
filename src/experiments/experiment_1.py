"""Experiment 1: constant norm, varying sparsity, gaussian noise."""

import numpy as np
from typing import Tuple, Dict, Any
from src.algorithms.gaussian import cr_gaussian, cbc_gaussian, cluster_gaussian
from src.algorithms.kmeans import kmeans_budget
from src.utils.clustering import clusters_equal, get_cluster_representatives
from src.utils.data_generation import generate_clusters, generate_data_matrix
from src.configs.config_1 import config_1

def simulation_iteration_1(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                              np.ndarray, np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:
    """
    Run one iteration of Experiment 1.
    
    This experiment tests all algorithms with:
    - Fixed matrix dimensions
    - Varying sparsity levels
    - Fixed signal strength
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - cr_errors: Error rates for CR algorithm
        - cr_budgets: Budgets used by CR algorithm
        - cbc_errors: Error rates for CBC algorithm
        - cbc_budgets: Budgets used by CBC algorithm
        - cluster_errors: Error rates for cluster algorithm
        - cluster_budgets: Budgets used by cluster algorithm
        - km_errors: Error rates for K-means algorithm
        - km_budgets: Budgets used by K-means algorithm
    """
    # Set random seed
    np.random.seed(seed)
    print(f"Running Experiment 1 with seed {seed}")
    # Initialize output arrays
    sparsity_grid_size = config_1["num_sparsity_params"]
    budget_steps = config_1["num_budget_steps"]
    
    # Initialize arrays for CR, CBC, and cluster algorithms
    cr_budgets = np.zeros(sparsity_grid_size)
    cr_errors = np.zeros(sparsity_grid_size)
    cbc_budgets = np.zeros(sparsity_grid_size)
    cbc_errors = np.zeros(sparsity_grid_size)
    cluster_budgets = np.zeros(sparsity_grid_size)
    cluster_errors = np.zeros(sparsity_grid_size)
    
    # Initialize arrays for K-means algorithm
    km_budgets = np.zeros([sparsity_grid_size, budget_steps])
    km_errors = np.zeros([sparsity_grid_size, budget_steps])
    
    # Generate true clusters
    true_clusters = generate_clusters(config_1["num_items"], config_1["theta"])
    
    # Get cluster representatives
    i_a, i_b = get_cluster_representatives(true_clusters)
    
    # Iterate over sparsity grid
    for s_idx in range(sparsity_grid_size):
        # Calculate sparsity
        sparsity = int((config_1["num_features"] - 1) * s_idx / (sparsity_grid_size - 1) + 1)
        
        # Generate data matrix
        M = generate_data_matrix(
            config_1["num_items"],
            config_1["num_features"],
            sparsity,
            config_1["norm"]/np.sqrt(sparsity),
            true_clusters
        )
        
        # Perform CR algorithm
        cr_arms, cr_budget = cr_gaussian(M, 0.4)  # delta/2 = 0.8/2
        cr_budgets[s_idx] = cr_budget
        cr_errors[s_idx] = int(true_clusters[0] == true_clusters[cr_arms[0]])
        
        # Perform CBC algorithm
        if true_clusters[0] == 0:
            cbc_clusters, cbc_budget = cbc_gaussian(M, i_a, 0.4)
        else:
            cbc_clusters, cbc_budget = cbc_gaussian(M, i_b, 0.4)
        cbc_budgets[s_idx] = cbc_budget
        cbc_errors[s_idx] = 1 - int(clusters_equal(true_clusters, cbc_clusters))
        
        # Perform cluster algorithm
        cluster_clusters, cluster_budget = cluster_gaussian(M, 0.8)
        cluster_budgets[s_idx] = cluster_budget
        cluster_errors[s_idx] = 1 - int(clusters_equal(true_clusters, cluster_clusters))
        
        # Perform K-means algorithm for different budgets
        for budget_idx in range(budget_steps):
            budget = int(config_1["min_budget"] + budget_idx * (config_1["max_budget"] - config_1["min_budget"]) / (budget_steps - 1))
            km_clusters = kmeans_budget(M, budget)
            km_errors[s_idx, budget_idx] = 1 - int(clusters_equal(true_clusters, km_clusters))
            km_budgets[s_idx, budget_idx] = budget
    
    return (cr_errors, cr_budgets, cbc_errors, cbc_budgets, 
            cluster_errors, cluster_budgets, km_errors, km_budgets)

