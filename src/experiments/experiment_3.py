"""Experiment 3: constant gap, varying proportion."""


import numpy as np
from typing import Tuple, Dict, Any
from src.algorithms.gaussian import cr_gaussian, cbc_gaussian, cluster_gaussian
from src.algorithms.kmeans import kmeans_budget
from src.utils.clustering import clusters_equal, get_cluster_representatives
from src.utils.data_generation import generate_clusters, generate_data_matrix
from src.configs.config_3 import config_3

def simulation_iteration_3(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                              np.ndarray, np.ndarray, np.ndarray,
                                              np.ndarray]:
    """
    Run one iteration of Experiment 3.
    
    This experiment tests all algorithms with:
    - Fixed matrix dimensions
    - Varying cluster proportions (theta)
    - Fixed signal strength and sparsity
    
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
    """
    # Set random seed
    print(seed)
    np.random.seed(seed)
    
    # Initialize output arrays
    theta_size = config_3["theta_size"]
    budget_steps = config_3["budget_steps"]
    
    # Initialize arrays for CR, CBC, and cluster algorithms
    cr_budgets = np.zeros(theta_size)
    cr_errors = np.zeros(theta_size)
    cbc_budgets = np.zeros(theta_size)
    cbc_errors = np.zeros(theta_size)
    cluster_budgets = np.zeros(theta_size)
    cluster_errors = np.zeros(theta_size)
    
    # Initialize array for K-means algorithm
    km_errors = np.zeros([theta_size, budget_steps])
    
    # Iterate over different theta values
    for theta_idx in range(theta_size):
        # Get theta proportion directly from config
        theta = config_3["theta_array"][theta_idx]
        
        # Generate true clusters and data matrix
        true_clusters = generate_clusters(config_3["num_items"], theta)
        M = generate_data_matrix(
            config_3["num_items"],
            config_3["num_features"],
            config_3["sparsity"],
            config_3["signal_strength"],
            true_clusters
        )
        # Get cluster representatives
        i_a, i_b = get_cluster_representatives(true_clusters)
        # Perform CR algorithm
        cr_arms, cr_budget = cr_gaussian(M, 0.4)  # delta/2 = 0.8/2
        cr_budgets[theta_idx] = cr_budget
        cr_errors[theta_idx] = 1 - int(true_clusters[0] == true_clusters[cr_arms[0]])
        
        # Perform CBC algorithm
        if true_clusters[0] == 0:
            cbc_clusters, cbc_budget = cbc_gaussian(M, i_a, 0.4)
        else:
            cbc_clusters, cbc_budget = cbc_gaussian(M, i_b, 0.4)
        cbc_budgets[theta_idx] = cbc_budget
        cbc_errors[theta_idx] = 1 - int(clusters_equal(true_clusters, cbc_clusters))
        
        # Perform cluster algorithm
        cluster_clusters, cluster_budget = cluster_gaussian(M, 0.8)
        cluster_budgets[theta_idx] = cluster_budget
        cluster_errors[theta_idx] = 1 - int(clusters_equal(true_clusters, cluster_clusters))
        
        # Perform K-means algorithm for different budgets
        for budget_idx in range(budget_steps):
            budget = int(config_3["min_budget"] + budget_idx * (config_3["max_budget"] - config_3["min_budget"]) / (budget_steps - 1))
            km_clusters = kmeans_budget(M, budget)
            km_errors[theta_idx, budget_idx] = 1 - int(clusters_equal(true_clusters, km_clusters))
    
    return (cr_errors, cr_budgets, cbc_errors, cbc_budgets, 
            cluster_errors, cluster_budgets, km_errors)

