"""Experiment 4: Comparison of Adaptive Sampling and Custom Algorithms."""

import numpy as np
from typing import Tuple, Dict, Any

# Import the algorithms to compare
from src.algorithms.adaptive_sampling import adaptive_clustering
# Import your modified algorithm(s) - adjust path/names as necessary
# from src.utils.adaptive import your_modified_algorithm_1, your_modified_algorithm_2

# Import the configuration
from src.configs.config_4 import config_4

def simulation_iteration_4(seed: int) -> Dict[str, Any]:
    """
    Run one iteration of Experiment 4.

    Compares adaptive_clustering against custom modified algorithms.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing results (e.g., errors, budgets) for each algorithm.
    """
    # Set random seed for this iteration
    np.random.seed(seed)
    print(f"Running iteration with seed: {seed}")

    # --- Data Generation ---
    # Generate the true underlying state (sigma) for this iteration.
    # If sigma should be fixed across runs, generate it once in config_4.
    # Otherwise, generate it here based on parameters in config_4.
    if "true_sigma" in config_4:
        true_sigma = config_4["true_sigma"]
        print("Using pre-defined true_sigma from config")
    else:
        print("Generating new true_sigma for this iteration")
        true_sigma = np.random.randint(0, config_4['K'], size=config_4['n'])

    # --- Run Adaptive Clustering Algorithm ---
    print("Running adaptive_clustering...")
    try:
        # Pass only the necessary parts of the config if the function expects that
        # Or pass the whole config if the function handles extraction
        # The refactored function returns: final_hat_sigma, final_error_rate, N_tot (budget)
        adapt_samp_hat_sigma, adapt_samp_error, adapt_samp_budget_matrix = adaptive_clustering(true_sigma, config_4)
        # Aggregate budget if needed (e.g., total pulls)
        adapt_samp_total_budget = np.sum(adapt_samp_budget_matrix)
        print(f"Adaptive Clustering finished. Error: {adapt_samp_error:.4f}, Budget: {adapt_samp_total_budget}")
    except Exception as e:
        print(f"ERROR running adaptive_clustering: {e}")
        adapt_samp_error = np.nan
        adapt_samp_total_budget = np.nan
        # Handle error case appropriately - maybe return NaNs or default error/budget

    # --- Collect Results ---
    results = {
        "seed": seed,
        "adaptive_sampling_error": adapt_samp_error,
        "adaptive_sampling_budget": adapt_samp_total_budget,
        # Add results for other custom algorithms if comparing more than one
    }

    return results 