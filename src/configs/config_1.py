"""Configuration parameters for Experiment 1: constant norm, varying sparsity, gaussian noise."""

import numpy as np

# Experiment 1 parameters
config_1 = {
    # Monte Carlo settings
    "monte_carlo_runs": 500,

    # Matrix dimensions
    "num_items": 20,
    "num_features": 1000,

    # Sparsity parameters
    "num_sparsity_params": 20,

    # Budget settings
    "num_budget_steps": 10,
    "min_budget": 170000,
    "max_budget": 300000,

    # Cluster settings
    "theta": 0.5,
    "norm": 15
}

# Generate true clusters
true_clusters = np.random.permutation(
    1 * (np.array(range(config_1["num_items"])) < config_1["theta"] * config_1["num_items"])
)
config_1["true_clusters"] = true_clusters
config_1["i_a"] = np.argmax(true_clusters)
config_1["i_b"] = np.argmin(true_clusters)
