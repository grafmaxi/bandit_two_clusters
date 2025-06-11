"""Configuration parameters for Experiment 2: constant norm and sparsity, gaussian noise."""

import numpy as np

# Experiment 2 parameters
config_2 = {
    # Monte Carlo runs
    "monte_carlo_runs": 500,
    # Matrix dimensions
    # "num_items_array": np.array([100, 200, 500, 1000, 2000, 5000]),
    "num_items_array": np.array([10, 20, 50, 100, 200, 500]),
    "multiplier": 10,
    "num_features_array": None,  # Will be calculated as multiplier * num_items_array

    # Confidence parameters
    "delta_array": np.array([0.8, 0.5, 0.2, 0.05]),

    # Array lengths
    "num_items_count": None,  # Will be calculated as len(num_items_array)
    "delta_count": None,      # Will be calculated as len(delta_array)

    # Cluster settings
    "theta": 0.5,
    "sparsity": 10,
    "signal_strength": 5
}

# Calculate derived parameters
config_2["num_features_array"] = config_2["multiplier"] * \
    config_2["num_items_array"]
config_2["num_items_count"] = len(config_2["num_items_array"])
config_2["delta_count"] = len(config_2["delta_array"])
