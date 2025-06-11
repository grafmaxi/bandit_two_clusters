"""Configuration parameters for Experiment 4: Algorithm Comparison."""

import numpy as np

# Define key parameters first
feature_grid_size = 5
# Experiment 4 parameters
config_4 = {
    # Monte Carlo settings
    "monte_carlo_runs": 500,  # Number of simulation runs

    # Common Parameters
    "num_items": 30,  # Number of items (users)
    # Number of features (questions/tasks)
    "feature_grid": [2 * 2**i for i in range(feature_grid_size)],
    "sparsity_grid": [2**i for i in range(feature_grid_size)],
    'theta': 0.5,
    'signal_strength': 0.5,

    "delta": 0.8,  # Risk level for cluster_bernoulli
}
