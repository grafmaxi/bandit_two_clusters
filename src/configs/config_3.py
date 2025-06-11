"""Configuration parameters for Experiment 3: constant gap, varying proportion."""

import numpy as np

# Experiment 3 parameters
config_3 = {
    # Monte Carlo settings
    "monte_carlo_runs": 500,

    # Matrix dimensions
    "num_items": 1000,
    "num_features": 1000,

    # Proportion parameters
    "theta_size": None,  # Will be calculated as int(log2(num_items))
    "theta_array": None,  # Will be calculated as powers of 2

    # Signal and noise settings
    "signal_strength": 1.5,
    "sparsity": 100,

    # Budget settings
    "min_budget": 10**6,
    "max_budget": 10**10,
    "budget_steps": 20,
    "budget_grid": None,  # Will be calculated as powers of 2
}

# Calculate derived parameters
config_3["theta_size"] = int(np.log2(config_3["num_items"]))
config_3["theta_array"] = np.array(
    [(2**i) / config_3["num_items"] for i in range(config_3["theta_size"])], dtype=float)
config_3["budget_grid"] = np.array([(2**i -
                                     1) /
                                    (2**config_3["budget_steps"] -
                                     1) *
                                    (config_3["max_budget"] -
                                     config_3["min_budget"]) +
                                    config_3["min_budget"] for i in range(config_3["budget_steps"])], dtype=float)
