"""Configuration parameters for Experiment 4: Algorithm Comparison."""

import numpy as np

# Experiment 4 parameters
config_4 = {
    # Monte Carlo settings
    "monte_carlo_runs": 1, # Adjust as needed

    # Adaptive Sampling Algorithm Parameters (extracted from adaptive_sampling.py)
    "n": 1000,  # number of users
    "K": 2,  # number of clusters
    "L": 5,  # number of questions
    "T": 1000000, # Total budget/time steps
    "p_norm": np.inf, # Norm for normalizing r vectors
    "w": 1,  # Parameter for min_w and do_uniform (context specific)
    "ucb_constant": 0.1,  # UCB constant for weight updates
    "num_rank": 2, # Number of ranks for h_rank calculation

    # Example cluster probabilities (adjust as needed for the experiment)
    "p": np.array([
        [0.7, 0.3, 0.3, 0.3, 0.3],  # cluster 1
        [0.2, 0.3, 0.3, 0.3, 0.3]   # cluster 2
    ]),

    # Parameters for *your* modified algorithms (from src/utils/adaptive.py?)
    # Add any specific parameters needed for your algorithms here
    "your_algo_param_1": "value1",
    "your_algo_param_2": 123,

    # Parameters for data generation (if needed, adapt from other configs)
    "data_gen_param": "example"

    # You might want to pre-generate true sigma like in config_1 if it's fixed
    # "true_sigma": np.random.randint(0, K, size=n)
}

# Ensure K matches the shape of p
assert config_4['p'].shape[0] == config_4['K'], "Number of rows in 'p' must match 'K'"
assert config_4['p'].shape[1] == config_4['L'], "Number of columns in 'p' must match 'L'"

# Example: Pre-generate true sigma if needed for consistency across runs
# This assumes K and n are defined within the config_4 dict itself
# np.random.seed(42) # Optional: Seed for deterministic sigma generation
# config_4["true_sigma"] = np.random.randint(0, config_4['K'], size=config_4['n']) 