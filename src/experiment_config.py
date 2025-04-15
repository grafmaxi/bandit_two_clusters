import numpy as np

experiment_1_config = {
    "num_items": 20,  # Number of rows
    "num_features": 1000,  # Number of columns
    "spars_grid_size": 20,  # Number of sparsity levels
    "time_grid_size": 10,  # Time grid for KMeans
    "t_min": 170000,  # Minimum time for KMeans
    "t_max": 300000,  # Maximum time for KMeans
    "theta": 0.5,  # Proportion of items in one cluster
    "average_signal_strength": 15,  # Norm of Delta
    "delta": 0.8,  # Confidence parameter
    "monte_carlo_runs": 32,  # Number of Monte Carlo runs
}

# Experiment 2: Constant norm and sparsity, varying dimensions and delta
experiment_2_config = {
    "num_items_array": [100, 200, 500, 1000, 2000, 5000],  # Array of n values
    "multip": 10,  # Multiplier for num_features (num_features = multip * num_items)
    "delta_array": [0.8, 0.5, 0.2, 0.05],  # Array of delta values
    "sparsity": 10,  # Fixed sparsity
    "signal_strength": 5,  # Signal of non-zero features
    "theta": 0.5,  # Proportion of items in one cluster
    "monte_carlo_runs": 32,  # Number of Monte Carlo runs
}

# Experiment 3: Constant gap, varying proportion
experiment_3_config = {
    "num_items": 1000,  # Number of rows
    "num_features": 1000,  # Number of columns
    "theta_size": None,  # Size of theta_vec (calculated dynamically)
    "theta_vec": None,  # Vector of theta values (calculated dynamically)
    "sparsity": 100,  # Sparsity
    "signal_strength": 1.5,  # Signal strength
    "t_min": 1 * 10**7,  # Minimum budget for KMeans
    "t_max": 100 * 10**7,  # Maximum budget for KMeans
    "time_grid_size": 100,  # Time grid size for KMeans
    "delta": 0.8,  # Confidence parameter
    "monte_carlo_runs": 32,  # Number of Monte Carlo runs (K)
}

# Dynamically calculate theta_sz and theta_vec based on num_items
num_items_3 = experiment_3_config["num_items"]
theta_size = int(np.log(num_items_3) / np.log(2))
theta_vec = np.zeros(theta_size)
for i in range(theta_size):
    theta_vec[i] = 2**i
experiment_3_config["theta_size"] = theta_size
experiment_3_config["theta_vec"] = theta_vec.astype(int)