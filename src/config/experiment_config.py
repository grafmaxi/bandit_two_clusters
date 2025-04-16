from dataclasses import dataclass
from typing import List, Optional
import numpy as np # type: ignore

@dataclass
class BaseConfig:
    """Base configuration class with common parameters"""
    monte_carlo_runs: int = 8  # Number of Monte Carlo runs
    delta: float = 0.8  # Confidence parameter
    theta: float = 0.5  # Proportion of items in one cluster

@dataclass
class Experiment1Config(BaseConfig):
    """Configuration for Experiment 1: Varying sparsity with constant norm"""
    num_items: int = 20  # Number of rows
    num_features: int = 1000  # Number of columns
    spars_grid_size: int = 20  # Number of sparsity levels
    time_grid_size: int = 10  # Time grid for KMeans
    t_min: int = 170000  # Minimum time for KMeans
    t_max: int = 300000  # Maximum time for KMeans
    average_signal_strength: float = 15  # Norm of Delta

@dataclass
class Experiment2Config(BaseConfig):
    """Configuration for Experiment 2: Varying dimensions with constant norm and sparsity"""
    num_items_array: List[int] = None  # Array of num_items values
    multip: int = 10  # Multiplier for num_features
    delta_array: List[float] = None  # Array of delta values
    sparsity: int = 10  # Fixed sparsity
    average_signal_strength: float = 5  # Signal of non-zero features

    def __post_init__(self):
        if self.num_items_array is None:
            self.num_items_array = [100, 200, 500, 1000, 2000, 5000]
        if self.delta_array is None:
            self.delta_array = [0.8, 0.5, 0.2, 0.05]

@dataclass
class Experiment3Config(BaseConfig):
    """Configuration for Experiment 3: Varying proportion with constant gap"""
    num_items: int = 1000  # Number of rows
    num_features: int = 1000  # Number of columns
    sparsity: int = 100  # Sparsity
    average_signal_strength: float = 1.5  # Signal strength
    t_min: int = 1 * 10**7  # Minimum budget for KMeans
    t_max: int = 100 * 10**7  # Maximum budget for KMeans
    time_grid_size: int = 100  # Time grid size for KMeans
    theta_size: Optional[int] = None  # Size of theta_vec
    theta_vec: Optional[np.ndarray] = None  # Vector of theta values

    def __post_init__(self):
        # Calculate theta_size and theta_vec if not provided
        if self.theta_size is None:
            self.theta_size = int(np.log(self.num_items) / np.log(2))
        if self.theta_vec is None:
            self.theta_vec = np.array([2**i for i in range(self.theta_size)])

# Default configurations for each experiment
experiment_1_config = Experiment1Config()
experiment_2_config = Experiment2Config()
experiment_3_config = Experiment3Config()

def get_config(experiment_name: str):
    """Get configuration for a specific experiment"""
    configs = {
        "experiment_1": experiment_1_config,
        "experiment_2": experiment_2_config,
        "experiment_3": experiment_3_config
    }
    return configs.get(experiment_name) 