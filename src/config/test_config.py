"""
Test configuration with reduced parameters for quick testing.
"""
from .experiment_config import (
    BaseConfig,
    Experiment1Config,
    Experiment2Config,
    Experiment3Config
)

class TestBaseConfig(BaseConfig):
    """Test base configuration with reduced Monte Carlo runs"""
    monte_carlo_runs: int = 1  # Reduced from 32 to 8

class TestExperiment1Config(Experiment1Config):
    """Test configuration for Experiment 1"""
    spars_grid_size: int = 5  # Reduced from 20 to 5
    time_grid_size: int = 5   # Reduced from 10 to 5

class TestExperiment2Config(Experiment2Config):
    """Test configuration for Experiment 2"""
    num_items_array: list = [100, 200, 500]  # Reduced from [100, 200, 500, 1000, 2000, 5000]
    average_signal_strength: float = 5  # Signal of non-zero features

class TestExperiment3Config(Experiment3Config):
    """Test configuration for Experiment 3"""
    theta_size: int = 5  # Reduced from log(n)/log(2)
    average_signal_strength: float = 1.5  # Signal strength

# Test configurations for each experiment
test_experiment_1_config = TestExperiment1Config()
test_experiment_2_config = TestExperiment2Config()
test_experiment_3_config = TestExperiment3Config()

def get_test_config(experiment_name: str):
    """Get test configuration for a specific experiment"""
    configs = {
        "experiment_1": test_experiment_1_config,
        "experiment_2": test_experiment_2_config,
        "experiment_3": test_experiment_3_config
    }
    return configs.get(experiment_name) 