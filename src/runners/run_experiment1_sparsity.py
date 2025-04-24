"""
Runner for Experiment 1: Constant norm, varying sparsity.
"""

import multiprocessing as mp
from src.runners.experiment_utils import setup_environment, save_results, run_parallel_experiment
from src.experiments.experiment_1 import simulation_iteration_1
from src.configs.config_1 import config_1

# Set up multiprocessing method
mp.set_start_method('spawn', force=True)

# Set up environment variables
setup_environment()

if __name__ == "__main__":
    print(f"Running Experiment 1: Varying sparsity with {config_1['monte_carlo_runs']} Monte Carlo runs")
    
    # Run the experiment in parallel
    results = run_parallel_experiment(
        simulation_iteration_1, 
        config_1["monte_carlo_runs"]
    )
    
    # Save the results
    save_results(results, "results_1", "sparsity_experiment")
    
    print("Experiment 1 completed successfully")