"""
Runner for Experiment 3: Constant gap, varying proportion.
"""

import multiprocessing as mp
from src.runners.experiment_utils import setup_environment, save_results, run_parallel_experiment
from src.experiments.experiment_3 import simulation_iteration_3
from src.configs.config_3 import config_3

# Set up multiprocessing method
mp.set_start_method('spawn', force=True)

# Set up environment variables
setup_environment()

if __name__ == "__main__":
    print(f"Running Experiment 3: Varying proportion with {config_3['monte_carlo_runs']} Monte Carlo runs")
    
    # Run the experiment in parallel
    results = run_parallel_experiment(
        simulation_iteration_3, 
        config_3["monte_carlo_runs"]
    )
    
    # Save the results
    save_results(results, "results_3")
    
    print("Experiment 3 completed successfully")