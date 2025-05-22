"""
Runner for Experiment 2: Varying signal strength.
"""

import multiprocessing as mp
from src.runners.experiment_utils import setup_environment, save_results, run_parallel_experiment
from src.experiments.experiment_2 import simulation_iteration_2
from src.configs.config_2 import config_2

# Set up multiprocessing method
mp.set_start_method('spawn', force=True)

# Set up environment variables
setup_environment()

if __name__ == "__main__":
    print(f"Running Experiment 2: Varying signal strength with {config_2['monte_carlo_runs']} Monte Carlo runs")
    
    # Run the experiment in parallel
    results = run_parallel_experiment(
        simulation_iteration_2, 
        config_2["monte_carlo_runs"]
    )
    
    # Save the results
    save_results(results, "results_2")
    
    print("Experiment 2 completed successfully")