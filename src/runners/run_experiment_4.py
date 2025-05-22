"""
Runner for Experiment 4: Algorithm Comparison.
"""

import multiprocessing as mp
import time
import os

from src.runners.experiment_utils import setup_environment, save_results, run_parallel_experiment
from src.experiments.experiment_4 import simulation_iteration_4_cluster
from src.configs.config_4 import config_4

def run_experiment4():
    """Main function to run experiment 4"""
    start_time = time.time()
    mc_runs = config_4['monte_carlo_runs']
    
    # Create results directory if it doesn't exist
    os.makedirs("src/results", exist_ok=True)

    # Run the experiment simulation iterations in parallel
    results_list = run_parallel_experiment(
        simulation_iteration_4_cluster,
        config_4["monte_carlo_runs"]
    )


    # Save the collected results
    save_results(results_list, "results_4")
    # save_results(adaptive_results_list, "results_4_adaptive")
    end_time = time.time()
    
    return results_list

if __name__ == "__main__":
    # Set up multiprocessing method - 'fork' is generally faster on Unix systems
    mp.set_start_method('spawn', force=True)
    
    # Set up environment variables for numerical libraries
    setup_environment()
    
    # Run the experiment
    run_experiment4() 