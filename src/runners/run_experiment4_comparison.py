"""
Runner for Experiment 4: Algorithm Comparison.
"""

import multiprocessing as mp
import time

from src.runners.experiment_utils import setup_environment, save_results, run_parallel_experiment
from src.experiments.experiment_4 import simulation_iteration_4
from src.configs.config_4 import config_4

# Set up multiprocessing method
# Consider using 'fork' on Linux/macOS if 'spawn' causes issues with imports/globals,
# but 'spawn' is generally safer and cross-platform.
mp.set_start_method('fork', force=True)

# Set up environment variables for numerical libraries
setup_environment()

if __name__ == "__main__":
    start_time = time.time()
    mc_runs = config_4['monte_carlo_runs']
    print(f"Running Experiment 4: Algorithm Comparison with {mc_runs} Monte Carlo runs")

    # Run the experiment simulation iterations in parallel
    results_list = run_parallel_experiment(
        simulation_iteration_4,
        num_runs=mc_runs
        # Optionally pass num_processes here if you don't want the default min(cpu_count, 4)
        # num_processes=config_4.get('num_processes', None)
    )

    # Save the collected results
    # The results_list contains dictionaries from each simulation_iteration_4 run
    save_results(results_list, "results_4")

    end_time = time.time()
    print(f"Experiment 4 completed successfully in {end_time - start_time:.2f} seconds") 