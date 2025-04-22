"""Script to conduct experiments."""

import multiprocessing as mp
import pickle
import os
from src.experiments.experiment_1 import simulation_iteration_1
from src.configs.config_1 import config_1
from src.experiments.experiment_2 import simulation_iteration_2
from src.configs.config_2 import config_2
from src.experiments.experiment_3 import simulation_iteration_3
from src.configs.config_3 import config_3

mp.set_start_method('spawn', force=True)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def save_results(results_1, results_2, results_3):
    """
    Save the three result lists as separate pickle files.
    
    Args:
        results_1: Results from simulation_iteration_1
        results_2: Results from simulation_iteration_2
        results_3: Results from simulation_iteration_3
    """
    # Create results directory within src folder
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save each result list as a separate pickle file
    with open(os.path.join(results_dir, "results_1.pkl"), 'wb') as f:
        pickle.dump(results_1, f)
    
    with open(os.path.join(results_dir, "results_2.pkl"), 'wb') as f:
        pickle.dump(results_2, f)
    
    with open(os.path.join(results_dir, "results_3.pkl"), 'wb') as f:
        pickle.dump(results_3, f)
    
    print(f"Results saved to {results_dir} as pickle files")

if __name__ == "__main__":
    num_processes = 4  # Use multiple kernels
    with mp.Pool(processes=num_processes) as pool:
        results_1 = pool.map(simulation_iteration_1, range(config_1["monte_carlo_runs"]))
        results_2 = pool.map(simulation_iteration_2, range(config_2["monte_carlo_runs"]))
        results_3 = pool.map(simulation_iteration_3, range(config_3["monte_carlo_runs"]))
    
    # Save results to pickle files
    save_results(results_1, results_2, results_3)

