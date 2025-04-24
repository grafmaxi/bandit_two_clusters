"""
Utility functions for running experiments and saving results.
"""

import os
import pickle
import multiprocessing as mp
import numpy as np

def setup_environment():
    """Set environment variables to control thread behavior in numerical libraries."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

def save_results(results, filename, subfolder=None):
    """
    Save experiment results as a pickle file.
    
    Args:
        results: Results from the experiment
        filename: Name of the pickle file (without extension)
        subfolder: Optional subfolder within results directory
    """
    # Create results directory within src folder
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(current_dir, "results")
    
    # Create subfolder if specified
    if subfolder:
        results_dir = os.path.join(results_dir, subfolder)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results as pickle file
    filepath = os.path.join(results_dir, f"{filename}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {filepath}")

def run_parallel_experiment(experiment_function, num_runs, num_processes=None):
    """
    Run an experiment function in parallel using multiprocessing.
    
    Args:
        experiment_function: Function that runs a single experiment iteration
        num_runs: Number of Monte Carlo runs
        num_processes: Number of processes to use (defaults to min(CPU count, 4))
        
    Returns:
        List of results from all experiment runs
    """
    # Use min(CPU count, 4) processes by default
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 4)
    
    # Set up process pool and map experiment function to seeds
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(experiment_function, range(num_runs))
    
    return results