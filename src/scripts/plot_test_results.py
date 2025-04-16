"""
Script to plot test experiment results.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt

# Get the absolute path to the project root
project_root = Path(__file__).parent.parent.parent
os.chdir(project_root)  # Change to project root directory
sys.path.insert(0, str(project_root))  # Add project root to Python path

from src.utils.visualization import plot_experiment_1, plot_experiment_2, plot_experiment_3

def load_results(results_dir: str) -> Dict[str, np.ndarray]:
    """
    Load results from CSV files in the results directory.
    
    Args:
        results_dir: Directory containing result files
        
    Returns:
        Dictionary containing loaded results
    """
    results = {}
    results_path = Path(results_dir)
    
    # Load each CSV file
    for file_path in results_path.glob("*.csv"):
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        results[file_path.stem] = data
    
    # Load time spent
    time_file = results_path / "time_spent.txt"
    if time_file.exists():
        with open(time_file, 'r') as f:
            results['time_spent'] = float(f.read().split(': ')[1])
    
    return results

def plot_test_results(results_dir: str = "test_results",
                     output_dir: str = "test_plots") -> None:
    """
    Generate plots from test experiment results.
    
    Args:
        results_dir: Directory containing test results
        output_dir: Directory to save plots
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get experiments to plot
    experiments = {
        "experiment_1": (plot_experiment_1, "sparsity_vs_budget.png"),
        "experiment_2": (plot_experiment_2, "dimensions_vs_budget.png"),
        "experiment_3": (plot_experiment_3, "proportion_vs_budget.png")
    }
    
    # Plot each experiment
    for name, (plot_func, output_file) in experiments.items():
        results_path = Path(results_dir) / name
        if not results_path.exists():
            print(f"Warning: No results found for {name}, skipping")
            continue
            
        print(f"\nPlotting {name}")
        
        # Load results
        results = load_results(str(results_path))
        
        # Generate plot
        plot_func(results, save_path=str(output_path / output_file))
        print(f"Plot saved to {output_path / output_file}")

if __name__ == "__main__":
    plot_test_results() 