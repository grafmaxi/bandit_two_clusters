"""
Script to generate plots from experiment results.
"""
import argparse
from pathlib import Path
from typing import List, Optional

from src.utils.visualization import (
    plot_experiment_1,
    plot_experiment_2,
    plot_experiment_3
)
import numpy as np # type: ignore

def load_results(results_dir: str) -> dict:
    """Load results from CSV files in the results directory"""
    results = {}
    results_path = Path(results_dir)
    
    # Load each CSV file
    for csv_file in results_path.glob("*.csv"):
        data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        results[csv_file.stem] = data
    
    return results

def plot_results(experiment_names: Optional[List[str]] = None,
                results_dir: str = "results",
                output_dir: str = "plots") -> None:
    """
    Generate plots from experiment results.
    
    Args:
        experiment_names: List of experiment names to plot (None for all)
        results_dir: Directory containing experiment results
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
    
    if experiment_names is None:
        experiment_names = list(experiments.keys())
    
    # Plot each experiment
    for name in experiment_names:
        if name not in experiments:
            print(f"Warning: Unknown experiment {name}, skipping")
            continue
            
        print(f"\nPlotting {name}")
        
        # Get plot function and output filename
        plot_func, output_file = experiments[name]
        
        # Load results
        results = load_results(str(Path(results_dir) / name))
        
        # Generate plot
        plot_func(results, save_path=str(output_path / output_file))
        print(f"Plot saved to {output_path / output_file}")

def main():
    """Parse command line arguments and generate plots"""
    parser = argparse.ArgumentParser(description="Generate plots from experiment results")
    parser.add_argument("--experiments", nargs="+", help="Experiments to plot (default: all)")
    parser.add_argument("--results-dir", default="results", help="Directory containing results")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    
    args = parser.parse_args()
    plot_results(args.experiments, args.results_dir, args.output_dir)

if __name__ == "__main__":
    main() 