"""
Script to run all experiments and save results.
"""
import argparse
from pathlib import Path
from typing import List, Optional

from src.experiments.experiment_1 import Experiment1 # type: ignore
from src.experiments.experiment_2 import Experiment2 # type: ignore
from src.experiments.experiment_3 import Experiment3 # type: ignore
from src.config.experiment_config import get_config

def run_experiments(experiment_names: Optional[List[str]] = None,
                   output_dir: str = "results",
                   num_processes: Optional[int] = None) -> None:
    """
    Run specified experiments and save results.
    
    Args:
        experiment_names: List of experiment names to run (None for all)
        output_dir: Directory to save results
        num_processes: Number of processes to use for parallelization
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get experiments to run
    experiments = {
        "experiment_1": (Experiment1, "Varying sparsity with constant norm"),
        "experiment_2": (Experiment2, "Varying dimensions and delta"),
        "experiment_3": (Experiment3, "Varying proportion with constant gap")
    }
    
    if experiment_names is None:
        experiment_names = list(experiments.keys())
    
    # Run each experiment
    for name in experiment_names:
        if name not in experiments:
            print(f"Warning: Unknown experiment {name}, skipping")
            continue
            
        print(f"\nRunning {name}: {experiments[name][1]}")
        
        # Get experiment class and config
        experiment_class, _ = experiments[name]
        config = get_config(name)
        
        # Run experiment
        experiment = experiment_class(config)
        results = experiment.run(num_processes=num_processes)
        
        # Save results
        experiment.save_results(str(output_path / name))
        print(f"Results saved to {output_path / name}")

def main():
    """Parse command line arguments and run experiments"""
    parser = argparse.ArgumentParser(description="Run clustering experiments")
    parser.add_argument("--experiments", nargs="+", help="Experiments to run (default: all)")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--num-processes", type=int, help="Number of processes to use")
    
    args = parser.parse_args()
    run_experiments(args.experiments, args.output_dir, args.num_processes)

if __name__ == "__main__":
    main() 