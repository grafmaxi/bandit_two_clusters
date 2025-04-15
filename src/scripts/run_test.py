"""
Script to run a single iteration of experiment 3 for debugging.
"""

from src.experiments.experiment_3 import Experiment3
from src.config.test_config import get_test_config

def run_single_iteration(output_dir: str = "test_results") -> None:
    """
    Run a single iteration of experiment 3 for debugging.
    
    Args:
        output_dir: Directory to save results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nRunning single iteration of experiment 3")
    
    # Get test config and create experiment
    config = get_test_config("experiment_3")
    experiment = Experiment3(config)
    
    # Run a single iteration with seed 0
    result = experiment.run_single_iteration(seed=0)
    
    # Print results
    print("\nResults:")
    for key, value in result.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    run_single_iteration() 