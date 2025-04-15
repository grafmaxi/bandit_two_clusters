from typing import Dict, Any, List
import numpy as np # type: ignore
import multiprocessing as mp
import time
from pathlib import Path
import csv

class BaseExperiment:
    """Base class for all experiments providing common functionality"""
    
    def __init__(self, config: Any):
        """
        Initialize experiment with configuration
        
        Args:
            config: Experiment configuration object
        """
        self.config = config
        self.results = {}
        
    def run(self, num_processes: int = None) -> Dict[str, Any]:
        """
        Run the experiment with parallel processing
        
        Args:
            num_processes: Number of processes to use (default: min(cpu_count, 16))
            
        Returns:
            Dictionary containing experiment results
        """
        # Set up parallel processing
        num_processes = num_processes or min(mp.cpu_count(), 16)
        seeds = range(self.config.monte_carlo_runs)
        
        # Run experiment
        start_time = time.time()
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(self.run_single_iteration, seeds)
        end_time = time.time()
        
        # Process results
        self.process_results(results)
        self.results['time_spent'] = end_time - start_time
        
        return self.results
        
    def run_single_iteration(self, seed: int) -> Dict[str, Any]:
        """
        Run a single Monte Carlo iteration
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing iteration results
        """
        raise NotImplementedError("Subclasses must implement run_single_iteration")
        
    def process_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Process and store results from all iterations
        
        Args:
            results: List of results from each iteration
        """
        raise NotImplementedError("Subclasses must implement process_results")
        
    def save_results(self, output_dir: str) -> None:
        """
        Save experiment results to CSV files
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each result array as a CSV file
        for name, data in self.results.items():
            if isinstance(data, np.ndarray):
                file_path = output_path / f"{name}.csv"
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Write header with indices
                    if data.ndim == 1:
                        writer.writerow([f"{i+1}" for i in range(len(data))])
                        writer.writerow(data)
                    else:
                        writer.writerow([f"{i+1}" for i in range(data.shape[1])])
                        writer.writerows(data)
                        
        # Save time spent
        time_file = output_path / "time_spent.txt"
        with open(time_file, 'w') as f:
            f.write(f"Time spent: {self.results['time_spent']:.2f} seconds") 