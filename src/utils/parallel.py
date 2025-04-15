"""
Utilities for parallel processing.
"""
import multiprocessing as mp
from typing import Callable, List, Any, Optional

def run_parallel(func: Callable, 
                args_list: List[Any], 
                num_processes: Optional[int] = None) -> List[Any]:
    """
    Run a function in parallel using multiple processes.
    
    Args:
        func: Function to run in parallel
        args_list: List of arguments to pass to the function
        num_processes: Number of processes to use (default: min(cpu_count, 16))
        
    Returns:
        List of results from each function call
    """
    num_processes = num_processes or min(mp.cpu_count(), 16)
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(func, args_list)
        
    return results 