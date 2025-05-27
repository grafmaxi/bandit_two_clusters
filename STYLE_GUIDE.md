# Bandit Two Clusters - Code Style Guide

## Overview
This document defines the coding standards and conventions for the Bandit Two Clusters repository.

## Variable Naming Conventions

### Core Data Structures
- `means_matrix`: Always use this for the main input matrix (num_items × num_features)
- `item_indices`: List or array of item indices to process
- `feature_indices`: List or array of feature indices to process  
- `sample_cost`: Total number of samples used by an algorithm
- `true_clusters`: Ground truth cluster assignments
- `predicted_clusters`: Algorithm-predicted cluster assignments

### Algorithm Parameters
- `delta`: Risk/confidence parameter (consistent across all algorithms)
- `budget`: Maximum number of samples allowed
- `num_items`: Number of rows in data matrix
- `num_features`: Number of columns in data matrix
- `num_clusters`: Number of clusters (typically 2)

### Configuration Parameters
- `monte_carlo_runs`: Number of simulation repetitions
- `signal_strength`: Strength of the clustering signal
- `sparsity`: Number of non-zero features

## Function Naming Conventions

### Algorithm Functions
- `cesh_*`: Confidence-based Elimination with Sequential Halving
- `cr_*`: Candidate Row identification
- `cbc_*`: Clustering given Best Candidate
- `cluster_*`: Full clustering algorithm (cr + cbc)

### Utility Functions  
- `generate_*`: Data generation functions
- `clusters_equal`: Compare two clustering assignments
- `simulation_iteration_*`: Single experiment iteration

### File Naming
- Algorithm files: `algorithm_name.py` (e.g., `gaussian.py`, `bernoulli.py`)
- Config files: `config_N.py` where N is experiment number
- Experiment files: `experiment_N.py` where N is experiment number
- Analysis files: `results_analysis_N.py` where N is experiment number

## Documentation Standards

### Module Docstrings
```python
"""
Brief one-line description.

Detailed description of the module's purpose and contents.
Explain the main algorithms or functionality provided.
"""
```

### Function Docstrings
```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of first parameter.
        param2: Description of second parameter.
        
    Returns:
        Description of return value.
        
    Raises:
        ExceptionType: When this exception is raised.
    """
```

## Code Organization

### Import Order
1. Standard library imports
2. Third-party imports (numpy, matplotlib, etc.)
3. Local imports (src.*)

### Error Handling
- Use appropriate exception types
- Provide meaningful error messages
- Validate input parameters where necessary

### Constants
- Define magic numbers as named constants at module level
- Use ALL_CAPS for constants

## Type Hints
- Always include type hints for function parameters and return values
- Use `from typing import` for complex types
- Remove unnecessary `# type: ignore` comments

## Comments
- Use inline comments sparingly, only for complex logic
- Prefer self-documenting code with clear variable names
- Remove commented-out code unless there's a specific reason to keep it 