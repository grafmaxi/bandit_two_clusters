"""
Utilities package for the experiments.
"""
from .parallel import run_parallel
from .visualization import (
    plot_error_vs_sparsity,
    plot_budget_comparison,
    plot_kmeans_performance
)

__all__ = [
    'run_parallel',
    'plot_error_vs_sparsity',
    'plot_budget_comparison',
    'plot_kmeans_performance'
] 