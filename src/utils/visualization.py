"""
Utilities for visualizing experiment results.
"""
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
from typing import Dict, List, Optional, Tuple
from pathlib import Path

def plot_error_vs_sparsity(results: Dict[str, np.ndarray],
                          algorithms: List[str],
                          save_path: Optional[str] = None) -> None:
    """
    Plot error rates vs sparsity for different algorithms.
    
    Args:
        results: Dictionary containing results for each algorithm
        algorithms: List of algorithm names to plot
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    for algo in algorithms:
        errors = np.mean(results[f"{algo}_errors"], axis=1)
        sparsity_levels = np.arange(len(errors))
        plt.plot(sparsity_levels, errors, label=algo, marker='o')
        
    plt.xlabel("Sparsity Level")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_budget_comparison(results: Dict[str, np.ndarray],
                         algorithms: List[str],
                         save_path: Optional[str] = None) -> None:
    """
    Plot budget comparison across algorithms.
    
    Args:
        results: Dictionary containing results for each algorithm
        algorithms: List of algorithm names to plot
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    for algo in algorithms:
        budgets = np.mean(results[f"{algo}_budgets"], axis=1)
        sparsity_levels = np.arange(len(budgets))
        plt.plot(sparsity_levels, budgets, label=algo, marker='s')
        
    plt.xlabel("Sparsity Level")
    plt.ylabel("Budget Used")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_kmeans_performance(results: Dict[str, np.ndarray],
                          save_path: Optional[str] = None) -> None:
    """
    Plot K-means performance across different budgets.
    
    Args:
        results: Dictionary containing K-means results
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot mean error across Monte Carlo runs
    mean_errors = np.mean(results['km_errors'], axis=2)
    budgets = results['km_budgets'][:, :, 0]  # Budgets are the same across runs
    
    for s in range(mean_errors.shape[0]):
        plt.plot(budgets[s], mean_errors[s], 
                label=f"Sparsity Level {s}", marker='o')
        
    plt.xlabel("Budget")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_experiment_1(results: Dict[str, np.ndarray],
                     sparsity_range: Tuple[int, int] = (1, 8),
                     save_path: Optional[str] = None) -> None:
    """
    Plot results for Experiment 1: Varying sparsity with constant norm.
    
    Args:
        results: Dictionary containing experiment results
        sparsity_range: Tuple of (min, max) sparsity levels to plot
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Extract data
    sparsity_min, sparsity_max = sparsity_range
    sparsity_indices = range(sparsity_min, sparsity_max)
    
    # Plot each algorithm
    plt.plot(results['sparsity_grid'][sparsity_indices],
             results['cbc_budgets_means'][sparsity_indices],
             label='CBC', color='red', linewidth=2, linestyle='--')
    
    plt.plot(results['sparsity_grid'][sparsity_indices],
             results['cluster_budgets_means'][sparsity_indices],
             label='Cluster', color='black', linewidth=3)
    
    plt.plot(results['sparsity_grid'][sparsity_indices],
             results['cr_budgets_means'][sparsity_indices],
             label='CR', color='blue', linewidth=2, linestyle='--')
    
    plt.plot(results['sparsity_grid'][sparsity_indices],
             results['km_budgets'][sparsity_indices],
             label='KMeans', color='black', linewidth=2, linestyle='--')
    
    # Add confidence intervals
    plt.fill_between(results['sparsity_grid'][sparsity_indices],
                    results['cluster_budgets_lower_quant'][sparsity_indices],
                    results['cluster_budgets_upper_quant'][sparsity_indices],
                    color='black', alpha=0.2, label='_nolegend_')
    
    # Set scales and labels
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('Sparsity', fontsize=10)
    plt.ylabel('Average Budget', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_experiment_2(results: Dict[str, np.ndarray],
                     save_path: Optional[str] = None) -> None:
    """
    Plot results for Experiment 2: Varying dimensions and delta.
    
    Args:
        results: Dictionary containing experiment results
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Plot theoretical bound
    n_range = np.array(range(5000))
    plt.plot(n_range, n_range**2,
             label=r"$n^2$", color="black", linewidth=1)
    
    # Plot each delta value
    styles = ["--", "--", "--", "-"]
    colors = ["violet", "orange", "blue", "red"]
    widths = [1, 1, 1, 2]
    
    for i in range(len(results['delta_array'])):
        plt.plot(results['n_array'],
                results['mean_budgets'][:, i],
                label=fr'$\delta$={results["delta_array"][i]}',
                color=colors[i],
                linewidth=widths[i],
                linestyle=styles[i])
    
    # Add confidence intervals for the last delta
    plt.fill_between(results['n_array'],
                    results['lower_budgets'][:, -1],
                    results['upper_budgets'][:, -1],
                    color='red', alpha=0.2, label='_nolegend_')
    
    # Set labels
    plt.xlabel(r'Number of Items $n$', fontsize=10)
    plt.ylabel('Average Budget', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_experiment_3(results: Dict[str, np.ndarray],
                     save_path: Optional[str] = None) -> None:
    """
    Plot results for Experiment 3: Varying proportion with constant gap.
    
    Args:
        results: Dictionary containing experiment results
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Plot each algorithm
    plt.plot(results['theta_vec'],
             results['cluster_budgets_means'],
             label='Cluster', color='black', linewidth=3)
    
    plt.plot(results['theta_vec'],
             results['cbc_budgets_means'],
             label='CBC', color='red', linewidth=2, linestyle='--')
    
    plt.plot(results['theta_vec'],
             results['cr_budgets_means'],
             label='CR', color='blue', linewidth=2, linestyle='--')
    
    # Set scales and labels
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('Theta', fontsize=10)
    plt.ylabel('Average Budget', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 