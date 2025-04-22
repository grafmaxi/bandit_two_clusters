"""
Utility functions for generating synthetic data for clustering experiments.
"""
import numpy as np # type: ignore
from typing import Tuple

def generate_clusters(num_items: int, theta: float) -> np.ndarray:
    """
    Generate random cluster assignments for a two-cluster problem.
    
    Args:
        num_items: Number of items to cluster
        theta: Proportion of items in one cluster (between 0 and 1)
        
    Returns:
        Array of cluster assignments (0 or 1)
    """
    return np.random.permutation(
        1 * (np.array(range(num_items)) < theta * num_items)
    )

def generate_data_matrix(num_items: int,
                        num_features: int,
                        sparsity: int,
                        signal_strength: float,
                        true_clusters: np.ndarray) -> np.ndarray:
    """
    Generate a data matrix with specified properties for a two-cluster problem.
    
    Args:
        num_items: Number of rows (items)
        num_features: Number of columns (features)
        sparsity: Number of non-zero features in the signal
        signal_strength: Strength of the non-zero features
        true_clusters: Array of true cluster assignments (0 or 1)
        
    Returns:
        Data matrix of shape (num_items, num_features)
    """
    # Generate cluster means
    m_a = np.zeros(num_features)
    m_b = np.zeros(num_features)
    m_b[range(sparsity)] = np.zeros(sparsity) + signal_strength
    
    # Generate data matrix
    M = np.zeros([num_items, num_features])
    for i in range(num_items):
        if true_clusters[i] == 1:
            M[i,:] = m_a
        else:
            M[i,:] = m_b
            
    return M 