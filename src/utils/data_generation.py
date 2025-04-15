"""
Utility functions for generating synthetic data.
"""
import numpy as np # type: ignore
from typing import Tuple

def generate_clusters(num_items: int, theta: float) -> np.ndarray:
    """
    Generate random cluster assignments.
    
    Args:
        num_items: Number of items to cluster
        theta: Proportion of items in one cluster
        
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
    Generate data matrix with specified properties.
    
    Args:
        num_items: Number of rows (items)
        num_features: Number of columns (features)
        sparsity: Number of non-zero features
        signal_strength: Strength of non-zero features
        true_clusters: Array of true cluster assignments
        
    Returns:
        Data matrix of shape (num_items, num_features)
    """
    # Generate cluster means
    m_a = np.zeros(num_features)
    m_b = np.zeros(num_features)
    m_b[range(sparsity)] = np.zeros(sparsity) + signal_strength/np.sqrt(sparsity)
    
    # Generate data matrix
    M = np.zeros([num_items, num_features])
    for i in range(num_items):
        if true_clusters[i] == 1:
            M[i,:] = m_a
        else:
            M[i,:] = m_b
            
    return M

def generate_experiment_data(config: object) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data for an experiment based on its configuration.
    
    Args:
        config: Experiment configuration object
        
    Returns:
        Tuple of (data matrix, true cluster assignments)
    """
    true_clusters = generate_clusters(config.num_items, config.theta)
    M = generate_data_matrix(
        config.num_items,
        config.num_features,
        config.sparsity,
        config.signal_strength,
        true_clusters
    )
    return M, true_clusters 