"""
Utility functions for generating synthetic data for clustering experiments.
"""
import numpy as np  # type: ignore
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
            M[i, :] = m_a
        else:
            M[i, :] = m_b
    M = M[:, np.random.permutation(num_features)]
    return M


def generate_data_matrix_for_bernoulli(num_items: int,
                                       num_features: int,
                                       true_clusters: np.ndarray,
                                       signal_strength: float,
                                       sparsity: int) -> np.ndarray:
    """
    Generate a data matrix for the Bernoulli problem.

    Args:
        num_items: Number of rows (items)
        num_features: Number of columns (features)
        true_clusters: Array of true cluster assignments (0 or 1)
        signal_strength: Signal level (probability offset from 0.5, e.g., 0.1 means probs are 0.6/0.4)
        sparsity: Integer number of features where the signal is present.

    Returns:
        Data matrix of shape (num_items, num_features) containing probabilities.
    """
    # Ensure sparsity is not greater than num_features and non-negative
    sparsity = max(0, min(sparsity, num_features))

    # Base probability matrix (all 0.5)
    M = np.full((num_items, num_features), 0.5)

    # Generate indices for sparse features once
    # Ensure we only choose if sparsity > 0
    if sparsity > 0:
        sparse_indices = np.random.choice(
            num_features, sparsity, replace=False)
    else:
        # Empty array if sparsity is 0
        sparse_indices = np.array([], dtype=int)

    # Assign probabilities based on cluster and sparsity
    # Calculate probabilities for cluster 1 and 0 on the sparse features
    prob_cluster_1 = 0.5 + signal_strength / 2
    prob_cluster_0 = 0.5 - signal_strength / 2

    # Clip probabilities to be within [0, 1]
    prob_cluster_1 = np.clip(prob_cluster_1, 0.0, 1.0)
    prob_cluster_0 = np.clip(prob_cluster_0, 0.0, 1.0)

    # Apply the signal to the selected sparse features for each item based on
    # its cluster
    for i in range(num_items):
        if true_clusters[i] == 1:
            # Assign the calculated probability directly to the sparse indices
            # for this item
            M[i, sparse_indices] = prob_cluster_1
        else:
            M[i, sparse_indices] = prob_cluster_0

    return M
