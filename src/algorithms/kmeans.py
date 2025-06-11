"""
K-means clustering algorithm with budget constraint.
"""
import numpy as np  # type: ignore
from typing import Tuple, Optional
from sklearn.cluster import KMeans  # type: ignore


def kmeans_budget(
        means_matrix: np.ndarray,
        budget: int,
        k: int = 2) -> np.ndarray:
    """
    Perform K-means clustering with a budget constraint.

    The algorithm:
    1. Calculates the number of samples per entry based on the budget
    2. Adds Gaussian noise to the data based on the sample size
    3. Performs K-means clustering on the noisy data

    Args:
        means_matrix: Matrix of arm means (n x d)
        budget: Maximum number of samples allowed
        k: Number of clusters (default: 2)

    Returns:
        Cluster assignments for each data point (n-dimensional array)
    """
    num_items, num_features = means_matrix.shape
    samples_per_entry = int(budget / (num_items * num_features))

    # Add Gaussian noise based on sample size
    noisy_means = np.random.normal(
        means_matrix, 1 / np.sqrt(samples_per_entry))

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, n_init='auto')
    clusters = kmeans.fit_predict(noisy_means)

    return clusters
