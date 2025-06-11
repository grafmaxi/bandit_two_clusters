"""
Utility functions for clustering operations.
"""
import numpy as np  # type: ignore
from typing import Tuple


def clusters_equal(clusters1: np.ndarray, clusters2: np.ndarray) -> bool:
    """
    Check if two cluster assignments are equal (up to permutation).

    Args:
        clusters1: First cluster assignment array
        clusters2: Second cluster assignment array

    Returns:
        True if the cluster assignments are equivalent (up to permutation), False otherwise
    """
    # Handle None values
    if clusters1 is None or clusters2 is None:
        return False

    # Ensure inputs are numpy arrays
    clusters1 = np.asarray(clusters1)
    clusters2 = np.asarray(clusters2)

    # Check if arrays have the same shape
    if clusters1.shape != clusters2.shape:
        return False

    # Check if arrays contain only 0s and 1s
    if not (
        np.all(
            np.isin(
            clusters1, [
                0, 1])) and np.all(
                    np.isin(
                        clusters2, [
                            0, 1]))):
        return False

    return (np.all(clusters1 == clusters2) or
            np.all(clusters1 == 1 - clusters2))


def get_cluster_representatives(clusters: np.ndarray) -> Tuple[int, int]:
    """
    Get representative indices for each cluster.

    Args:
        clusters: Cluster assignment array

    Returns:
        Tuple of (index of first cluster representative, index of second cluster representative)
    """
    # Handle None values
    if clusters is None:
        raise ValueError("Cluster assignments cannot be None")

    # Ensure input is numpy array
    clusters = np.asarray(clusters)

    # Check if array contains only 0s and 1s
    if not np.all(np.isin(clusters, [0, 1])):
        raise ValueError("Cluster assignments must contain only 0s and 1s")

    return np.argmax(clusters), np.argmin(clusters)
