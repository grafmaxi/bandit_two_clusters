"""
Algorithms for binary clustering when the noise is standard-normal.
"""
import numpy as np # type: ignore
from random import choices


def cesh_gaussian(means_matrix, rows, depth, budget):
    """
    Perform sequential halving for a subsampled set of arms.

    Parameters
    ----------
    means_matrix : np.ndarray
        Matrix of arm means.
    rows : list
        Rows to sample indices from.
    depth : int
        Number of halving steps.
    budget : int
        Maximal number of samples.

    Returns
    -------
    indices : list
        Index pair that remains after the last halving step.
    cost : int
        Number of samples that were required.
    """
    num_features = means_matrix.shape[1]
    # Subsample index-pairs
    indices = [choices(rows, k=2**depth), choices(range(num_features), k=2**depth)]
    cost = 0
    for halving_step in range(depth):
        sample_num = int(budget / (2**(depth - halving_step + 1) * depth))
        # Set sample effort
        step_means = np.zeros(2**(depth - halving_step))
        for i in range(2**(depth - halving_step)):
            step_means[i] = abs(
                np.random.normal(
                    means_matrix[0, indices[1][i]] -
                    means_matrix[indices[0][i], indices[1][i]],
                    np.sqrt(2 / sample_num)
                )
            )
            cost += 2 * sample_num
        # Eliminate indices corresponding to smallest means in absolute value
        index_sort = np.argsort(-step_means)
        indices[0] = [indices[0][i] for i in index_sort[: int(2**(depth - halving_step - 1))]]
        indices[1] = [indices[1][i] for i in index_sort[: int(2**(depth - halving_step - 1))]]
    return indices, cost


def cr_gaussian(means_matrix, delta):
    """
    Find an index from a different cluster than index 0.

    Parameters
    ----------
    means_matrix : np.ndarray
        Matrix of arm means.
    delta : float
        Risk level.

    Returns
    -------
    candidate_indices : list
        Row and column indices.
    cost : int
        Number of samples that were required.
    """
    num_items, num_features = means_matrix.shape
    terminate = False
    iteration = 1
    cost = 0  # Start with budget 2**1 and double at each iteration
    while not terminate:
        halving_step = 1
        while (
            halving_step * 2**(halving_step + 1) <= 2**(iteration + 1) and
            halving_step < int(
                np.log(16 * num_items * num_features *
                       np.log(16 * np.log(8 * num_items * num_features) / delta)) /
                np.log(2)
            ) + 1
        ):
            candidate_row_result = cesh_gaussian(
                means_matrix, range(num_items), halving_step, 2**(iteration + 1)
            )
            cost += candidate_row_result[1]
            candidate_indices = candidate_row_result[0]
            if (
                abs(
                    np.random.normal(
                        means_matrix[0, candidate_indices[1]] -
                        means_matrix[candidate_indices[0], candidate_indices[1]],
                        np.sqrt(1 / 2**(iteration - 1))
                    )
                ) > 2**((2 - iteration) / 2) * np.sqrt(np.log(iteration**3 / (0.15 * delta)))
            ):
                cost += 2**(iteration + 1)
                terminate = True
                break
            halving_step += 1
            cost += 2**(iteration + 1)
        iteration += 1
    return candidate_indices, cost


def cbc_gaussian(means_matrix, candidate_row_index, delta):
    """
    Perform clustering if we are given candidate_row_index from a different
    cluster than index 0.

    Parameters
    ----------
    means_matrix : np.ndarray
        Matrix of arm means.
    candidate_row_index : int
        Row index.
    delta : float
        Risk level.

    Returns
    -------
    clusters : np.ndarray
        Labels indices by detected clusters.
    cost : int
        Number of samples that were required.
    """
    num_items, num_features = means_matrix.shape
    clusters = np.zeros(num_items)
    iteration = int(np.ceil(np.log(num_items) / np.log(2)))
    cost = 0  # Start with budget 2**iteration and double at each iteration
    while clusters[0] == 0:
        halving_step = 1
        while (
            halving_step * 2**(halving_step + 1) <= 2**(iteration + 1) and
            halving_step < int(
                np.log(
                    16 * num_items * num_features *
                    np.log(16 * np.log(8 * num_items * num_features) / delta)
                ) / np.log(2)
            ) + 1
        ):
            candidate_column_result = cesh_gaussian(
                means_matrix, [candidate_row_index], halving_step, 2**(iteration + 1)
            )
            cost += candidate_column_result[1]
            candidate_column_index = candidate_column_result[0][1]
            if (
                abs(
                    np.random.normal(
                        int(2**iteration / num_items) *
                        (means_matrix[0, candidate_column_index] -
                         means_matrix[candidate_row_index, candidate_column_index]),
                        np.sqrt(2 * int(2**iteration / num_items))
                    )
                ) > 3 * np.sqrt(
                    4 * int(2**iteration / num_items) *
                    np.log(num_items * iteration**3 / (0.15 * delta))
                )
            ):
                clusters[0] = 1
                cost += 2 * int(2**iteration / num_items)
                for ind in range(1, num_items):
                    if (
                        abs(
                            np.random.normal(
                                int(2**iteration / num_items) *
                                (means_matrix[candidate_row_index, candidate_column_index] -
                                 means_matrix[ind, candidate_column_index]),
                                np.sqrt(2 * int(2**iteration / num_items))
                            )
                        ) > np.sqrt(
                            4 * int(2**iteration / num_items) *
                            np.log(num_items * iteration**3 / (0.15 * delta))
                        )
                    ):
                        clusters[ind] = 1
                        cost += 2 * int(2**iteration / num_items)
                break
            cost += 2 * int(2**iteration / num_items)
            halving_step += 1
        iteration += 1
    return clusters, cost


def cluster_gaussian(means_matrix, delta):
    """
    Clustering algorithm, combining cr and cbc.

    Parameters
    ----------
    means_matrix : np.ndarray
        Matrix of arm means.
    delta : float
        Risk level.

    Returns
    -------
    clusters : np.ndarray
        Labels indices by detected clusters.
    cost : int
        Number of samples that were required.
    """
    candidate_indices, cr_cost = cr_gaussian(means_matrix, delta / 2)
    if (means_matrix[0, :] == means_matrix[candidate_indices[0], :]).all():
        return np.nan, np.nan
    else:
        clusters, cbc_cost = cbc_gaussian(means_matrix, candidate_indices[0], delta / 2)
        cost = cbc_cost + cr_cost
        return clusters, cost