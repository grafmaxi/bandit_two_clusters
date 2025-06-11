"""
Algorithms for binary clustering with Bernoulli-distributed responses.

This module implements sequential halving algorithms for clustering with Bernoulli trials.
The algorithms use individual Bernoulli draws for sampling.
"""
import numpy as np
from typing import Tuple, List, Optional
from random import choices

# Helper function for sum of Bernoullis


def sample_sum_bernoulli(n: int, p: float) -> int:
    """
    Simulate sum of n Bernoulli(p) trials.

    Args:
        n: Number of trials.
        p: Success probability for each trial.

    Returns:
        Sum of n Bernoulli random variables.
    """
    if n <= 0:
        return 0
    # Ensure p is within valid probability range [0, 1]
    p = np.clip(p, 0.0, 1.0)
    value = 0
    for i in range(n):
        value += np.random.rand() < p
    return value


def cesh_bernoulli(means_matrix: np.ndarray,
                   item_indices: List[int],
                   depth: int,
                   budget: int) -> Tuple[List[List[int]], int]:
    """
    Perform sequential halving using sum of Bernoulli samples.

    Args:
        means_matrix: Matrix of arm means (num_items x num_features).
        item_indices: Item indices to sample from.
        depth: Number of halving steps.
        budget: Maximum number of samples.

    Returns:
        Tuple containing:
        - indices: Index pairs (item, feature) that remain after the last halving step.
        - sample_cost: Number of samples that were used.
    """
    num_items, num_features = means_matrix.shape
    # Subsample index-pairs
    indices = [
        choices(
            item_indices,
            k=2**depth),
        choices(
            range(num_features),
            k=2**depth)]
    sample_cost = 0

    # Handle potential division by zero if depth is large relative to budget
    if depth <= 0:
        return indices, sample_cost

    for halving_step in range(depth):
        # Avoid division by zero and ensure sample_num is integer >= 0
        denominator = (2**(depth - halving_step + 1) * depth)
        if denominator == 0:
            sample_num = 0
        else:
            sample_num = max(0, int(budget / denominator))

        num_arms_in_step = 2**(depth - halving_step)
        step_means = np.zeros(num_arms_in_step)

        # Ensure sample_num is valid
        if sample_num < 0:
            continue

        for i in range(num_arms_in_step):
            # Check indices are within bounds
            item_idx_0 = 0  # Reference item is always 0
            item_idx_i = indices[0][i]
            feature_idx_i = indices[1][i]

            if (item_idx_i >= num_items or feature_idx_i >= num_features):
                print(f"Warning: Index out of bounds in cesh_bernoulli. Skipping step.")
                continue

            # Estimate means using sum of Bernoulli samples
            mean_0 = sample_sum_bernoulli(
                sample_num, means_matrix[item_idx_0, feature_idx_i])
            mean_i = sample_sum_bernoulli(
                sample_num, means_matrix[item_idx_i, feature_idx_i])

            if sample_num > 0:
                step_means[i] = abs(mean_0 / sample_num - mean_i / sample_num)
            else:
                step_means[i] = 0

            sample_cost += 2 * sample_num

        # Eliminate indices corresponding to smallest means in absolute value
        num_to_keep = max(0, int(num_arms_in_step / 2))
        # Sort descending by absolute difference
        index_sort = np.argsort(-step_means)
        indices[0] = [indices[0][i] for i in index_sort[:num_to_keep]]
        indices[1] = [indices[1][i] for i in index_sort[:num_to_keep]]

    return indices, sample_cost


def cr_bernoulli(means_matrix: np.ndarray,
                 delta: float) -> Tuple[List[int], int]:
    """
    Find an index from a different cluster than index 0 using sum of Bernoulli samples.

    Args:
        means_matrix: Matrix of arm means (num_items x num_features).
        delta: Risk level.

    Returns:
        Tuple containing:
        - candidate_indices: Item and feature indices from different cluster.
        - sample_cost: Number of samples that were used.
    """
    num_items, num_features = means_matrix.shape
    terminate = False
    iteration = 1
    sample_cost = 0

    while not terminate:
        budget_iteration = 2**(iteration + 1)
        log_arg = 16 * num_items * num_features * \
            np.log(max(1e-9, 16 * np.log(max(1e-9, 8 * num_items * num_features)) / delta))
        if log_arg <= 1:
            max_halving_steps = 1
        else:
            max_halving_steps = int(np.log(log_arg) / np.log(2)) + 1

        halving_step = 1
        while (halving_step * 2**(halving_step + 1) <=
               budget_iteration and halving_step < max_halving_steps):

            candidate_row_result = cesh_bernoulli(means_matrix, list(
                range(num_items)), halving_step, budget_iteration)
            step_cost = candidate_row_result[1]
            candidate_indices = candidate_row_result[0]
            sample_cost += step_cost

            if not candidate_indices or not candidate_indices[0] or not candidate_indices[1]:
                halving_step += 1
                continue

            samples_per_arm = 2**iteration
            item_idx_0 = 0
            item_idx_cand = candidate_indices[0][0]
            feature_idx_cand = candidate_indices[1][0]

            # Estimate means using sum of Bernoulli samples
            mean_0 = sample_sum_bernoulli(
                samples_per_arm, means_matrix[item_idx_0, feature_idx_cand])
            mean_cand = sample_sum_bernoulli(
                samples_per_arm, means_matrix[item_idx_cand, feature_idx_cand])

            if samples_per_arm > 0:
                diff = abs(
                    mean_0 /
                    samples_per_arm -
                    mean_cand /
                    samples_per_arm)
                log_thresh_arg = max(1e-9, iteration**3 / (0.15 * delta))
                threshold = (2.0**(-iteration / 2.0)) * \
                    np.sqrt(np.log(log_thresh_arg))

                if diff > threshold:
                    sample_cost += 2 * samples_per_arm
                    terminate = True
                    final_candidate_indices = [item_idx_cand, feature_idx_cand]
                    return final_candidate_indices, sample_cost

            sample_cost += 2 * samples_per_arm
            halving_step += 1

        iteration += 1
        if iteration > 100:
            print("Warning: cr_bernoulli exceeded max iterations.")
            dummy_indices = [
                0, 0] if num_items > 0 and num_features > 0 else []
            return dummy_indices, sample_cost

    print("Warning: cr_bernoulli exited loop unexpectedly.")
    final_candidate_indices = [
        0, 0] if num_items > 0 and num_features > 0 else []
    return final_candidate_indices, sample_cost


def cbc_bernoulli(means_matrix: np.ndarray,
                  candidate_row_index: int,
                  delta: float) -> Tuple[np.ndarray, int]:
    """
    Perform clustering using sum of Bernoulli samples.

    Args:
        means_matrix: Matrix of arm means (num_items x num_features).
        candidate_row_index: Row index presumed to be in a different cluster than item 0.
        delta: Risk level.

    Returns:
        Tuple containing:
        - clusters: Labels indices by detected clusters (0 or 1).
        - sample_cost: Number of samples that were used.
    """
    num_items, num_features = means_matrix.shape
    # Initialize clusters as integers
    clusters = np.zeros(num_items, dtype=int)
    iteration = int(np.ceil(np.log(max(1, num_items)) /
                    np.log(2))) if num_items > 0 else 1
    sample_cost = 0
    found_feature = False  # Flag to indicate if a separating feature is found

    while not found_feature:
        budget_iteration = 2**(iteration + 1)
        log_arg = 16 * num_items * num_features * \
            np.log(max(1e-9, 16 * np.log(max(1e-9, 8 * num_items * num_features)) / delta))
        if log_arg <= 1:
            max_halving_steps = 1
        else:
            max_halving_steps = int(np.log(log_arg) / np.log(2)) + 1

        halving_step = 1
        while (halving_step * 2**(halving_step + 1) <=
               budget_iteration and halving_step < max_halving_steps):

            candidate_column_result = cesh_bernoulli(
                means_matrix, [candidate_row_index], halving_step, budget_iteration)
            step_cost = candidate_column_result[1]
            candidate_indices = candidate_column_result[0]
            sample_cost += step_cost

            if not candidate_indices or not candidate_indices[1]:
                halving_step += 1
                continue

            candidate_column_index = candidate_indices[1][0]
            # Avoid division by zero
            samples_per_comp = max(1, int(2**iteration / max(1, num_items)))

            # Compare item 0 and candidate_row_index using sum of Bernoulli
            # samples
            mean_0 = sample_sum_bernoulli(
                samples_per_comp, means_matrix[0, candidate_column_index])
            mean_cand = sample_sum_bernoulli(
                samples_per_comp, means_matrix[candidate_row_index, candidate_column_index])
            sample_cost += 2 * samples_per_comp

            log_thresh_arg = max(1e-9,
                                 num_items * iteration**3 / (0.15 * delta))
            if samples_per_comp <= 0:
                threshold_main = np.inf
            else:
                threshold_main = 3 * \
                    np.sqrt(samples_per_comp * np.log(log_thresh_arg))

            if abs(mean_0 - mean_cand) > threshold_main:
                found_feature = True
                clusters[0] = 0
                clusters[candidate_row_index] = 1
                threshold_others = np.sqrt(
                    samples_per_comp * np.log(log_thresh_arg))

                for ind in range(num_items):
                    if ind == 0 or ind == candidate_row_index:
                        continue

                    # Compare item 'ind' with candidate_row_index using sum of
                    # Bernoulli samples
                    mean_ind = sample_sum_bernoulli(
                        samples_per_comp, means_matrix[ind, candidate_column_index])
                    mean_cand_again = sample_sum_bernoulli(
                        samples_per_comp, means_matrix[candidate_row_index, candidate_column_index])
                    sample_cost += 2 * samples_per_comp

                    if abs(mean_cand_again - mean_ind) > threshold_others:
                        clusters[ind] = 0
                    else:
                        clusters[ind] = 1

                return clusters, sample_cost

            halving_step += 1

        iteration += 1
        if iteration > 100 + \
                int(np.ceil(np.log(max(1, num_items)) / np.log(2))):
            print("Warning: cbc_bernoulli exceeded max iterations.")
            return clusters, sample_cost

    print("Warning: cbc_bernoulli exited loop unexpectedly.")
    return clusters, sample_cost


def cluster_bernoulli(means_matrix: np.ndarray,
                      delta: float) -> Tuple[Optional[np.ndarray], int]:
    """
    Full clustering algorithm combining CR and CBC with Bernoulli sampling.

    Args:
        means_matrix: Matrix of arm means (num_items x num_features).
        delta: Risk level.

    Returns:
        Tuple containing:
        - clusters: Labels indices by detected clusters (0 or 1), or None if clustering failed.
        - sample_cost: Total number of samples that were used.
    """
    sample_cost = 0

    # Find a candidate item from the other cluster
    candidate_indices, cr_cost = cr_bernoulli(means_matrix, delta / 2)
    sample_cost += cr_cost

    # Check if cr_bernoulli returned valid indices
    if not candidate_indices or len(candidate_indices) < 2:
        print("Warning: cr_bernoulli failed to return valid candidate indices.")
        return None, sample_cost

    candidate_row_index = candidate_indices[0]

    mean_0_check = means_matrix[0, :]
    mean_cand_check = means_matrix[candidate_row_index, :]

    if np.all(mean_0_check == mean_cand_check):
        return None, sample_cost

    # If candidate seems different, proceed with CBC
    else:
        clusters, cbc_cost = cbc_bernoulli(
            means_matrix, candidate_row_index, delta / 2)
        sample_cost += cbc_cost
        return clusters, sample_cost
