"""
Algorithms for binary clustering with Bernoulli-distributed responses.
"""
import numpy as np
# import time # No longer needed here
from typing import Tuple, List, Optional
from random import choices


def cesh_bernoulli(means_matrix: np.ndarray, 
                   rows: List[int], 
                   depth: int, 
                   budget: int) -> Tuple[List[List[int]], int]:
    """
    Perform sequential halving for a subsampled set of arms.
    
    Args:
        means_matrix: Matrix of arm means (num_items x num_features).
        rows: Rows (item indices) to sample indices from.
        depth: Number of halving steps.
        budget: Maximal number of samples.
        
    Returns:
        Tuple containing:
        - indices: Index pairs (item, feature) that remain after the last halving step.
        - cost: Number of samples that were required.
    """
    num_items, num_features = means_matrix.shape
    # Subsample index-pairs
    indices = [choices(rows, k=2**depth), choices(range(num_features), k=2**depth)]
    cost = 0
    
    # Handle potential division by zero if depth is large relative to budget
    if depth <= 0:
        return indices, cost # Or raise error
    
    for halving_step in range(depth):
        # Avoid division by zero and ensure sample_num is integer >= 0
        denominator = (2**(depth - halving_step + 1) * depth)
        if denominator == 0: 
            sample_num = 0
        else:
            sample_num = max(0, int(budget / denominator))
        
        num_arms_in_step = 2**(depth - halving_step)
        step_means = np.zeros(num_arms_in_step)
        
        # Ensure sample_num is valid for binomial distribution
        if sample_num < 0:
             continue # or handle error

        for i in range(num_arms_in_step):
            # Check indices are within bounds
            item_idx_0 = 0 # Reference item is always 0
            item_idx_i = indices[0][i]
            feature_idx_i = indices[1][i]
            
            if (item_idx_i >= num_items or feature_idx_i >= num_features):
                 print(f"Warning: Index out of bounds in cesh_bernoulli. Skipping step.")
                 # Handle out-of-bounds, maybe assign default mean or skip
                 continue
                 
            # Estimate means using binomial samples
            mean_0 = np.random.binomial(sample_num, means_matrix[item_idx_0, feature_idx_i])
            mean_i = np.random.binomial(sample_num, means_matrix[item_idx_i, feature_idx_i])

            if sample_num > 0:
                step_means[i] = abs(mean_0 / sample_num - mean_i / sample_num)
            else:
                step_means[i] = 0 # Define behavior for zero samples
            
            cost += 2 * sample_num
            
        # Eliminate indices corresponding to smallest means in absolute value
        num_to_keep = max(0, int(num_arms_in_step / 2))
        index_sort = np.argsort(-step_means) # Sort descending by absolute difference
        indices[0] = [indices[0][i] for i in index_sort[:num_to_keep]]
        indices[1] = [indices[1][i] for i in index_sort[:num_to_keep]]
        
    return indices, cost


def cr_bernoulli(means_matrix: np.ndarray, delta: float) -> Tuple[List[int], int]:
    """
    Find an index from a different cluster than index 0.
    
    Args:
        means_matrix: Matrix of arm means (num_items x num_features).
        delta: Risk level.
        
    Returns:
        Tuple containing:
        - candidate_indices: Row (item) and column (feature) indices.
        - cost: Number of samples that were required.
    """
    num_items, num_features = means_matrix.shape
    terminate = False
    iteration = 1
    cost = 0
    
    while not terminate:
        budget_iteration = 2**(iteration + 1)
        # Calculate max halving steps based on theoretical bound
        log_arg = 16 * num_items * num_features * np.log(16 * np.log(8 * num_items * num_features) / delta)
        if log_arg <= 1: # Prevent log of non-positive
             max_halving_steps = 1
        else:
             max_halving_steps = int(np.log(log_arg) / np.log(2)) + 1
        
        halving_step = 1
        # Loop through possible halving steps for the current budget
        while (halving_step * 2**(halving_step + 1) <= budget_iteration and halving_step < max_halving_steps):
            
            candidate_row_result = cesh_bernoulli(
                means_matrix, list(range(num_items)), halving_step, budget_iteration
            )
            step_cost = candidate_row_result[1]
            candidate_indices = candidate_row_result[0]
            cost += step_cost
            
            # Ensure candidate_indices is not empty after CESH
            if not candidate_indices or not candidate_indices[0] or not candidate_indices[1]:
                # print(f"Warning: CESH returned empty indices in cr_bernoulli iteration {iteration}, step {halving_step}. Continuing.")
                halving_step += 1
                continue # Skip to next halving step if no candidates found
            
            # Check termination condition
            samples_per_arm = 2**iteration
            item_idx_0 = 0
            item_idx_cand = candidate_indices[0][0] # Take the first candidate item
            feature_idx_cand = candidate_indices[1][0] # Take the first candidate feature

            # Estimate means
            mean_0 = np.random.binomial(samples_per_arm, means_matrix[item_idx_0, feature_idx_cand])
            mean_cand = np.random.binomial(samples_per_arm, means_matrix[item_idx_cand, feature_idx_cand])
            
            # Check indices before calculating diff and threshold
            if samples_per_arm > 0: 
                diff = abs(mean_0 / samples_per_arm - mean_cand / samples_per_arm)
                # Calculate threshold (ensure log argument is positive)
                log_thresh_arg = iteration**3 / (0.15 * delta)
                if log_thresh_arg <= 0: 
                    threshold = np.inf # Or handle error as appropriate
                else:
                    threshold = (2.0**(-iteration / 2.0)) * np.sqrt(np.log(log_thresh_arg))
                
                # Check termination condition
                if diff > threshold:
                    cost += 2 * samples_per_arm # Cost for this final check
                    terminate = True
                    # Ensure candidate_indices has the right format before returning
                    final_candidate_indices = [item_idx_cand, feature_idx_cand]
                    return final_candidate_indices, cost
            
            # Cost for the check if not terminated yet (samples already taken)
            cost += 2 * samples_per_arm 
            halving_step += 1
            
        # If loop finishes without termination, increment iteration
        iteration += 1
        # Safety break for potential infinite loops
        if iteration > 100: # Adjust limit as needed
             print("Warning: cr_bernoulli exceeded max iterations.")
             # Return something reasonable or raise error
             # Returning dummy indices and current cost
             dummy_indices = [0, 0] if num_items > 0 and num_features > 0 else []
             return dummy_indices, cost

    # Should be unreachable if logic is correct, but added for safety
    # This part might be reached if terminate becomes true but the return inside the loop isn't hit
    # Returning the last candidate found or a default
    # The return statement inside the loop should handle the successful exit.
    # If we reach here, it means the loop exited unexpectedly. 
    print("Warning: cr_bernoulli exited loop unexpectedly.")
    # Return a default value or raise an error
    final_candidate_indices = [0, 0] if num_items > 0 and num_features > 0 else []
    return final_candidate_indices, cost


def cbc_bernoulli(means_matrix: np.ndarray, 
                  candidate_row_index: int, 
                  delta: float) -> Tuple[np.ndarray, int]:
    """
    Perform clustering given a candidate row index from a different cluster than index 0.
    
    Args:
        means_matrix: Matrix of arm means (num_items x num_features).
        candidate_row_index: Row index (item) presumed to be in a different cluster than item 0.
        delta: Risk level.
        
    Returns:
        Tuple containing:
        - clusters: Labels indices by detected clusters (0 or 1).
        - cost: Number of samples that were required.
    """
    num_items, num_features = means_matrix.shape
    clusters = np.zeros(num_items, dtype=int) # Initialize clusters as integers
    # Initial iteration based on num_items
    iteration = int(np.ceil(np.log(num_items) / np.log(2))) if num_items > 0 else 1
    cost = 0
    found_feature = False # Flag to indicate if a separating feature is found
    
    while not found_feature:
        budget_iteration = 2**(iteration + 1)
        # Calculate max halving steps
        log_arg = 16 * num_items * num_features * np.log(16 * np.log(8 * num_items * num_features) / delta)
        if log_arg <= 1: max_halving_steps = 1
        else: max_halving_steps = int(np.log(log_arg) / np.log(2)) + 1
        
        halving_step = 1
        while (halving_step * 2**(halving_step + 1) <= budget_iteration and halving_step < max_halving_steps):
            
            # Run CESH to find a good candidate feature
            candidate_column_result = cesh_bernoulli(
                means_matrix, [candidate_row_index], halving_step, budget_iteration
            )
            step_cost = candidate_column_result[1]
            candidate_indices = candidate_column_result[0]
            cost += step_cost
            
            # Ensure candidate_indices is valid
            if not candidate_indices or not candidate_indices[1]:
                # print(f"Warning: CESH returned empty indices in cbc_bernoulli iteration {iteration}, step {halving_step}.")
                halving_step += 1
                continue
                
            candidate_column_index = candidate_indices[1][0]
            
            # Calculate samples per comparison
            samples_per_comp = max(1, int(2**iteration / num_items)) # Ensure at least 1 sample

            # Compare item 0 and candidate_row_index using the candidate feature
            mean_0 = np.random.binomial(samples_per_comp, means_matrix[0, candidate_column_index])
            mean_cand = np.random.binomial(samples_per_comp, means_matrix[candidate_row_index, candidate_column_index])
            cost_comparison = 2 * samples_per_comp
            cost += cost_comparison

            # Calculate threshold for distinguishing 0 and candidate_row_index
            log_thresh_arg = num_items * iteration**3 / (0.15 * delta)
            if log_thresh_arg <= 0 or samples_per_comp <= 0: 
                threshold_main = np.inf
            else:
                threshold_main = 3 * np.sqrt(samples_per_comp * np.log(log_thresh_arg))
            
            # Check if item 0 and candidate_row_index are different enough
            if abs(mean_0 - mean_cand) > threshold_main:
                found_feature = True # Found a feature that likely separates the clusters
                clusters[0] = 0 # Assign item 0 to cluster 0 (by convention)
                clusters[candidate_row_index] = 1 # Assign candidate to cluster 1

                # Now, compare all other items to the candidate_row_index using this feature
                threshold_others = np.sqrt(samples_per_comp * np.log(log_thresh_arg)) # Threshold for other items
                
                for ind in range(num_items):
                    if ind == 0 or ind == candidate_row_index: continue # Skip items already assigned
                    
                    # Compare item 'ind' with candidate_row_index
                    mean_ind = np.random.binomial(samples_per_comp, means_matrix[ind, candidate_column_index])
                    mean_cand_again = np.random.binomial(samples_per_comp, means_matrix[candidate_row_index, candidate_column_index])
                    cost += 2 * samples_per_comp
                    
                    # If item 'ind' is different from candidate_row_index, assign to cluster 0
                    if abs(mean_cand_again - mean_ind) > threshold_others:
                        clusters[ind] = 0 
                    # Otherwise, assign to cluster 1 (same as candidate_row_index)
                    else:
                        clusters[ind] = 1
                        
                return clusters, cost # Return results once clustering is done
            
            # If feature didn't separate, proceed to next halving step
            halving_step += 1
            
        # If loop finishes without finding a feature, increment iteration
        iteration += 1
        # Safety break
        if iteration > 100 + int(np.ceil(np.log(num_items) / np.log(2))): # Adjust limit
             print("Warning: cbc_bernoulli exceeded max iterations.")
             # Return current clusters (likely all zeros) and cost
             return clusters, cost
             
    # Should be unreachable if logic is correct
    print("Warning: cbc_bernoulli exited loop unexpectedly.")
    return clusters, cost


def cluster_bernoulli(means_matrix: np.ndarray, delta: float) -> Tuple[Optional[np.ndarray], int]:
    """
    Clustering algorithm, combining cr and cbc. Assumes K=2.
    
    Args:
        means_matrix: Matrix of arm means (num_items x num_features).
        delta: Risk level.
        
    Returns:
        Tuple containing:
        - clusters: Labels indices by detected clusters (0 or 1), or None if clustering failed.
        - cost: Total number of samples required.
    """
    # Step 1: Find a candidate item from the other cluster
    candidate_indices, cr_cost = cr_bernoulli(means_matrix, delta / 2)
    
    # Check if cr_bernoulli returned valid indices
    if not candidate_indices or len(candidate_indices) < 2:
         print("Warning: cr_bernoulli failed to return valid candidate indices.")
         return None, cr_cost # Indicate failure
         
    candidate_row_index = candidate_indices[0]
    candidate_feature_index = candidate_indices[1]

    # Step 2: Check if the candidate is actually different from item 0 using the found feature
    # This check uses the logic similar to cr_bernoulli's termination condition but with potentially more samples
    # Determine samples needed for a reliable check (can be adjusted)
    # Using a fixed number or one based on delta, e.g., related to cbc's iterations
    num_items = means_matrix.shape[0]
    check_samples = max(10, int(np.log(1/delta))) # Example: Use more samples for check
    
    mean_0_check = np.random.binomial(check_samples, means_matrix[0, candidate_feature_index])
    mean_cand_check = np.random.binomial(check_samples, means_matrix[candidate_row_index, candidate_feature_index])
    check_cost = 2 * check_samples
    total_cost = cr_cost + check_cost
    
    # A simple check: if the means are empirically identical after check_samples, assume same cluster
    if mean_0_check == mean_cand_check: 
         print("Candidate item determined to be in the same cluster as item 0.")
         # Decide if returning None or assigning all to cluster 0 is appropriate
         # Returning None indicates failure to find distinct clusters.
         return None, total_cost
         
    # Step 3: If candidate seems different, proceed with CBC
    else:
        clusters, cbc_cost = cbc_bernoulli(means_matrix, candidate_row_index, delta / 2)
        total_cost += cbc_cost
        return clusters, total_cost