import numpy as np
import random
from sklearn.cluster import KMeans
# from src.utils.adaptive import update_wgt, min_w, do_uniform # Original import
# from src.utils import adaptive as adaptive_utils # Import module
# from src.utils import clustering as clustering_utils # Import module
# from src.utils.clustering import calc_err_by_sigma # Original import

# Removed global parameter definitions (algidx, h_max, h_min, nTrial, p_norm, n, K, L, T, w, ucb_constant, p, sigma, tag, num_rank)

def adaptive_clustering(sigma, config: dict):
    """
    Runs the adaptive clustering algorithm based on the provided configuration.

    Args:
        sigma: The true cluster assignments for users.
        config: Dictionary containing configuration parameters like:
            n, K, L, T, p_norm, p, h_min, h_max, algidx, w, ucb_constant, num_rank
    """
    # Import utilities here
    from src.utils import adaptive as adaptive_utils
    # from src.utils import clustering as clustering_utils # No longer needed for calc_err_by_sigma

    # Extract parameters from config
    n = config['n']
    K = config['K']
    L = config['L']
    T = config['T']
    p_norm = config.get('p_norm', np.inf) # Use .get for optional params or provide defaults
    p = config['p']
    # algidx = config.get('algidx', 1) # Remove algidx extraction
    w_param = config.get('w', 1) # Renamed from w to avoid conflict
    ucb_constant = config.get('ucb_constant', 0.1)
    num_rank = config.get('num_rank', 2)

    period = np.ceil(T / (4 * np.log(T / n))).astype(int)
    # Randomly generate h (user response probabilities) based on config min/max
    
    # Random cluster assignments (using provided sigma, not random)
    q = np.zeros((n, L))
    r = np.zeros((n, L))
    r_ = np.zeros((n, L))
    h_rank = np.zeros(n) # Assuming h_rank logic remains or is passed via config if needed
    for i in range(n):
        # Example h_rank logic, might need adjustment based on config/requirements
        for ell in range(L):
            # Use p from config
            q[i, ell] = p[sigma[i], ell] 
        r[i, :] = 2 * q[i, :] - 1
        # Use p_norm from config
        norm_r = np.linalg.norm(r[i, :], ord=p_norm)
        r_[i, :] = r[i, :] / norm_r if norm_r > 0 else r[i, :]
    
    # Initialize N_tot and N_pos
    N_tot = np.zeros((n, L))
    N_pos = np.zeros((n, L))
    hat_q = np.full((n, L), 0.5)
    hat_r = np.zeros((n, L))
    hat_r_ = np.zeros((n, L))
    hat_p = np.zeros((K, L))
    
    L_stat = np.zeros((K, L))
    # Reinstate initialization of hat_sigma
    hat_sigma = np.random.randint(0, K, size=n)
    xi = np.zeros((K, L))
    
    wgt = np.ones((n, L))
    idx = np.zeros(n)

    # Placeholder for results if needed within the loop, adjust as necessary
    # results_over_time = []
    
    cnt = -1
    for t in range(1, T + 1):
        # Sampling scheme - Keep only algidx == 1 logic
            if np.min(N_tot) <= max(np.sqrt(t / n), 10):
            # Use w_param from config
            ell_t, W_t = adaptive_utils.do_uniform(N_tot, w_param)
            else:
             # Use w_param from config
            W_t = adaptive_utils.min_w(idx, w_param)
            ell_min_val = np.min(N_tot[W_t[0], :])
            N_tot_i = np.sum(N_tot[W_t[0], :])
                if ell_min_val <= max(np.sqrt(N_tot_i), 10):
                ell_t = np.argmin(N_tot[W_t[0], :])
            else:
                ell_t = np.argmax(wgt[W_t[0], :])
        
        # Ask user and update statistics
        for i in W_t:
            N_tot[i, ell_t] += 1
            Xil = 2 * (random.random() < q[i, ell_t]) - 1
            if Xil > 0:
                N_pos[i, ell_t] += 1
            
            # Avoid division by zero if N_tot is 0
            if N_tot[i, ell_t] > 0:
            hat_q[i, ell_t] = N_pos[i, ell_t] / N_tot[i, ell_t]
            else:
                hat_q[i, ell_t] = 0.5 # Default initial estimate

            hat_r[i, ell_t] = 2 * hat_q[i, ell_t] - 1
            L_stat[sigma[i], ell_t] += 1
            
            # Update weights for the adaptive algorithm (always run this now)
            wgt[i, :] = adaptive_utils.update_wgt(N_tot[i, :], hat_q[i, :], hat_p, hat_sigma[i], t, ucb_constant)
                idx[i] = np.dot(wgt[i, :], N_tot[i, :])
        
        # Periodic clustering and error update
        if t % period == 0:
            cnt += 1
            for i in range(n):
                 # Use p_norm from config
                norm_hat_r = np.linalg.norm(hat_r[i, :], ord=p_norm)
                if norm_hat_r > 0:
                    hat_r_[i, :] = hat_r[i, :] / norm_hat_r
                else:
                    hat_r_[i, :] = hat_r[i, :] # Keep as zeros if norm is zero
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=K, max_iter=1000, n_init=5).fit(hat_r_)
            hat_sigma_ = kmeans.labels_
            
            # Use adaptive_utils.calc_err_by_sigma
            err, hat_sigma, k_vec = adaptive_utils.calc_err_by_sigma(hat_sigma_, sigma, K)
            
            # Update xi (clusters' characteristics)
            for k in range(K):
                 # Check if cluster k exists in results, handle empty clusters
                 if k_vec[k] < len(kmeans.cluster_centers_):
                xi[k, :] = kmeans.cluster_centers_[k_vec[k]]
                 # else: handle case where a cluster has no members assigned by kmeans
            
            # Update p_kl estimates
            for k in range(K):
                cluster_members = (hat_sigma == k)
                num_members = np.sum(cluster_members)
                if num_members > 0:
                    # Original logic: sum((2*hat_q - 1) * cluster_mask) / num_members
                    # Simplified equivalent for p_hat = 0.5 * (term + 1)
                    # term = sum(hat_r * cluster_mask) / num_members
                    hat_p[k, :] = np.sum(hat_r[cluster_members, :], axis=0) / num_members
                    # Convert back to probability p = (r + 1) / 2
                    hat_p[k, :] = (hat_p[k, :] + 1) / 2
                # else: handle empty cluster, maybe keep previous estimate or default?
            
            # Update weights and indices for adaptive algorithm (always run this now)
                for i in range(n):
                wgt[i, :] = adaptive_utils.update_wgt(N_tot[i, :], hat_q[i, :], hat_p, hat_sigma[i], t, ucb_constant)
                idx[i] = np.dot(wgt[i, :], N_tot[i, :])
        
            # Store results periodically if needed, e.g., error rate at this time step
            # current_error = err / n
            # results_over_time.append({'t': t, 'error': current_error, 'hat_sigma': hat_sigma.copy()})


    # Return the final clustering result and potentially other metrics
    # The original script didn't explicitly return, decide what's needed for experiment_4
    final_error, final_hat_sigma, _ = adaptive_utils.calc_err_by_sigma(hat_sigma, sigma, K)
    # Consider returning final_error/n, final_hat_sigma, N_tot (budget used per user/question), etc.
    return final_hat_sigma, final_error / n, N_tot

# Removed trailing script code / duplicate logic
