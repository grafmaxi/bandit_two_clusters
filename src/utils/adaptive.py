import numpy as np
from sklearn.cluster import KMeans
from itertools import permutations

def avg_kmeans(observe, hat_k):
    m = observe['m']
    n = observe['n']
    X = observe['X']
    W = observe['W']  # Not used in the code
    R = observe['R']
    lambda_ = observe['lambda']  # Not used in the code
    w = observe['w']  # Not used in the code

    sum_X = np.zeros(n)
    sum_cnt = np.zeros(n)

    for u in range(m):
        for v in R[u]:
            sum_cnt[v] += 1
            if X[u, v] == 1:
                sum_X[v] += 1

    for v in range(n):
        if sum_cnt[v] > 0:
            sum_X[v] /= sum_cnt[v]
        else:
            sum_X[v] = 0

    kmeans = KMeans(n_clusters=hat_k, random_state=0).fit(sum_X.reshape(-1, 1))
    hat_sigma = kmeans.labels_

    hat_v = [[] for _ in range(hat_k)]
    for k in range(hat_k):
        hat_v[k] = np.where(hat_sigma == k)[0].tolist()

    return hat_v

def calc_err_by_sigma(hat_sigma, sigma, K):
    n = len(sigma)

    # Generate all permutations of range(1, K+1)
    k_perms = list(permutations(range(K)))
    len_perm = len(k_perms)
    score_perm = np.zeros(len_perm)

    for perm_idx in range(len_perm):
        k_vec = k_perms[perm_idx]
        for v in range(n):
            if sigma[v] != k_vec[hat_sigma[v] - 1]:
                score_perm[perm_idx] += 1

    err = np.min(score_perm)
    best_idx = np.argmin(score_perm)

    k_vec = k_perms[best_idx]
    best_hat_sigma = np.zeros(n, dtype=int)
    for v in range(n):
        best_hat_sigma[v] = k_vec[hat_sigma[v] - 1]

    return err, best_hat_sigma, k_vec

def do_uniform(N_tot, w):
    ell_idx = np.min(N_tot, axis=0)
    ell_t = rand_argmin(ell_idx)
    W_t = min_w(N_tot[:, ell_t], w)
    return ell_t, W_t

def fast_weighted_prob(w):
    # w is the weight of interest (row vector)

    X = np.random.rand() * np.sum(w)
    R = 1
    for i in range(len(w)):
        X -= w[i]
        if X <= 0:
            R = i + 1  # Adjust for 0-based indexing in Python
            break
    return R

def kl_divergence(p, q):
    eps = np.finfo(float).eps
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    kl = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    return kl

def kmeans_with_threshold(R, threshold, n, K, p):
    L = R.shape[1]
    I_R = np.arange(1, n + 1)
    T_v = np.zeros((n, I_R.size))

    for ind in range(I_R.size):
        for i in range(n):
            T_v[i, ind] = (np.linalg.norm(R[I_R[ind] - 1, :] - R[i, :], ord=p) ** 2 <= threshold)

    S = np.zeros(n, dtype=int)
    xi = np.zeros((K, L))

    for k in range(1, K + 1):
        card_T_V_minus_S = np.zeros(I_R.size)

        for ind in range(I_R.size):
            T_V_minus_S = T_v[:, ind] * (S == 0)
            card_T_V_minus_S[ind] = np.sum(T_V_minus_S)

        argmax = np.argmax(card_T_V_minus_S)

        S += k * (T_v[:, argmax] * (S == 0)).astype(int)

        for i in range(n):
            if S[i] == k:
                xi[k - 1, :] += R[i, :]

        xi[k - 1, :] /= np.sum(S == k)

    for i in range(n):
        if S[i] == 0:
            norms_k = np.zeros(K)
            for k in range(1, K + 1):
                norms_k[k - 1] = np.linalg.norm(R[i, :] - xi[k - 1, :], ord=p)

            S[i] = np.argmin(norms_k) + 1

    return S, xi

def log_(x):
    y = np.log(x)
    y[x < 1] = 0
    return y

def make_row(in_vec):
    if in_vec.ndim == 2 and in_vec.shape[1] == 1:  # Check if column vector
        return in_vec.T
    return in_vec

def measure_error(hat_V, sigma, K):
    n = len(sigma)
    hat_sigma = np.zeros(n, dtype=int)

    for k in range(1, K + 1):
        hat_V_k = np.array(hat_V[k - 1])
        if hat_V_k.ndim == 2 and hat_V_k.shape[1] == 1:  # If column vector
            hat_V_k = hat_V_k.T
        for v in hat_V_k.flatten():
            hat_sigma[v - 1] = k

    k_perm = np.array(list(np.permutations(range(1, K + 1))))
    len_perm = k_perm.shape[0]
    score_perm = np.zeros(len_perm)

    for perm_idx in range(len_perm):
        k_vec = k_perm[perm_idx]
        for v in range(n):
            if sigma[v] != k_vec[hat_sigma[v] - 1]:
                score_perm[perm_idx] += 1

    err = np.min(score_perm)
    best_idx = np.argmin(score_perm)

    k_vec = k_perm[best_idx]
    best_hat_sigma = np.zeros(n, dtype=int)
    for v in range(n):
        best_hat_sigma[v] = k_vec[hat_sigma[v] - 1]

    best_hat_V = [None] * K
    for k in range(1, K + 1):
        best_hat_V[k_vec[k - 1] - 1] = hat_V[k - 1]

    return err, best_hat_V, best_hat_sigma

def min_w(idx, w):
    W_t = np.zeros(w, dtype=int)
    idx_ = idx.copy()

    for j in range(w):
        W_t[j] = rand_argmin(idx_)
        idx_[W_t[j]] = np.inf

    return W_t

def rand_argmax(vec):
    # Find the index of the maximum value
    max_val = np.max(vec)
    argmax_idx = np.argmax(vec)
    
    # Find all indices where the value equals the maximum
    all_argmax = np.where(vec == max_val)[0]
    
    # If there are multiple indices with the maximum value, randomly pick one
    if len(all_argmax) > 1:
        argmax_idx = np.random.choice(all_argmax)
    
    return argmax_idx

def rand_argmin(vec):
    # Find the index of the minimum value
    min_val = np.min(vec)
    argmin_idx = np.argmin(vec)
    
    # Find all indices where the value equals the minimum
    all_argmin = np.where(vec == min_val)[0]
    
    # If there are multiple indices with the minimum value, randomly pick one
    if len(all_argmin) > 1:
        argmin_idx = np.random.choice(all_argmin)
    
    return argmin_idx
    
def update_wgt(y_i, q_i, P_hat, sigma_i, t, ucbconstant):
    """
    Simplified weight update: assume h_i_ = 1, so weights depend on squared differences of P_hat clusters.
    """
    K, L = P_hat.shape
    wgt = np.ones(L)
    wgt_val = np.inf
    k = sigma_i
    # With h_i_ always 1, q_i_k simplifies to P_hat[k]
    q_i_k = P_hat[k, :]
    
    # Compare q_i_k to each other cluster P_hat[k_]
    for k_ in range(K):
        if k_ == k:  # skip if same cluster
            continue
        diff_sq = (q_i_k - P_hat[k_, :]) ** 2
        wgt_val_k = y_i @ diff_sq
        if wgt_val_k <= wgt_val:
            wgt = np.maximum(diff_sq, 1e-5)
            wgt_val = wgt_val_k
    
    return wgt

