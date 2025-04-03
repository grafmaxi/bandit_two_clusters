"""
Algorithms for binary clustering when the noise is standard-normal

"""
import numpy as np


def cesh_gaussian(m, rows, depth, budget):
    """
    performing sequential halving for a subsampled set of arms
    
    Parameters
    -------
    m: MATRIX OF ARMS MEANS.
    rows: ROWS TO SAMPLE INDICES FROM.
    depth: NUMBER OF HALVING STEPS.
    budget: MAXIMAL NUMBER OF SAMPLES.
    
    Returns
    -------
    indices: INDEX PAIR THAT REMAINS AFTER THE LAST HALVING STEP.
    cost: NUMBER OF SAMPLES THAT WERE REQUIRED.
    """
    d = m.shape[1]
    # subsample index-pairs
    indices = [choices(rows, k=2**depth), choices(range(d), k=2**depth)]
    cost = 0
    for l in range(depth):
        sample_num = int(budget/(2**(depth-l+1)*depth))
        # set sample effort
        step_means = np.zeros(2**(depth-l))
        for i in range(2**(depth-l)):
            step_means[i] = abs(
                np.random.normal(
                    m[0, indices[1][i]]
                    -m[indices[0][i], indices[1][i]],
                    np.sqrt(2/sample_num)))
            cost += 2*sample_num
        # eliminating indices corresponding to smallest means in absolute value
        index_sort = np.argsort(-step_means)
        indices[0] = [indices[0][i] for i in index_sort[: int(2**(depth-l-1))]]
        indices[1] = [indices[1][i] for i in index_sort[: int(2**(depth-l-1))]]
    return indices, cost

def cr_gaussian(m, delta):
    """
    finding an index from a different cluster than the index 0

    Parameters
    ----------
    m : MATRIX OF ARMS MEANS.
    delta : RISK LEVEL.

    Returns
    -------
    arm : ROW INDEX.
    count : NUMBER OF SAMPLES THAT WERE REQUIRED.

    """
    [n, d] = m.shape
    terminate = False
    k = 1
    # start with budget 2**1 and double at each iteration
    cost = 0
    while not terminate:
        l = 1
        while (l*2**(l+1) <= 2**(k+1) and
               l < int(
                   np.log(16*n*d*np.log(16*np.log(8*n*d)/delta))/np.log(2))+1):
            cand = cesh_gaussian(m, range(n), l, 2**(k+1))
            cost += cand[ 1 ]
            arm = cand[ 0 ]
            if (abs(
                    np.random.normal(m[0, arm[1]]-m[arm[0], arm[1]],
                                     np.sqrt(1/2**(k-1))))
                    > 2**((2-k)/2)*np.sqrt(np.log(k**3/(0.15*delta)))):
                cost += 2**(k+1)
                terminate = True
                break
            l += 1
            cost += 2**(k+1)
        k += 1
    return arm, cost

def cbc_gaussian(m, can_row_index, delta):
    """
    clusteering if we are given can_row_index from a different cluster than
    index 0

    Parameters
    ----------
    m : MATRIX OF ARM MEANS.
    can_row_index : ROW INDEX.
    delta : RISK LEVEK.

    Returns
    -------
    clusters: LABELS INDICES BY DETECTED CLUSTERS
    count: NUMBER OF SAMPLES THAT WERE REQUIRED.

    """
    [n, d] = m.shape
    cost = 0
    clusters = np.zeros(n)
    k = int(np.ceil(np.log(n)/np.log(2)))
    # start with budget 2**k and double at each iteration
    cost = 0
    while clusters[0] == 0:
        l = 1
        while (l*2**(l+1) <= 2**(k+1) and
                l < int(
                    np.log(
                        16*n*d*np.log(
                            16*np.log(8*n*d)/delta))/np.log(2))+1):
            can_col = cesh_gaussian(m, [can_row_index], l, 2**(k+1))
            cost += can_col[1]
            can_col_index = can_col[0][1]
            if (abs(
                    np.random.normal(int(2**k/n)*(m[0, can_col_index]
                                      - m[can_row_index,can_col_index]),
                                      np.sqrt(2*int(2**k/n))))
                    > 3*np.sqrt(
                        4*int(2**k/n)*np.log(n*k**3/(0.15*delta)))):
                clusters[0] = 1
                cost += 2*int(2**k/n)
                for ind in range(1, n):
                    if (abs(
                            np.random.normal(int(2**k/n)
                                             *(m[can_row_index, can_col_index]
                                               -m[ind, can_col_index]),
                                             np.sqrt(2*int(2**k/n))))
                            >np.sqrt(
                                4*int(2**k/n)*np.log(
                                    n*k**3/(0.15*delta)))):
                        clusters[ind] = 1
                        cost += 2*int(2**k/n)
                break
            cost += 2*int(2**k/n)
            l += 1
        k += 1
    return(clusters, cost)

def cluster_gaussian(m, delta):
    """
    clusteering algorithm, combining cr and cbc
    Parameters
    ----------
    m : MATRIX OF ARM MEANS.
    delta : RISK LEVEK.

    Returns
    -------
    clusters: LABELS INDICES BY DETECTED CLUSTERS
    count: NUMBER OF SAMPLES THAT WERE REQUIRED.

    """
    [can_row_indices, cr_cost] = cr_gaussian(m, delta/2)
    if(m[0,:]== m[can_row_indices[0],:]).all():
        return(np.nan, np.nan)
    else:
        [clusters, cbc_cost] = cbc_gaussian(m, can_row_indices[0], delta/2)
        cost = cbc_cost + cr_cost
        return(clusters, cost)
