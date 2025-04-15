"""
perform numerical experiments for final paper 
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from algorithms_gaussian import *
from algorition_gaussian_full_sampling import *
from toolbox import *
import multiprocessing as mp
mp.set_start_method('spawn', force=True)


import time
import csv

# number of Monte-Carlo runs for each experiments
K = 500

# Experiment 3: constant gap, vayring proportion

[n_3, d_3] = [1000, 1000]
theta_sz = int(np.log(n_3)/np.log(2))
theta_vec = np.zeros(theta_sz)
for i in range(theta_sz):
    theta_vec[i] = 2**i
theta_vec = theta_vec.astype(int)
s_3 = 100
h_3 = 1.5
t_min = 1*10**7
t_max = 100 * 10**7
t_grd_sz = 100

def simulation_iteration_4(seed):
    """
    for Experiment 3
    
    run cr, cbc and cluster algorithm to obtain respective errors and budgets
    for different value of theta
    
    """
    delta = .8
    np.random.seed(seed)
    # initialize output data
    cr_budgets = np.zeros(theta_sz)
    cr_errors = np.zeros(theta_sz)
    cbc_budgets = np.zeros(theta_sz)
    cbc_errors = np.zeros(theta_sz)
    cluster_budgets = np.zeros(theta_sz)
    cluster_errors = np.zeros(theta_sz)
    # iterate over grid
    for i in range(theta_sz):
        # setup matrix for experiment, depending on sparsity
        true_clusters_3 = np.zeros(n_3)
        true_clusters_3[range(theta_vec[i])] = np.zeros(theta_vec[i]) + 1
        true_clusters_3 = true_clusters_3.astype(int)
        np.random.shuffle(true_clusters_3)
        i_a = np.argmax(true_clusters_3)
        i_b = np.argmin(true_clusters_3)
        m_a = np.zeros(d_3)
        m_b = np.zeros(d_3)
        m_b[range(s_3)] = np.zeros(s_3) + h_3
        M = np.zeros([n_3,d_3])
        for j in range(n_3):
            if true_clusters_3[j] == 1:
                M[j,:] = m_a
            else:
                M[j,:] = m_b
        [cr_trial_arms, cr_trial_budget] = cr_gaussian(M, delta/2)
        # store budget and error
        cr_budgets[i] = cr_trial_budget
        if true_clusters_3[0] == true_clusters_3[cr_trial_arms[0]]:
            cr_errors[i] = 1
        if true_clusters_3[0] == 0:
            [cbc_trial_clusters, cbc_trial_budget] = cbc_gaussian(M, 
                                                                  i_a, delta/2)
        else:
            [cbc_trial_clusters, cbc_trial_budget] = cbc_gaussian(M, 
                                                                  i_b, delta/2)
        # perform cbc, store budget and error
        cbc_budgets[i] = cbc_trial_budget
        cbc_errors[i] = 1-clusters_equal(true_clusters_3, 
                                              cbc_trial_clusters)
        # perform cluster
        [trial_clusters, trial_budgets] = cluster_gaussian(M, delta)
        # store budget and error
        cluster_budgets[i] = trial_budgets
        cluster_errors[i] = 1-clusters_equal(true_clusters_3, 
                                                  trial_clusters)
    return(cr_errors, cr_budgets, cbc_errors, cbc_budgets, cluster_errors, 
           cluster_budgets)
     
def simulation_iteration_5(seed):
    km_error = np.zeros([theta_sz,t_grd_sz])
    for i in range(theta_sz):
        # setup matrix for experiment, depending on sparsity
        true_clusters_3 = np.zeros(n_3)
        true_clusters_3[range(theta_vec[i])] = np.zeros(theta_vec[i]) + 1
        true_clusters_3 = true_clusters_3.astype(int)
        np.random.shuffle(true_clusters_3)
        i_a = np.argmax(true_clusters_3)
        i_b = np.argmin(true_clusters_3)
        m_a = np.zeros(d_3)
        m_b = np.zeros(d_3)
        m_b[range(s_3)] = np.zeros(s_3) + h_3
        M = np.zeros([n_3,d_3])
        for j in range(n_3):
            if true_clusters_3[j] == 1:
                M[j,:] = m_a
            else:
                M[j,:] = m_b
        for j in range(t_grd_sz):
            budget = int(t_min + j*(t_max-t_min)/(t_grd_sz-1))
            trial_clusters = kmeans_budget(M, budget)
            km_error[i,j] = clusters_equal(true_clusters_3, trial_clusters)
    return km_error
        
        




if __name__ == "__main__":
    # parameters
    num_processes = min(mp.cpu_count(),16)  # Number of processes to run in parallel
    seeds = range(K)  # Example seeds for reproducibility

    # start timing
    start_time = time.time()

    # run the function in parallel
    with mp.Pool(processes=num_processes) as pool:
        #results = pool.map(simulation_iteration_4, seeds)
        results_km = pool.map(simulation_iteration_5, seeds)

    # end timing
    end_time = time.time()
    time_spent = end_time - start_time

    # store results in arrays
    cr_errors = np.zeros([theta_sz,K])
    cr_budgets = np.zeros([theta_sz, K])
    cbc_errors = np.zeros([theta_sz,K])
    cbc_budgets = np.zeros([theta_sz,K])
    cluster_errors = np.zeros([theta_sz,K])
    cluster_budgets = np.zeros([theta_sz,K])
    km_errors = np.zeros([theta_sz, t_grd_sz,K])
    
    for k in range(K):
        # cr_errors[:, k] = results[k][0]
        # cr_budgets[:, k] = results[k][1]
        # cbc_errors[:, k] = results[k][2]
        # cbc_budgets[:, k] = results[k][3]
        # cluster_errors[:, k] = results[k][4]
        # cluster_budgets[:, k] = results[k][5]
        km_errors[:,:, k] = results_km[k]
        
    subfolder = os.path.join(os.getcwd(), "results")
    os.makedirs(subfolder, exist_ok=True)

    cr_errors_file = os.path.join(subfolder, "3_exp_cr_errors.csv")
    cr_budgets_file = os.path.join(subfolder, "3_exp_cr_budgets.csv")
    cbc_errors_file = os.path.join(subfolder, "3_exp_cbc_errors.csv")
    cbc_budgets_file = os.path.join(subfolder, "3_exp_cbc_budgets.csv")
    cluster_errors_file = os.path.join(subfolder, "3_exp_cluster_errors.csv")
    cluster_budgets_file = os.path.join(subfolder, "3_exp_cluster_budgets.csv")
    
    # with open(cr_errors_file, mode="w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow([f"{i+1}" for i in range(theta_sz)])  
    #     writer.writerows(cr_errors.T)  
    # with open(cr_budgets_file, mode="w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow([f"{i+1}" for i in range(theta_sz)])  
    #     writer.writerows(cr_budgets.T)  
    # with open(cbc_errors_file, mode="w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow([f"{i+1}" for i in range(theta_sz)]) 
    #     writer.writerows(cbc_errors.T) 
    # with open(cbc_budgets_file, mode="w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow([f"{i+1}" for i in range(theta_sz)]) 
    #     writer.writerows(cbc_budgets.T) 
    # with open(cluster_errors_file, mode="w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow([f"{i+1}" for i in range(theta_sz)])  
    #     writer.writerows(cluster_errors.T)  
    # with open(cluster_budgets_file, mode="w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow([f"{i+1}" for i in range(theta_sz)]) 
    #     writer.writerows(cluster_budgets.T)  
        
    for t in range(t_grd_sz):
        km_error_file_name = f"3_exp_{t}_km_errors.csv"
        km_error_file = os.path.join(subfolder, km_error_file_name)
        with open(km_error_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"{i+1}" for i in range(theta_sz)])  
            writer.writerows(km_errors[:,t,:].T)  
