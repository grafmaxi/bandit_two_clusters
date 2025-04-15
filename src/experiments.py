"""
perform numerical experiments for final paper 
"""

import numpy as np
from algorithms_gaussian import *
from algorition_gaussian_full_sampling import *
from toolbox import *
import multiprocessing as mp

mp.set_start_method('fork', force=True)



import time
import csv
import os

# number of Monte-Carlo runs for each experiments
K = 32

# Experiment 1: constant norm, varying sparsity, gaussian noise

# dimensions of M
[n_1, d_1] = [20, 1000]
# number of sparsity parameters we consider
spars_grid_size = 20
# time grid for KMeans by uniform sampling
time_grid_size = 10
t_min = 170000
t_max = 300000
# setting up clusters
theta_1 = .5
true_clusters_1 = np.random.permutation(1*(np.array(range(n_1)) < theta_1*n_1))
i_a = np.argmax(true_clusters_1)
i_b = np.argmin(true_clusters_1)
# norm of Delta
h_1 = 15

# Experiment 2: constant norm and sparsity, gaussian noise

# number of items n, d will multip * n 
n_array = np.array([100,200,500,1000,2000,5000])
multip = 10
d_array = multip * n_array
# different confidence parameters delta
delta_array = np.array([0.8,0.5,0.2,0.05])

n_num = len(n_array)
delta_num = len(delta_array)
# parameters for clusters and matrix M
theta = .5
s_2 = 10 
h_2 = 5 


def simulation_iteration_1(seed):
    """
    for Experiment 1
    
    run cr, cbc and cluster algorithm to obtain respective errors and budgets
    for different sparsity levels
    
    """
    delta = .8
    np.random.seed(seed)
    # initialize output data
    cr_budgets = np.zeros(spars_grid_size)
    cr_errors = np.zeros(spars_grid_size)
    cbc_budgets = np.zeros(spars_grid_size)
    cbc_errors = np.zeros(spars_grid_size)
    cluster_budgets = np.zeros(spars_grid_size)
    cluster_errors = np.zeros(spars_grid_size)
    # iterate over grid
    for s_ind in range(spars_grid_size):
        # setup matrix for experiment, depending on sparsity
        sparsity = int((d_1-1)*s_ind/(spars_grid_size-1)+1)
        m_a = np.zeros(d_1)
        m_b = np.zeros(d_1)
        m_b[range(sparsity)] = np.zeros(sparsity) + h_1/np.sqrt(sparsity)
        M = np.zeros([n_1,d_1])
        for i in range(n_1):
            if true_clusters_1[i] == 1:
                M[i,:] = m_a
            else:
                M[i,:] = m_b
        # perform cr
        [cr_trial_arms, cr_trial_budget] = cr_gaussian(M, delta/2)
        # store budget and error
        cr_budgets[s_ind] = cr_trial_budget
        if true_clusters_1[0] == true_clusters_1[cr_trial_arms[0]]:
            cr_errors[s_ind] = 1
        if true_clusters_1[0] == 0:
            [cbc_trial_clusters, cbc_trial_budget] = cbc_gaussian(M, 
                                                                  i_a, delta/2)
        else:
            [cbc_trial_clusters, cbc_trial_budget] = cbc_gaussian(M, 
                                                                  i_b, delta/2)
        # perform cbc, store budget and error
        cbc_budgets[s_ind] = cbc_trial_budget
        cbc_errors[s_ind] = 1-clusters_equal(true_clusters_1, 
                                             cbc_trial_clusters)
        # perform cluster
        [trial_clusters, trial_budgets] = cluster_gaussian(M, delta)
        # store budget and error
        cluster_budgets[s_ind] = trial_budgets
        cluster_errors[s_ind] = 1-clusters_equal(true_clusters_1, 
                                                 trial_clusters)
    return(cr_errors, cr_budgets, cbc_errors, cbc_budgets, cluster_errors, 
           cluster_budgets)

def simulation_iteration_2(seed):
    """
    for Experiment 1
    
    perform uniform sampling and kmeans for dfferent sparsity levels
    """
    np.random.seed(seed)
    # initialize output data
    km_budgets = np.zeros([spars_grid_size,time_grid_size])
    km_errors = np.zeros([spars_grid_size,time_grid_size])
    for s_ind in range(spars_grid_size):
        # setup matrix for experiment, depending on sparsity
        sparsity = int((d_1-1)*s_ind/(spars_grid_size-1)+1)
        m_a = np.zeros(d_1)
        m_b = np.zeros(d_1)
        m_b[range(sparsity)] = np.zeros(sparsity) + h_1/np.sqrt(sparsity)
        M = np.zeros([n_1,d_1])
        for i in range(n_1):
            if true_clusters_1[i] == 1:
                M[i,:] = m_a
            else:
                M[i,:] = m_b
        # perform algorithm for different budgets
        for t_num in range(time_grid_size):
            t = t_min + (t_max-t_min)*(t_num/(time_grid_size-1))
            km_budgets[s_ind,t_num] = int(t/(n_1*d_1))*(n_1*d_1)
            km_cluster = kmeans_budget(M, t) 
            km_errors[s_ind,t_num] = 1-clusters_equal(true_clusters_1, 
                                                      km_cluster)
    return km_errors, km_budgets

def simulation_iteration_3(seed):
    """
    for Experiment 2
    
    run cluster algorithm for matrices of different dimension and 
    """
    np.random.seed(seed)
    cluster_budgets = np.zeros([n_num, delta_num])
    cluster_errors = np.zeros([n_num, delta_num])
    for n_ind in range(n_num):
        n = n_array[n_ind]
        d = d_array[n_ind]
        m_a = np.zeros(d)
        m_b = np.zeros(d)
        m_b[range(s_2)] = np.zeros(s_2) + h_2
        M = np.zeros([n,d])
        clusters = np.random.permutation(1*(np.array(range(n)) < theta*n))
        for i in range(n):
            if clusters[i] == 1:
                M[i,:] = m_a
            else:
                M[i,:] = m_b
        for j in range(delta_num):
            [trial_clusters, trial_budgets] = cluster_gaussian(M, 
                                                               delta_array[j])
            cluster_budgets[n_ind,j] = trial_budgets
            cluster_errors[n_ind,j] = 1-clusters_equal(clusters, 
                                                       trial_clusters)
    return cluster_errors, cluster_budgets




if __name__ == "__main__":
    # parameters
    num_processes = min(mp.cpu_count(),16)  # Number of processes to run in parallel
    seeds = range(K)  # Example seeds for reproducibility

    # start timing
    start_time = time.time()

    # run the function in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(simulation_iteration_1, seeds)
        results_km = pool.map(simulation_iteration_2, seeds)
        results_2 = pool.map(simulation_iteration_3, seeds)

    # end timing
    end_time = time.time()
    time_spent = end_time - start_time

    # store results in arrays
    cr_errors = np.zeros([spars_grid_size,K])
    cr_budgets = np.zeros([spars_grid_size, K])
    cbc_errors = np.zeros([spars_grid_size,K])
    cbc_budgets = np.zeros([spars_grid_size,K])
    cluster_errors = np.zeros([spars_grid_size,K])
    cluster_budgets = np.zeros([spars_grid_size,K])
    km_budgets = np.zeros([spars_grid_size, time_grid_size,K])
    km_errors = np.zeros([spars_grid_size, time_grid_size,K])
    cluster_errors_2 = np.zeros([n_num,delta_num,K])
    cluster_budgets_2 = np.zeros([n_num,delta_num,K])

    
    for k in range(K):
        cr_errors[:, k] = results[k][0]
        cr_budgets[:, k] = results[k][1]
        cbc_errors[:, k] = results[k][2]
        cbc_budgets[:, k] = results[k][3]
        cluster_errors[:, k] = results[k][4]
        cluster_budgets[:, k] = results[k][5]
        km_errors[:,:, k] = results_km[k][0]
        km_budgets[:,:,k] = results_km[k][1]
        cluster_errors_2[:, :, k] = results_2[k][0]
        cluster_budgets_2[:,:,  k] = results_2[k][1]
        
    
    
    # storing results as CSV
    subfolder = os.path.join(os.getcwd(), "results")
    os.makedirs(subfolder, exist_ok=True)

    cr_errors_file = os.path.join(subfolder, "1_exp_cr_errors.csv")
    cr_budgets_file = os.path.join(subfolder, "1_exp_cr_budgets.csv")
    cbc_errors_file = os.path.join(subfolder, "1_exp_cbc_errors.csv")
    cbc_budgets_file = os.path.join(subfolder, "1_exp_cbc_budgets.csv")
    cluster_errors_file = os.path.join(subfolder, "1_exp_cluster_errors.csv")
    cluster_budgets_file = os.path.join(subfolder, "1_exp_cluster_budgets.csv")
    km_budgets_file = os.path.join(subfolder, "1_exp_km_budgets.csv")
    time_file = os.path.join(subfolder, "1_exp_time_spent.txt")
    
    with open(cr_errors_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"{i+1}" for i in range(spars_grid_size)])  
        writer.writerows(cr_errors.T)  
    with open(cr_budgets_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"{i+1}" for i in range(spars_grid_size)])  
        writer.writerows(cr_budgets.T)  
    with open(cbc_errors_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"{i+1}" for i in range(spars_grid_size)]) 
        writer.writerows(cbc_errors.T) 
    with open(cbc_budgets_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"{i+1}" for i in range(spars_grid_size)]) 
        writer.writerows(cbc_budgets.T) 
    with open(cluster_errors_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"{i+1}" for i in range(spars_grid_size)])  
        writer.writerows(cluster_errors.T)  
    with open(cluster_budgets_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"{i+1}" for i in range(spars_grid_size)]) 
        writer.writerows(cluster_budgets.T)  
    with open(km_budgets_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"{i+1}" for i in range(time_grid_size)])  
        writer.writerows(km_budgets[0,:,:].T) 
    for t in range(time_grid_size):
        km_error_file_name = f"1_exp_{t}_km_errors.csv"
        km_error_file = os.path.join(subfolder, km_error_file_name)
        with open(km_error_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"{i+1}" for i in range(spars_grid_size)])  
            writer.writerows(km_errors[:,t,:].T)  
    for j in range(delta_num):
        budget_file = os.path.join(subfolder,f"2_exp_{j}_cluster_budgets.csv")
        with open(budget_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"{i+1}" for i in range(n_num)])  
            writer.writerows(cluster_budgets_2[:,j,:].T) 
    for j in range(delta_num):
        error_file = os.path.join(subfolder,f"2_exp_{j}_cluster_errors.csv")
        with open(error_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"{i+1}" for i in range(n_num)])  
            writer.writerows(cluster_errors_2[:,j,:].T)  
    with open(time_file, mode="w") as f:
        f.write(f"Total time spent: {time_spent:.2f} seconds")