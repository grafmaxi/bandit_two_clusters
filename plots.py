"""
using the CSV files from results to create plots

"""
import numpy as np
import matplotlib.pyplot as plt

# IMPORTANT: copy parameters from experiments.py
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

# additional parameters for plots
# quantiles
ql = 0.05
qu = 0.95
# x-margins
spars_min = 1
spars_max = 8


# setup respective grids
time_grid = np.zeros(time_grid_size)
for t_num in range(time_grid_size):
    temp = t_min + (t_max-t_min)*(t_num/(time_grid_size-1))
    time_grid[t_num] = n_1*d_1*int(temp/(n_1*d_1))
    
spars_grid = np.zeros(spars_grid_size)
for s_ind in range(spars_grid_size):
    spars_grid[s_ind] = int((d_1-1)*s_ind/(spars_grid_size-1)+1)

# restore data from csv files
cr_budgets = np.genfromtxt('results/1_exp_cr_budgets.csv',
                           delimiter=',', skip_header=1)
cbc_budgets = np.genfromtxt('results/1_exp_cbc_budgets.csv',
                            delimiter=',', skip_header=1)
cluster_budgets = np.genfromtxt('results/1_exp_cluster_budgets.csv',
                                delimiter=',', skip_header=1)
cr_errors = np.genfromtxt('results/1_exp_cr_errors.csv',
                          delimiter=',', skip_header=1)
cbc_errors = np.genfromtxt('results/1_exp_cbc_errors.csv',
                           delimiter=',', skip_header=1)
cluster_errors = np.genfromtxt('results/1_exp_cluster_errors.csv',
                               delimiter=',', skip_header=1)
cr_budgets_means = np.zeros(spars_grid_size)
cr_budgets_upper_quant = np.zeros(spars_grid_size)
cr_budgets_lower_quant = np.zeros(spars_grid_size)
cr_errors_means = np.zeros(spars_grid_size)
cbc_budgets_means = np.zeros(spars_grid_size)
cbc_budgets_upper_quant = np.zeros(spars_grid_size)
cbc_budgets_lower_quant = np.zeros(spars_grid_size)
cbc_errors_means = np.zeros(spars_grid_size)
cluster_budgets_means = np.zeros(spars_grid_size)
cluster_budgets_upper_quant = np.zeros(spars_grid_size)
cluster_budgets_lower_quant = np.zeros(spars_grid_size)
cluster_errors_means = np.zeros(spars_grid_size)
for s_ind in range(spars_grid_size):
    cr_budgets_means[s_ind] = np.mean(cr_budgets[:,s_ind])
    [qlow, qhigh] = np.quantile(cr_budgets[:,s_ind], [ql, qu])
    cr_budgets_lower_quant[s_ind] = qlow
    cr_budgets_upper_quant[s_ind] = qhigh
    cr_errors_means[s_ind] = np.mean(cr_errors[:,s_ind])
    cbc_budgets_means[s_ind] = np.mean(cbc_budgets[:,s_ind])
    [qlow, qhigh] = np.quantile(cbc_budgets[:,s_ind], [ql, qu])
    cbc_budgets_lower_quant[s_ind] = qlow
    cbc_budgets_upper_quant[s_ind] = qhigh
    cbc_errors_means[s_ind] = np.mean(cbc_errors[:,s_ind])
    temp_budgets = cluster_budgets[:,s_ind]
    success_budgets = temp_budgets[~np.isnan(temp_budgets)]
    cluster_budgets_means[s_ind] = np.mean(success_budgets)
    [qlow, qhigh] = np.quantile(success_budgets, [ql, qu])
    cluster_budgets_lower_quant[s_ind] = qlow
    cluster_budgets_upper_quant[s_ind] = qhigh
    cluster_errors_means[s_ind] = np.mean(cluster_errors[:,s_ind])
    
    
K = cr_budgets.shape[0]
    
km_errors = np.zeros([spars_grid_size, time_grid_size,K])
for t_num in range(time_grid_size):
    filename =  f"results/1_exp_{t_num}_km_errors.csv"
    km_errors[:, t_num, :] = np.genfromtxt(filename,
                                           delimiter=',', skip_header=1).T
km_error_means = np.zeros([spars_grid_size, time_grid_size])
for s_ind in range(spars_grid_size):
    for t_num in range(time_grid_size):
        km_error_means[s_ind,t_num] = np.mean(km_errors[s_ind,t_num,:])
km_budgets = np.zeros(spars_grid_size) 
for s_ind in range(spars_grid_size):
    for t_num in range(time_grid_size):
        if km_error_means[s_ind, t_num] < 0.01:
            km_budgets[s_ind] = time_grid[t_num]
            break
errors_2 = np.zeros([n_num, K, delta_num])
budgets_2 = np.zeros([n_num, K, delta_num])
for i in range(delta_num):
    filename =  f"results/2_exp_{i}_cluster_errors.csv"
    errors_2[:, :, i] = np.genfromtxt(filename,
                                      delimiter=',', skip_header=1).T
    filename =  f"results/2_exp_{i}_cluster_budgets.csv"
    budgets_2[:, :, i] = np.genfromtxt(filename,
                                       delimiter=',', skip_header=1).T

mean_errors_2 = np.zeros([n_num, delta_num])
mean_budgets_2 = np.zeros([n_num, delta_num])
upper_budgets_2 = np.zeros([n_num, delta_num])
lower_budgets_2 = np.zeros([n_num, delta_num])

for i in range(n_num):
    for j in range(delta_num):
        mean_errors_2[i,j] = np.mean(errors_2[i,:,j])
        vec = budgets_2[i,:,j]
        clean_vec = vec[~np.isnan(vec)]
        mean_budgets_2[i,j] = np.mean(clean_vec)
        [qlow, qhigh] = np.quantile(clean_vec, [ql, qu])
        upper_budgets_2[i,j] = qhigh
        lower_budgets_2[i,j] = qlow


# plot for experiment 1
plt.figure(dpi=300)   
plt.plot(spars_grid[range(spars_min,spars_max)], 
         cbc_budgets_means[range(spars_min,spars_max)], 
         label='Cluster', color='black', linewidth=3)
plt.plot(spars_grid[range(spars_min,spars_max)], 
         cluster_budgets_means[range(spars_min,spars_max)], 
         label='CBC', color='red', linewidth=2, linestyle = '--')
plt.plot(spars_grid[range(spars_min,spars_max)],
         cr_budgets_means[range(spars_min,spars_max)], 
         label='CR', color='blue', linewidth=2, linestyle = '--')
plt.plot(spars_grid[range(spars_min,spars_max)],
         km_budgets[range(spars_min,spars_max)],
         label='KMeans', color='black', linewidth=2, linestyle = '--')


plt.fill_between(spars_grid[range(spars_min,spars_max)],
                 cluster_budgets_lower_quant[range(spars_min,spars_max)], 
                 cluster_budgets_upper_quant[range(spars_min,spars_max)], 
                 color='black', alpha=0.2, label='_nolegend_')


plt.xscale("log")
plt.yscale("log")

plt.xlabel('sparsity', fontsize=10)
plt.ylabel('average budget', fontsize=10)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

# Show the plot
plt.savefig("plot_1.pdf", dpi=300)


# plot for experiment 2
styles = [ "--", "--", "--", "-"]
colors = ["violet", "orange", "blue", "red"]
widths = [1, 1, 1, 2]
plt.figure(dpi=300)  
plt.plot(np.array(range(5000)), np.array(range(5000))**2,
         label = r"$n^2$", color = "black", linewidth = 1)      
plt.fill_between(n_array, lower_budgets_2[:,3], upper_budgets_2[:,3], 
                 color='red', alpha=0.2, label='_nolegend_') 
for i in range(delta_num):
    plt.plot(n_array, mean_budgets_2[:,i], label=fr'$\delta$={delta_array[i]}', 
             color=colors[i], linewidth= widths[i], linestyle = styles[i])

plt.xlabel(r'number of items $n$', fontsize=10)
plt.ylabel('average budget', fontsize=10)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

plt.savefig("plot_2.pdf", dpi=300)

