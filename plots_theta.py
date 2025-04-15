"""
using the CSV files from results to create plots

"""
import numpy as np
import matplotlib.pyplot as plt

[n_3, d_3] = [1000, 1000]
t_min = 1*10**7
t_max = 10 * 10**7
t_grd_sz = 10

theta_sz = int(np.log(n_3)/np.log(2))
theta_vec = np.zeros(theta_sz)
for i in range(theta_sz):
    theta_vec[i] = 2**i/n_3


[ql, qu] = [0.05, 0.95]
cr_budgets = np.genfromtxt('results/3_exp_cr_budgets.csv',
                           delimiter=',', skip_header=1)
cbc_budgets = np.genfromtxt('results/3_exp_cbc_budgets.csv',
                            delimiter=',', skip_header=1)
cluster_budgets = np.genfromtxt('results/3_exp_cluster_budgets.csv',
                                delimiter=',', skip_header=1)
cr_errors = np.genfromtxt('results/3_exp_cr_errors.csv',
                          delimiter=',', skip_header=1)
cbc_errors = np.genfromtxt('results/3_exp_cbc_errors.csv',
                           delimiter=',', skip_header=1)
cluster_errors = np.genfromtxt('results/3_exp_cluster_errors.csv',
                               delimiter=',', skip_header=1)
cr_budgets_means = np.zeros(theta_sz)
cr_budgets_upper_quant = np.zeros(theta_sz)
cr_budgets_lower_quant = np.zeros(theta_sz)
cr_errors_means = np.zeros(theta_sz)
cbc_budgets_means = np.zeros(theta_sz)
cbc_budgets_upper_quant = np.zeros(theta_sz)
cbc_budgets_lower_quant = np.zeros(theta_sz)
cbc_errors_means = np.zeros(theta_sz)
cluster_budgets_means = np.zeros(theta_sz)
cluster_budgets_upper_quant = np.zeros(theta_sz)
cluster_budgets_lower_quant = np.zeros(theta_sz)
cluster_errors_means = np.zeros(theta_sz)
for s_ind in range(theta_sz):
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

    
km_errors = np.zeros([theta_sz, t_grd_sz,K])
for t_num in range(t_grd_sz):
    filename =  f"results/3_exp_{t_num}_km_errors.csv"
    km_errors[:, t_num, :] = np.genfromtxt(filename,
                                           delimiter=',', skip_header=1).T
km_error_means = np.zeros([theta_sz, t_grd_sz])
for s_ind in range(theta_sz):
    for t_num in range(t_grd_sz):
        km_error_means[s_ind,t_num] = np.mean(km_errors[s_ind,t_num,:])
km_budgets = np.zeros(theta_sz) 
for s_ind in range(theta_sz):
    for t_num in range(t_grd_sz):
        if km_error_means[s_ind, t_num] < 0.01:
            km_budgets[s_ind] = int(t_min + t_num*(t_max-t_min)/(t_grd_sz-1))
            break

plt.figure(dpi=300)   
plt.plot(theta_vec, 
         cluster_budgets_means, 
         label='Cluster', color='black', linewidth=3)
plt.plot(theta_vec, 
         cbc_budgets_means, 
         label='CBC', color='red', linewidth=2, linestyle = '--')
plt.plot(theta_vec,
         cr_budgets_means, 
         label='CR', color='blue', linewidth=2, linestyle = '--')

plt.xscale("log")
plt.yscale("log")

plt.xlabel('theta', fontsize=10)
plt.ylabel('average budget', fontsize=10)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

# Show the plot
plt.savefig("plot_3.pdf", dpi=300)