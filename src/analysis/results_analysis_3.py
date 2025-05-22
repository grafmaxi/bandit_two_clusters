import pickle
import numpy as np
import matplotlib.pyplot as plt
from src.configs.config_3 import config_3


with open('src/results/results_3.pkl', 'rb') as f:
    data = pickle.load(f)

monte_carlo_runs = config_3["monte_carlo_runs"]
theta_size = config_3["theta_size"]
theta_array = config_3["theta_array"]
budget_steps = config_3["budget_steps"]
budget_grid = config_3["budget_grid"]
qlow = 0.05
qhigh = 0.95
thetaind_min = 1
thetaind_max = 8

cr_errors = np.zeros((theta_size,  monte_carlo_runs))
cr_budgets = np.zeros((theta_size,  monte_carlo_runs))
cbc_errors = np.zeros((theta_size,  monte_carlo_runs))
cbc_budgets = np.zeros((theta_size,  monte_carlo_runs))
km_errors = np.zeros((theta_size, budget_steps, monte_carlo_runs))
cluster_errors = np.zeros((theta_size,  monte_carlo_runs))
cluster_budgets = [[] for _ in range(theta_size)]


for k in range(monte_carlo_runs):
    km_errors[:,:,k] = data[k][6]
    cluster_errors[:,k] = data[k][4]
    for thetaind in range(theta_size):
        if cluster_errors[thetaind,k] == 0:
            cluster_budgets[thetaind].append(data[k][5][thetaind])
    cr_errors[:,k] = data[k][0]
    cr_budgets[:,k] = data[k][1]
    cbc_errors[:,k] = data[k][2]
    cbc_budgets[:,k] = data[k][3]
km_error_means = np.mean(km_errors, axis=2)
cluster_error_means = np.mean(cluster_errors, axis=1)
cluster_budgets_means = np.zeros(theta_size)
lower_quant_budgets = np.zeros(theta_size)
upper_quant_budgets = np.zeros(theta_size)
for i in range(theta_size):
    cluster_budgets_means[i] = np.mean(cluster_budgets[i])
    lower_quant_budgets[i] = np.quantile(cluster_budgets[i], qlow)
    upper_quant_budgets[i] = np.quantile(cluster_budgets[i], qhigh)

cr_error_means = np.mean(cr_errors, axis=1)
cr_budgets_means = np.mean(cr_budgets, axis=1)
cbc_error_means = np.mean(cbc_errors, axis=1)
cbc_budgets_means = np.mean(cbc_budgets, axis=1)
km_budgets = np.full(thetaind_max - thetaind_min + 1, budget_grid[-1])
for thetaind in range(thetaind_min, thetaind_max + 1):
    for tind in range(budget_steps):
        if km_error_means[thetaind, tind] < 0.02:
            km_budgets[thetaind-thetaind_min] = budget_grid[tind]
            break

for thetaind in range(thetaind_min, thetaind_max + 1):
    theta_array[thetaind] = theta_array[thetaind]**(-1)

plt.figure(dpi=300)   
plt.plot(theta_array[thetaind_min:(thetaind_max+1)], 
         cluster_budgets_means[thetaind_min:(thetaind_max+1)], 
         label='BanditClustering', color='black', linewidth=3)
plt.plot(theta_array[thetaind_min:(thetaind_max+1)], 
         cbc_budgets_means[thetaind_min:(thetaind_max+1)], 
         label='CBC', color='red', linewidth=2, linestyle = '--')
plt.plot(theta_array[thetaind_min:(thetaind_max+1)], 
         cr_budgets_means[thetaind_min:(thetaind_max+1)], 
         label='CR', color='blue', linewidth=2, linestyle = '--')
plt.plot(theta_array[thetaind_min:(thetaind_max+1)], 
         km_budgets, 
         label='KMeans', color='green', linewidth=2, linestyle = ':')
plt.fill_between(
    theta_array[thetaind_min:(thetaind_max+1)], 
    lower_quant_budgets[thetaind_min:(thetaind_max+1)], 
    upper_quant_budgets[thetaind_min:(thetaind_max+1)],
    color='black', alpha=0.2, label=r'$(5\%, 95\%)$-confidence interval' + '\n'+ 'for BanditClustering')
plt.plot(theta_array[thetaind_min:(thetaind_max+1)], 
         theta_array[thetaind_min:(thetaind_max+1)], 
         label=r'$1/\theta$', color='gray', linewidth=1)

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r'$1/\theta$', fontsize=10)
plt.ylabel('average budget', fontsize=10)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

# Show the plot
plt.savefig('src/results/plot_3.pdf', dpi = 300)






    









