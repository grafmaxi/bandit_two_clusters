import pickle
import numpy as np
from src.configs.config_1 import config_1
import matplotlib.pyplot as plt
with open('src/results/results_1.pkl', 'rb') as f:
    data = pickle.load(f)
    
monte_carlo_runs = config_1["monte_carlo_runs"]
sparsity_grid_size = config_1["num_sparsity_params"]
budget_steps = config_1["num_budget_steps"]
qlow = 0.05
qhigh = 0.95
sind_min = 1
sind_max = 8

cr_errors = np.zeros((sparsity_grid_size, monte_carlo_runs))
cr_budgets = np.zeros((sparsity_grid_size, monte_carlo_runs))
cbc_errors = np.zeros((sparsity_grid_size, monte_carlo_runs))
cbc_budgets = np.zeros((sparsity_grid_size, monte_carlo_runs))
cluster_errors = np.zeros((sparsity_grid_size, monte_carlo_runs))
cluster_budgets = [[] for _ in range(sparsity_grid_size)]
km_errors = np.zeros((sparsity_grid_size, budget_steps, monte_carlo_runs))
km_budgets = np.zeros((sparsity_grid_size, budget_steps, monte_carlo_runs))


for k in range(monte_carlo_runs):
    cr_errors[:, k] = data[k][0]
    cr_budgets[:, k] = data[k][1]
    cbc_errors[:, k] = data[k][2]
    cbc_budgets[:, k] = data[k][3]
    cluster_errors[:, k] = data[k][4]
    for sind in range(sparsity_grid_size):
        if cluster_errors[sind, k] == 0:
            cluster_budgets[sind].append(data[k][5][sind])
    km_errors[:, :, k] = data[k][6]
    km_budgets[:, :, k] = data[k][7]

cluster_budgets_mean = np.zeros(sparsity_grid_size)
cluster_budgets_lower_quantile = np.zeros(sparsity_grid_size)
cluster_budgets_upper_quantile = np.zeros(sparsity_grid_size)
for sind in range(sparsity_grid_size):
    cluster_budgets_mean[sind] = np.mean(cluster_budgets[sind])
    cluster_budgets_lower_quantile[sind] = np.quantile(cluster_budgets[sind], qlow)
    cluster_budgets_upper_quantile[sind] = np.quantile(cluster_budgets[sind], qhigh)

print(np.mean(cluster_errors, axis=1))

sparsity_grid = np.zeros(sparsity_grid_size)
for i in range(sparsity_grid_size):
    sparsity_grid[i] = int((config_1["num_features"] - 1) * i / (sparsity_grid_size - 1) + 1)

km_confidence_budget = np.zeros(sparsity_grid_size)

for sind in range(sparsity_grid_size):
    for budget_idx in range(budget_steps):
        if np.mean(km_errors[sind, budget_idx, :]) < 0.01:
            km_confidence_budget[sind] = km_budgets[sind, budget_idx, 0]
            break



plt.figure(dpi=300)   
plt.plot(sparsity_grid[sind_min:sind_max], 
         cluster_budgets_mean[sind_min:sind_max], 
         label='BanditClustering', color='black', linewidth=3)
plt.plot(sparsity_grid[sind_min:sind_max], 
         np.mean(cbc_budgets[sind_min:sind_max], axis=1), 
         label='CBC', color='red', linewidth=2, linestyle = '--')
plt.plot(sparsity_grid[sind_min:sind_max], 
         np.mean(cr_budgets[sind_min:sind_max], axis=1), 
         label='CR', color='blue', linewidth=2, linestyle = '--')
plt.plot(sparsity_grid[sind_min:sind_max], 
         km_confidence_budget[sind_min:sind_max], 
         label='K-means', color='green', linewidth=2, linestyle = ':')
plt.fill_between(sparsity_grid[sind_min:sind_max], 
                 cluster_budgets_lower_quantile[sind_min:sind_max], 
                 cluster_budgets_upper_quantile[sind_min:sind_max], 
                 color='black', alpha=0.2, label=r'$(5\%, 95\%)$-confidence interval' + '\n'+ 'for BanditClustering')
plt.plot(sparsity_grid[sind_min:sind_max], 
         sparsity_grid[sind_min:sind_max], 
         label=r'$s$', color='gray', linewidth=1)

plt.xscale("log")
plt.yscale("log")

plt.xlabel('sparsity', fontsize=10)
plt.ylabel('average budget', fontsize=10)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

# Show the plot
plt.savefig('src/results/plot_1.pdf', dpi=300)
