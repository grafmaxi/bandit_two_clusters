"""
Analysis of results from Experiment 4: Comparison of Adaptive Sampling and Bernoulli Sequential Algorithms.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from src.configs.config_4 import config_4
import pandas as pd

with open('src/results/results_4.pkl', 'rb') as f:
    data = pickle.load(f)

feature_grid = config_4["feature_grid"]
feature_size = len(feature_grid)
monte_carlo_runs = config_4["monte_carlo_runs"]
sparsity_grid = config_4["sparsity_grid"]
cluster_budgets = [[] for _ in range(feature_size)]
cluster_errors = np.zeros((feature_size, monte_carlo_runs))
qlow = 0.05
qhigh = 0.95

# Load the CSV file containing error rates for adaptive sampling
data_adaptive = pd.read_csv('src/results/err_rates.csv', header=None).values

for k in range(monte_carlo_runs):
    cluster_errors[:, k] = data[k][0]
    # Only append budget if error is 0
    for i in range(feature_size):
        if cluster_errors[i, k] == 0:
            cluster_budgets[i].append(data[k][1][i])

# Convert list of lists to numpy array for plotting
mean_cluster_budgets = np.array(
    [np.mean(budgets) if budgets else 0 for budgets in cluster_budgets])
mean_adaptive = np.mean(1 * (data_adaptive > 0), axis=0)
mean_adaptive_partial = np.mean(data_adaptive, axis=0)
mean_cluster_errors = np.mean(cluster_errors, axis=1)
lower_quantiles_cluster_budgets = np.array(
    [np.quantile(budgets, qlow) if budgets else 0 for budgets in cluster_budgets])
upper_quantiles_cluster_budgets = np.array(
    [np.quantile(budgets, qhigh) if budgets else 0 for budgets in cluster_budgets])

plt.figure(dpi=300, figsize=(10, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plot error rates on left y-axis
ax1.plot(
    feature_grid,
    mean_adaptive,
    label='Error rate for Adaptive Clustering',
    color='blue',
    linewidth=2)
ax1.plot(
    feature_grid,
    mean_cluster_errors,
    label='Error rate for BanditClustering',
    color='red',
    linewidth=2)
ax1.set_ylabel('Error Rate', color='black', fontsize=16)
ax1.set_ylim(bottom=0, top=0.6)
ax1.tick_params(axis='y', labelcolor='black', labelsize=14)
ax1.set_xlabel('Number of features', fontsize=16)
ax1.tick_params(axis='x', labelsize=14)

# Plot budgets on right y-axis
ax2.plot(
    feature_grid,
    mean_cluster_budgets,
    label='Average budget for BanditClustering',
    color='red',
    linewidth=2,
    linestyle=':')
ax2.plot(
    feature_grid,
    np.full(
        feature_size,
        400000),
    label='Budget used for Adaptive Clustering',
    color='blue',
    linewidth=2,
    linestyle=':')
ax2.set_ylabel('Budget', color='black', fontsize=16)
ax2.tick_params(axis='y', labelcolor='black', labelsize=14)
ax2.set_ylim(bottom=0, top=450000)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1,
    labels1,
    bbox_to_anchor=(
        0.05,
        0.65),
    loc='center left',
    title='Error Rates',
    fontsize=14,
    title_fontsize=16)
ax2.legend(
    lines2,
    labels2,
    bbox_to_anchor=(
        0.4,
        0.2),
    loc='center left',
    title='Budgets',
    fontsize=14,
    title_fontsize=16)
plt.tight_layout()
plt.savefig('src/results/plot_4.pdf', dpi=300)
