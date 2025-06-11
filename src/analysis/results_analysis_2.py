import pickle
import numpy as np
from src.configs.config_2 import config_2
import matplotlib.pyplot as plt

with open('src/results/results_2.pkl', 'rb') as f:
    data = pickle.load(f)


monte_carlo_runs = config_2["monte_carlo_runs"]
num_items_array = config_2["num_items_array"]
delta_array = config_2["delta_array"]
qhigh = 0.95
qlow = 0.05

errors = np.zeros((len(num_items_array), len(delta_array), monte_carlo_runs))
budgets = [[[] for _ in range(len(delta_array))]
           for _ in range(len(num_items_array))]

for k in range(monte_carlo_runs):
    errors[:, :, k] = data[k][0]
    for i in range(len(num_items_array)):
        for j in range(len(delta_array)):
            if errors[i, j, k] == 0:
                budgets[i][j].append(data[k][1][i, j])

styles = ["--", "--", "--", "-"]
colors = ["violet", "orange", "blue", "red"]
widths = [1, 1, 1, 2]
plt.figure(dpi=300)
plt.plot(np.array(range(num_items_array[-1])),
         np.array(range(num_items_array[-1]))**2,
         label=r"$n^2$",
         color="black",
         linewidth=1)

for i in range(len(delta_array)):
    temp_mean_budgets = np.zeros(len(num_items_array))
    for j in range(len(num_items_array)):
        temp_mean_budgets[j] = np.mean(budgets[j][i])
    plt.plot(
        num_items_array,
        temp_mean_budgets,
        label=fr'$\delta$={delta_array[i]}',
        color=colors[i],
        linewidth=widths[i],
        linestyle=styles[i])
lower_quant = np.zeros(len(num_items_array))
upper_quant = np.zeros(len(num_items_array))
for j in range(len(num_items_array)):
    lower_quant[j] = np.quantile(budgets[j][3], qlow)
    upper_quant[j] = np.quantile(budgets[j][3], qhigh)

plt.fill_between(
    num_items_array,
    lower_quant,
    upper_quant,
    color='red',
    alpha=0.2,
    label=r'$(5\%, 95\%)$-confidence interval' +
    '\n' +
    r'for BanditClustering with $\delta=0.05$')

plt.xlabel(r'number of items $n$', fontsize=10)
plt.ylabel('average budget', fontsize=10)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

plt.savefig('src/results/plot_2.pdf', dpi=300)
