import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 12})

metric = 'mcc'
num_nodes = 4
local_dict = {}
fair_dict = {}
unfair_dict = {}

for i in range(1, num_nodes + 1):
    local_dict[i] = np.load('./results/local_node_' + str(i) + '_' + str(metric) + '.npy')
    fair_dict[i] = np.load('./results/fair_node_' + str(i) + '_' + str(metric) + '.npy')
    unfair_dict[i] = np.load('./results/unfair_node_' + str(i) + '_' + str(metric) + '.npy')

# Create a list to hold all the DataFrames
data_frames = []

for i in range(1, num_nodes + 1):
    df = pd.DataFrame({'Local': local_dict[i], 'Fair SL': fair_dict[i], 'Standard SL': unfair_dict[i]}).assign(Nodes=i)
    data_frames.append(df)

# Concatenate all DataFrames
cdf = pd.concat(data_frames)
mdf = pd.melt(cdf, id_vars=['Nodes'], var_name=['Method'])

plt.figure(dpi=300)
ax = sns.boxplot(x="Nodes", y="value", data=mdf, hue="Method")
if metric == 'acc':
    ax.set_ylabel("Accuracy")
elif metric == 'mcc':
    ax.set_ylabel("MCC")
ax.set_xlabel("Nodes")
sns.move_legend(ax, "lower right")
plt.savefig("maternal_" + str(num_nodes) + ".eps", dpi=300, format='eps', bbox_inches='tight', pad_inches=0.2)
# plt.ylim(50, 100)
plt.show()

plt.clf()
plt.close()


def ttest (dist1, dist2):
    t_stat, p_value = ttest_ind(dist1, dist2, alternative= 'less')
    mean1 = np.mean(dist1)
    mean2 = np.mean(dist2)
    increase = ((mean2 - mean1) / mean1) * 100
    print("first mean: ", mean1)
    print("second mean: ", mean2)

    print("increase: ", increase)
    print(f't statistic = {t_stat}')
    print(f'p_value = {p_value}')
    print()

# for i in range (1, num_nodes+1):
#     print(f'Node {i}:')
#     ttest (local_dict[i], fair_dict[i])

print('----------------------------------------')
print('ttest - Comparing fair SL models of nodes:\n')
for i in range (1, num_nodes):
    print(f'Node {i} and {i+1}:')
    ttest (fair_dict[i], fair_dict[i+1])