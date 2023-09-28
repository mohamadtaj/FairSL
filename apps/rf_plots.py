import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import pandas as pd

metric = 'mcc'
num_nodes = 3
local_dict = {}
fair_dict = {}
unfair_dict = {}

for i in range(1, num_nodes+1):

    local_dict[i] = np.load('./results/local_node_'+str(i)+'_'+str(metric)+'.npy')
    fair_dict[i] = np.load('./results/fair_node_'+str(i)+'_'+str(metric)+'.npy')
    unfair_dict[i] = np.load('./results/unfair_node_'+str(i)+'_'+str(metric)+'.npy')

Node1 = pd.DataFrame({'Local': local_dict[1], 'Fair SL': fair_dict[1], 'Typical SL': unfair_dict[1]}).assign(Nodes=1)
Node2 = pd.DataFrame({'Local': local_dict[2], 'Fair SL': fair_dict[2], 'Typical SL': unfair_dict[2]}).assign(Nodes=2)
Node3 = pd.DataFrame({'Local': local_dict[3], 'Fair SL': fair_dict[3], 'Typical SL': unfair_dict[3]}).assign(Nodes=3)

cdf = pd.concat([Node1, Node2, Node3])
mdf = pd.melt(cdf, id_vars=['Nodes'], var_name=['Method'])

plt.figure(figsize=(6, 4 ), dpi = 80)
ax = sns.boxplot(x="Nodes", y="value", data=mdf, hue="Method")
if (metric == 'acc'):
    ax.set_ylabel("Accuracy",fontsize=12)
elif (metric == 'mcc'):
    ax.set_ylabel("MCC",fontsize=12)
ax.set_xlabel("Nodes",fontsize=12)
sns.move_legend(ax, "lower right")
#plt.ylim(50, 100)
plt.show()

plt.clf()
plt.close()

print('Comparing local and fair SL models at each node\n')
def ttest (dist1, dist2):
    t_stat, p_value = ttest_ind(dist1, dist2, alternative= 'less')
    print(f't statistic = {t_stat}')
    print(f'p_value = {p_value}')
    print()

for i in range (1, num_nodes+1):
    print(f'Node {i}:')
    ttest (local_dict[i], fair_dict[i])

print('--------------------------------')
print('ttest - Comparing fair SL models of nodes:\n')
for i in range (1, num_nodes):
    print(f'Node {i} and {i+1}:')
    ttest (fair_dict[i], fair_dict[i+1])
print(f'Node {1} and {3}:')
ttest (fair_dict[1], fair_dict[3])