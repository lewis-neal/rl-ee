from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import csv, os, sys
strats = ['boltzmann', 'controlability', 'epsilon-greedy', 'mbie-eb', 'random-play', 'random-search', 'ucb-1', 'vdbe']
datas = []
data = []
env = 'roulette'

for strat in strats:
    with open('data/' + strat + '_' + env + '.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    temp = []
    for row in data:
        temp.append(float(row[0]))
    data = temp
    datas.append(data)

graph_dir = 'graphs/' + env

os.makedirs(graph_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.boxplot(datas)
plt.title(env.capitalize())
plt.xticks(np.arange(1, len(strats) + 1), strats)
plt.grid(axis='y')
plt.savefig(graph_dir + '/box-plots')
