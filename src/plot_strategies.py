from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import csv, os

with open('data/env_stats.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

strats = ['boltzmann', 'controlability', 'epsilon-greedy', 'mbie-eb', 'random', 'ucb-1', 'vdbe']
working_list = strats
for strat in strats:
    working_list = working_list[1:]
    for strat2 in working_list:
        if strat == strat2:
            continue
        temp = []
        for row in data:
            if row[1] == strat and row[2] == strat2:
                temp.append(row)
            elif row[1] == strat2 and row[2] == strat:
                row[3] = 1 - float(row[3])
                temp_row = row[1]
                row[1] = row[2]
                row[2] = temp_row
                temp.append(row)

        graph_dir = 'graphs/between_strats/'
        os.makedirs(graph_dir, exist_ok=True)
        temp.reverse()

        colours = []
        envs = []
        a = []
        for row in temp:
            envs.append(row[0])
            val = float(row[3])
            a.append(val)
            if val > 0.71 or val < 0.29:
                colours.append('red')
            elif val > 0.64 or val < 0.36:
                colours.append('yellow')
            elif val > 0.56 or val < 0.44:
                colours.append('green')
            else:
                colours.append('blue')
        y_pos = np.arange(len(envs))
        plt.figure(figsize=(45, 30))
        plt.barh(y_pos, a, align='center', color=colours)
        plt.ylabel('Environment', fontsize=30)
        plt.xlabel('A', fontsize=30)
        if strat == 'random':
            name = 'Random-Search'
        else:
            name = strat.capitalize()
        if strat2 == 'random':
            name2 = 'Random-Search'
        else:
            name2 = strat2.capitalize()
        plt.title(name + ' vs ' + name2, fontsize=30)
        plt.yticks(y_pos, envs, fontsize=30)
        plt.xticks(fontsize=30)
        plt.xlim(0, 1)
        plt.axvline(0.5, color='black', label='Equal', linestyle='--')
        plt.grid(axis='x')
        legend_data = [
            Patch(facecolor='red', label='Large Effect'),
            Patch(facecolor='yellow', label='Medium Effect'),
            Patch(facecolor='green', label='Small Effect'),
            Patch(facecolor='blue', label='No Effect'),
        ]
        plt.legend(handles=legend_data, loc=1, title='Bar Colour', fontsize=30, title_fontsize=30)
        plt.savefig(graph_dir + strat + '_vs_' + strat2)# + '_' + str(loop))
        plt.close()
