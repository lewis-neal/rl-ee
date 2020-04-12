from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import csv, os

with open('data/env_stats.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

temp = []
for row in data:
    if row[2] == 'random-play':
        temp.append(row)
    elif row[1] == 'random-play':
        row[3] = 1 - float(row[3])
        row[1] = row[2]
        row[2] = 'random-play'
        temp.append(row)
data = temp


envs = {}
os.makedirs('graphs/vs-random-play', exist_ok=True)

# Example data
for row in data:
    if row[0] in envs:
        envs[row[0]].append(row[1:])
    else:
        envs[row[0]] = [row[1:]]
for env in envs:
    exps = []
    a = []
    for exp in envs[env]:
        exps.append(exp[0])
        a.append(float(exp[2]))
    x_pos = np.arange(len(exps))
    exps = tuple(exps)
    colours = []
    for val in a:
        if val > 0.71 or val < 0.29:
            colours.append('red')
        elif val > 0.64 or val < 0.36:
            colours.append('yellow')
        elif val > 0.56 or val < 0.44:
            colours.append('green')
        else:
            colours.append('blue')

    plt.figure(figsize=(10, 6))
    plt.bar(x_pos, a, align='center', color=colours)
    plt.xlabel('Exploration Strategy')
    plt.ylabel('A')
    plt.title(env + ': Performance vs Random-Play')
    plt.xticks(x_pos, exps)
    plt.ylim(0, 1)
    plt.axhline(0.5, color='black', label='Equal', linestyle='--')
    plt.grid(axis='y')
    legend_data = [
        Patch(facecolor='red', label='Large Effect'),
        Patch(facecolor='yellow', label='Medium Effect'),
        Patch(facecolor='green', label='Small Effect'),
        Patch(facecolor='blue', label='No Effect'),
    ]
    plt.legend(handles=legend_data, loc=1, title='Bar Colour')
    plt.savefig('graphs/vs-random-play/' + env)
