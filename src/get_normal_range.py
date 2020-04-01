import numpy as np
import sys, csv

env_names = ['Acrobot-v1', 'CartPole-v1', 'MountainCar-v0', 'Pendulum-v0', \
'Copy-v0', 'DuplicatedInput-v0', 'RepeatCopy-v0', 'Reverse-v0', 'ReversedAddition-v0', 'ReversedAddition3-v0', \
'Blackjack-v0', 'Roulette-v0', 'FrozenLake-v0', 'FrozenLake8x8-v0', 'NChain-v0', 'Taxi-v3']

action_selection_names = ['boltzmann', 'controlability', 'epsilon-greedy', 'mbie-eb', 'random', 'ucb-1', 'vdbe']

args = sys.argv[1:]
base_dir = args[0] + 'data/'
env_vals = []

for env in env_names:
    maximum = -99999999
    minimum = 99999999
    for a in action_selection_names:
        data = np.loadtxt(base_dir + env + '/' + a + '/results.csv', delimiter=',')
        for row in data:
            maximum = max(maximum, row[0])
            minimum = min(minimum, row[0])
    env_vals.append([env, minimum, maximum])
with open(base_dir + '/normals.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(env_vals)
