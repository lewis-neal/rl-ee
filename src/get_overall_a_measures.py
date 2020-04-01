import numpy as np
from a_measure import a_measure
import csv, sys

args = sys.argv[1:]

env_names = ['Acrobot-v1', 'CartPole-v1', 'MountainCar-v0', 'Pendulum-v0', \
'Copy-v0', 'DuplicatedInput-v0', 'RepeatCopy-v0', 'Reverse-v0', 'ReversedAddition-v0', 'ReversedAddition3-v0', \
'Blackjack-v0', 'Roulette-v0', 'FrozenLake-v0', 'FrozenLake8x8-v0', 'NChain-v0', 'Taxi-v3']

action_selection_names = ['boltzmann', 'controlability', 'epsilon-greedy', 'mbie-eb', 'random', 'ucb-1', 'vdbe']
base_dir = args[0] + 'data/'
measures = []
data_file_name = 'normal_reward_results.csv'
working_list = action_selection_names

for a in action_selection_names:
    for b in working_list:
        print(a + ' with ' + b)
        if a == b:
            continue
        results_a = np.array([])
        results_b = np.array([])
        for env in env_names:
            env_dir = base_dir + env + '/'
            results_a = np.concatenate((results_a, np.loadtxt(env_dir + a + '/' + data_file_name, delimiter=',')))
            results_b = np.concatenate((results_b, np.loadtxt(env_dir + b + '/' + data_file_name, delimiter=',')))
        am = a_measure(results_a, results_b)
        print(am)
        measure = [a, b, str(am)]
        measures.append(measure)
    working_list = working_list[1:]
with open(base_dir + '/measures.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(measures)
