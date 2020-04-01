import numpy as np
import sys, csv

env_names = ['Acrobot-v1', 'CartPole-v1', 'MountainCar-v0', 'Pendulum-v0', \
'Copy-v0', 'DuplicatedInput-v0', 'RepeatCopy-v0', 'Reverse-v0', 'ReversedAddition-v0', 'ReversedAddition3-v0', \
'Blackjack-v0', 'Roulette-v0', 'FrozenLake-v0', 'FrozenLake8x8-v0', 'NChain-v0', 'Taxi-v3']

action_selection_names = ['boltzmann', 'controlability', 'epsilon-greedy', 'mbie-eb', 'random', 'ucb-1', 'vdbe']

args = sys.argv[1:]
base_dir = args[0] + 'data/'
vals = {
    'Acrobot-v1': {
        'min': -200.0,
        'max': -87.0
    },
    'CartPole-v1': {
        'min': 8.0,
        'max': 85.0
    },
    'MountainCar-v0': {
        'min': -200.0,
        'max': -200.0
    },
    'Pendulum-v0': {
        'min': -1969.1814580136293,
        'max': -0.10779888241761089
    },
    'Copy-v0': {
        'min': -0.5,
        'max': 32.0
    },
    'DuplicatedInput-v0': {
        'min': -0.5,
        'max': 3.0
    },
    'RepeatCopy-v0': {
        'min': -0.5,
        'max': 12.0
    },
    'Reverse-v0': {
        'min': -0.5,
        'max': 3.0
    },
    'ReversedAddition-v0': {
        'min': -1.0,
        'max': 5.0
    },
    'ReversedAddition3-v0': {
        'min': -0.5,
        'max': 5.0
    },
    'FrozenLake-v0': {
        'min': 0.0,
        'max': 1.0
    },
    'FrozenLake8x8-v0': {
        'min': 0.0,
        'max': 1.0
    },
    'Blackjack-v0': {
        'min': -1.0,
        'max': 1.0
    },
    'Roulette-v0': {
        'min': -100.0,
        'max': 344.0 
    },
    'NChain-v0': {
        'min': 272.0,
        'max': 1286.0
    },
    'Taxi-v3': {
        'min': -2000.0,
        'max': 15.0
    },
}

def normalise(value, minimum, maximum):
    return (value - minimum) / (maximum - minimum)

for env in env_names:
    env_dir = base_dir + '/' + env
    maximum = vals[env]['max']
    minimum = vals[env]['min']
    for a in action_selection_names:
        act_dir = env_dir + '/' + a
        data = np.loadtxt(base_dir + env + '/' + a + '/results.csv', delimiter=',')
        new = []
        for row in data:
            new.append([normalise(row[0], minimum, maximum)])
        with open(act_dir + '/normal_reward_results.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerows(new)
