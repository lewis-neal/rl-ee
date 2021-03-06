import os
import numpy as np

env_names = ['Acrobot-v1', 'CartPole-v1', 'MountainCar-v0', 'Pendulum-v0', \
'Copy-v0', 'DuplicatedInput-v0', 'RepeatCopy-v0', 'Reverse-v0', 'ReversedAddition-v0', 'ReversedAddition3-v0', \
'Blackjack-v0', 'Roulette-v0', 'FrozenLake-v0', 'FrozenLake8x8-v0', 'NChain-v0', 'Taxi-v3']

action_selectors = ['boltzmann', 'controlability', 'epsilon-greedy', 'mbie-eb', 'vdbe', 'ucb-1']

base_dir = 'data/'

#format val, mean, median, max, min, std

def get_maxes(arr):
    maxes = []
    max_val = None
    for key in arr:
        if max_val == None:
            max_val = arr[key]
            maxes.append(key)
            continue
        if arr[key] > max_val:
            max_val = arr[key]
            maxes = [key]
            continue
        if arr[key] == max_val:
            maxes.append(key)
    return maxes

def get_mins(arr):
    mins = []
    min_val = None
    for key in arr:
        if min_val == None:
            min_val = arr[key]
            mins.append(key)
            continue
        if arr[key] < min_val:
            min_val = arr[key]
            mins = [key]
            continue
        if arr[key] == min_val:
            mins.append(key)
    return mins

def get_median(arr):
    for key in arr:
        arr[key] = np.median(arr[key])
    return arr

def get_values(keys_arr, vals_arr):
    new_vals = {}
    for key in keys_arr:
       new_vals[key] = vals_arr[key]
    return new_vals

for env_name in env_names:
    print(env_name)
    env_dir = base_dir + env_name + '/'
    for act in action_selectors:
        print(act)
        try:
            benchmark = np.loadtxt(env_dir + act + '/benchmark.csv', delimiter=',')
        except:
            continue
        medians = {}
        means = {}
        maximums = {}
        minimums = {}
        stands = {}
        for row in benchmark:
            key = str(row[0])
            if key in medians:
                means[key].append(row[1])
                medians[key].append(row[2])
                maximums[key].append(row[3])
                minimums[key].append(row[4])
                stands[key].append(row[5])
            else:
                means[key] = [row[1]]
                medians[key] = [row[2]]
                maximums[key] = [row[3]]
                minimums[key] = [row[4]]
                stands[key] = [row[5]]
        maxes = get_maxes(medians)
        if len(maxes) == 1:
            print(maxes)
            continue
        maxes_m = get_maxes(get_values(maxes, means))
        if len(maxes_m) == 1:
            print(maxes_m)
            continue

        min_s = get_mins(get_values(maxes_m, stands))
        if len(min_s) == 1:
            print(min_s)
            continue

        minm = get_maxes(get_values(min_s, minimums))
        if len(minm) == 1:
            print(minm)
            continue

        maxm = get_maxes(get_values(minm, maximums))
        if len(maxm) == 1:
            print(maxm)
            continue

        print('Arbitration required between values: ' + str(maxm))
        if len(maxm) == 10:
            print('All results identical')
