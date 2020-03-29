import gym, os, sys
import numpy as np
from env_handler import EnvHandler

# Parameters
episodes = 100
steps = 200
episode_reward = 0
seed = 1

args = sys.argv[1:]

betas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
epsilon_disc = [0.9999, 0.999, 0.99, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]
temps = [1000000, 500000, 100000, 10000, 5000, 1000]
deltas = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 25]
c_vals = [0.1, 0.5, 1, 5, 10, 25, 50, 100, 250, 500]

env_handler = EnvHandler()
action_selector_name = args[0]
base_dir = args[1] + '/data/'
env_names = ['Acrobot-v1', 'CartPole-v1', 'MountainCar-v0', 'Pendulum-v0', \
'Copy-v0', 'DuplicatedInput-v0', 'RepeatCopy-v0', 'Reverse-v0', 'ReversedAddition-v0', 'ReversedAddition3-v0', \
'Blackjack-v0', 'Roulette-v0', 'FrozenLake-v0', 'FrozenLake8x8-v0', 'NChain-v0', 'Taxi-v3']
# format: hyperparam_val,median,mean,max,min,std

if action_selector_name == 'epsilon-greedy':
    vals = epsilon_disc
elif action_selector_name == 'boltzmann':
    vals = temps
elif action_selector_name == 'ucb-1':
    vals = c_vals
elif action_selector_name == 'vdbe':
    vals = deltas
elif action_selector_name == 'controlability' or action_selector_name == 'mbie-eb':
    vals = betas

for env_name in env_names:
    env_dir = base_dir + env_name + '/' + action_selector_name
    data = []
    for val in vals:
        val_dir = env_dir + '/' + str(val)
        q_dir = val_dir + '/q_function'
        files = os.listdir(q_dir)
        rewards = []
        for q in files:
            try:
                q_function = np.loadtxt(q_dir + '/' + q, delimiter=',')
            except:
                continue
            env = env_handler.get_env(env_name)
            env.seed(seed)
            for ep in range(episodes):
                current_state = env.reset()
                episode_reward = 0
                for i in range(steps):
                    if env_name == 'Roulette-v0':
                        action = np.argmax(q_function)
                    else:
                        action = np.argmax(q_function[current_state,:])
                    next_state, reward, done, info = env.step(action)
                    current_state = next_state
                    episode_reward += reward
                    if done:
                        break
                rewards.append(episode_reward)
        median = np.median(rewards)
        mean = np.mean(rewards)
        maximum = max(rewards)
        minimum = min(rewards)
        std = np.std(rewards)
        benchmark = [val, median, mean, maximum, minimum, std]
        data.append(benchmark)
    fname = env_dir + '/benchmark.csv'
    np.savetxt(fname, np.array(data), delimiter=',')
env.close()
