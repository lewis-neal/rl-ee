import gym, os
import numpy as np
from env_handler import EnvHandler

# Parameters
episodes = 100
steps = 200
episode_reward = 0
seed = 1

env_handler = EnvHandler()
rewards = []
action_strat = 'epsilon-greedy'
base_dir = 'data/'
env_names = ['Acrobot-v1', 'CartPole-v1', 'MountainCar-v0', 'Pendulum-v0', \
'Copy-v0', 'DuplicatedInput-v0', 'RepeatCopy-v0', 'Reverse-v0', 'ReversedAddition-v0', 'ReversedAddition3-v0', \
'Blackjack-v0', 'Roulette-v0', 'FrozenLake-v0', 'FrozenLake8x8-v0', 'NChain-v0', 'Taxi-v3', 'GuessingGame-v0', 'HotterColder-v0']
# format: hyperparam_val,median,mean,max,min,std

for env_name in env_names:
    data = []
    env_dir = base_dir + env_name + '/' + action_strat
    q_dir = env_dir + '/q_function'
    files = os.listdir(q_dir)
    files.sort()
    for q in files:
        try:
            q_function = np.loadtxt(q_dir + '/' + q, delimiter=',')
        except:
            continue
        total_reward = 0
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
            total_reward += episode_reward
        average_reward = np.median(rewards)
        mean = total_reward / episodes
        benchmark = []
        name = q.split('-')
        benchmark.append(float(name[2]))
        benchmark.append(average_reward)
        benchmark.append(mean)
        benchmark.append(max(rewards))
        benchmark.append(min(rewards))
        benchmark.append(np.std(rewards))
        data.append(benchmark)
    fname = env_dir + '/benchmark.csv'
    np.savetxt(fname, np.array(data), delimiter=',')
env.close()
