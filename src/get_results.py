import gym, os, sys
import numpy as np
from env_handler import EnvHandler

# Parameters
episodes = 100
steps = 200
seed = 1

args = sys.argv[1:]

env_handler = EnvHandler()
action_selector_name = args[0]
base_dir = args[1] + '/data/'
env_names = ['Acrobot-v1', 'CartPole-v1', 'MountainCar-v0', 'Pendulum-v0', \
'Copy-v0', 'DuplicatedInput-v0', 'RepeatCopy-v0', 'Reverse-v0', 'ReversedAddition-v0', 'ReversedAddition3-v0', \
'Blackjack-v0', 'Roulette-v0', 'FrozenLake-v0', 'FrozenLake8x8-v0', 'NChain-v0', 'Taxi-v3']

for env_name in env_names:
    print(env_name)
    env_dir = base_dir + env_name + '/' + action_selector_name
    q_dir = env_dir + '/final/q_function'
    if action_selector_name == 'random-play':
        os.makedirs(env_dir, exist_ok=True)
        episodes = 1000
    else:
        files = os.listdir(q_dir)
    data = []
    results = []
    if action_selector_name == 'random-play':
        os.makedirs(env_dir, exist_ok=True)
        files = [1]
    for q in files:
        print(q)
        if not action_selector_name == 'random-play':
            try:
                q_function = np.loadtxt(q_dir + '/' + q, delimiter=',')
            except:
                continue
        print('Loaded')
        env = env_handler.get_env(env_name)
        env.seed(seed)
        for ep in range(episodes):
            current_state = env.reset()
            episode_reward = 0
            episode_steps = 0
            for i in range(steps):
                if action_selector_name == 'random-play':
                    action = env.get_random_action()
                elif env_name == 'Roulette-v0':
                    action = np.argmax(q_function)
                else:
                    action = np.argmax(q_function[current_state,:])
                next_state, reward, done, info = env.step(action)
                current_state = next_state
                episode_reward += reward
                episode_steps += 1
                if done:
                    break
            result = [episode_reward, episode_steps]
            results.append(result)
    fname = env_dir + '/results.csv'
    np.savetxt(fname, np.array(results), delimiter=',')
env.close()
