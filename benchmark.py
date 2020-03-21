import gym, os
import numpy as np
from env_handler import EnvHandler

# Parameters
episodes = 10000
steps = 200
episode_reward = 0
total_reward = 0
env_name = 'Taxi-v3'
seed = 101

env_handler = EnvHandler()
env = env_handler.get_env(env_name)
env.seed(seed)
rewards = []
action_strat = 'mbie-eb'
log_dir = 'data/' + env_name + '/' + action_strat + '/q_function'
files = os.listdir(log_dir)

for q in files:
    try:
        q_function = np.loadtxt(log_dir + '/' + q, delimiter=',')
    except:
        continue
    total_reward = 0
    for ep in range(episodes):
        current_state = env.reset()
        episode_reward = 0
        for i in range(steps):
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

    print(q)
    print('Median reward per episode after ' + str(episodes)  + ' episodes = ' + str(average_reward))
    print('Mean reward per episode after ' + str(episodes)  + ' episodes = ' + str(mean))
    print('Max: ' + str(max(rewards)))
    print('Min: ' + str(min(rewards)))
    print('Standard deviation: ' + str(np.std(rewards)))
env.close()
