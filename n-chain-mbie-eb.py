import gym, datetime
import numpy as np
from mbie_eb import MBIE_EB
from logger import Logger
env = gym.make('NChain-v0')

# Parameters
learning_rate = 0.1
discount_factor = 0.9
episodes = 5000
beta = 0.05
steps = 1000
iterations = 3

log_dir = 'data/n-chain'
date_string = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
filepath = log_dir + '/n-chain-mbie-eb' + date_string

logger = Logger(episodes, iterations)

def update_q_function(current_state, next_state, action, reward):
    q_function[current_state, action] = q_function[current_state, action] + learning_rate * (reward + discount_factor * np.max(q_function[next_state, :]) - q_function[current_state, action])

def reset_q_function():
    return np.zeros([env.observation_space.n, env.action_space.n])

action_selector = MBIE_EB(beta, env.observation_space.n, env.action_space.n, discount_factor)

episode_reward = 0
total_reward = 0
q_function = reset_q_function()

for iteration in range(iterations):
    total_reward = 0
    for episode in range(episodes):
        current_state = env.reset()
        episode_reward = 0
        episode_length = 0
        for t in range(steps):
            action = action_selector.select_action(current_state, q_function)
            next_state, reward, done, info = env.step(action)
            update_q_function(current_state, next_state, action, reward)
            current_state = next_state
            episode_reward += reward
            episode_length += 1
        total_reward += episode_reward
        logger.log(iteration, episode, episode_reward, total_reward, episode_length)

episode_reward = 0
current_state = env.reset()

for i in range(steps):
    action = np.argmax(q_function[current_state,:])
    next_state, reward, done, info = env.step(action)
    current_state = next_state
    episode_reward += reward
print("Episode finished after {} timesteps".format(i+1))
print("Cumulative reward at end = " + str(episode_reward))
env.close()

np.savetxt(filepath + '-q-function-', q_function, delimiter=',')

logger.write(filepath)
