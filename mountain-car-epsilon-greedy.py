import gym, datetime
import numpy as np
from epsilon_greedy import EpsilonGreedy
from logger import Logger
from discrete import Discrete
env = gym.make('MountainCar-v0')

def update_q_function(current_state, next_state, action, reward):
    q_function[current_state, action] = q_function[current_state, action] + learning_rate * (reward + discount_factor * np.max(q_function[next_state, :]) - q_function[current_state, action])

def reset_q_function():
    return np.zeros([100, env.action_space.n])

# Parameters
learning_rate = 0.1
discount_factor = 0.9
episodes = 1000
epsilon = 1
epsilon_discount_factor = 0.9999
steps = 1000
iterations = 1

log_dir = 'data/mountain-car'
date_string = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
filepath = log_dir + '/mountain-car-epsilon-greedy' + date_string

logger = Logger(episodes, iterations)

epsilon_greedy = EpsilonGreedy(epsilon, epsilon_discount_factor)

episode_reward = 0
total_reward = 0
q_function = reset_q_function()
discrete = Discrete([10, 10], env)

for iteration in range(iterations):
    total_reward = 0
    q_function = reset_q_function()
    epsilon_greedy.reset()
    for episode in range(episodes):
        current_state = env.reset()
        episode_reward = 0
        episode_length = 0
        for t in range(steps):
            action = epsilon_greedy.select_action(discrete.discretise(current_state), q_function, env)
            next_state, reward, done, info = env.step(action)
            update_q_function(discrete.discretise(current_state), discrete.discretise(next_state), action, reward)
            current_state = next_state
            episode_reward += reward
            episode_length += 1
            if done:
                break
        total_reward += episode_reward
        logger.log(iteration, episode, episode_reward, total_reward, episode_length)

episode_reward = 0
current_state = env.reset()
env.render()

for i in range(steps):
    action = np.argmax(q_function[discrete.discretise(current_state),:])
    next_state, reward, done, info = env.step(action)
    current_state = next_state
    episode_reward += reward
    env.render()
    if done:
        break
print("Episode finished after {} timesteps".format(i+1))
print("Cumulative reward at end = " + str(episode_reward))
env.close()

np.savetxt(filepath + '-q-function', q_function, delimiter=',')

logger.write(filepath)
