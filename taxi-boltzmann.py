import gym, datetime
import numpy as np
from boltzmann import Boltzmann
from logger import Logger
from q_function import Q
env = gym.make('Taxi-v3')

# Parameters
learning_rate = 0.1
discount_factor = 0.9
episodes = 5000
epsilon = 1
epsilon_discount_factor = 0.9999
steps = 200
iterations = 1
temperature = 1000000

log_dir = 'data/taxi'
date_string = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
filepath = log_dir + '/taxi-boltzmann' + date_string

logger = Logger(episodes, iterations)

boltzmann = Boltzmann(temperature)

episode_reward = 0
total_reward = 0
q_function = Q(env.observation_space.n, env.action_space.n, learning_rate, discount_factor)

for iteration in range(iterations):
    total_reward = 0
    q_function.reset()
    boltzmann.reset()
    for episode in range(episodes):
        current_state = env.reset()
        episode_reward = 0
        episode_length = 0
        for t in range(steps):
            action = boltzmann.select_action(current_state, q_function, env)
            next_state, reward, done, info = env.step(action)
            q_function.update_q_function(current_state, next_state, action, reward)
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
    action = q_function.get_best_action(current_state)
    next_state, reward, done, info = env.step(action)
    current_state = next_state
    episode_reward += reward
    env.render()
    if done:
        break
print("Episode finished after {} timesteps".format(i+1))
print("Cumulative reward at end = " + str(episode_reward))
env.close()

np.savetxt(filepath + '-q-function.csv', q_function.get_q_function(), delimiter=',')

logger.write(filepath +' .csv')
