import gym
import numpy as np
env = gym.make('Taxi-v3')

# Parameters
episodes = 10000
steps = 1000
episode_reward = 0
total_reward = 0

q_function = np.loadtxt('data/taxi/taxi-mbie-eb-q-function-2020-03-07_16:43:44', delimiter=',')

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
    total_reward += episode_reward
average_reward = total_reward / episodes
print('Average reward per episode after ' + str(episodes)  + ' episodes = ' + str(average_reward))
print("Total reward at end = " + str(total_reward))
env.close()
