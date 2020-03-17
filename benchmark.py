import gym
import numpy as np
from discrete import Discrete
from env_wrapper import EnvWrapper
env = gym.make('CartPole-v1')

# Parameters
episodes = 100
steps = 1000
episode_reward = 0
total_reward = 0
num_states = [10, 0, 10, 0]

q_function = np.loadtxt('data/cart-pole/epsilon-greedy2020-03-17_13:08:34-q-function.csv', delimiter=',')
discrete = Discrete(num_states, env)
env = EnvWrapper(env, num_states, discrete)

for ep in range(episodes):
    current_state = env.reset()
    episode_reward = 0
    for i in range(steps):
        action = np.argmax(q_function[current_state,:])
        next_state, reward, done, info = env.step(action)
        current_state = next_state
        episode_reward += reward
        env.render()
        if done:
            print(episode_reward)
            break
    total_reward += episode_reward
average_reward = total_reward / episodes
print('Average reward per episode after ' + str(episodes)  + ' episodes = ' + str(average_reward))
print("Total reward at end = " + str(total_reward))
env.close()
