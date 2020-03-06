import gym
import numpy as np
from epsilon_greedy import EpsilonGreedy
env = gym.make('NChain-v0')

def update_q_function(current_state, next_state, action, reward):
    q_function[current_state, action] = q_function[current_state, action] + learning_rate * (reward + discount_factor * np.max(q_function[next_state, :]) - q_function[current_state, action])

def reset_q_function():
    return np.zeros([env.observation_space.n, env.action_space.n])

# Parameters
learning_rate = 0.1
discount_factor = 0.9
episodes = 1
epsilon = 1
epsilon_discount_factor = 0.9999
steps = 1000000

epsilon_greedy = EpsilonGreedy(epsilon, epsilon_discount_factor)

cumulative_reward = 0

for i_episode in range(episodes):
    q_function = reset_q_function()
    epsilon_greedy.reset()
    current_state = env.reset()
    cumulative_reward = 0
    for t in range(steps):
        action = epsilon_greedy.select_action(current_state, q_function, env)
        next_state, reward, done, info = env.step(action)
        update_q_function(current_state, next_state, action, reward)
        current_state = next_state
        cumulative_reward += reward
        if t % (steps / 20) == 0:
            print("Cumulative reward so far = " + str(cumulative_reward))
    print("Episode finished after {} timesteps".format(t+1))
    print("Cumulative reward at end = " + str(cumulative_reward))
    print(q_function)

cumulative_reward = 0
current_state = env.reset()

for i in range(10000):
    action = np.argmax(q_function[current_state,:])
    next_state, reward, done, info = env.step(action)
    current_state = next_state
    cumulative_reward += reward
print("Episode finished after {} timesteps".format(i+1))
print("Cumulative reward at end = " + str(cumulative_reward))
env.close()

np.savetxt('data/n-chain-epsilon-greedy-01.txt', q_function, delimiter=',')
