import gym
import numpy as np
env = gym.make('CliffWalking-v0')

def select_action(current_state, beta):
    return np.argmax(q_function[current_state,:] + beta * (np.power(counts[current_state], -0.5)))

def update_q_function(current_state, next_state, action, reward):
    q_function[current_state, action] = q_function[current_state, action] + learning_rate * (reward + discount_factor * np.max(q_function[next_state, :]) - q_function[current_state, action])

def reset_q_function():
    return np.zeros([env.observation_space.n, env.action_space.n])

def reset_counts():
    return np.zeros(env.observation_space.n)

def update_counts(state, counts):
    counts[state] += 1
    return counts

# Parameters
learning_rate = 0.5
discount_factor = 0.9
episodes = 2
beta = 0.05
steps = 1000000
switch_to_random = False

cumulative_reward = 0

for i_episode in range(episodes):
    q_function = reset_q_function()
    counts = reset_counts()
    current_state = env.reset()
    counts = update_counts(current_state, counts)
    cumulative_reward = 0
    for t in range(steps):
        action = select_action(current_state, beta)
        next_state, reward, done, info = env.step(action)
        update_q_function(current_state, next_state, action, reward)
        current_state = next_state
        counts = update_counts(current_state, counts)
        cumulative_reward += reward
        if t % (steps / 20) == 0:
            print("Cumulative reward so far = " + str(cumulative_reward))
    print("Episode finished after {} timesteps".format(t+1))
    print("Cumulative reward at end = " + str(cumulative_reward))
    print(q_function)
    print(counts)

done = False
cumulative_reward = 0

for i in range(1000):
    action = np.argmax(q_function[current_state,:])
    next_state, reward, done, info = env.step(action)
    current_state = next_state
    cumulative_reward += reward
    if done:
        break
print("Cumulative reward at end = " + str(cumulative_reward))

env.close()

