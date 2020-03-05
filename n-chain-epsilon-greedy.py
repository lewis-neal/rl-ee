import gym
import numpy as np
env = gym.make('NChain-v0')

def select_action(current_state):
    num = np.random.uniform()
    if num > epsilon:
        update_epsilon()
        return np.argmax(q_function[current_state,:])
    update_epsilon()
    return env.action_space.sample()

def update_epsilon():
    global switch_to_random
    if switch_to_random:
        return
    global epsilon
    epsilon *= epsilon_discount_factor

def update_q_function(current_state, next_state, action, reward):
    q_function[current_state, action] = q_function[current_state, action] + learning_rate * (reward + discount_factor * np.max(q_function[next_state, :]) - q_function[current_state, action])

def reset_q_function():
    return np.zeros([env.observation_space.n, env.action_space.n])

def reset_epsilon():
    return 1

# Parameters
learning_rate = 0.5
discount_factor = 0.9
episodes = 2
epsilon = reset_epsilon() 
epsilon_discount_factor = 0.9999
steps = 1000000
switch_to_random = False

cumulative_reward = 0

for i_episode in range(episodes):
    q_function = reset_q_function()
    epsilon = reset_epsilon()
    current_state = env.reset()
    cumulative_reward = 0
    for t in range(steps):
        #env.render() some environments can't be rendered
        action = select_action(current_state)
        next_state, reward, done, info = env.step(action)
        update_q_function(current_state, next_state, action, reward)
        current_state = next_state
        cumulative_reward += reward
        if t % (steps / 20) == 0:
            print("Cumulative reward so far = " + str(cumulative_reward))
    print("Episode finished after {} timesteps".format(t+1))
    print("Cumulative reward at end = " + str(cumulative_reward))
    print(q_function)
    switch_to_random = True
env.close()

