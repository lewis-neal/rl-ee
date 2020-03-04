import gym
import numpy as np
env = gym.make('NChain-v0')
q_function = np.zeros([env.observation_space.n, env.action_space.n])

# Parameters
learning_rate = 0.1
discount_factor = 0.9
episodes = 20
steps = 10000

cumulative_reward = 0

def select_action(current_state):
    return np.argmax(q_function[current_state,:])

def update_q_function(current_state, next_state, action, reward):
    q_function[current_state, action] = \
        q_function[current_state, action] + \
        learning_rate * (reward + discount_factor * \
        np.max(q_function[next_state, :]) - \
        q_function[current_state, action])

for i_episode in range(episodes):
    current_state = env.reset()
    cumulative_reward = 0
    for t in range(steps):
        #env.render() some environments can't be rendered
        action = select_action(current_state)
        next_state, reward, done, info = env.step(action)
        update_q_function(current_state, next_state, action, reward)
        current_state = next_state
        cumulative_reward += reward
        if done:
            break
    print("Episode finished after {} timesteps".format(t+1))
    print("Cumulative reward = " + str(cumulative_reward))
env.close()

