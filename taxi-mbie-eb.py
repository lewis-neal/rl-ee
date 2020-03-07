import gym, datetime
import numpy as np
from mbie_eb import MBIE_EB
env = gym.make('Taxi-v3')

# Parameters
learning_rate = 0.1
discount_factor = 0.9
episodes = 2000
beta = 0.05
steps = 1000

log_dir = 'data/taxi'
filepath = log_dir + '/taxi-mbie-eb'
date_string = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

def update_q_function(current_state, next_state, action, reward):
    q_function[current_state, action] = q_function[current_state, action] + learning_rate * (reward + discount_factor * np.max(q_function[next_state, :]) - q_function[current_state, action])

def reset_q_function():
    return np.zeros([env.observation_space.n, env.action_space.n])

action_selector = MBIE_EB(beta, env.observation_space.n, env.action_space.n, discount_factor)

cumulative_reward = 0
logs = []
q_function = reset_q_function()

for i_episode in range(episodes):
    current_state = env.reset()
    cumulative_reward = 0
    for t in range(steps):
        action = action_selector.select_action(current_state, q_function)
        next_state, reward, done, info = env.step(action)
        update_q_function(current_state, next_state, action, reward)
        current_state = next_state
        cumulative_reward += reward
        if done:
            break
        logs.append([i_episode, t, reward, cumulative_reward])

cumulative_reward = 0
current_state = env.reset()
env.render()

for i in range(steps):
    action = np.argmax(q_function[current_state,:])
    next_state, reward, done, info = env.step(action)
    current_state = next_state
    cumulative_reward += reward
    env.render()
    if done:
        break
print("Episode finished after {} timesteps".format(i+1))
print("Cumulative reward at end = " + str(cumulative_reward))
env.close()

np.savetxt(filepath + '-q-function-' + date_string, q_function, delimiter=',')

# data is logged in the format: episode, step, reward, cumulative reward
np.savetxt(filepath + date_string, logs, delimiter=',')

