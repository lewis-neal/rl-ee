import gym, datetime
import numpy as np
from epsilon_greedy import EpsilonGreedy
from logger import Logger
env = gym.make('MountainCar-v0')

def update_q_function(current_state, next_state, action, reward):
    q_function[current_state, action] = q_function[current_state, action] + learning_rate * (reward + discount_factor * np.max(q_function[next_state, :]) - q_function[current_state, action])

def reset_q_function():
    return np.zeros([100, env.action_space.n])

def get_states(env, dim, num_states):
    high = env.observation_space.high[dim]
    low = env.observation_space.low[dim]
    state_list = []
    num_states -= 1
    for i in np.arange(low, high + ((high - low) / num_states), (high - low) / num_states):
        state_list.append(i)
    return state_list

def discretise(state):
    num_states_a = 10
    num_states_b = 10
    states_a = get_states(env, 0, num_states_a)
    states_b = get_states(env, 1, num_states_b)

    state_a = get_state(state[0], states_a)
    state_b = get_state(state[1], states_b)

    return (state_a * (num_states_a ** 0)) + (state_b * (num_states_b ** 1))

def get_state(value, state_list):
    low = -9999999999
    high = 9999999999
    low_ind = 0
    high_ind = 0
    for ind, s in enumerate(state_list):
        if value < s:
            high = s
            high_ind = ind
            break
        low = s
        low_ind = ind

    diff_low = abs(value - low)
    diff_high = abs(value - high)

    if diff_low > diff_high:
        return high_ind
    return low_ind



# Parameters
learning_rate = 0.1
discount_factor = 0.9
episodes = 500
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

for iteration in range(iterations):
    total_reward = 0
    q_function = reset_q_function()
    epsilon_greedy.reset()
    for episode in range(episodes):
        current_state = env.reset()
        episode_reward = 0
        episode_length = 0
        for t in range(steps):
            print(discretise(current_state))
            action = epsilon_greedy.select_action(discretise(current_state), q_function, env)
            next_state, reward, done, info = env.step(action)
            update_q_function(discretise(current_state), discretise(next_state), action, reward)
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
    action = np.argmax(q_function[discretise(current_state),:])
    next_state, reward, done, info = env.step(action)
    current_state = next_state
    episode_reward += reward
    env.render()
    if done:
        print(done)
        break
print("Episode finished after {} timesteps".format(i+1))
print("Cumulative reward at end = " + str(episode_reward))
env.close()

np.savetxt(filepath + '-q-function', q_function, delimiter=',')

logger.write(filepath)
