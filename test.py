import gym, datetime
from agent import Agent
from q_function import Q
from epsilon_greedy import EpsilonGreedy
from logger import Logger

# Parameters
learning_rate = 0.1
discount_factor = 0.9
episodes = 5000
epsilon = 1
epsilon_discount_factor = 0.9999
steps = 200

log_dir = 'data/n-chain'
date_string = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
filepath = log_dir + '/n-chain-epsilon-greedy' + date_string

env = gym.make('NChain-v0')
q_function = Q(env.observation_space.n, env.action_space.n, learning_rate, discount_factor)
action_selector = EpsilonGreedy(epsilon, epsilon_discount_factor)
logger = Logger(episodes, 1)

agent = Agent(env, q_function, action_selector, logger)

agent.train(steps, episodes)

agent.solve(steps, filepath, False)
