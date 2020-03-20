import numpy as np
from env_handler import EnvHandler
from q_function import Q
from mbie_eb import MBIE_EB
from agent import Agent
from logger import Logger
from epsilon_greedy import EpsilonGreedy
from boltzmann import Boltzmann

env_handler = EnvHandler()

episodes = 1000
steps = 200
learning_rate = 0.1
discount_factor = 0.9999
env_names = ['Blackjack-v0', 'GuessingGame-v0', 'HotterColder-v0', 'Roulette-v0', 'FrozenLake-v0', 'FrozenLake8x8-v0', 'NChain-v0', 'Taxi-v3']
epsilon = 1
betas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
epsilon_disc = [0.9999, 0.999, 0.99, 0.9]
temps = [1000000, 500000, 100000, 10000, 1000]

for env_name in env_names:
    print(env_name)
    env = env_handler.get_env(env_name)
    q_function = Q(env.get_total_states(), env.get_total_actions(), learning_rate, discount_factor)
    log_dir = 'data/' + env_name
    for val in betas:
        print('Beta = ' + str(val))
        action_selector = MBIE_EB(val, env.get_total_states(), env.get_total_actions(), discount_factor)#Boltzmann(val)#EpsilonGreedy(epsilon, val)
        q_function.reset()
        logger = Logger(episodes)
        agent = Agent(env, q_function, action_selector, logger)
        filepath = log_dir + '/mbie-eb_beta-' + str(val)
        agent.train(steps, episodes)
        agent.solve(steps, filepath, False)
print('Done')
