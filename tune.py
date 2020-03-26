import numpy as np
from env_handler import EnvHandler
from q_function import Q
from logger import Logger
from agent import Agent
from action_selectors.mbie_eb import MBIE_EB
from action_selectors.epsilon_greedy import EpsilonGreedy
from action_selectors.boltzmann import Boltzmann
from action_selectors.ucb_1 import UCB_1
from action_selectors.controlability import Controlability
from action_selectors.vdbe import VDBE

env_handler = EnvHandler()

episodes = 1000
steps = 200
learning_rate = 0.1
discount_factor = 0.9
env_names = ['Acrobot-v1', 'CartPole-v1', 'MountainCar-v0', 'Pendulum-v0', \
'Copy-v0', 'DuplicatedInput-v0', 'RepeatCopy-v0', 'Reverse-v0', 'ReversedAddition-v0', 'ReversedAddition3-v0', \
'Blackjack-v0', 'Roulette-v0', 'FrozenLake-v0', 'FrozenLake8x8-v0', 'NChain-v0', 'Taxi-v3', 'GuessingGame-v0', 'HotterColder-v0']
epsilon = 1
betas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
epsilon_disc = [0.9999, 0.999, 0.99, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]
temps = [1000000, 500000, 100000, 10000, 1000]
c_vals = [1, 5, 10, 25, 50, 100, 250, 500]
seeds = [101, 100, 99, 98, 97]
action_selector_name = 'epsilon-greedy'

for env_name in env_names:
    print(env_name)
    env = env_handler.get_env(env_name)
    q_function = Q(env.get_total_states(), env.get_total_actions(), learning_rate, discount_factor)
    log_dir = 'data/' + env_name + '/' + action_selector_name
    for val in c_vals:
        print('EDF =  ' + str(val))
        for seed in seeds:
            print('Seed = ' + str(seed))
            env.seed(seed)
            action_selector = EpsilonGreedy(epsilon, val, seed)#MBIE_EB(val, env.get_total_states(), env.get_total_actions(), discount_factor)#Boltzmann(val, seed)UCB_1(val, env.get_total_states(), env.get_total_actions(), discount_factor)
            q_function.reset()
            logger = Logger(episodes)
            filepath = action_selector_name + '-edf-' + str(val) + '-seed-' + str(seed)
            agent = Agent(env, q_function, action_selector, logger)
            agent.train(steps, episodes, log_dir, filepath)
print('Done')
