import numpy as np
from env_handler import EnvHandler
from q_function import Q
from agent import Agent
from logger import Logger
from action_selectors.mbie_eb import MBIE_EB
from action_selectors.epsilon_greedy import EpsilonGreedy
from action_selectors.boltzmann import Boltzmann

env_handler = EnvHandler()

episodes = 1000
steps = 200
learning_rate = 0.1
discount_factor = 0.9
env_names = [['Acrobot-v1', 1000000], ['CartPole-v1', 500000], ['MountainCar-v0', 500000], ['Pendulum-v0', 500000], \
['Copy-v0', 5000], ['DuplicatedInput-v0', 500000], ['RepeatCopy-v0', 10000], ['Reverse-v0', 100000], ['ReversedAddition-v0', 10000], ['ReversedAddition3-v0', 50000], \
['Blackjack-v0', 1000], ['Roulette-v0', 1000], ['FrozenLake-v0', 500000], ['FrozenLake8x8-v0', 250000], ['NChain-v0', 100000], ['Taxi-v3', 500000], ['GuessingGame-v0', 1000], ['HotterColder-v0', 500000]]
epsilon = 1
seed = 101
action_selector_name = 'boltzmann'

for env_name, val in env_names:
    print(env_name)
    print(val)
    env = env_handler.get_env(env_name)
    env.seed(seed)
    q_function = Q(env.get_total_states(), env.get_total_actions(), learning_rate, discount_factor)
    log_dir = 'data/' + env_name + '/' + action_selector_name
    action_selector = Boltzmann(val, seed)#MBIE_EB(val, env.get_total_states(), env.get_total_actions(), discount_factor)#EpsilonGreedy(epsilon, val, seed)
    q_function.reset()
    logger = Logger(episodes)
    agent = Agent(env, q_function, action_selector, logger)
    filepath = log_dir + '/' + action_selector_name + '-FINAL'
    agent.train(steps, episodes, filepath)
print('Done')
