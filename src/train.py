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
env_names = [['Acrobot-v1', 1], ['CartPole-v1', 1], ['MountainCar-v0', 1], ['Pendulum-v0', 1], \
['Copy-v0', 1], ['DuplicatedInput-v0', 1], ['RepeatCopy-v0', 1], ['Reverse-v0', 1], ['ReversedAddition-v0', 1], ['ReversedAddition3-v0', 1], \
['Blackjack-v0', 1], ['Roulette-v0', 1], ['FrozenLake-v0', 1], ['FrozenLake8x8-v0', 1], ['NChain-v0', 1], ['Taxi-v3', 1], ['GuessingGame-v0', 1], ['HotterColder-v0', 1]]
epsilon = 1
seeds = range(20, 30)
action_selector_name = 'random'

for seed in seeds:
    for env_name, val in env_names:
        print(env_name)
        print(seed)
        env = env_handler.get_env(env_name)
        env.seed(seed)
        q_function = Q(env.get_total_states(), env.get_total_actions(), learning_rate, discount_factor)
        log_dir = 'data/' + env_name + '/' + action_selector_name
        action_selector = EpsilonGreedy(epsilon, val, seed)#action_selector = Boltzmann(val, seed)#MBIE_EB(val, env.get_total_states(), env.get_total_actions(), discount_factor)
        q_function.reset()
        logger = Logger(episodes)
        agent = Agent(env, q_function, action_selector, logger)
        filename = action_selector_name + '-seed-' + str(seed)
        agent.train(steps, episodes, log_dir, filename)
print('Done')
