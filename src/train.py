import sys, csv, os
import numpy as np
from env_handler import EnvHandler
from q_function import Q
from agent import Agent
from logger import Logger
from action_selectors.mbie_eb import MBIE_EB
from action_selectors.epsilon_greedy import EpsilonGreedy
from action_selectors.boltzmann import Boltzmann
from action_selectors.controlability import Controlability
from action_selectors.ucb_1 import UCB_1
from action_selectors.vdbe import VDBE

env_handler = EnvHandler()
# args = [action selector name, base dir, path to params file]
args = sys.argv[1:]

episodes = 1
steps = 200
learning_rate = 0.1
discount_factor = 0.9
epsilon = 1
omega = 0.5
seeds = range(20, 30)
action_selector_name = args[0]
base_dir = args[1] + '/data/'
params_path = args[2]
with open(params_path, newline='') as csvfile:
    env_names = list(csv.reader(csvfile))

for env_name, val in env_names:
    val = float(val)
    for seed in seeds:
        print(env_name)
        print(seed)
        env = env_handler.get_env(env_name)
        env.seed(seed)
        q_function = Q(env.get_total_states(), env.get_total_actions(), learning_rate, discount_factor)
        log_dir = base_dir + env_name + '/' + action_selector_name + '/' + 'final'
        os.makedirs(log_dir + '/q_function', exist_ok=True)
        os.makedirs(log_dir + '/training-data', exist_ok=True)
        if action_selector_name == 'epsilon-greedy' or action_selector_name == 'random':
            action_selector = EpsilonGreedy(epsilon, val, seed)
        elif action_selector_name == 'boltzmann':
            action_selector = Boltzmann(val, seed)
        elif action_selector_name == 'ucb-1':
            action_selector = UCB_1(val, env.get_total_states(), env.get_total_actions())
        elif action_selector_name == 'vdbe':
            action_selector = VDBE(env.get_total_states(), val, 1 / env.get_total_actions(), learning_rate, seed)
        elif action_selector_name == 'mbie-eb':
            action_selector = MBIE_EB(val, env.get_total_states(), env.get_total_actions(), discount_factor)
        elif action_selector_name == 'controlability':
            action_selector = Controlability(val, env.get_total_states(), env.get_total_actions(), learning_rate, omega)
        q_function.reset()
        logger = Logger(episodes)
        agent = Agent(env, q_function, action_selector, logger)
        filename = str(seed)
        agent.train(steps, episodes, log_dir, filename)
print('Done')
