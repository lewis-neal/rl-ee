import gym, datetime
from agent import Agent
from q_function import Q
from epsilon_greedy import EpsilonGreedy
from mbie_eb import MBIE_EB
from boltzmann import Boltzmann
from logger import Logger
from env_handler import EnvHandler

# Parameters
learning_rate = 0.1
discount_factor = 0.9
episodes = 1000
epsilon = 1
epsilon_discount_factor = 0.9999
steps = 200
beta = 0.05
temperature = 1000000
env_name = 'Taxi-v3'

log_dir = 'data/' + env_name
date_string = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
filepath = log_dir + '/epsilon-greedy' + date_string
env_handler = EnvHandler()
env = env_handler.get_env(env_name)

q_function = Q(env.get_total_states(), env.get_total_actions(), learning_rate, discount_factor)
#action_selector = EpsilonGreedy(epsilon, epsilon_discount_factor)
action_selector = MBIE_EB(beta, env.get_total_states(), env.get_total_actions(), discount_factor)
#action_selector = Boltzmann(temperature)
logger = Logger(episodes)

agent = Agent(env, q_function, action_selector, logger)

agent.train(steps, episodes)

agent.solve(steps, filepath, False)
