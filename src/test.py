import gym, datetime
from agent import Agent
from q_function import Q
from logger import Logger
from env_handler import EnvHandler
from action_selectors.epsilon_greedy import EpsilonGreedy
from action_selectors.mbie_eb import MBIE_EB
from action_selectors.boltzmann import Boltzmann
from action_selectors.ucb_1 import UCB_1
from action_selectors.controlability import Controlability
from action_selectors.vdbe import VDBE

# Parameters
learning_rate = 0.1
discount_factor = 0.9
episodes = 5000
epsilon = 1
epsilon_discount_factor = 0.9999
steps = 200
beta = 0.05
temperature = 1000000
env_name = 'Taxi-v3'
seed = 101
c = 10
beta_c = 0.5
omega = 0.5
inv_sens = 1 / 6
delta = 0.33

log_dir = 'data/' + env_name
date_string = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
filepath = log_dir + '/epsilon-greedy' + date_string
env_handler = EnvHandler()
env = env_handler.get_env(env_name)
env.seed(seed)

q_function = Q(env.get_total_states(), env.get_total_actions(), learning_rate, discount_factor)
#action_selector = EpsilonGreedy(epsilon, epsilon_discount_factor)
#action_selector = MBIE_EB(beta, env.get_total_states(), env.get_total_actions(), discount_factor)
action_selector = Boltzmann(temperature)
#action_selector = UCB_1(c, env.get_total_states(), env.get_total_actions(), discount_factor)
#action_selector = Controlability(beta_c, env.get_total_states(), env.get_total_actions(), learning_rate, omega)
#action_selector = VDBE(env.get_total_states(), delta, inv_sens, learning_rate)
logger = Logger(episodes)

agent = Agent(env, q_function, action_selector, logger)

agent.train(steps, episodes, filepath)

env.seed(seed)
agent.solve(steps, True)
agent.solve(steps, True)
env.seed(seed)
agent.solve(steps, True)
agent.solve(steps, True)

env.close()
