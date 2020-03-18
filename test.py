import gym, datetime
from agent import Agent
from q_function import Q
from epsilon_greedy import EpsilonGreedy
from mbie_eb import MBIE_EB
from boltzmann import Boltzmann
from logger import Logger
from env_wrapper import EnvWrapper
from discrete import Discrete
from action_space_wrapper import ActionWrapper
from env_handler import EnvHandler

# Parameters
learning_rate = 0.1
discount_factor = 0.9
episodes = 5000
epsilon = 1
epsilon_discount_factor = 0.9999
steps = 200
beta = 0.05
temperature = 1000000
num_states = [32, 11, 2]

log_dir = 'data/cart-pole'
date_string = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
filepath = log_dir + '/epsilon-greedy' + date_string
env_handler = EnvHandler()
env = env_handler.get_env('ReversedAddition3-v0')
# add discrete for non-discrete state spaces
discrete = Discrete(num_states, env)
# add action_wrapper for non-discrete action spaces
#action_wrapper = ActionWrapper(env.action_space.high[0], env.action_space.low[0], 100)
total_states = discrete.get_total_states()

q_function = Q(env.get_total_states(), env.get_total_actions(), learning_rate, discount_factor)
action_selector = EpsilonGreedy(epsilon, epsilon_discount_factor)
#action_selector = MBIE_EB(beta, total_states, env.action_space.n, discount_factor)
#action_selector = Boltzmann(temperature)
logger = Logger(episodes, 1)

agent = Agent(env, q_function, action_selector, logger)

agent.train(steps, episodes)

agent.solve(steps, filepath, True)
