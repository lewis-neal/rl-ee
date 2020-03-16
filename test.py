import gym, datetime
from agent import Agent
from q_function import Q
from epsilon_greedy import EpsilonGreedy
from mbie_eb import MBIE_EB
from boltzmann import Boltzmann
from logger import Logger
from env_wrapper import EnvWrapper
from discrete import Discrete

# Parameters
learning_rate = 0.1
discount_factor = 0.9
episodes = 5000
epsilon = 1
epsilon_discount_factor = 0.9999
steps = 200
beta = 0.05
temperature = 1000000
num_states = [100, 0, 10, 0]

log_dir = 'data/n-chain'
date_string = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
filepath = log_dir + '/n-chain-epsilon-greedy' + date_string
env = gym.make('CartPole-v1')
discrete = Discrete(num_states, env)

env = EnvWrapper(env, num_states, discrete)
q_function = Q(1000, env.action_space.n, learning_rate, discount_factor)
action_selector = EpsilonGreedy(epsilon, epsilon_discount_factor)
#action_selector = MBIE_EB(beta, env.observation_space.n, env.action_space.n, discount_factor)
#action_selector = Boltzmann(temperature)
logger = Logger(episodes, 1)

agent = Agent(env, q_function, action_selector, logger)

agent.train(steps, episodes)

agent.solve(steps, filepath, False)
