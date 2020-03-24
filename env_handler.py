import gym
from envs.tape_env_wrapper import TapeEnvWrapper
from envs.discrete_env_wrapper import DiscreteEnvWrapper
from envs.blackjack_env_wrapper import BlackjackEnvWrapper
from envs.guess_env_wrapper import GuessEnvWrapper
from envs.continuous_state_env_wrapper import ContinuousStateEnvWrapper
from envs.pendulum_env_wrapper import PendulumEnvWrapper

class EnvHandler:
    def get_env(self, name):
        if name in ['Copy-v0', 'DuplicatedInput-v0', 'RepeatCopy-v0', 'Reverse-v0', 'ReversedAddition-v0', 'ReversedAddition3-v0']:
            return TapeEnvWrapper(gym.make(name))

        if name in ['FrozenLake-v0', 'FrozenLake8x8-v0', 'NChain-v0', 'Roulette-v0', 'Taxi-v3']:
            return DiscreteEnvWrapper(gym.make(name))

        if name in ['Blackjack-v0']:
            return BlackjackEnvWrapper(gym.make(name))

        if name in ['GuessingGame-v0', 'HotterColder-v0']:
            return GuessEnvWrapper(gym.make(name))

        if name in ['MountainCar-v0']:
            return ContinuousStateEnvWrapper(gym.make(name), [36, 28])

        if name in ['CartPole-v1']:
            return ContinuousStateEnvWrapper(gym.make(name), [96, 0, 48, 0])

        if name in ['Acrobot-v1']:
            return ContinuousStateEnvWrapper(gym.make(name), [10, 10, 10, 10, 25, 57])

        if name in ['Pendulum-v0']:
            return PendulumEnvWrapper(gym.make(name), [10, 10, 16], [20])
