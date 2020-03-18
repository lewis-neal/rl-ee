import gym
from tape_env_wrapper import TapeEnvWrapper
from discrete_env_wrapper import DiscreteEnvWrapper
from blackjack_env_wrapper import BlackjackEnvWrapper
from guess_env_wrapper import GuessEnvWrapper
from continuous_state_env_wrapper import ContinuousStateEnvWrapper

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
            return ContinuousStateEnvWrapper(gym.make(name), [10, 10])

        if name in ['CartPole-v1']:
            return ContinuousStateEnvWrapper(gym.make(name), [100, 0, 10, 0])

        if name in ['Acrobot-v1']:
            return ContinuousStateEnvWrapper(gym.make(name), [10, 10, 10, 10, 10, 10])
