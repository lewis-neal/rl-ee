import gym
from tape_env_wrapper import TapeEnvWrapper
from discrete_env_wrapper import DiscreteEnvWrapper
from blackjack_env_wrapper import BlackjackEnvWrapper

class EnvHandler:
    def get_env(self, name):
        if name in ['Copy-v0', 'DuplicatedInput-v0', 'RepeatCopy-v0', 'Reverse-v0', 'ReversedAddition-v0', 'ReversedAddition3-v0']:
            return TapeEnvWrapper(gym.make(name))

        if name in ['FrozenLake-v0', 'FrozenLake8x8-v0', 'NChain-v0', 'Roulette-v0', 'Taxi-v3']:
            return DiscreteEnvWrapper(gym.make(name))

        if name in ['Blackjack-v0']:
            return BlackjackEnvWrapper(gym.make(name))
