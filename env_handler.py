import gym
from tape_env_wrapper import TapeEnvWrapper

class EnvHandler:
    def get_env(self, name):
        if name in ['Copy-v0', 'DuplicatedInput-v0', 'RepeatCopy-v0', 'Reverse-v0', 'ReversedAddition-v0', 'ReversedAddition3-v0']:
            return TapeEnvWrapper(gym.make(name))
