import numpy as np

class GuessEnvWrapper:
    def __init__(self, env):
        self.__env = env
        self.__range = (env.action_space.high[0] - env.action_space.low[0]) / 2

    def reset(self):
        return self.__env.reset()

    def step(self, action):
        action = np.array([float(action - self.__range)])
        next_state, reward, done, info = self.__env.step(action)
        return next_state, reward, done, info

    def render(self):
        self.__env.render()

    def close(self):
        self.__env.close()

    def get_random_action(self):
        action = self.__env.action_space.sample()
        return int(round(action[0]) + self.__range)

    def get_total_actions(self):
        return int(self.__env.action_space.high[0] - self.__env.action_space.low[0]) + 1

    def get_total_states(self):
        return self.__env.observation_space.n

    def seed(self, seed=None, set_action_seed=True):
        if set_action_seed:
            self.__env.action_space.seed(seed)
        return self.__env.seed(seed)
